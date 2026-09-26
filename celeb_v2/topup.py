"""
Builds topup.json: a regional top-up for celebs.json (agreed 2026-09-24). celebs.json ranks by English
Wikipedia views, so Asian, Latin American, African and Middle Eastern celebrities are badly under-represented.
Here each region is ranked by views on its OWN languages' Wikipedias, about half women each:

    score = log10(views_6mo on the region's wikis) + 0.5 * log10(sitelinks)     (same formula as build_list)

Candidates: living humans with a citizenship (P27) in the region, >= SL_MIN language editions and an article on
one of the region's wikis. Only the top few per gender by sitelinks get their views looked up. Clickstream dumps
don't exist for ko/hi/ar/th..., so views come from the Pageviews REST API, at most 3 requests/second.
Same exclusions as build_list.py (18+, adult industry, serious crime). Anyone already in data/ is left out.

    .venv\\Scripts\\python topup.py
    .venv\\Scripts\\python collect.py --list topup.json
"""
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import threading
import time
from urllib.parse import quote, unquote
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_list import (CACHE, HERE, S, age_on, cached, details, exclusion, latest_birth, last_months, qid,
                        sparql, GENDER_QIDS, MONTHS)

OUT = os.path.join(HERE, "topup.json")
SL_MIN = 5    # lower than build_list's 10: regional stars are on fewer wikis
POOL = 4      # views looked up for POOL x the target per gender, by sitelinks

A = lambda *q: " ".join(f"wd:{x}" for x in q)
REGIONS = {  # name: (target, countries as a SPARQL pattern binding ?country, wikis)
    "East Asia": (400, f"VALUES ?country {{ {A('Q148', 'Q17', 'Q884', 'Q423', 'Q865', 'Q8646', 'Q14773', 'Q711')} }}",
                  "zh ja ko mn"),
    "South Asia": (400, f"VALUES ?country {{ {A('Q668', 'Q843', 'Q902', 'Q854', 'Q837', 'Q889', 'Q917', 'Q826')} }}",
                   "hi ur bn ta te ml mr kn gu pa si ne ps"),
    "Latin America": (300, f"VALUES ?country {{ {A('Q96', 'Q155', 'Q414', 'Q739', 'Q298', 'Q419', 'Q717', 'Q241', 'Q736', 'Q750', 'Q77', 'Q733', 'Q786', 'Q1183', 'Q774', 'Q800', 'Q804', 'Q783', 'Q792', 'Q811')} }}",
                      "es pt"),
    "Southeast Asia": (200, f"VALUES ?country {{ {A('Q252', 'Q928', 'Q881', 'Q869', 'Q833', 'Q334', 'Q836', 'Q424', 'Q819', 'Q921', 'Q574')} }}",
                       "id tl vi th ms my km lo"),
    # Africa minus Egypt (in Middle East); English/French are many African countries' own wikis
    "Africa": (200, "?country wdt:P31 wd:Q6256 ; wdt:P30 wd:Q15 . FILTER(?country != wd:Q79)",
               "en fr ar pt sw af am ha yo ig zu so"),
    "Middle East": (200, f"VALUES ?country {{ {A('Q851', 'Q878', 'Q794', 'Q796', 'Q801', 'Q43', 'Q79', 'Q810', 'Q822', 'Q858', 'Q817', 'Q846', 'Q398', 'Q842', 'Q805', 'Q219060')} }}",
                    "ar fa he tr"),
}


def candidates(countries, wikis):
    """qid -> {sl, gender, titles: {lang: article title}}"""
    rows = sparql(f"""SELECT ?item ?sl ?article ?wiki ?g WHERE {{
      {countries}
      ?item wdt:P27 ?country ; wdt:P31 wd:Q5 ; wikibase:sitelinks ?sl .  # sl filtered below: QLever's FILTER on it returns nothing (2026-09)
      MINUS {{ ?item wdt:P570 ?died }}
      VALUES ?wiki {{ {" ".join(f"<https://{w}.wikipedia.org/>" for w in wikis.split())} }}
      ?article schema:about ?item ; schema:isPartOf ?wiki .
      OPTIONAL {{ ?item wdt:P21 ?g }}
    }}""")
    out = {}
    for r in rows:
        if int(r["sl"]) < SL_MIN:
            continue
        d = out.setdefault(qid(r["item"]), {"sl": int(r["sl"]), "g": set(), "titles": {}})
        d["titles"][r["wiki"].split("//")[1].split(".")[0]] = unquote(r["article"].split("/wiki/", 1)[1])
        if "g" in r:
            d["g"].add(GENDER_QIDS.get(qid(r["g"]), "?"))
    for d in out.values():  # same rule as build_list.genders(): one clear F/M, else ""
        g = d.pop("g")
        d["gender"] = next(iter(g)) if len(g) == 1 and g != {"?"} else ""
    return out


_gate = threading.Lock()
_last = [0.0]


def views(lang, title, months):
    with _gate:  # Wikimedia's limit is 200 requests/minute for an identified client: stay at 180
        time.sleep(max(0.0, _last[0] + 1 / 3 - time.monotonic()))
        _last[0] = time.monotonic()
    start, end = months[0].replace("-", "") + "0100", months[-1].replace("-", "") + "2800"
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/"
           f"{quote(title, safe='')}/monthly/{start}/{end}")
    r = S.get(url, timeout=30)
    return sum(i["views"] for i in r.json().get("items", [])) if r.ok else 0


def pick(target, people, info, today, taken):
    """Top target/2 women and target/2 men (people with no recorded gender fill in), best score first."""
    ok = [q for q in sorted(people, key=lambda q: people[q]["score"], reverse=True)
          if q not in taken and q in info and not exclusion(info[q], today)]
    women = [q for q in ok if people[q]["gender"] == "F"][:target // 2]
    rest = [q for q in ok if q not in set(women)][:target - len(women)]
    return women + rest


def main():
    today = dt.date.today()
    months = last_months(MONTHS)
    taken = {name.rsplit("_", 1)[-1] for name in os.listdir(os.path.join(HERE, "data"))}  # folders end in _QID
    out = []
    for region, (target, countries, wikis) in REGIONS.items():
        key = region.lower().replace(" ", "_")
        people = cached(f"topup_{key}_candidates.json", lambda: candidates(countries, wikis))
        pool = []
        for g in ("F", "M", ""):
            same = sorted((q for q, d in people.items() if d["gender"] == g and q not in taken),
                          key=lambda q: people[q]["sl"], reverse=True)
            pool += same[:POOL * target // (2 if g else 8)]
        print(f"{region}: {len(people)} candidates, views for {len(pool)}", flush=True)

        def lookup():  # a few in flight, paced by views()
            with cf.ThreadPoolExecutor(3) as ex:
                got = list(ex.map(lambda q: sum(views(lang, t, months) for lang, t in people[q]["titles"].items()), pool))
            return dict(zip(pool, got))
        seen = cached(f"topup_{key}_views_{months[-1]}.json", lookup)
        people = {q: {**people[q], "views": seen[q],
                      "score": round(math.log10(seen[q] + 1) + 0.5 * math.log10(people[q]["sl"]), 4)} for q in pool}
        info = cached(f"topup_{key}_details.json", lambda: {
            q: {**d, "images": sorted(d["images"]), "cats": sorted(d["cats"]), "native": sorted(d["native"]),
                "occupations": sorted(d["occupations"]), "convictions": sorted(d["convictions"])}
            for q, d in details(list(pool)).items()})
        chosen = pick(target, people, info, today, taken)
        taken |= set(chosen)
        for q in chosen:
            d, p = info[q], people[q]
            native = [t.replace("_", " ").split(" (")[0] for lang, t in p["titles"].items() if lang != "en"]
            name = d["label"] or (native or [p["titles"].get("en", q).replace("_", " ")])[0]
            out.append({
                "rank": len(out) + 1, "qid": q, "name": name, "region": region,
                "description": d["desc"], "occupations": d["occupations"],
                "age": age_on(latest_birth(d["births"]), today), "sitelinks": p["sl"],
                f"views_{MONTHS}mo": p["views"], "score": p["score"], "enwiki": p["titles"].get("en", ""),
                "images": d["images"], "commons_categories": d["cats"],
                "native_names": list(dict.fromkeys(n for n in native + d["native"] if n.lower() != name.lower()))[:2],
                "gender": p["gender"],
            })
        women = sum(people[q]["gender"] == "F" for q in chosen)
        print(f"  picked {len(chosen)} ({women} women): " + ", ".join(info[q]["label"] or q for q in chosen[:8]),
              flush=True)

    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {len(out)} people to topup.json")


if __name__ == "__main__":
    main()
