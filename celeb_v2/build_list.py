"""
Builds celebs.json: the most famous living adults, ranked.

Fame = English Wikipedia views over the last 6 full months (current fame), boosted by how many
language Wikipedias cover the person (global reach):
    score = log10(views_6mo) + 0.5 * log10(sitelinks)

Candidates: every living person with >= 10 Wikipedia language editions and an English article
(Wikidata via QLever). Views come from Wikimedia's monthly clickstream dumps (every article's
traffic in one file), per Wikimedia's robot policy: bulk data from dumps, not live API calls.
Kept: humans, alive, 18+ (birth date must be known; an imprecise date counts as the latest
possible day, so nobody under 18 slips through), at least one occupation, and not someone
known for porn or for serious crimes (occupation or conviction; see EXCLUDE_*).

Data: Wikidata (CC0) and Wikimedia clickstream (CC0). Everything is cached in cache/
(~2.8 GB of dumps), so reruns are quick and an interrupted run resumes.

    celeb_v2\\.venv\\Scripts\\python build_list.py
"""
import datetime as dt
import gzip
import json
import math
import os
import re
from urllib.parse import unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "celebs.json")
UA = "OllieCelebIndex/2.0 (https://ollie.ml)"
QLEVER = "https://qlever.dev/api/wikidata"
MONTHS = 6           # fame window
KEEP = 12000         # ranked list length: 5000 target + backfill for people without enough licensed photos
SL_MIN = 10          # language editions needed to be a candidate (filters one-off news subjects)

# Whole words only ("rapist" must not hit "therapist"). Porn is judged by the description, i.e. what the
# person is known for: Wikidata tags many mainstream stars "erotic photography model" for one photo shoot.
CRIME_WORDS = re.compile(r"\b(criminal|murderer|serial killer|spree killer|mass murderer|terrorist|drug lord|"
                         r"drug trafficker|gangster|mobster|hitman|assassin|rapist|serial rapist|sex offender|"
                         r"fraudster|con artist|scammer|convicted)\b")
PORN_WORDS = re.compile(r"\b(porn\w*|adult (film|content|entertainment|performer|actress|actor|model))\b")
BAD_CONVICTIONS = re.compile(r"\b(murder|homicide|manslaughter|rape|sexual|child|terror\w*|genocide|war crimes?|"
                             r"crimes against humanity|kidnapping)\b|\b(human|sex|child\w*|people) trafficking|"
                             r"trafficking of (children|persons|women)")
# no occupation on Wikidata is fine if the description shows public fame (Kris Jenner, Jay Shah)
FAME_WORDS = re.compile(r"\b(personality|business\w*|administrator|actor|actress|singer|rapper|model|socialite|"
                        r"influencer|presenter|host|athlete|player|politician|royal|prince|princess|king|queen)\b")

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    retry = Retry(total=8, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  respect_retry_after_header=True, allowed_methods=None)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


S = session()


def cached(name, fn):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    data = fn()
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return data


def sparql(query):
    # POST: queries with long VALUES lists don't fit in a URL
    r = S.post(QLEVER, data={"query": PREFIXES + query}, timeout=300,
               headers={"Accept": "application/sparql-results+json"})
    r.raise_for_status()
    return [{k: v["value"] for k, v in row.items()} for row in r.json()["results"]["bindings"]]


def qid(iri):
    return iri.rsplit("/", 1)[1]


def living_people():
    """Every living human with >= SL_MIN language editions and an English Wikipedia article."""
    rows = sparql(f"""SELECT ?item ?sl ?article WHERE {{
      ?item wdt:P31 wd:Q5 ; wikibase:sitelinks ?sl . FILTER(?sl >= {SL_MIN})
      MINUS {{ ?item wdt:P570 ?died }}
      ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
    }}""")
    return {qid(r["item"]): {"sl": int(r["sl"]),
                             "title": unquote(r["article"].split("/wiki/", 1)[1])} for r in rows}


def last_months(n):
    first = dt.date.today().replace(day=1)
    months = []
    for _ in range(n):
        first = (first - dt.timedelta(days=1)).replace(day=1)
        months.append(f"{first:%Y-%m}")
    return months[::-1]


def month_views(month, titles):
    """Views per title for one month, from the enwiki clickstream dump (sum over all referrers)."""
    path = os.path.join(CACHE, "clickstream", f"clickstream-enwiki-{month}.tsv.gz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://dumps.wikimedia.org/other/clickstream/{month}/clickstream-enwiki-{month}.tsv.gz"
        print(f"  downloading {url}", flush=True)
        with S.get(url, stream=True, timeout=120) as r:  # one download at a time, per the robot policy
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(path + ".part", path)
    counts = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4 and parts[1] in titles:
                counts[parts[1]] = counts.get(parts[1], 0) + int(parts[3])
    print(f"  {month}: views for {len(counts)} of {len(titles)} candidates", flush=True)
    return counts


def details(qids):
    """Birth date (+precision), photo, Commons category, label, description, occupations, convictions."""
    out = {q: {"births": [], "images": set(), "cats": set(), "label": "", "desc": "",
               "occupations": set(), "convictions": set(), "native": set()} for q in qids}
    for i in range(0, len(qids), 500):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 500])
        for r in sparql(f"""SELECT ?item ?birth ?prec ?img ?cat ?commons ?label ?desc WHERE {{
          VALUES ?item {{ {values} }}
          OPTIONAL {{ ?item p:P569/psv:P569 ?bv . ?bv wikibase:timeValue ?birth ; wikibase:timePrecision ?prec }}
          OPTIONAL {{ ?item wdt:P18 ?img }}
          OPTIONAL {{ ?item wdt:P373 ?cat }}
          OPTIONAL {{ ?commons schema:about ?item ; schema:isPartOf <https://commons.wikimedia.org/> }}
          OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
          OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc) = "en") }}
        }}"""):
            d = out[qid(r["item"])]
            if "birth" in r:
                d["births"].append((r["birth"], int(r.get("prec", 9))))
            if "img" in r:
                d["images"].add(unquote(r["img"].rsplit("/", 1)[1]))
            if "cat" in r:
                d["cats"].add(r["cat"])
            if "commons" in r and "/wiki/Category:" in r["commons"]:
                d["cats"].add(unquote(r["commons"].split("/wiki/Category:", 1)[1]).replace("_", " "))
            d["label"] = d["label"] or r.get("label", "")
            d["desc"] = d["desc"] or r.get("desc", "")
        for r in sparql(f"""SELECT ?item ?kind ?lab WHERE {{
          VALUES ?item {{ {values} }}
          {{ ?item wdt:P106 ?x . BIND("occ" AS ?kind) }} UNION {{ ?item wdt:P1399 ?x . BIND("conv" AS ?kind) }}
          ?x rdfs:label ?lab FILTER(LANG(?lab) = "en")
        }}"""):
            out[qid(r["item"])]["occupations" if r["kind"] == "occ" else "convictions"].add(r["lab"])
        # "name in native language": Commons often titles e.g. K-pop or Bollywood photos in that script
        for r in sparql(f"""SELECT ?item ?n WHERE {{ VALUES ?item {{ {values} }} ?item wdt:P1559 ?n }}"""):
            out[qid(r["item"])]["native"].add(r["n"])
        print(f"  details {min(i + 500, len(qids))}/{len(qids)}", flush=True)
    return out


def latest_birth(births):
    """Latest possible birth date across all statements, honouring precision (9=year, 10=month, 11=day)."""
    latest = None
    for value, prec in births:
        if value.startswith("-"):
            continue
        try:
            y, m, d = int(value[:4]), max(int(value[5:7]), 1), max(int(value[8:10]), 1)
            if prec <= 9:
                day = dt.date(y, 12, 31)
            elif prec == 10:
                day = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
            else:
                day = dt.date(y, m, d)
        except ValueError:
            continue
        latest = day if latest is None or day > latest else latest
    return latest


def age_on(born, today):
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def exclusion(d, today):
    born = latest_birth(d["births"])
    desc = d["desc"].lower()
    occs = " | ".join(d["occupations"]).lower()
    if born is None:
        return "no birth date"
    if age_on(born, today) < 18:
        return "under 18"
    if age_on(born, today) > 110:
        return "not a living person"  # legendary/ancient figures Wikidata lists without a death date
    if re.search(r"\b(anonymous|unidentified|unknown)\b", desc):
        return "no known identity"
    if not d["occupations"] and not FAME_WORDS.search(desc):
        return "no occupation"
    if PORN_WORDS.search(desc) or (not desc and PORN_WORDS.search(occs)):
        return f"adult industry: {d['desc']}"
    for text in d["occupations"] + [d["desc"]]:
        if CRIME_WORDS.search(text.lower()):
            return f"crime: {text}"
    for c in d["convictions"]:
        if BAD_CONVICTIONS.search(c.lower()):
            return f"convicted: {c}"
    return None


def _check():
    """The under-18 filter is a safety rule: imprecise dates must resolve to the LATEST possible day."""
    t = dt.date(2026, 9, 23)
    assert latest_birth([("2008-01-01T00:00:00Z", 9)]) == dt.date(2008, 12, 31)   # year only
    assert age_on(latest_birth([("2008-01-01T00:00:00Z", 9)]), t) == 17           # -> still a minor
    assert latest_birth([("2008-02-00T00:00:00Z", 10)]) == dt.date(2008, 2, 29)   # month only
    assert age_on(dt.date(2008, 9, 23), t) == 18 and age_on(dt.date(2008, 9, 24), t) == 17
    assert latest_birth([("1964-01-23T00:00:00Z", 11), ("1964-06-23T00:00:00Z", 11)]) == dt.date(1964, 6, 23)
    adult = [("1990-05-05T00:00:00Z", 11)]
    person = lambda occ, desc="", conv=(): {"births": adult, "occupations": list(occ), "desc": desc,
                                            "convictions": list(conv)}
    assert exclusion({"births": [], "occupations": ["actor"], "desc": "", "convictions": []}, t) == "no birth date"
    assert exclusion(person(["politician", "hypnotherapist"], "British politician"), t) is None   # not "rapist"
    assert exclusion(person(["erotic photography model", "actor"], "American actress (born 1997)"), t) is None
    assert exclusion(person(["pornographic film actor"], "American pornographic actress"), t).startswith("adult")
    assert exclusion(person(["actor"], "actress and former pornographic actress"), t).startswith("adult")
    assert exclusion(person(["actor"], "American actor", ["drug trafficking"]), t) is None
    assert exclusion(person(["singer"], "American singer", ["sex trafficking"]), t).startswith("convicted")
    assert exclusion(person([], "American media personality, socialite, and businesswoman"), t) is None
    assert exclusion(person([], "Ukrainian-born American"), t) == "no occupation"
    assert exclusion(person(["activist"], "anonymous man who stood in front of tanks"), t) == "no known identity"
    assert exclusion(person(["criminal"], "Syrian politician"), t).startswith("crime")
    assert last_months(2)[-1] == f"{(dt.date.today().replace(day=1) - dt.timedelta(days=1)):%Y-%m}"


def main():
    _check()
    os.makedirs(CACHE, exist_ok=True)
    today = dt.date.today()

    people = cached("living_people.json", living_people)
    print(f"living people with >= {SL_MIN} language editions + enwiki: {len(people)}", flush=True)

    months = last_months(MONTHS)
    titles = {p["title"] for p in people.values()}
    views_by_title = {}
    for month in months:
        for t, n in cached(f"views_{month}.json", lambda m=month: month_views(m, titles)).items():
            views_by_title[t] = views_by_title.get(t, 0) + n
    views = {q: views_by_title.get(p["title"], 0) for q, p in people.items()}

    def score(q):
        return math.log10(views[q] + 1) + 0.5 * math.log10(people[q]["sl"])

    ranked = sorted(people, key=score, reverse=True)[:int(KEEP * 1.6)]
    info = cached(f"details_{len(ranked)}.json", lambda: {
        q: {**d, "images": sorted(d["images"]), "cats": sorted(d["cats"]), "native": sorted(d["native"]),
            "occupations": sorted(d["occupations"]), "convictions": sorted(d["convictions"])}
        for q, d in details(ranked).items()})

    celebs, dropped = [], {}
    for q in ranked:
        d = info[q]
        why = exclusion(d, today)
        if why:
            dropped[why.split(":")[0]] = dropped.get(why.split(":")[0], 0) + 1
            continue
        title = people[q]["title"]
        celebs.append({
            "rank": len(celebs) + 1, "qid": q, "name": d["label"] or title.replace("_", " ").split(" (")[0],
            "description": d["desc"], "occupations": d["occupations"],
            "age": age_on(latest_birth(d["births"]), today), "sitelinks": people[q]["sl"],
            f"views_{MONTHS}mo": views[q], "score": round(score(q), 4), "enwiki": title,
            "images": d["images"], "commons_categories": d["cats"],
            "native_names": [n for n in d["native"] if n.lower() != (d["label"] or title).lower()][:2],
        })
        if len(celebs) == KEEP:
            break

    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(celebs, f, ensure_ascii=False, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"\nwrote {len(celebs)} people to celebs.json (views {months[0]}..{months[-1]}); dropped: {dropped}")
    for c in celebs[:30] + celebs[4990:5005]:
        print(f"{c['rank']:5d}  {c['name'][:30]:30s} {c[f'views_{MONTHS}mo']:>11,} views  {c['sitelinks']:3d} langs  "
              f"{', '.join(c['occupations'][:2])[:45]}")


if __name__ == "__main__":
    main()
