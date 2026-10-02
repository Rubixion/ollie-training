"""Face shapes of the celebrities in the live index, for the blog's "celebrities with X faces" posts.

Same measurement and classifier as the /ai-stylist face scan (ollie-frontend/lib/style/face-shape.ts: NORMS,
PROTOTYPES, classify), applied to each person's average ratios from face_norms.json (up to 3 straight-on photos).

    .venv\\Scripts\\python celeb_v2\\celeb_face_shapes.py   (from "neural network learning")  -> celeb_face_shapes.json
      + ollie-frontend/lib/style/celeb-shapes.json and public/style/celebs/<index row>.webp ("celebrities with your
        face shape" on /ai-stylist: entertainers and athletes only, each with one licensed photo and its credit)
"""
import json, math, os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Copied from lib/style/face-shape.ts; keep in sync.
NORMS = {"mean": [1.184, 0.933, 0.903, 0.578], "std": [0.058, 0.021, 0.02, 0.027]}
PROTOTYPES = {
    "oval": [0.4, 0, -0.2, -0.2],
    "round": [-1.2, 0, 0.4, 0.6],
    "square": [-0.6, 0.2, 1, 1],
    "oblong": [1.4, 0, 0.2, 0],
    "heart": [0, 1, -0.8, -1],
    "diamond": [0.2, -1, -0.6, -0.6],
    "triangle": [-0.2, -1, 1, 0.6],
}
CONFIDENT = 0.5  # examples named in posts must have at least this probability for their shape
OVAL_TOP = 0.25  # no one reaches CONFIDENT for oval (it sits in the middle of the others): use its most certain quarter
PER_GENDER = 8   # people per shape per gender on /ai-stylist

# /ai-stylist list: only people famous for entertainment or sport, never politics, business, religion or royalty
SHOW = {"actor", "film actor", "television actor", "stage actor", "voice actor", "character actor", "singer", "musician",
        "singer-songwriter", "rapper", "recording artist", "model", "comedian", "stand-up comedian", "dancer",
        "television presenter", "association football player", "basketball player", "tennis player", "American football player",
        "baseball player", "cricketer", "athlete", "boxer", "racing automobile driver", "professional wrestler", "guitarist"}
NEVER = {"politician", "businessperson", "entrepreneur", "lawyer", "jurist", "activist", "journalist", "economist", "diplomat",
         "military officer", "monarch", "aristocrat", "judge", "religious leader", "priest", "cleric", "podcaster", "manufacturer",
         "political activist", "trade unionist", "civil servant", "military personnel", "socialite"}
NEVER_WORDS = ("politician", "president", "prime minister", "minister", "prince", "princess", "king of", "queen of", "royal",
               "pope", "cardinal", "imam", "ayatollah", "senator", "governor", "activist", "businessman", "businesswoman", "ceo")
# Controversial for a style page (criminal cases, abuse allegations, hate speech or major public feuds). Takedowns go here too.
BLOCK = {"Kanye West", "Chris Brown", "Andrew Tate", "Sean Combs", "R. Kelly", "Harvey Weinstein", "Kevin Spacey", "Bill Cosby",
         "Marilyn Manson", "Ezra Miller", "Mel Gibson", "Woody Allen", "Roman Polanski", "Danny Masterson", "Armie Hammer",
         "Logan Paul", "Jake Paul", "O. J. Simpson", "Lance Armstrong", "Gina Carano", "Roseanne Barr", "Johnny Depp",
         "Amber Heard", "Tekashi 6ix9ine", "Tory Lanez", "DaBaby", "Morgan Wallen", "Russell Brand", "Conor McGregor",
         "Shia LaBeouf", "Jim Caviezel", "Benjamin Mendy", "Mason Greenwood", "Dani Alves", "Trevor Bauer", "Michael Jackson", "Gary Glitter", "Phil Spector"}


def classify(r):
    z = [(v - NORMS["mean"][i]) / NORMS["std"][i] for i, v in enumerate(r)]
    w = {s: math.exp(-sum((z[i] - p[i]) ** 2 for i in range(4)) / 2) for s, p in PROTOTYPES.items()}
    total = sum(w.values()) or 1
    probs = {s: v / total for s, v in w.items()}
    shape = max(probs, key=probs.get)
    return shape, probs


def main():
    norms = json.load(open(os.path.join(HERE, "face_norms.json"), encoding="utf8"))["people"]
    celebs = {c["qid"]: c for c in json.load(open(os.path.join(HERE, "celebs.json"), encoding="utf8"))}
    live = set(np.load(os.path.join(HERE, "..", "hf_space", "index.npz"), allow_pickle=True)["names"].tolist())

    rows = []
    for key, ratios in norms.items():
        qid = key.rsplit("_", 1)[-1]
        c = celebs.get(qid)
        if not c or c["name"] not in live:
            continue
        shape, probs = classify(ratios)
        rows.append({"name": c["name"], "rank": c["rank"], "gender": c.get("gender"), "description": c.get("description", ""),
                     "shape": shape, "p": round(probs[shape], 3), "ratios": ratios})

    n = len(rows)
    out = {"n": n, "live_index_people": len(live), "shares": {}, "by_gender": {}, "examples": {}}
    for s, k in Counter(r["shape"] for r in rows).most_common():
        out["shares"][s] = {"count": k, "pct": round(100 * k / n, 1)}
    for g in ("M", "F"):
        sub = [r for r in rows if r["gender"] == g]
        out["by_gender"][g] = {"n": len(sub), **{s: round(100 * k / len(sub), 1) for s, k in Counter(r["shape"] for r in sub).most_common()}}
    for s in PROTOTYPES:
        sure = sorted((r for r in rows if r["shape"] == s and r["p"] >= CONFIDENT), key=lambda r: r["rank"])
        out["examples"][s] = [{k: r[k] for k in ("name", "rank", "gender", "p", "description")} for r in sure[:40]]
        out["shares"].setdefault(s, {"count": 0, "pct": 0})["confident"] = sum(1 for r in rows if r["shape"] == s and r["p"] >= CONFIDENT)
    json.dump(out, open(os.path.join(HERE, "celeb_face_shapes.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    stylist_list(rows, celebs_by_name={c["name"]: c for c in celebs.values()})
    print(n, "people classified of", len(live), "in the live index")
    print(json.dumps(out["shares"]), "\n", json.dumps(out["by_gender"]))


def shown(c):
    occ = set(c.get("occupations", []))
    desc = c.get("description", "").lower()
    return c["name"] not in BLOCK and occ & SHOW and not occ & NEVER and not any(w in desc for w in NEVER_WORDS)


def stylist_list(rows, celebs_by_name):
    """celeb-shapes.json + thumbnails for /ai-stylist. Photo = the person's most typical one in the index (closest to
    their other photos), the same thumbnail and credit the lookalike finder shows."""
    from PIL import Image
    root = os.path.join(HERE, "..", "..", "ollie-frontend")
    d = np.load(os.path.join(HERE, "..", "hf_space", "index.npz"))
    emb, names, credits = d["embeddings"], d["names"], d["credits"]
    out, imgs = {}, os.path.join(root, "public", "style", "celebs")
    os.makedirs(imgs, exist_ok=True)
    oval_cut = np.quantile([r["p"] for r in rows if r["shape"] == "oval"], 1 - OVAL_TOP)
    for s in PROTOTYPES:
        sure = [r for r in rows if r["shape"] == s and r["p"] >= (oval_cut if s == "oval" else CONFIDENT)
                and shown(celebs_by_name.get(r["name"], {"name": r["name"]}))]
        people = []
        for g in ("M", "F"):
            for r in sorted((r for r in sure if r["gender"] == g), key=lambda r: r["rank"])[:PER_GENDER]:
                mine = np.flatnonzero(names == r["name"])
                row = int(mine[np.argmax((emb[mine] @ emb[mine].T).sum(1))]) if len(mine) > 1 else int(mine[0])
                Image.open(os.path.join(HERE, "..", "hf_space", "imgthumbs", f"{row}.jpg")).convert("RGB").save(os.path.join(imgs, f"{row}.webp"), quality=82)
                people.append({"name": r["name"], "gender": g, "p": r["p"], "knownFor": r["description"], "img": row,
                               "credit": json.loads(credits[row])})
        out[s] = people
    json.dump(out, open(os.path.join(root, "lib", "style", "celeb-shapes.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("celeb-shapes.json:", {s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    main()
