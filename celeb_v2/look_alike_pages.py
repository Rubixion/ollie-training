"""Data for the /look-alike/<slug> pages: for each chosen celebrity, the celebrities the model puts closest.

Same pair score as the "celebrities who look alike" blog post: every photo of A against every photo of B,
the two closest photo pairs averaged, mapped to a percent like the live finder (1 - distance / 2).
Same gender only, like the finder's default.

    python celeb_v2/look_alike_pages.py   (from "neural network learning")
      -> ollie-frontend/lib/look-alike-data.json + ollie-frontend/public/look-alike/img/<index>.webp

PAGES is the list of people to build, picked by US search volume for "<name> look alike"
(DataForSEO via OpenSEO, 2026-10-01; >= 260 searches/month, politicians left out).
"""
import json, os, re, unicodedata

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
INDEX = os.path.join(HERE, "..", "hf_space", "index.npz")
THUMBS = os.path.join(HERE, "..", "hf_space", "imgthumbs")
OUT_JSON = os.path.join(ROOT, "ollie-frontend", "lib", "look-alike-data.json")
OUT_IMG = os.path.join(ROOT, "ollie-frontend", "public", "look-alike", "img")
TOP = 8

# name -> monthly US searches for "<name> look alike"
PAGES = json.load(open(os.path.join(HERE, "look_alike_pages.json"), encoding="utf-8"))


def slug(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def main():
    d = np.load(INDEX)
    emb, names, genders = d["embeddings"], d["names"], d["genders"]
    known, cats, credits = d["known_for"], d["categories"], d["credits"]
    people, inv = np.unique(names, return_inverse=True)
    photos_of = {p: np.flatnonzero(inv == i) for i, p in enumerate(people)}

    pages, used = [], set()
    for name, volume in PAGES.items():
        if name not in photos_of:
            print("not in index, skipped:", name)
            continue
        mine = photos_of[name]
        g = genders[mine[0]]
        dist = np.sqrt(np.maximum(0, 2 - 2 * emb[mine] @ emb.T)) if np.allclose(np.linalg.norm(emb[:5], axis=1), 1) \
            else np.linalg.norm(emb[mine][:, None] - emb[None], axis=2)
        # The two closest pairs between two people are among each photo's two best rows, so keep only those
        two = np.sort(dist, axis=0)[:2]                       # 2 x N
        rows = np.lexsort((two.ravel(), np.tile(inv, 2)))      # grouped by person, closest first
        who, val = np.tile(inv, 2)[rows], two.ravel()[rows]
        first = np.r_[True, who[1:] != who[:-1]]
        start = np.flatnonzero(first)
        pair = (val[start] + val[np.minimum(start + 1, len(val) - 1)]) / 2
        col = np.tile(np.arange(len(names)), 2)[rows][start]  # each person's closest photo (index into the index)
        matches = []
        for k in np.argsort(pair):
            other = people[who[start][k]]
            if other == name or genders[col[k]] != g:
                continue
            matches.append({
                "name": str(other),
                "slug": slug(other),
                "knownFor": str(known[col[k]]),
                "score": round(float(max(0, 1 - pair[k] / 2) * 100), 1),
                "img": int(col[k]),
                "credit": json.loads(credits[col[k]]),
            })
            if len(matches) == TOP:
                break
        # Their own photo: the one closest to their own other photos (most typical), else the first
        own = mine[np.argmin(dist[:, mine].sum(axis=0))] if len(mine) > 1 else mine[0]
        pages.append({
            "name": name,
            "slug": slug(name),
            "volume": volume,
            "knownFor": str(known[own]),
            "category": str(cats[own]),
            "gender": str(g),
            "photos": int(len(mine)),
            "img": int(own),
            "credit": json.loads(credits[own]),
            "matches": matches,
        })
        used.update([int(own)] + [m["img"] for m in matches])

    # Who else lists this person in their top matches (for "also looks like" links between pages)
    by_slug = {p["slug"]: p for p in pages}
    for p in pages:
        p["alsoIn"] = [q["slug"] for q in pages if q is not p and any(m["slug"] == p["slug"] for m in q["matches"])]
        for m in p["matches"]:
            m["hasPage"] = m["slug"] in by_slug

    os.makedirs(OUT_IMG, exist_ok=True)
    for i in sorted(used):
        out = os.path.join(OUT_IMG, f"{i}.webp")
        if not os.path.exists(out):
            Image.open(os.path.join(THUMBS, f"{i}.jpg")).save(out, "WEBP", quality=82)
    json.dump({"pages": pages}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(pages)} pages, {len(used)} images -> {OUT_JSON}")


if __name__ == "__main__":
    main()
