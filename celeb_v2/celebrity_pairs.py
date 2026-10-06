"""Numbers for the blog post /blog/celebrities-who-look-alike: every celebrity compared with every other.

Same pair score as look_alike_pages.py: every photo of A against every photo of B, the two closest photo pairs
averaged, raw = (1 - distance / 2) * 100, shown on the finder's scale (shown(), lib/look-alike.ts). Rerun after any
index rebuild and update the post by hand from the printout.

    python celeb_v2/celebrity_pairs.py   (from "neural network learning")
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "..", "hf_space", "index.npz")
RAW_LO, RAW_HI, OUT_LO, OUT_HI = 30, 41.7, 20, 80  # keep in sync with lib/look-alike.ts


def shown(raw):
    return np.clip(np.round(OUT_LO + (raw - RAW_LO) * (OUT_HI - OUT_LO) / (RAW_HI - RAW_LO)), 0, 99)


def main():
    d = np.load(INDEX)
    emb, names, genders, known, cats = d["embeddings"], d["names"], d["genders"], d["known_for"], d["categories"]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    people, inv = np.unique(names, return_inverse=True)
    P = len(people)
    first = np.array([np.flatnonzero(inv == i)[0] for i in range(P)])
    print(f"{P} people, {len(names)} photos, {P * (P - 1) // 2:,} pairs")

    # raw[p, q]: mean of the two closest photo pairs between person p and person q
    raw = np.zeros((P, P), np.float32)
    order = np.argsort(inv, kind="stable")
    for p in range(P):
        mine = np.flatnonzero(inv == p)
        dist = np.sqrt(np.maximum(0, 2 - 2 * emb[mine] @ emb.T))  # n_mine x N
        two = np.sort(dist, axis=0)[:2]                           # each photo's two best distances to p's photos
        # per other person q: the two smallest values among their photos' columns (both rows)
        vals = np.concatenate([two[0], two[1] if len(mine) > 1 else np.full(len(names), np.inf)])
        who = np.concatenate([inv, inv])
        srt = np.lexsort((vals, who))
        w, v = who[srt], vals[srt]
        start = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
        b = np.minimum(start + 1, len(v) - 1)
        second = np.where(w[b] == w[start], v[b], v[start])
        raw[p, w[start]] = (1 - (v[start] + second) / 2 / 2) * 100
    iu = np.triu_indices(P, 1)
    pr = raw[iu]
    print(f"random pair: median raw {np.median(pr):.1f} -> shown {shown(np.median(pr)):.0f}%; "
          f"top 1%: raw {np.percentile(pr, 99):.1f} -> shown {shown(np.percentile(pr, 99)):.0f}%")

    def label(i):
        f = first[i]
        return f"{people[i]} ({known[f]}; {cats[f]}; {genders[f]})"

    top = np.argsort(pr)[::-1][:300]
    print("\n== closest 300 pairs ==")
    for k, t in enumerate(top):
        a, b = iu[0][t], iu[1][t]
        print(f"{k + 1:3d}. {shown(pr[t]):.0f}% raw {pr[t]:.1f}  {label(a)}  |  {label(b)}")

    # who appears most in the closest 1,000 pairs
    top1k = np.argsort(pr)[::-1][:1000]
    cnt = np.bincount(np.concatenate([iu[0][top1k], iu[1][top1k]]), minlength=P)
    print("\n== most frequent in the closest 1,000 pairs ==")
    for i in np.argsort(cnt)[::-1][:15]:
        print(f"{cnt[i]:3d}  {label(i)}")

    # closest man-woman pair
    g = np.array([genders[f] for f in first])
    mixed = np.flatnonzero((g[iu[0]] != g[iu[1]]) & np.isin(g[iu[0]], ["M", "F"]) & np.isin(g[iu[1]], ["M", "F"]))
    print("\n== closest man-woman pairs ==")
    for t in mixed[np.argsort(pr[mixed])[::-1][:5]]:
        print(f"{shown(pr[t]):.0f}%  {label(iu[0][t])}  |  {label(iu[1][t])}")

    # most 'average' faces: closest to the centre of all per-person mean embeddings
    means = np.stack([emb[inv == i].mean(0) for i in range(P)])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    c = means.mean(0)
    c /= np.linalg.norm(c)
    print("\n== closest to the average face ==")
    for i in np.argsort(means @ c)[::-1][:10]:
        print(label(i))

    # pairs the old post named: where are they now?
    rank = np.empty(len(pr), int)
    rank[np.argsort(pr)[::-1]] = np.arange(1, len(pr) + 1)
    idx = {n: i for i, n in enumerate(people)}
    print("\n== pairs named in the post (rank of", f"{len(pr):,})")
    for a, b in [("Dakota Fanning", "Sadie Sink"), ("Amanda Seyfried", "Dakota Fanning"), ("Winona Ryder", "Rachael Leigh Cook"),
                 ("Mila Kunis", "Camila Mendes"), ("Kate Bosworth", "Danielle Panabaker"), ("Carrie Underwood", "Emily Osment"),
                 ("Katrina Kaif", "Zarine Khan"), ("Kareena Kapoor", "Kritika Kamra"), ("Alia Bhatt", "Nidhhi Agerwal"),
                 ("Rory McIlroy", "Matthew Fitzpatrick"), ("Alexander Zverev", "Aryna Sabalenka"), ("Akihito", "Naruhito"),
                 ("Naruhito", "Fumihito"), ("Mako Komuro", "Prince Hisahito"), ("Hamad II of Bahrain", "Salman bin Hamad"),
                 ("Praggnanandhaa", "Vaishali Rameshbabu"), ("Jonah Hill", "Beanie Feldstein"),
                 ("Dakota Johnson", "Renate Reinsve"), ("Dayot Upamecano", "Dávinson Sánchez"), ("Michelle Williams", "Riki Lindhome")]:
        ia = next((i for n, i in idx.items() if a.lower() in n.lower()), None)
        ib = next((i for n, i in idx.items() if b.lower() in n.lower()), None)
        if ia is None or ib is None:
            print(f"  {a} / {b}: not in index")
            continue
        r = raw[min(ia, ib), max(ia, ib)]
        print(f"  {people[ia]} / {people[ib]}: {shown(r):.0f}%, rank {rank[np.flatnonzero((iu[0] == min(ia, ib)) & (iu[1] == max(ia, ib)))[0]]:,}")


if __name__ == "__main__":
    main()
