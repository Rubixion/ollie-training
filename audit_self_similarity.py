"""
Self-similarity audit: for every player with 2+ images, find each image's nearest sibling
(same player) by embedding distance. A genuine photo of a player should have a close sibling;
a mislabeled/wrong photo (like the Haaland images found in Tomasson/Jefte/Jean_Nicolas) sits far
from every photo of that player because it isn't actually them.

    python audit_self_similarity.py            # top 40 suspects
    python audit_self_similarity.py --all       # every image over the cutoff
    python audit_self_similarity.py --n 100

Reads embed_cache_soccer_best.npz (same embeddings the live index is built from).
"""
import argparse

import numpy as np

CACHE = "embed_cache_soccer_best.npz"

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=40, help="how many top suspects to print")
p.add_argument("--all", action="store_true", help="print every image over the cutoff, not just top N")
p.add_argument("--pct", type=float, default=99.5, help="percentile of nearest-sibling distance used as the cutoff")
args = p.parse_args()

with np.load(CACHE, allow_pickle=True) as d:
    names = d["names"]
    paths = d["paths"]
    embs = d["embeddings"].astype(np.float64)

order = np.argsort(names, kind="stable")
names, paths, embs = names[order], paths[order], embs[order]
uniq, starts, counts = np.unique(names, return_index=True, return_counts=True)

nn_dist = np.full(len(names), np.nan)  # each image's distance to its closest same-player sibling
singles = 0
for start, count in zip(starts, counts):
    if count < 2:
        singles += 1
        continue
    block = embs[start:start + count]
    d2 = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=2)
    np.fill_diagonal(d2, np.inf)
    nn_dist[start:start + count] = d2.min(axis=1)

valid = ~np.isnan(nn_dist)
# "normal" = a real photo has at least one close sibling; use the bulk of the distribution
# (not the tail we're trying to find) to set what "close" means.
cutoff = np.percentile(nn_dist[valid], args.pct)
suspects = np.argsort(-np.where(valid, nn_dist, -1))
suspects = [i for i in suspects if valid[i] and nn_dist[i] > cutoff]

print(f"{len(names)} images, {len(uniq)} players ({singles} with only 1 image — can't self-check).")
print(f"nearest-sibling distance: median {np.median(nn_dist[valid]):.3f}, {args.pct}th pct {cutoff:.3f}")
print(f"{len(suspects)} image(s) above the {args.pct}th percentile — likely mislabeled or a bad crop"
      " (some will just be old/grainy photos, not wrong people).\n")

shown = suspects if args.all else suspects[:args.n]
for i in shown:
    print(f"{nn_dist[i]:.3f}  {names[i]:<28} {paths[i]}")
if not args.all and len(suspects) > len(shown):
    print(f"... and {len(suspects) - len(shown)} more (--all to see everything)")
