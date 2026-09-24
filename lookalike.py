"""
Lookalike ranking, shared by the Gradio app (app.py) and the API server (server.py).
Pure numpy: no torch / gradio / training imports, so it stays cheap to import.
"""
import hashlib
import re
import unicodedata
from collections import Counter

import numpy as np

MARGIN = 2.0  # embedding distance that maps to 0% similarity
TOP_K = 2     # "top2": a person's score is the mean of their TOP_K best-matching photos

# (label, agg — see rank_players). No skin-tone filter any more: scores are the CNN alone.
SEARCH_MODES = [
    ("CNN only",              "top2"),
    ("CNN only (best image)", "best"),
]


def thumb_name(player):
    """Filename of a player's thumbnail — hashed so accents/spaces never matter."""
    return hashlib.md5(player.encode("utf-8")).hexdigest()[:16] + ".jpg"


def merge_duplicate_names(names):
    """One player stored under reordered/accented names ("Heung-Min_Son" / "Son_Heung-min") -> one name,
    the spelling with the most images. ponytail: same words in any order = same player; the current index
    has exactly one such pair, recheck if a rebuilt index ever merges two different people."""
    def key(n):
        n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
        return " ".join(sorted(re.findall(r"[a-z0-9]+", n)))
    counts = Counter(names)
    best = {}
    for n in counts:
        k = key(n)
        if k not in best or (counts[n], n) > (counts[best[k]], best[k]):
            best[k] = n
    canon = {n: best[key(n)] for n in counts}
    return [canon[n] for n in names]


def rank_players(names, dist, agg="top2", allowed=None):
    """
    names: person per indexed image. dist: L2 embedding distance from the query to every indexed
    image, in the same order. allowed: optional bool per image (e.g. the gender filter); people
    with no allowed image are left out.

    agg: how a person is scored. "top2" = mean of their TOP_K best photos, so someone with 12 photos
    doesn't beat someone with 3 just by having more chances at one lucky match; "avg" = mean over all
    their photos; "best" = their single best photo; "first" = their first photo.

    Returns [(name, percent, image_index)] best first. image_index (into the full index) is the
    photo to show: the person's best-matching one (their first one for "first").
    """
    pct = np.maximum(0.0, 1.0 - np.asarray(dist, dtype=np.float64) / MARGIN) * 100
    idx = np.arange(len(pct)) if allowed is None else np.flatnonzero(allowed)
    pct = pct[idx]
    players, inv = np.unique(np.asarray(names)[idx], return_inverse=True)
    n = len(players)
    by_pct = np.argsort(-pct, kind="stable")
    best_i = by_pct[np.unique(inv[by_pct], return_index=True)[1]]  # per person: their best image
    first_i = np.unique(inv, return_index=True)[1]                  # per person: first image (index order)
    if agg == "avg":
        score, show = np.bincount(inv, weights=pct, minlength=n) / np.bincount(inv, minlength=n), best_i
    elif agg == "first":
        score, show = pct[first_i], first_i
    elif agg == "best":
        score, show = pct[best_i], best_i
    else:  # "top2"
        o = np.lexsort((-pct, inv))                                 # grouped by person, best photo first
        start = np.flatnonzero(np.r_[True, inv[o][1:] != inv[o][:-1]])
        rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
        top = o[rank < TOP_K]
        score = np.bincount(inv[top], weights=pct[top], minlength=n) / np.bincount(inv[top], minlength=n)
        show = best_i
    order = np.argsort(-score, kind="stable")
    return [(str(players[k]), float(score[k]), int(idx[show[k]])) for k in order]


def _demo():
    # A: 3 near-identical strong photos + a weak first one; B: one mid photo; C: one better photo
    names = ["A", "A", "A", "A", "B", "C"]
    dist = [1.0, 0.2, 0.2, 0.2, 0.7, 0.5]  # -> 50, 90, 90, 90, 65, 75 %
    top = lambda agg, **kw: [(n, round(s)) for n, s, _ in rank_players(names, dist, agg, **kw)]
    assert top("avg") == [("A", 80), ("C", 75), ("B", 65)]
    assert top("first") == [("C", 75), ("B", 65), ("A", 50)]
    assert top("best") == [("A", 90), ("C", 75), ("B", 65)]
    assert top("top2") == [("A", 90), ("C", 75), ("B", 65)]
    assert rank_players(names, dist, "best")[0][2] == 1   # A's best photo = index 1
    assert rank_players(names, dist, "first")[2][2] == 0  # A's first photo = index 0
    # top2 averages a person's two best photos: one lucky photo doesn't win on its own
    assert top("top2", allowed=None) == top("top2")
    d2 = [0.1, 1.2, 1.2, 1.2, 0.3, 0.3]                     # A: 95 once, then 40s; B and C: 85
    assert [n for n, _, _ in rank_players(names, d2, "best")][0] == "A"
    assert [(n, round(s)) for n, s, _ in rank_players(names, d2, "top2")][0] == ("B", 85)
    # allowed mask (gender filter): A's photos hidden -> A is gone, indices still point into the full index
    mask = np.array([False, False, False, False, True, True])
    assert rank_players(names, dist, "top2", allowed=mask) == [("C", 75.0, 5), ("B", 65.0, 4)]
    assert merge_duplicate_names(["A_B", "B_A", "B_A", "C"]) == ["B_A", "B_A", "B_A", "C"]
    print("ok")


if __name__ == "__main__":
    _demo()
