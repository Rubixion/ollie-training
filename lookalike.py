"""
Player-lookalike ranking, shared by the Gradio app (app.py) and the API server (server.py).
Pure numpy: no torch / gradio / training imports, so it stays cheap to import.
"""
import hashlib
import re
import unicodedata
from collections import Counter

import numpy as np

MARGIN = 2.0  # embedding distance that maps to 0% similarity

# Sanity gate: a player whose typical skin lightness (feature 20) is further than this from the
# query's is dropped from the results (never re-scored). Leave-one-out on the index: 0.20 blocks
# ~28% of players and costs ~7 points of same-player top-1 (95.8 -> 88.4); 0.25 costs ~4, 0.30 ~1.
# Cheek lightness only: the forehead patch (feature 31) reads fringe/hair as skin.
SKIN_GATE = 0.20
GATE_MIN_KEEP = 20  # if the gate would leave fewer players than this (e.g. a very dark photo), skip it

# (label, use the query's face features for the sanity gate, agg — see rank_players)
SEARCH_MODES = [
    ("CNN Only",              True, "avg"),
    ("CNN Only (best image)", True, "best"),
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


def rank_players(names, dist, index_features, q_feats, agg="avg", min_keep=GATE_MIN_KEEP):
    """
    names: player name per indexed image. dist: L2 embedding distance from the query
    to every indexed image, in the same order. index_features: (N, FEAT_DIM).
    q_feats: query features; all-zero (no face found) = no sanity gate.

    Scores are pure embedding similarity; the skin-tone gate only removes implausible players.

    agg: how a player is scored — "avg" = mean over all their images, "first" = first
    image only, "best" = their single best image (every image ranked individually,
    one row per player).

    Returns [(name, percent, image_index)] for every allowed player, best first. image_index is
    the image to show: the player's best-matching image (first image for "first").
    """
    pct = np.maximum(0.0, 1.0 - np.asarray(dist, dtype=np.float64) / MARGIN) * 100

    players, inv = np.unique(np.asarray(names), return_inverse=True)
    by_pct = np.argsort(-pct, kind="stable")
    best_i = by_pct[np.unique(inv[by_pct], return_index=True)[1]]  # per player: its best image
    first_i = np.unique(inv, return_index=True)[1]                  # per player: first image (index order)
    if agg == "avg":
        score, show = np.bincount(inv, weights=pct) / np.bincount(inv), best_i
    elif agg == "first":
        score, show = pct[first_i], first_i
    else:  # "best"
        score, show = pct[best_i], best_i
    order = np.argsort(-score, kind="stable")

    if q_feats[20] > 0:  # face found: drop players whose typical skin tone is far from the query's
        skin = np.asarray(index_features, dtype=np.float64)[:, 20]
        cnt = np.bincount(inv)
        srt = skin[np.lexsort((skin, inv))]                          # skin values grouped by player, ascending
        lo = np.cumsum(cnt) - cnt
        median = (srt[lo + (cnt - 1) // 2] + srt[lo + cnt // 2]) / 2  # per-player median
        ok = np.abs(median - q_feats[20]) <= SKIN_GATE
        if ok.sum() >= min(min_keep, len(players)):
            order = order[ok[order]]
    return [(str(players[k]), float(score[k]), int(show[k])) for k in order]


def _demo():
    # A: 3 near-identical strong photos + a weak first one; B: one mid photo; C: one better photo
    names = ["A", "A", "A", "A", "B", "C"]
    dist = [1.0, 0.2, 0.2, 0.2, 0.7, 0.5]  # -> 50, 90, 90, 90, 65, 75 %
    zeros = np.zeros((6, 32), np.float32)
    q0 = np.zeros(32, np.float32)
    top = lambda agg: [(n, round(s)) for n, s, _ in rank_players(names, dist, zeros, q0, agg)]
    assert top("avg") == [("A", 80), ("C", 75), ("B", 65)]
    assert top("first") == [("C", 75), ("B", 65), ("A", 50)]
    assert top("best") == [("A", 90), ("C", 75), ("B", 65)]
    assert rank_players(names, dist, zeros, q0, "best")[0][2] == 1   # A's best photo = index 1
    assert rank_players(names, dist, zeros, q0, "first")[2][2] == 0  # A's first photo = index 0
    # sanity gate: B's skin (feat 20) is far from the query's -> B is dropped, everyone else keeps the plain score
    feats = np.zeros((6, 32), np.float32); feats[:, 20] = 0.5; feats[4, 20] = 0.9
    q = np.zeros(32, np.float32); q[20] = 0.5
    gated = rank_players(names, dist, feats, q, "best", min_keep=1)
    assert [(n, round(s)) for n, s, _ in gated] == [("A", 90), ("C", 75)]
    # ...but never leaves too few players: with the default floor the gate is skipped on a 3-player index
    assert len(rank_players(names, dist, feats, q, "best")) == 3
    assert merge_duplicate_names(["A_B", "B_A", "B_A", "C"]) == ["B_A", "B_A", "B_A", "C"]
    print("ok")


if __name__ == "__main__":
    _demo()
