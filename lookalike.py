"""
Player-lookalike ranking, shared by the Gradio app (app.py) and the API server (server.py).
Pure numpy: no torch / gradio / training imports, so it stays cheap to import.
"""
import hashlib

import numpy as np

MARGIN = 2.0  # embedding distance that maps to 0% similarity

# (label, use feature re-ranking, agg — see rank_players)
SEARCH_MODES = [
    ("CNN + Features",              True,  "avg"),
    ("CNN Only",                    False, "avg"),
    ("CNN + Features (1 image)",    True,  "first"),
    ("CNN Only (1 image)",          False, "first"),
    ("CNN + Features (best image)", True,  "best"),
    ("CNN Only (best image)",       False, "best"),
]


def thumb_name(player):
    """Filename of a player's thumbnail — hashed so accents/spaces never matter."""
    return hashlib.md5(player.encode("utf-8")).hexdigest()[:16] + ".jpg"


def rank_players(names, dist, index_features, q_feats, agg="avg"):
    """
    names: player name per indexed image. dist: L2 embedding distance from the query
    to every indexed image, in the same order. index_features: (N, FEAT_DIM).
    q_feats: query features; all-zero (CNN Only, or no face found) = no feature penalty.

    agg: how a player is scored — "avg" = mean over all their images, "first" = first
    image only, "best" = their single best image (every image ranked individually,
    one row per player).

    Returns [(name, percent, image_index)] for every player, best first. image_index is
    the image to show: the player's best-matching image (first image for "first").
    """
    dist = np.asarray(dist, dtype=np.float64)
    q_feats = np.asarray(q_feats, dtype=np.float64)  # float32 penalty sums drift a few 1e-6 %
    penalty = np.zeros(len(dist))
    if np.any(q_feats != 0):
        f = np.asarray(index_features, dtype=np.float64)
        hh = np.abs(q_feats[17] - f[:, 17])
        p = (5.0 * np.abs(q_feats[20] - f[:, 20])
             + 2.0 * np.minimum(hh, 1.0 - hh)
             + np.abs(q_feats[11] - f[:, 11]))
        if q_feats[28] > 0:
            p += np.where(f[:, 28] > 0, 3.0 * np.abs(q_feats[28] - f[:, 28]), 0.0)
        if q_feats[29] > 0:
            p += np.where(f[:, 29] > 0, 4.0 * np.abs(q_feats[29] - f[:, 29]), 0.0)
        penalty = np.where(np.any(f != 0, axis=1), p, 0.30)  # no features stored = flat 0.30
    pct = np.maximum(0.0, 1.0 - (dist + penalty) / MARGIN) * 100

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
    return [(str(players[k]), float(score[k]), int(show[k])) for k in np.argsort(-score, kind="stable")]


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
    # feature penalty: same distances, but B's skin tone (feat 20) is far from the query -> B drops out
    feats = np.zeros((6, 32), np.float32); feats[:, 20] = 0.5; feats[4, 20] = 0.9
    q = np.zeros(32, np.float32); q[20] = 0.5
    assert [n for n, *_ in rank_players(names, dist, feats, q, "best")][-1] == "B"
    print("ok")


if __name__ == "__main__":
    _demo()
