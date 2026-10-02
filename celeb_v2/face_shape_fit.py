"""Checks and fits the face-shape prototypes used by the /ai-stylist scan (ollie-frontend/lib/style/face-shape.ts).

A. Repeatability: people with 2-3 straight-on photos (face_norms.json "photos"): how often every photo gets the same shape.
B. Agreement with stylists: face_shape_labels.json (published "X has a square face" claims, names only). Scores the
   current hand-set prototypes, then fits new ones as the average face of each labelled shape (pulled toward the
   hand-set ones when a shape has few labels), scored leave-one-out so the number is honest.

    .venv\\Scripts\\python celeb_v2\\face_shape_fit.py      (from "neural network learning")  -> face_shape_fit.json
"""
import json, math, os, unicodedata
from collections import Counter, defaultdict

import numpy as np

from celeb_face_shapes import NORMS, PROTOTYPES

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = list(PROTOTYPES)
import sys
PULL = float(sys.argv[1]) if len(sys.argv) > 1 else 4  # labelled faces' worth of weight on the hand-set prototype; matters only for shapes with few labels


def norm(name):
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).lower().replace(".", "").replace("-", " ").strip()


def z(r):
    return (np.array(r) - NORMS["mean"]) / NORMS["std"]


def probs(zv, protos):
    w = np.array([math.exp(-np.sum((zv - protos[s]) ** 2) / 2) for s in SHAPES])
    return w / (w.sum() or 1)


def top(zv, protos):
    return SHAPES[int(np.argmax(probs(zv, protos)))]


def main():
    fn = json.load(open(os.path.join(HERE, "face_norms.json"), encoding="utf8"))
    celebs = {c["qid"]: c for c in json.load(open(os.path.join(HERE, "celebs.json"), encoding="utf8"))}
    by_name = {}
    for key, r in fn["people"].items():
        c = celebs.get(key.rsplit("_", 1)[-1])
        if c:
            by_name[norm(c["name"])] = key
    hand = {s: np.array(p, float) for s, p in PROTOTYPES.items()}

    # ── labels: majority shape across sources; ties are conflicts and dropped ──
    votes = defaultdict(Counter)
    for shape, srcs in json.load(open(os.path.join(HERE, "face_shape_labels.json"), encoding="utf8"))["labels"].items():
        for names in srcs.values():
            for n in set(names):
                votes[norm(n)][shape] += 1
    labels, conflicts, missing = {}, [], []
    for n, c in votes.items():
        (s1, k1), *rest = c.most_common()
        if rest and rest[0][1] == k1:
            conflicts.append(n)
        elif n not in by_name:
            missing.append(n)
        else:
            labels[by_name[n]] = s1
    keys = sorted(labels)
    Z = {k: z(fn["people"][k]) for k in keys}

    def accuracy(protos_for):
        hit = hit2 = 0
        for k in keys:
            p = probs(Z[k], protos_for(k))
            order = [SHAPES[i] for i in np.argsort(-p)]
            hit += order[0] == labels[k]
            hit2 += labels[k] in order[:2]
        return hit / len(keys), hit2 / len(keys)

    def fit(exclude=None):
        out = {}
        for s in SHAPES:
            pts = [Z[k] for k in keys if labels[k] == s and k != exclude]
            out[s] = (np.sum(pts, 0) + PULL * hand[s]) / (len(pts) + PULL) if pts else hand[s]
        return out

    fitted = fit()
    acc_hand = accuracy(lambda k: hand)
    acc_fit = accuracy(lambda k: fit(exclude=k))  # leave-one-out
    chance = max(Counter(labels.values()).values()) / len(keys)

    # ── A. repeatability over each person's photos ──
    def repeat(protos):
        same = n = 0
        for ph in fn.get("photos", {}).values():
            if len(ph) >= 2:
                n += 1
                same += len({top(z(r), protos) for r in ph}) == 1
        return same / max(n, 1), n

    rep_hand, n_rep = repeat(hand)
    rep_fit, _ = repeat(fitted)

    def shares(protos):
        c = Counter(top(z(r), protos) for r in fn["people"].values())
        return {s: round(100 * c[s] / len(fn["people"]), 1) for s in SHAPES}

    out = {
        "labelled": len(keys), "per_shape": Counter(labels.values()), "conflicts_dropped": len(conflicts), "not_in_index": len(missing),
        "accuracy_hand": [round(a, 3) for a in acc_hand], "accuracy_fitted_loo": [round(a, 3) for a in acc_fit], "chance": round(chance, 3),
        "repeatable_hand": round(rep_hand, 3), "repeatable_fitted": round(rep_fit, 3), "people_with_2plus_photos": n_rep,
        "shares_hand": shares(hand), "shares_fitted": shares(fitted),
        "prototypes_fitted": {s: [round(float(v), 2) for v in p] for s, p in fitted.items()},
        "conflicts": sorted(conflicts), "missing": sorted(missing),
    }
    json.dump(out, open(os.path.join(HERE, "face_shape_fit.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=dict)
    for k in ("labelled", "per_shape", "conflicts_dropped", "not_in_index", "chance", "accuracy_hand", "accuracy_fitted_loo",
              "people_with_2plus_photos", "repeatable_hand", "repeatable_fitted", "shares_hand", "shares_fitted", "prototypes_fitted"):
        print(f"{k:26}", json.dumps(out[k], default=dict))


if __name__ == "__main__":
    main()
