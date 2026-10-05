"""
LFW / CALFW / CPLFW / AgeDB_30 verification accuracy, using the exact protocol of
yakhyo/face-recognition (evaluate.py): features = concat(f(img), f(flip(img))),
cosine similarity, 10 sequential folds, threshold grid -1..1 step 0.005.
Same aligned 112x112 data as their table, so the numbers compare directly.

Usage:  .venv/Scripts/python eval_benchmarks.py [--ref]
  --ref  also runs yakhyo's released sphere20 weights through this script (protocol sanity check)
"""
import os
import sys
import urllib.request

import kagglehub
import numpy as np
import torch
from PIL import Image

from lfw_pytorch import SphereFaceNet, EMBEDDING_SIZE

SETS = ["lfw", "calfw", "cplfw", "agedb_30"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference", "face-recognition")


def find_val_root(path):
    for root, _, files in os.walk(path):
        if "lfw_ann.txt" in files:
            return root
    raise FileNotFoundError("lfw_ann.txt not found under " + path)


def to_tensor(imgs):
    x = np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs])
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2)


@torch.no_grad()
def embed(fn, root, paths):
    out = {}
    for i in range(0, len(paths), 128):
        chunk = paths[i:i + 128]
        imgs = [Image.open(os.path.join(root, p)).convert("RGB") for p in chunk]
        x = to_tensor(imgs).to(DEV)
        f = torch.cat([fn(x), fn(torch.flip(x, dims=[3]))], dim=1).cpu().numpy()
        out.update(zip(chunk, f))
    return out


def kfold_acc(sims, labels):
    # identical to yakhyo k_fold_split / find_best_threshold / eval_accuracy
    thresholds = np.arange(-1.0, 1.0, 0.005)
    n, size = len(sims), len(sims) // 10
    accs = []
    for k in range(10):
        test = np.zeros(n, bool)
        test[k * size:(k + 1) * size] = True
        train_acc = [((sims[~test] > t) == labels[~test]).mean() for t in thresholds]
        t = thresholds[int(np.argmax(train_acc))]
        accs.append(((sims[test] > t) == labels[test]).mean())
    return np.mean(accs), np.std(accs)


def evaluate(name, fn, root):
    row = []
    for s in SETS:
        with open(os.path.join(root, f"{s}_ann.txt")) as f:
            pairs = [l.split() for l in f.readlines()[1:] if len(l.split()) == 3]
        paths = sorted({p for _, a, b in pairs for p in (a, b)})
        feats = embed(fn, root, paths)
        sims = np.array([
            feats[a] @ feats[b] / (np.linalg.norm(feats[a]) * np.linalg.norm(feats[b]) + 1e-5)
            for _, a, b in pairs])
        labels = np.array([int(g) for g, _, _ in pairs]).astype(bool)
        acc, std = kfold_acc(sims, labels)
        print(f"  {name:10s} {s:9s} {acc*100:6.2f}% +/- {std*100:.2f}  ({len(pairs)} pairs)")
        row.append(acc * 100)
    return row


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    root = find_val_root(kagglehub.dataset_download("yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112"))
    print("data:", root)
    results = {}

    ours = SphereFaceNet(EMBEDDING_SIZE)
    ckpt = torch.load("app_best.pt", map_location="cpu", weights_only=False)
    ours.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    ours.to(DEV).eval()
    # ponytail: params counted on the full module; if it holds a training-only head this overcounts
    results["Ollie Sphere20 (ours)"] = evaluate("ours", ours.get_embedding, root) + [n_params(ours)]

    if "--ref" in sys.argv:
        sys.path.insert(0, REF_DIR)
        from models import sphere20
        w = os.path.join(REF_DIR, "weights", "sphere20_mcp.pth")
        if not os.path.exists(w):
            urllib.request.urlretrieve(
                "https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere20_mcp.pth", w)
        ref = sphere20(512)
        ref.load_state_dict(torch.load(w, map_location="cpu"))
        ref.to(DEV).eval()
        results["yakhyo Sphere20 (rerun)"] = evaluate("yakhyo", ref, root) + [n_params(ref)]

    print("\n| Model | LFW | CALFW | CPLFW | AgeDB_30 | Params |\n|---|---|---|---|---|---|")
    for k, (a, b, c, d, p) in results.items():
        print(f"| {k} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {p:.2f}M |")


if __name__ == "__main__":
    main()
