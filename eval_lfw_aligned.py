"""
Evaluate a saved checkpoint on properly ArcFace-aligned LFW pairs.

Why this exists: app.py's live eval feeds LFW's raw "deepfunneled" images
(loose crop, full head/shoulders, black funnel corners) through a plain
Resize+CenterCrop — a completely different framing than the tight,
5-point-landmark-aligned 112x112 crops the model is trained on (MS1MV2).
That mismatch alone can cap verification accuracy regardless of how good
the embeddings actually are. This script re-aligns LFW with InsightFace's
detector + norm_crop (the same ArcFace alignment protocol MS1MV2 uses) and
re-runs the 10-fold LFW eval so you can see the model's *real* accuracy.

Runs entirely on CPU so it can be run alongside a live GPU training run
without contending for VRAM. Aligned crops are cached to disk on first run
(lfw_aligned_cache/), so re-checking a later checkpoint is fast.

Usage:
    python eval_lfw_aligned.py                # evaluates app_checkpoint.pt
    python eval_lfw_aligned.py best            # evaluates app_best.pt
    python eval_lfw_aligned.py path/to/ckpt.pt # evaluates an explicit file
"""

import os
import sys
import time
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lfw_pytorch import (
    SphereFaceNet, EMBEDDING_SIZE,
    load_csv_pairs, load_pairs_csv, k_fold_eval,
)

APP_CHECKPOINT = "app_checkpoint.pt"
APP_BEST       = "app_best.pt"
ALIGN_SIZE     = 112
ALIGN_CACHE    = "lfw_aligned_cache"
DET_SIZE       = (320, 320)   # small + fast on CPU; LFW faces are large in-frame


def get_lfw_dataset_path():
    import kagglehub
    cached = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub",
                           "datasets", "jessicali9530", "lfw-dataset", "versions", "4")
    if os.path.isdir(cached):
        return cached
    return kagglehub.dataset_download("jessicali9530/lfw-dataset")


def load_test_pairs():
    dp = get_lfw_dataset_path()
    pairs = load_pairs_csv(dp, exclude_people=set())
    if pairs:
        print(f"Loaded {len(pairs)} pairs from pairs.csv (reference 10-fold protocol)")
        return pairs
    pairs = load_csv_pairs(dp, "matchpairsDevTest.csv", "mismatchpairsDevTest.csv")
    print(f"pairs.csv not found — using DevTest ({len(pairs)} pairs)")
    return pairs


def get_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        allowed_modules=['detection'],
        providers=['CPUExecutionProvider'],   # force CPU — GPU is busy training
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def cache_path(img_path):
    h = hashlib.md5(img_path.encode('utf-8')).hexdigest()
    return os.path.join(ALIGN_CACHE, f"{h}.jpg")


def align_one(face_app, img_path):
    out_path = cache_path(img_path)
    if os.path.exists(out_path):
        return out_path

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    faces = face_app.get(img_bgr)
    if not faces:
        return None

    # largest face, in case of stray detections
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    from insightface.utils import face_align
    aligned = face_align.norm_crop(img_bgr, face.kps, image_size=ALIGN_SIZE)

    os.makedirs(ALIGN_CACHE, exist_ok=True)
    cv2.imwrite(out_path, aligned)
    return out_path


def load_checkpoint_model(which):
    if which in ("best", APP_BEST):
        path = APP_BEST
    elif which in ("checkpoint", "ckpt", APP_CHECKPOINT):
        path = APP_CHECKPOINT
    else:
        path = which

    model = SphereFaceNet(EMBEDDING_SIZE)

    for attempt in range(2):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            break
        except Exception as e:
            if attempt == 0:
                print(f"  Checkpoint read failed ({e}) — probably mid-write by the "
                      f"training process, retrying in 3s...")
                time.sleep(3)
            else:
                raise

    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    epoch = ckpt.get('epoch') if isinstance(ckpt, dict) else None
    live_best = ckpt.get('best_acc') if isinstance(ckpt, dict) else None
    return model, path, epoch, live_best


def embed_all(model, aligned_paths):
    """Returns dict {img_path: np.ndarray[512]} for every successfully aligned image."""
    embeddings = {}
    batch_paths, batch_tensors = [], []
    total = sum(1 for a in aligned_paths.values() if a is not None)
    done = 0
    t0 = time.time()

    def flush():
        nonlocal done
        if not batch_tensors:
            return
        x = torch.from_numpy(np.stack(batch_tensors))
        with torch.no_grad():
            emb = model.get_embedding(x)
        emb = emb.numpy()
        for p, e in zip(batch_paths, emb):
            embeddings[p] = e
        done += len(batch_paths)
        batch_paths.clear()
        batch_tensors.clear()
        if done % 512 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done:5d}/{total}]  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    for orig_path, aligned in aligned_paths.items():
        if aligned is None:
            continue
        img = cv2.imread(aligned)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        batch_paths.append(orig_path)
        batch_tensors.append(img)
        if len(batch_tensors) >= 64:
            flush()
    flush()
    return embeddings


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else APP_CHECKPOINT

    print(f"Loading model from: {which}")
    model, path, epoch, live_best = load_checkpoint_model(which)
    print(f"  epoch={epoch}  live (misaligned) best_acc={live_best}")

    test_pairs = load_test_pairs()

    unique_paths = sorted({p for p1, p2, _ in test_pairs for p in (p1, p2)})
    print(f"\nAligning {len(unique_paths)} unique LFW images "
          f"(cached under {ALIGN_CACHE}/ — instant on reruns)...")

    face_app = get_face_app()
    aligned_paths = {}
    t0 = time.time()
    n_fail = 0
    for i, p in enumerate(unique_paths):
        aligned = align_one(face_app, p)
        aligned_paths[p] = aligned
        if aligned is None:
            n_fail += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(unique_paths):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(unique_paths) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1:5d}/{len(unique_paths)}]  "
                  f"failed={n_fail}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    if n_fail:
        print(f"  {n_fail} images had no detectable face — those pairs are skipped.")

    print("\nComputing embeddings (CPU)...")
    embeddings = embed_all(model, aligned_paths)

    all_dists, all_labels = [], []
    skipped = 0
    for p1, p2, label in test_pairs:
        e1, e2 = embeddings.get(p1), embeddings.get(p2)
        if e1 is None or e2 is None:
            skipped += 1
            continue
        dist = float(np.linalg.norm(e1 - e2))
        all_dists.append(dist)
        all_labels.append(label)

    if skipped:
        print(f"  Skipped {skipped}/{len(test_pairs)} pairs (alignment failed on one side).")

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)
    acc, std, thr = k_fold_eval(all_dists, all_labels)

    print("\n" + "=" * 60)
    print(f"Checkpoint:            {path}  (epoch {epoch})")
    print(f"Live (misaligned) LFW: {live_best*100:.2f}%" if live_best is not None else "Live LFW: n/a")
    print(f"ArcFace-aligned LFW:   {acc*100:.2f}% ± {std*100:.2f}%   (threshold {thr:.3f})")
    print(f"Pairs evaluated:       {len(all_dists)}/{len(test_pairs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
