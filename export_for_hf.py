"""
Bundle everything the Hugging Face Docker Space needs into hf_space/:
server code, app_best.pt (copied — the original is never touched), index.npz, and one
thumbnail per player. Re-run whenever the model, index, or player images change:

    1. In app.py: Rebuild Index (from scratch), then Build Feature Index
    2. python export_for_hf.py
    3. cd hf_space && git add . && git commit && git push   (see hf_space/README.md)
"""
import os
import shutil

import numpy as np
from PIL import Image, ImageOps

from face_features import aligned_face
from lookalike import thumb_name

SRC_CACHE = "embed_cache_soccer_best.npz"  # written by app.py's Build Index (EMBED_CACHE_BEST)
OUT       = "hf_space"
COPY      = ["server.py", "lookalike.py", "face_features.py", "lfw_pytorch.py", "app_best.pt"]
THUMB     = 128

os.makedirs(f"{OUT}/thumbs", exist_ok=True)
for f in COPY:
    shutil.copy(f, OUT)

with np.load(SRC_CACHE, allow_pickle=True) as d:
    names, paths = d["names"].tolist(), d["paths"].tolist()
    embs, feats  = d["embeddings"].astype(np.float32), d["features"].astype(np.float32)

np.savez_compressed(f"{OUT}/index.npz", names=np.array(names, dtype=str), embeddings=embs, features=feats)

# one thumbnail per player: the aligned face from their first readable image
players = set(names)
print(f"Making thumbnails for {len(players)} players (one face detection each, ~10+ min on CPU). "
      "Safe to Ctrl+C and re-run: finished ones are skipped.", flush=True)
have = set()
for name, path in zip(names, paths):
    if name in have:
        continue
    out = f"{OUT}/thumbs/{thumb_name(name)}"
    try:
        if not os.path.exists(out):
            ImageOps.fit(aligned_face(Image.open(path).convert("RGB")), (THUMB, THUMB)).save(out, quality=85)
        have.add(name)
        if len(have) % 100 == 0:
            print(f"  {len(have)}/{len(players)} thumbnails", flush=True)
    except Exception:
        continue  # deleted/unreadable image — the player's next image gets tried

print(f"index: {len(names)} images, {len(players)} players | thumbnails: {len(have)}/{len(players)}")
if players - have:
    print("no thumbnail for:", ", ".join(sorted(players - have)[:10]), "..." if len(players - have) > 10 else "")
