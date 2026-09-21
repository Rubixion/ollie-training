"""
One thumbnail per indexed image (hf_space/imgthumbs/<index>.jpg), so a result can show the photo that
actually matched instead of the player's first one. Resumable: finished thumbnails are skipped.

    python export_image_thumbs.py     # ~16K face detections, CPU: takes a while; Ctrl+C and re-run is safe

Reads the image paths from the cache export_for_hf.py used and refuses to run if it no longer
lines up with hf_space/index.npz. Then redeploy (modal deploy modal_app.py, from hf_space).
"""
import os

import numpy as np
from PIL import Image, ImageOps

from face_features import aligned_face

SRC_CACHE = "embed_cache_soccer_best.npz"
OUT       = "hf_space/imgthumbs"
THUMB     = 128

with np.load(SRC_CACHE, allow_pickle=True) as d:
    names, paths = d["names"].tolist(), d["paths"].tolist()
with np.load("hf_space/index.npz") as d:
    assert names == d["names"].tolist(), "cache and hf_space/index.npz differ: re-run export_for_hf.py first"

os.makedirs(OUT, exist_ok=True)
done = failed = 0
for i, path in enumerate(paths):
    out = f"{OUT}/{i}.jpg"
    if os.path.exists(out):
        continue
    try:
        ImageOps.fit(aligned_face(Image.open(path).convert("RGB")), (THUMB, THUMB)).save(out + ".tmp", "JPEG", quality=80)
        os.replace(out + ".tmp", out)  # never leaves a half-written thumbnail
        done += 1
    except Exception:
        failed += 1  # the server falls back to the player's default thumbnail
    if (done + failed) % 500 == 0:
        print(f"  {i + 1}/{len(paths)} images ({failed} failed)", flush=True)
print(f"done: {len(os.listdir(OUT))}/{len(paths)} thumbnails")
