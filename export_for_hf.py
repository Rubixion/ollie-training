"""
Bundle everything the Modal deployment needs into hf_space/, for the v2 celebrity index:
server code + app_best.pt (copied; the original is never touched), index.npz and imgthumbs/<row>.jpg.
Re-run whenever the model or celeb_v2/data change (e.g. after the top-up or a takedown), then deploy:

    python export_for_hf.py
    cd hf_space && modal deploy modal_app.py

Source: every celeb_v2/data/*/manifest.json with status "ok". Each photo is re-aligned with InsightFace
around its verified face_box (the photo can hold other people) and embedded with app_best.pt, the model
the server uses for the query. Aligned faces are cached per person (aligned.npz), so a rebuild only
detects faces for new or changed people.
index.npz, one row per photo: names, embeddings, genders ("F"/"M"/""), categories ("actor|musician"),
credits (JSON: author, license, license_url, page) and known_for (the Wikidata description).
"""
import collections
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, "celeb_v2")
from build_list import categories                    # noqa: E402  (celeb_v2)
from collect import credit_license                   # noqa: E402
from lfw_pytorch import EMBEDDING_SIZE, SphereFaceNet, test_transform
from lookalike import merge_duplicate_names

OUT   = "hf_space"
DATA  = "celeb_v2/data"
LISTS = ["celeb_v2/celebs.json", "celeb_v2/topup.json"]
COPY  = ["server.py", "lookalike.py", "face_features.py", "lfw_pytorch.py", "app_best.pt"]
THUMB = 128

_det = None


def detector():
    global _det
    if _det is None:
        from insightface.app import FaceAnalysis
        _det = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        _det.prepare(ctx_id=-1, det_size=(320, 320))  # a crop around one face: small det_size is plenty
    return _det


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h + 1e-6)


def align(path, box):
    """-> 112x112 aligned face (uint8 RGB) of the verified face in `box`, or None."""
    from insightface.utils.face_align import norm_crop
    rgb = np.asarray(Image.open(path).convert("RGB"))
    x1, y1, x2, y2 = box
    m = 0.6 * max(x2 - x1, y2 - y1)
    cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    crop = np.ascontiguousarray(rgb[cy1:int(min(rgb.shape[0], y2 + m)), cx1:int(min(rgb.shape[1], x2 + m))])
    want = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
    faces = detector().get(crop[:, :, ::-1])  # detector expects BGR
    best = max(faces, key=lambda f: iou(f.bbox, want), default=None)
    if best is None or iou(best.bbox, want) < 0.3:
        return None
    return norm_crop(crop, best.kps, image_size=112)


def aligned_faces(folder, manifest):
    """-> (files, faces uint8 [N,112,112,3]) for the manifest's photos, cached in aligned.npz."""
    path = os.path.join(folder, "aligned.npz")
    files = [i["file"] for i in manifest["images"]]
    if os.path.exists(path):
        with np.load(path) as d:
            if d["files"].tolist() == files:
                return files, d["faces"]
    got, faces = [], []
    for img in manifest["images"]:
        face = align(os.path.join(folder, img["file"]), img["face_box"])
        if face is not None:
            got.append(img["file"])
            faces.append(face)
    faces = np.stack(faces) if faces else np.zeros((0, 112, 112, 3), np.uint8)
    if got == files:  # only cache complete sets, so a failed photo is retried next run
        np.savez_compressed(path, files=np.array(files), faces=faces)
    return got, faces


def unique_names(people):
    """Two different people must never share a name: the server groups photos by name and merges
    reordered spellings. Colliding ones get their description added, e.g. "Kim Ji-won (South Korean actress)"."""
    names = [p["name"] for p in people]
    canon = merge_duplicate_names(names)
    clash = {c for c, n in collections.Counter(canon).items() if n > 1}
    return [f"{n} ({p['description'] or p['qid']})" if c in clash else n for n, c, p in zip(names, canon, people)]


def main():
    info = {}
    for path in LISTS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                info.update({c["qid"]: c for c in json.load(f)})

    people = []
    for mpath in sorted(glob.glob(os.path.join(DATA, "*", "manifest.json"))):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        if m["status"] == "ok":
            c = info.get(m["qid"], {})
            people.append({"qid": m["qid"], "name": m["name"], "folder": os.path.dirname(mpath), "manifest": m,
                           "description": c.get("description", ""), "gender": c.get("gender", "")})
    for p, name in zip(people, unique_names(people)):
        p["name"] = name
    print(f"{len(people)} people", flush=True)

    model = SphereFaceNet(EMBEDDING_SIZE)
    raw = torch.load("app_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)
    model.eval()

    os.makedirs(OUT, exist_ok=True)
    for f in COPY:
        shutil.copy(f, OUT)
    thumbs = os.path.join(OUT, "imgthumbs")
    shutil.rmtree(thumbs, ignore_errors=True)  # row numbers change on every rebuild
    shutil.rmtree(os.path.join(OUT, "thumbs"), ignore_errors=True)  # old per-player soccer thumbs
    os.makedirs(thumbs)

    cols = collections.defaultdict(list)
    missing = 0
    for n, p in enumerate(people, 1):
        files, faces = aligned_faces(p["folder"], p["manifest"])
        missing += len(p["manifest"]["images"]) - len(files)
        if not files:
            continue
        with torch.no_grad():
            batch = torch.stack([test_transform(Image.fromarray(f)) for f in faces])
            embs = model.get_embedding(batch).numpy()
        by_file = {i["file"]: i for i in p["manifest"]["images"]}
        for file, face, emb in zip(files, faces, embs):
            img = by_file[file]
            ImageOps.fit(Image.fromarray(face), (THUMB, THUMB)).save(
                os.path.join(thumbs, f"{len(cols['names'])}.jpg"), quality=80)
            cols["names"].append(p["name"])
            cols["embeddings"].append(emb)
            cols["genders"].append(p["gender"])
            cols["categories"].append(categories(p["description"]))
            cols["known_for"].append(p["description"])
            cols["credits"].append(json.dumps({"author": img["author"], "license": credit_license(img),
                                               "license_url": img["license_url"], "page": img["page"]},
                                              ensure_ascii=False))
        if n % 250 == 0:
            print(f"  {n}/{len(people)} people, {len(cols['names'])} photos", flush=True)

    np.savez_compressed(os.path.join(OUT, "index.npz"),
                        embeddings=np.stack(cols.pop("embeddings")).astype(np.float32),
                        **{k: np.array(v, dtype=str) for k, v in cols.items()})
    g = collections.Counter({p["name"]: p["gender"] for p in people}.values())
    print(f"index: {len(cols['names'])} photos, {len(set(cols['names']))} people "
          f"({g['F']} women, {g['M']} men, {g['']} unknown) | {missing} photos without a face on re-detection")


if __name__ == "__main__":
    main()
