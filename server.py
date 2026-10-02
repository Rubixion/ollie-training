"""
Lookalike API for the Modal / Hugging Face Docker deployment (or any host).

    API_KEY=secret uvicorn server:app --port 7860

POST /search   multipart: "file" (photo), optional "gender" = any | female | male | auto (default any),
               optional "category" = any | actor | musician | footballer (default any)
               header "X-Api-Key"
  -> {"face_found": bool,
      "gender_used": "female" | "male" | null,     # null = nobody filtered out
      "category_used": "actor" | "musician" | "footballer" | null,   # null = not filtered
      "modes":  {"CNN Only (best image)": [{"name": "Zendaya", "score": 72.4}, ... top 5]},
      "thumbs": {"Zendaya": "data:image/jpeg;base64,..."},          # the photo that matched best
      "credits": {"Zendaya": {"author", "license", "license_url", "page"}},   # only if the index has them
      "known_for": {"Zendaya": "American actress"}}                          # only if the index has them

POST /kirk     multipart: "file", header "X-Api-Key" (the hidden /kirk-meter page)
  -> {"face_found": bool, "score": 72.4, "thumb": "data:image/jpeg;base64,..."}   # vs the photos in kirk/

POST /compare  multipart: "file", "file2", header "X-Api-Key" (the hidden /compare page)
  -> {"face_found": [bool, bool], "score": 72.4}   # how alike the two faces are, same scale as /search

POST /landmarks  multipart: "file", header "X-Api-Key" (the ChatGPT apps: face symmetry test and Ollie Stylist)
  -> {"face_found": bool, "width", "height", "yaw", "pitch", "landmarks": [[x, y], ... 478, normalised 0..1]}
     MediaPipe Face Landmarker, the same model the browser runs, so lib/symmetry.ts and lib/style/face-shape.ts work unchanged.

POST /style-scan  multipart: "file", header "X-Api-Key" (the hidden /style page)
  -> {"face_found": bool, "age": int | None, "gender": "female" | "male" | None}   # InsightFace apparent age/sex

One mode only (the skin-tone filter is gone). Its key stays "CNN Only (best image)" so frontends
built before this change still find it; new frontends just take the first mode.

Files next to this one (made by export_for_hf.py): app_best.pt, index.npz, thumbs/, imgthumbs/.
index.npz: names + embeddings (one row per image). Optional, per image, for the celebrity index:
genders ("F"/"M"/""), categories ("actor|musician", from celeb_v2/build_list.categories), credits (JSON: author, license, license_url, page) and known_for.
Uploaded photos live in memory for the request only — never written to disk or logged.
"""
import base64
import hmac
import io
import json
import os
import threading

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from face_features import _get_insight_app, face_and_sex
from lfw_pytorch import EMBEDDING_SIZE, SphereFaceNet, test_transform
from lookalike import merge_duplicate_names, rank_players, thumb_name

HERE      = os.path.dirname(os.path.abspath(__file__))
API_KEY   = os.environ.get("API_KEY")
MAX_BYTES = 8 * 1024 * 1024  # upload cap
TOP_N     = 5
MODE      = "CNN Only (best image)"  # response key kept for older frontends
SCORE     = os.environ.get("SCORE", "best")  # each person's score = their single best photo (owner, 2026-09-25); SCORE=top2 averages their 2 best
GENDER    = {"female": "F", "male": "M"}
CATEGORY_NAMES = ("actor", "musician", "footballer")

if not API_KEY:
    raise SystemExit("Set the API_KEY env var (a Space secret on Hugging Face) — refusing to start an open API.")

# ── load once at startup ──────────────────────────────────────────────────────
model = SphereFaceNet(EMBEDDING_SIZE)
raw   = torch.load(os.path.join(HERE, "app_best.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)  # strict: fail loudly
model.eval()

with np.load(os.path.join(HERE, "index.npz")) as d:
    NAMES = merge_duplicate_names(d["names"].tolist())
    EMBS  = d["embeddings"].astype(np.float32)
    GENDERS   = d["genders"].astype(str) if "genders" in d.files else None      # soccer index: none (all men)
    CREDITS   = d["credits"].tolist() if "credits" in d.files else None
    KNOWN_FOR = d["known_for"].tolist() if "known_for" in d.files else None
    # per category, a bool per image; unlike gender, people with no category are left out
    CATEGORY  = ({k: np.array([k in c.split("|") for c in d["categories"].astype(str)]) for k in CATEGORY_NAMES}
                 if "categories" in d.files else None)


def _embed(img):
    """PIL RGB image -> (embedding, sex, face found, aligned face)."""
    face, sex, found = face_and_sex(img)
    with torch.no_grad():
        return model.get_embedding(test_transform(face).unsqueeze(0)).numpy()[0], sex, found, face


def _data_url(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Kirk meter (hidden /kirk-meter page): reference photos dropped into kirk/ next to this file, embedded once here.
# Photos without a detectable face are skipped. No kirk/ folder = /kirk answers 503, /search is unaffected.
KIRK_EMBS, KIRK_THUMBS = [], []
for _f in sorted(os.listdir(os.path.join(HERE, "kirk")) if os.path.isdir(os.path.join(HERE, "kirk")) else []):
    try:
        _emb, _, _found, _face = _embed(Image.open(os.path.join(HERE, "kirk", _f)).convert("RGB"))
    except Exception:
        continue
    if _found:
        KIRK_EMBS.append(_emb)
        KIRK_THUMBS.append(_data_url(_face))
KIRK_EMBS = np.array(KIRK_EMBS, dtype=np.float32)
print(f"kirk meter: {len(KIRK_THUMBS)} reference photos", flush=True)

# ponytail: one search at a time (CPU-bound anyway, and the InsightFace session isn't shared-safe);
# run more than one worker/replica if this ever queues up.
_lock = threading.Lock()
app   = FastAPI()


def _thumb(player, idx):
    """Thumbnail of the exact image that matched (imgthumbs/, made by export_image_thumbs.py),
    else the player's default one (thumbs/)."""
    for path in (os.path.join(HERE, "imgthumbs", f"{idx}.jpg"), os.path.join(HERE, "thumbs", thumb_name(player))):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return None


def _allowed(gender, sex):
    """-> (bool mask per image or None, the gender filtered to or None). Only people whose recorded
    gender is known and different are left out; unknown/other genders are always shown."""
    want = GENDER.get(gender, sex if gender == "auto" else None)
    if GENDERS is None or want is None:
        return None, None
    return GENDERS != ("M" if want == "F" else "F"), {"F": "female", "M": "male"}[want]


def _category(category):
    """-> (bool mask per image or None, the category filtered to or None). Unknown values = no filter."""
    if CATEGORY is None or category not in CATEGORY:
        return None, None
    return CATEGORY[category], category


@app.get("/health")
def health():
    return {"ok": True, "images": len(NAMES), "people": len(set(NAMES)), "genders": GENDERS is not None,
            "categories": CATEGORY is not None}


def _read_upload(file, x_api_key):
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Bad API key")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Not a readable image")
    img.thumbnail((1600, 1600))  # keeps face detection fast on phone-sized photos
    return img


@app.post("/kirk")
def kirk(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    """-> {"face_found": bool, "score": raw percent (same scale as /search), "thumb": the reference photo that matched}"""
    img = _read_upload(file, x_api_key)
    if not KIRK_THUMBS:
        raise HTTPException(503, "No kirk/ reference photos loaded")
    with _lock:
        q_emb, _, found, _ = _embed(img)
    _, score, i = rank_players(["kirk"] * len(KIRK_EMBS), np.linalg.norm(KIRK_EMBS - q_emb, axis=1), SCORE)[0]
    return {"face_found": found, "score": round(score, 1), "thumb": KIRK_THUMBS[i]}


@app.post("/compare")
def compare(file: UploadFile = File(...), file2: UploadFile = File(...), x_api_key: str = Header(default="")):
    """Hidden /compare page: two photos -> {"face_found": [bool, bool], "score": raw percent (same scale as /search)}"""
    a, b = _read_upload(file, x_api_key), _read_upload(file2, x_api_key)
    with _lock:
        emb_a, _, found_a, _ = _embed(a)
        emb_b, _, found_b, _ = _embed(b)
    _, score, _ = rank_players(["b"], [np.linalg.norm(emb_a - emb_b)], "best")[0]
    return {"face_found": [found_a, found_b], "score": round(score, 1)}


_landmarker = None


def _get_landmarker():
    """MediaPipe Face Landmarker (face_landmarker.task next to this file), loaded on first use."""
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import BaseOptions, vision
        _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(HERE, "face_landmarker.task")),
            output_facial_transformation_matrixes=True, num_faces=1))
    return _landmarker


@app.post("/landmarks")
def landmarks(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    """ChatGPT apps: one photo -> the 478-point face mesh plus head turn (same yaw/pitch formula as the browser)."""
    import math
    import mediapipe as mp
    img = _read_upload(file, x_api_key)
    with _lock:
        res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.array(img))))
    if not res.face_landmarks:
        return {"face_found": False}
    m = res.facial_transformation_matrixes[0]
    return {
        "face_found": True, "width": img.width, "height": img.height,
        "yaw": math.degrees(math.atan2(m[0][2], m[2][2])),
        "pitch": math.degrees(math.atan2(-m[1][2], math.hypot(m[0][2], m[2][2]))),
        "landmarks": [[round(q.x, 5), round(q.y, 5)] for q in res.face_landmarks[0]],
    }


@app.post("/style-scan")
def style_scan(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    """Hidden /style page: one camera frame -> {"face_found": bool, "age": int | None, "gender": "female" | "male" | None}.
    InsightFace's apparent age (about ±5 years), used only to offer a "look older / younger" goal."""
    img = _read_upload(file, x_api_key)
    fa = _get_insight_app()
    if fa is None:
        raise HTTPException(503, "Face model not loaded")
    with _lock:
        faces = fa.get(np.array(img)[:, :, ::-1])
    if not faces:
        return {"face_found": False, "age": None, "gender": None}
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return {"face_found": True, "age": int(face.age), "gender": {"F": "female", "M": "male"}.get(getattr(face, "sex", None))}


@app.post("/search")
def search(file: UploadFile = File(...), gender: str = Form("any"), category: str = Form("any"),
           x_api_key: str = Header(default="")):
    img = _read_upload(file, x_api_key)

    with _lock:
        q_emb, sex, found, _ = _embed(img)
        dist = np.linalg.norm(EMBS - q_emb, axis=1)
        allowed, gender_used = _allowed(gender, sex)
        in_cat, category_used = _category(category)
        if in_cat is not None:
            allowed = in_cat if allowed is None else allowed & in_cat
        rows = rank_players(NAMES, dist, SCORE, allowed)[:TOP_N]

    pretty = lambda n: n.replace("_", " ")
    out = {
        "face_found": found,
        "gender_used": gender_used,
        "category_used": category_used,
        "modes": {MODE: [{"name": pretty(n), "score": round(s, 1)} for n, s, _ in rows]},
        "thumbs": {pretty(n): t for n, _, i in rows if (t := _thumb(n, i))},
    }
    if CREDITS is not None:
        out["credits"] = {pretty(n): json.loads(CREDITS[i]) for n, _, i in rows if CREDITS[i]}
    if KNOWN_FOR is not None:
        out["known_for"] = {pretty(n): KNOWN_FOR[i] for n, _, i in rows if KNOWN_FOR[i]}
    return out
