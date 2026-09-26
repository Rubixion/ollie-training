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

from face_features import face_and_sex
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


@app.post("/search")
def search(file: UploadFile = File(...), gender: str = Form("any"), category: str = Form("any"),
           x_api_key: str = Header(default="")):
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

    with _lock:
        face, sex, found = face_and_sex(img)
        with torch.no_grad():
            q_emb = model.get_embedding(test_transform(face).unsqueeze(0)).numpy()[0]
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
