"""
Lookalike API for the Hugging Face Docker Space (or any host).

    API_KEY=secret uvicorn server:app --port 7860

POST /search   multipart field "file" (photo) + header "X-Api-Key"
  -> {"face_found": bool,     # modes: "CNN + Features", "CNN Only", "CNN + Features (best image)", "CNN Only (best image)"
      "modes":  {"<mode label>": [{"name": "Declan Rice", "score": 72.4}, ... top 5]},
      "thumbs": {"Declan Rice": "data:image/jpeg;base64,..."}}     # one thumbnail per player, shared by all modes

Files next to this one (made by export_for_hf.py): app_best.pt, index.npz, thumbs/.
Uploaded photos live in memory for the request only — never written to disk or logged.
"""
import base64
import hmac
import io
import os
import threading

import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

from face_features import aligned_face, extract_face_features
from lfw_pytorch import EMBEDDING_SIZE, SphereFaceNet, test_transform
from lookalike import SEARCH_MODES, rank_players, thumb_name

HERE      = os.path.dirname(os.path.abspath(__file__))
API_KEY   = os.environ.get("API_KEY")
MAX_BYTES = 8 * 1024 * 1024  # upload cap
TOP_N     = 5
# The site shows 4 modes: CNN Only / CNN + Features, each averaged over all a player's images or by best image.
# (The Gradio app in app.py still shows all 6, including the first-image-only ones.)
MODES     = [m for m in SEARCH_MODES if m[2] != "first"]

if not API_KEY:
    raise SystemExit("Set the API_KEY env var (a Space secret on Hugging Face) — refusing to start an open API.")

# ── load once at startup ──────────────────────────────────────────────────────
model = SphereFaceNet(EMBEDDING_SIZE)
raw   = torch.load(os.path.join(HERE, "app_best.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)  # strict: fail loudly
model.eval()

with np.load(os.path.join(HERE, "index.npz")) as d:
    NAMES, EMBS, FEATS = d["names"].tolist(), d["embeddings"].astype(np.float32), d["features"].astype(np.float32)

# ponytail: one search at a time (CPU-bound anyway, and the InsightFace session isn't shared-safe);
# run more than one worker/replica if this ever queues up.
_lock = threading.Lock()
app   = FastAPI()


def _thumb(player):
    path = os.path.join(HERE, "thumbs", thumb_name(player))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


@app.get("/health")
def health():
    return {"ok": True, "images": len(NAMES), "players": len(set(NAMES))}


@app.post("/search")
def search(file: UploadFile = File(...), x_api_key: str = Header(default="")):
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
        q_feats = extract_face_features(img)  # all-zero if no face found
        with torch.no_grad():
            q_emb = model.get_embedding(test_transform(aligned_face(img)).unsqueeze(0)).numpy()[0]
        dist = np.linalg.norm(EMBS - q_emb, axis=1)
        modes = {label: rank_players(NAMES, dist, FEATS, q_feats if use_feats else np.zeros_like(q_feats), agg)[:TOP_N]
                 for label, use_feats, agg in MODES}

    shown = {name for rows in modes.values() for name, _, _ in rows}
    return {
        "face_found": bool(np.any(q_feats != 0)),
        "modes": {label: [{"name": n.replace("_", " "), "score": round(s, 1)} for n, s, _ in rows]
                  for label, rows in modes.items()},
        "thumbs": {n.replace("_", " "): t for n in shown if (t := _thumb(n))},
    }
