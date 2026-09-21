"""
Lookalike API for the Hugging Face Docker Space (or any host).

    API_KEY=secret uvicorn server:app --port 7860

POST /search   multipart field "file" (photo) + header "X-Api-Key"
  -> {"face_found": bool,     # modes: "CNN Only (best image)"
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
from lookalike import merge_duplicate_names, rank_players, thumb_name

HERE      = os.path.dirname(os.path.abspath(__file__))
API_KEY   = os.environ.get("API_KEY")
MAX_BYTES = 8 * 1024 * 1024  # upload cap
TOP_N     = 5
# The site offers two modes, each scored by a player's best image: the CNN with the skin-tone sanity
# gate (default; keeps its old label so older frontends still work) and the raw CNN with no gate.
# (The Gradio app in app.py has its own SEARCH_MODES.)
MODES     = [("CNN Only (best image)", True, "best"), ("CNN Only (best image, no tweaks)", False, "best")]

if not API_KEY:
    raise SystemExit("Set the API_KEY env var (a Space secret on Hugging Face) — refusing to start an open API.")

# ── load once at startup ──────────────────────────────────────────────────────
model = SphereFaceNet(EMBEDDING_SIZE)
raw   = torch.load(os.path.join(HERE, "app_best.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)  # strict: fail loudly
model.eval()

with np.load(os.path.join(HERE, "index.npz")) as d:
    NAMES, EMBS, FEATS = merge_duplicate_names(d["names"].tolist()), d["embeddings"].astype(np.float32), d["features"].astype(np.float32)

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

    shown = {name: idx for rows in modes.values() for name, _, idx in rows}  # player -> its best-matching image
    return {
        "face_found": bool(np.any(q_feats != 0)),
        "modes": {label: [{"name": n.replace("_", " "), "score": round(s, 1)} for n, s, _ in rows]
                  for label, rows in modes.items()},
        "thumbs": {n.replace("_", " "): t for n, idx in shown.items() if (t := _thumb(n, idx))},
    }
