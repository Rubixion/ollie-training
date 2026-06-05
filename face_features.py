"""
Geometric and colour features extracted from face images.

Primary path:  InsightFace (GPU via ONNX Runtime CUDA) — ~500 img/sec on GTX 1650
Fallback path: MediaPipe FaceMesh (CPU, thread-local) — full 478-landmark mesh

Feature vector (FEAT_DIM = 24):
  [0]    normalised inter-eye distance
  [1-2]  eye aspect ratio left/right  (EAR — MediaPipe only; 0.3 default on GPU path)
  [3-5]  right iris HSV (normalised)
  [6-8]  left  iris HSV (normalised)
  [9-10] iris texture variance right/left
  [11]   face width / height ratio          ← re-ranking key
  [12]   chin width ratio
  [13]   mid-jaw width ratio
  [14]   nose width/height ratio
  [15]   lip width (normalised)
  [16]   lip height (normalised)
  [17]   hair hue (HSV H, normalised)       ← re-ranking key
  [18]   hair saturation
  [19]   hair value
  [20]   skin LAB L (normalised)            ← re-ranking key
  [21]   skin LAB A
  [22]   skin LAB B
  [23]   forehead ratio
"""

import os
import pickle
import threading
import numpy as np
import cv2
from PIL import Image

FEAT_DIM  = 24
_ZERO_VEC = np.zeros(FEAT_DIM, dtype=np.float32)

# ── InsightFace (GPU) ──────────────────────────────────────────────────────────

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _HAVE_INSIGHT = True
except ImportError:
    _HAVE_INSIGHT = False
    _FaceAnalysis = None

_insight_app = None
_insight_lock = threading.Lock()


def _get_insight_app():
    global _insight_app
    if not _HAVE_INSIGHT:
        return None
    if _insight_app is None:
        with _insight_lock:
            if _insight_app is None:
                app = _FaceAnalysis(
                    allowed_modules=['detection'],
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _insight_app = app
    return _insight_app


def _iris_stats(img_np, center, radius=7):
    h, w = img_np.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    crop = img_np[max(0, cy-radius):min(h, cy+radius),
                  max(0, cx-radius):min(w, cx+radius)]
    if crop.size == 0:
        return np.zeros(3, np.float32), 0.0
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_hsv = (hsv.mean(axis=(0, 1)) / np.array([179., 255., 255.])).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return mean_hsv, float(gray.var()) / (255. ** 2)


def _hair_and_skin(img_np, x1, y1, x2, y2):
    h, w = img_np.shape[:2]
    face_w = float(x2 - x1) + 1e-6
    face_h = float(y2 - y1) + 1e-6

    # Hair: strip above forehead
    hair_y0  = max(0, y1 - int(0.3 * face_h))
    hair_crop = img_np[hair_y0:y1, x1:x2]
    if hair_crop.size > 0:
        bgr_h    = cv2.cvtColor(hair_crop, cv2.COLOR_RGB2BGR)
        hsv_h    = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hair_hsv = (hsv_h.mean(axis=(0, 1)) / np.array([179., 255., 255.])).astype(np.float32)
    else:
        hair_hsv = np.zeros(3, np.float32)

    # Skin: two cheek patches
    r       = max(4, int(0.08 * face_w))
    cheek_y = y1 + int(0.60 * face_h)
    lx      = x1 + int(0.18 * face_w)
    rx      = x2 - int(0.18 * face_w)
    patches = []
    for cx in (lx, rx):
        p = img_np[max(0, cheek_y-r):min(h, cheek_y+r),
                   max(0, cx-r):min(w, cx+r)]
        if p.size > 0:
            patches.append(p.reshape(-1, 3))
    if patches:
        combined = np.concatenate(patches, axis=0).reshape(1, -1, 3)
        bgr_s    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        lab_s    = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_lab = (lab_s.mean(axis=(0, 1)) / 255.).astype(np.float32)
    else:
        skin_lab = np.zeros(3, np.float32)

    return hair_hsv, skin_lab


def _feats_from_insight(face, img_np):
    """
    Compute feature vector from an InsightFace detection result.
    Uses 5 keypoints + bbox — covers all 3 re-ranking features exactly.
    EAR, jaw, nose, lip_h are set to sensible defaults (not used for re-ranking).
    """
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = (int(c) for c in face.bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face_w  = float(x2 - x1) + 1e-6
    face_h  = float(y2 - y1) + 1e-6

    kps = face.kps   # (5,2): left_eye, right_eye, nose, left_mouth, right_mouth

    eye_dist   = float(np.linalg.norm(kps[1] - kps[0])) / face_w
    r_hsv, r_var = _iris_stats(img_np, kps[1])
    l_hsv, l_var = _iris_stats(img_np, kps[0])
    lip_w_norm = float(np.linalg.norm(kps[4] - kps[3])) / face_w
    eye_y_mid  = float((kps[0][1] + kps[1][1]) / 2)
    fore_ratio = (eye_y_mid - y1) / face_h
    hair_hsv, skin_lab = _hair_and_skin(img_np, x1, y1, x2, y2)

    return np.array([
        eye_dist,
        0.30, 0.30,            # EAR — 5-kp path can't compute, use typical
        *r_hsv, *l_hsv,
        r_var, l_var,
        face_w / face_h,       # [11] face shape  ← re-ranking
        0.0, 0.0,              # [12-13] jaw — not available
        0.0,                   # [14] nose ratio — not available
        lip_w_norm, 0.0,       # [15-16] lip w / h
        *hair_hsv,             # [17-19] hair HSV  ← re-ranking
        *skin_lab,             # [20-22] skin LAB  ← re-ranking
        fore_ratio,            # [23]
    ], dtype=np.float32)


# ── MediaPipe FaceMesh (CPU fallback, thread-local) ───────────────────────────

try:
    import mediapipe as mp
    _mp_mesh_cls = mp.solutions.face_mesh.FaceMesh
    _HAVE_MP = True
except (ImportError, AttributeError):
    _HAVE_MP = False
    _mp_mesh_cls = None

_tls = threading.local()


def _get_mesh():
    if not _HAVE_MP:
        return None
    if not hasattr(_tls, 'mesh') or _tls.mesh is None:
        _tls.mesh = _mp_mesh_cls(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
        )
    return _tls.mesh


# ── public API ────────────────────────────────────────────────────────────────

def using_gpu() -> bool:
    """True when InsightFace GPU path is active."""
    return _HAVE_INSIGHT


def extract_face_features(img) -> np.ndarray:
    """
    img: PIL.Image (RGB) or np.ndarray (H, W, 3) uint8 RGB.
    Returns ndarray shape (FEAT_DIM,).
    Tries InsightFace GPU first; falls back to MediaPipe CPU.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert('RGB'), dtype=np.uint8)
    else:
        img_np = np.asarray(img, dtype=np.uint8)

    if max(img_np.shape[:2]) < 64:
        return _ZERO_VEC.copy()

    # ── GPU path ──────────────────────────────────────────────────────────────
    app = _get_insight_app()
    if app is not None:
        try:
            faces = app.get(img_np)
            if faces:
                face = max(faces, key=lambda f: float(f.det_score))
                return _feats_from_insight(face, img_np)
        except Exception:
            pass
        return _ZERO_VEC.copy()

    # ── CPU fallback (MediaPipe) ───────────────────────────────────────────────
    img_pil = Image.fromarray(img_np)
    orig_w, orig_h = img_pil.size
    mesh       = _get_mesh()
    if mesh is None:
        return _ZERO_VEC.copy()

    best_result, best_np = None, None
    tried = set()
    for target in [480, orig_w, 320, 640, 240]:
        max_side = max(orig_w, orig_h)
        scale    = target / max_side
        new_w    = max(1, int(orig_w * scale))
        new_h    = max(1, int(orig_h * scale))
        if (new_w, new_h) in tried:
            continue
        tried.add((new_w, new_h))
        attempt = img_pil.resize((new_w, new_h), Image.LANCZOS) if (new_w, new_h) != (orig_w, orig_h) else img_pil
        arr     = np.array(attempt, dtype=np.uint8).copy()
        result  = mesh.process(arr)
        if result.multi_face_landmarks:
            best_result, best_np = result, arr
            break

    if best_result is None:
        return _ZERO_VEC.copy()

    img_np2 = best_np
    h, w    = img_np2.shape[:2]
    lm      = best_result.multi_face_landmarks[0].landmark
    pts     = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)

    if len(pts) < 478:
        return _ZERO_VEC.copy()

    face_w = float(np.linalg.norm(pts[454] - pts[234])) + 1e-6
    face_h = float(np.linalg.norm(pts[152] - pts[10]))  + 1e-6
    eye_dist = float(np.linalg.norm(pts[473] - pts[468])) / face_w

    def ear(p_outer, p_inner, p_top, p_bot):
        return float(np.linalg.norm(pts[p_top] - pts[p_bot]) /
                     (np.linalg.norm(pts[p_inner] - pts[p_outer]) + 1e-6))

    left_ear  = ear(33,  133, 159, 145)
    right_ear = ear(362, 263, 386, 374)

    r_hsv, r_var = _iris_stats(img_np2, pts[468])
    l_hsv, l_var = _iris_stats(img_np2, pts[473])

    mid_jaw_w  = float(np.linalg.norm(pts[397] - pts[172])) / face_w
    chin_w     = float(np.linalg.norm(pts[288] - pts[58]))  / face_w
    nose_w     = float(np.linalg.norm(pts[358] - pts[129])) + 1e-6
    nose_h     = float(np.linalg.norm(pts[4]   - pts[168])) + 1e-6
    nose_ratio = nose_w / nose_h
    lip_w      = float(np.linalg.norm(pts[291] - pts[61]))
    lip_h      = float(np.linalg.norm(pts[14]  - pts[13])) + 1e-6
    lip_w_norm = lip_w / face_w
    lip_h_norm = lip_h / face_h

    fore_y  = int(pts[10][1])
    hair_y0 = max(0, fore_y - int(0.45 * face_h))
    fx1     = max(0, int(pts[234][0]))
    fx2     = min(w, int(pts[454][0]))
    hair_crop = img_np2[hair_y0:fore_y, fx1:fx2]
    if hair_crop.size > 0:
        bgr_h    = cv2.cvtColor(hair_crop, cv2.COLOR_RGB2BGR)
        hsv_h    = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hair_hsv = (hsv_h.mean(axis=(0, 1)) / np.array([179., 255., 255.])).astype(np.float32)
    else:
        hair_hsv = np.zeros(3, np.float32)

    skin_patches = []
    for lm_idx in (234, 454):
        cx, cy = int(pts[lm_idx][0]), int(pts[lm_idx][1])
        r = 12
        patch = img_np2[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
        if patch.size > 0:
            skin_patches.append(patch.reshape(-1, 3))
    if skin_patches:
        combined = np.concatenate(skin_patches, axis=0).reshape(1, -1, 3)
        bgr_s    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        lab_s    = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_lab = (lab_s.mean(axis=(0, 1)) / 255.).astype(np.float32)
    else:
        skin_lab = np.zeros(3, np.float32)

    eye_y_mid      = float((pts[468][1] + pts[473][1]) / 2)
    forehead_ratio = (eye_y_mid - pts[10][1]) / face_h

    return np.array([
        eye_dist, left_ear, right_ear,
        *r_hsv, *l_hsv,
        r_var, l_var,
        face_w / face_h,
        chin_w, mid_jaw_w,
        nose_ratio,
        lip_w_norm, lip_h_norm,
        *hair_hsv,
        *skin_lab,
        forehead_ratio,
    ], dtype=np.float32)


# ── disk-cached batch extraction ──────────────────────────────────────────────

def build_feature_cache(image_paths, cache_path="feature_cache.pkl", progress_cb=None):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            loaded = pickle.load(f)
        sample = next(iter(loaded.values()), None) if loaded else None
        if sample is not None and len(sample) == FEAT_DIM:
            cache = loaded

    todo  = [p for p in image_paths if p not in cache]
    total = len(todo)
    dirty = False

    for i, path in enumerate(todo):
        try:
            cache[path] = extract_face_features(Image.open(path).convert('RGB'))
        except Exception:
            cache[path] = _ZERO_VEC.copy()
        dirty = True
        if progress_cb:
            progress_cb(i + 1, total)

    if dirty:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    return cache
