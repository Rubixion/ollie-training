"""
Geometric and colour features extracted from face images.

Primary path:  InsightFace GPU (ONNX Runtime CUDA)
Fallback path: MediaPipe FaceMesh CPU (thread-local)

Feature vector — FEAT_DIM = 32
  [0]    inter-eye distance / face_w
  [1-2]  EAR left/right  (0.30 default on GPU path)
  [3-5]  right iris HSV (normalised 0-1)
  [6-8]  left  iris HSV (normalised 0-1)
  [9-10] iris texture variance right/left
  [11]   face w/h ratio                  ← re-rank ×1
  [12]   chin width ratio
  [13]   mid-jaw width ratio
  [14]   nose width/height ratio
  [15]   lip width (norm)
  [16]   lip height (norm)
  [17]   hair hue  (H/179)               ← re-rank ×2
  [18]   hair saturation
  [19]   hair value
  [20]   skin LAB L (norm)               ← re-rank ×5
  [21]   skin LAB A
  [22]   skin LAB B
  [23]   forehead ratio
  [24]   eye-to-nose vertical ratio
  [25]   nose-to-mouth vertical ratio
  [26]   mouth-to-chin ratio
  [27]   hair length score  (0=short, 1=long)
  [28]   age / 100                        ← re-rank ×3
  [29]   gender score (0=female, 1=male)  ← re-rank ×4
  [30]   face area ratio (face px / image px, log-scaled)
  [31]   forehead skin L (norm)
"""

import os
import pickle
import threading
import numpy as np
import cv2
from PIL import Image

FEAT_DIM  = 32
_ZERO_VEC = np.zeros(FEAT_DIM, dtype=np.float32)

FEAT_NAMES = [
    "eye_dist", "EAR_L", "EAR_R",
    "iris_R_H", "iris_R_S", "iris_R_V",
    "iris_L_H", "iris_L_S", "iris_L_V",
    "iris_R_var", "iris_L_var",
    "face_ratio",
    "chin_w", "midjaw_w", "nose_ratio",
    "lip_w", "lip_h",
    "hair_H", "hair_S", "hair_V",
    "skin_L", "skin_A", "skin_B",
    "forehead_ratio",
    "eye_nose_ratio", "nose_mouth_ratio", "mouth_chin_ratio",
    "hair_length",
    "age_norm", "gender",
    "face_area", "forehead_L",
]

# ── shared helpers ─────────────────────────────────────────────────────────────

def _iris_stats(img_np, center, radius=7):
    h, w = img_np.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    crop = img_np[max(0,cy-radius):min(h,cy+radius),
                  max(0,cx-radius):min(w,cx+radius)]
    if crop.size == 0:
        return np.zeros(3, np.float32), 0.0
    bgr  = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2HSV).astype(np.float32)
    mean = (hsv.mean(axis=(0,1)) / np.array([179.,255.,255.])).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return mean, float(gray.var()) / (255.**2)


def _hair_and_skin(img_np, x1, y1, x2, y2):
    h, w    = img_np.shape[:2]
    face_w  = float(x2-x1)+1e-6
    face_h  = float(y2-y1)+1e-6

    # Hair strip above forehead
    hair_y0   = max(0, y1 - int(0.3*face_h))
    hair_crop = img_np[hair_y0:y1, x1:x2]
    if hair_crop.size > 0:
        bgr_h    = cv2.cvtColor(hair_crop, cv2.COLOR_RGB2BGR)
        hsv_h    = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hair_hsv = (hsv_h.mean(axis=(0,1)) / np.array([179.,255.,255.])).astype(np.float32)
    else:
        hair_hsv = np.zeros(3, np.float32)

    # Cheek patches
    r       = max(4, int(0.08*face_w))
    cheek_y = y1 + int(0.60*face_h)
    lx, rx  = x1+int(0.18*face_w), x2-int(0.18*face_w)
    patches = []
    for cx in (int(lx), int(rx)):
        p = img_np[max(0,cheek_y-r):min(h,cheek_y+r),
                   max(0,cx-r):min(w,cx+r)]
        if p.size > 0:
            patches.append(p.reshape(-1,3))
    if patches:
        combined = np.concatenate(patches).reshape(1,-1,3)
        bgr_s    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        lab_s    = cv2.cvtColor(bgr_s,    cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_lab = (lab_s.mean(axis=(0,1)) / 255.).astype(np.float32)
    else:
        skin_lab = np.zeros(3, np.float32)

    return hair_hsv, skin_lab


def _forehead_L(img_np, x1, y1, x2, y2, eye_y):
    h, w    = img_np.shape[:2]
    face_h  = float(y2-y1)+1e-6
    fy1 = max(0,  y1 + int(0.05*face_h))
    fy2 = max(fy1+1, int(eye_y) - int(0.05*face_h))
    fx1 = max(0,  x1 + int(0.25*(x2-x1)))
    fx2 = min(w,  x2 - int(0.25*(x2-x1)))
    patch = img_np[fy1:fy2, fx1:fx2]
    if patch.size == 0:
        return 0.0
    bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr,   cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lab[:,:,0].mean()) / 255.0


def _hair_length(img_np, x1, y1, x2, y2, hair_hsv):
    h, w   = img_np.shape[:2]
    face_h = float(y2-y1)+1e-6
    face_w = float(x2-x1)+1e-6

    cy1 = y2
    cy2 = min(h, int(y2 + 1.5*face_h))
    margin = max(0, int(0.1*face_w))
    cx1 = max(0, x1+margin)
    cx2 = min(w, x2-margin)
    if cy2 <= cy1 or cx2 <= cx1:
        return 0.0

    region = img_np[cy1:cy2, cx1:cx2]
    if region.size == 0:
        return 0.0

    bgr      = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    hsv      = cv2.cvtColor(bgr,    cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_norm = hsv / np.array([179.,255.,255.])

    hue_diff = np.abs(hsv_norm[:,:,0] - float(hair_hsv[0]))
    hue_diff = np.minimum(hue_diff, 1.0-hue_diff)
    val      = hsv_norm[:,:,2]
    sat      = hsv_norm[:,:,1]

    hair_pixels = ((hue_diff < 0.12) & (sat > 0.10)) | (val < 0.20)
    row_frac    = hair_pixels.mean(axis=1)
    n = len(row_frac)
    if n == 0:
        return 0.0
    weights = np.linspace(1.0, 2.0, n)
    score   = float((row_frac * weights).sum()) / (weights.sum()+1e-6)
    return min(1.0, score * 2.0)


# ── InsightFace GPU ────────────────────────────────────────────────────────────

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _HAVE_INSIGHT = True
except ImportError:
    _HAVE_INSIGHT = False
    _FaceAnalysis  = None

_insight_app  = None
_insight_lock = threading.Lock()


def _get_insight_app():
    global _insight_app
    if not _HAVE_INSIGHT:
        return None
    if _insight_app is None:
        with _insight_lock:
            if _insight_app is None:
                app = _FaceAnalysis(
                    allowed_modules=['detection', 'genderage'],
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _insight_app = app
    return _insight_app


def _feats_from_insight(face, img_np):
    h, w    = img_np.shape[:2]
    x1,y1,x2,y2 = (int(c) for c in face.bbox)
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w,x2), min(h,y2)
    face_w = float(x2-x1)+1e-6
    face_h = float(y2-y1)+1e-6

    kps = face.kps  # left_eye[0], right_eye[1], nose[2], l_mouth[3], r_mouth[4]

    eye_dist   = float(np.linalg.norm(kps[1]-kps[0])) / face_w
    r_hsv,r_var = _iris_stats(img_np, kps[1])
    l_hsv,l_var = _iris_stats(img_np, kps[0])
    lip_w_norm  = float(np.linalg.norm(kps[4]-kps[3])) / face_w
    eye_y       = float((kps[0][1]+kps[1][1])/2)
    fore_ratio  = (eye_y - y1) / face_h
    mouth_y     = float((kps[3][1]+kps[4][1])/2)
    nose_y      = float(kps[2][1])

    hair_hsv, skin_lab = _hair_and_skin(img_np, x1, y1, x2, y2)
    fore_l             = _forehead_L(img_np, x1, y1, x2, y2, eye_y)
    hair_len           = _hair_length(img_np, x1, y1, x2, y2, hair_hsv)

    eye_nose_ratio   = (nose_y  - eye_y)  / face_h
    nose_mouth_ratio = (mouth_y - nose_y) / face_h
    mouth_chin_ratio = (y2      - mouth_y)/ face_h
    face_area        = min(1.0, float(np.log1p((face_w*face_h)/(w*h+1e-6))))

    age_norm = 0.0
    gender   = 0.0
    if hasattr(face, 'age') and face.age is not None:
        age_norm = float(face.age) / 100.0
    sex = getattr(face, 'sex', None) or getattr(face, 'gender', None)
    if sex is not None:
        if isinstance(sex, str):
            gender = 1.0 if sex == 'M' else 0.0
        else:
            gender = float(sex)

    return np.array([
        eye_dist,
        0.30, 0.30,
        *r_hsv, *l_hsv,
        r_var, l_var,
        face_w/face_h,
        0.0, 0.0,
        0.0,
        lip_w_norm, 0.0,
        *hair_hsv,
        *skin_lab,
        fore_ratio,
        eye_nose_ratio, nose_mouth_ratio, mouth_chin_ratio,
        hair_len,
        age_norm, gender,
        face_area, fore_l,
    ], dtype=np.float32)


# ── MediaPipe CPU fallback ─────────────────────────────────────────────────────

try:
    import mediapipe as mp
    _mp_mesh_cls = mp.solutions.face_mesh.FaceMesh
    _HAVE_MP = True
except (ImportError, AttributeError):
    _HAVE_MP     = False
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
    return _HAVE_INSIGHT


def extract_face_features(img) -> np.ndarray:
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
    mesh = _get_mesh()
    if mesh is None:
        return _ZERO_VEC.copy()

    best_result = best_np = None
    tried = set()
    for target in [480, orig_w, 320, 640, 240]:
        scale = target / max(orig_w, orig_h)
        nw = max(1, int(orig_w*scale)); nh = max(1, int(orig_h*scale))
        if (nw,nh) in tried: continue
        tried.add((nw,nh))
        attempt = img_pil.resize((nw,nh), Image.LANCZOS) if (nw,nh)!=(orig_w,orig_h) else img_pil
        arr    = np.array(attempt, dtype=np.uint8).copy()
        result = mesh.process(arr)
        if result.multi_face_landmarks:
            best_result, best_np = result, arr; break

    if best_result is None:
        return _ZERO_VEC.copy()

    img_np2 = best_np
    h, w    = img_np2.shape[:2]
    lm      = best_result.multi_face_landmarks[0].landmark
    pts     = np.array([[l.x*w, l.y*h] for l in lm], dtype=np.float32)
    if len(pts) < 478:
        return _ZERO_VEC.copy()

    face_w = float(np.linalg.norm(pts[454]-pts[234]))+1e-6
    face_h = float(np.linalg.norm(pts[152]-pts[10]))+1e-6

    eye_dist  = float(np.linalg.norm(pts[473]-pts[468])) / face_w

    def ear(po,pi,pt,pb):
        return float(np.linalg.norm(pts[pt]-pts[pb]) /
                     (np.linalg.norm(pts[pi]-pts[po])+1e-6))
    left_ear  = ear(33,  133, 159, 145)
    right_ear = ear(362, 263, 386, 374)

    r_hsv,r_var = _iris_stats(img_np2, pts[468])
    l_hsv,l_var = _iris_stats(img_np2, pts[473])

    mid_jaw_w  = float(np.linalg.norm(pts[397]-pts[172])) / face_w
    chin_w     = float(np.linalg.norm(pts[288]-pts[58]))  / face_w
    nose_w     = float(np.linalg.norm(pts[358]-pts[129]))+1e-6
    nose_h     = float(np.linalg.norm(pts[4]-pts[168]))+1e-6
    nose_ratio = nose_w / nose_h
    lip_w      = float(np.linalg.norm(pts[291]-pts[61]))
    lip_h      = float(np.linalg.norm(pts[14]-pts[13]))+1e-6
    lip_w_norm = lip_w / face_w
    lip_h_norm = lip_h / face_h

    fore_y  = int(pts[10][1])
    fx1,fx2 = max(0,int(pts[234][0])), min(w,int(pts[454][0]))
    hair_crop = img_np2[max(0,fore_y-int(0.45*face_h)):fore_y, fx1:fx2]
    if hair_crop.size > 0:
        bgr_h    = cv2.cvtColor(hair_crop, cv2.COLOR_RGB2BGR)
        hsv_h    = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hair_hsv = (hsv_h.mean(axis=(0,1)) / np.array([179.,255.,255.])).astype(np.float32)
    else:
        hair_hsv = np.zeros(3, np.float32)

    skin_patches = []
    for idx in (234, 454):
        cx,cy = int(pts[idx][0]), int(pts[idx][1])
        p = img_np2[max(0,cy-12):min(h,cy+12), max(0,cx-12):min(w,cx+12)]
        if p.size > 0: skin_patches.append(p.reshape(-1,3))
    if skin_patches:
        combined = np.concatenate(skin_patches).reshape(1,-1,3)
        bgr_s    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        lab_s    = cv2.cvtColor(bgr_s,    cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_lab = (lab_s.mean(axis=(0,1)) / 255.).astype(np.float32)
    else:
        skin_lab = np.zeros(3, np.float32)

    eye_y      = float((pts[468][1]+pts[473][1])/2)
    fore_ratio = (eye_y - pts[10][1]) / face_h

    # New features
    # Face bbox approximation from landmarks
    x1b = int(pts[234][0]); y1b = int(pts[10][1])
    x2b = int(pts[454][0]); y2b = int(pts[152][1])
    fore_l  = _forehead_L(img_np2, x1b, y1b, x2b, y2b, eye_y)
    hair_len = _hair_length(img_np2, x1b, y1b, x2b, y2b, hair_hsv)
    face_area = min(1.0, float(np.log1p((face_w*face_h)/(w*h+1e-6))))

    nose_tip_y   = float(pts[4][1])
    mouth_ctr_y  = float((pts[13][1]+pts[14][1])/2)
    chin_y       = float(pts[152][1])
    top_y        = float(pts[10][1])

    eye_nose_ratio   = (nose_tip_y   - eye_y)    / face_h
    nose_mouth_ratio = (mouth_ctr_y  - nose_tip_y) / face_h
    mouth_chin_ratio = (chin_y       - mouth_ctr_y) / face_h

    return np.array([
        eye_dist, left_ear, right_ear,
        *r_hsv, *l_hsv,
        r_var, l_var,
        face_w/face_h,
        chin_w, mid_jaw_w,
        nose_ratio,
        lip_w_norm, lip_h_norm,
        *hair_hsv,
        *skin_lab,
        fore_ratio,
        eye_nose_ratio, nose_mouth_ratio, mouth_chin_ratio,
        hair_len,
        0.0, 0.0,          # age/gender not available in MediaPipe path
        face_area, fore_l,
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
    dirty = False
    for i, path in enumerate(todo):
        try:
            cache[path] = extract_face_features(Image.open(path).convert('RGB'))
        except Exception:
            cache[path] = _ZERO_VEC.copy()
        dirty = True
        if progress_cb:
            progress_cb(i+1, len(todo))

    if dirty:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    return cache
