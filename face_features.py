"""
Geometric and colour features extracted from face images via MediaPipe FaceMesh.

Features (FEAT_DIM = 12):
  [0]   normalised inter-eye distance  (eye_dist / face_width)
  [1]   left  eye aspect ratio (EAR)
  [2]   right eye aspect ratio (EAR)
  [3-5] right iris mean HSV  (each normalised to [0, 1])
  [6-8] left  iris mean HSV  (each normalised to [0, 1])
  [9]   right iris texture variance (normalised)
  [10]  left  iris texture variance (normalised)
  [11]  face width / height ratio

Returns zero vector when MediaPipe is absent or no face is detected.
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image

FEAT_DIM  = 24   # 12 geometric + 2 jaw ratios + nose + 2 lip + 3 hair HSV + 3 skin LAB + forehead
_ZERO_VEC = np.zeros(FEAT_DIM, dtype=np.float32)

# ── optional mediapipe ─────────────────────────────────────────────────────────

try:
    import mediapipe as mp
    _mp_mesh_cls = mp.solutions.face_mesh.FaceMesh
    _HAVE_MP = True
except (ImportError, AttributeError):
    # AttributeError = mediapipe >=0.10.14 dropped mp.solutions; pin to 0.10.9
    _HAVE_MP = False
    _mp_mesh_cls = None

_mesh = None


def _get_mesh():
    global _mesh
    if _mesh is None and _HAVE_MP:
        _mesh = _mp_mesh_cls(
            static_image_mode=True,
            refine_landmarks=True,   # adds iris landmarks 468–477
            max_num_faces=1,
            min_detection_confidence=0.3,
        )
    return _mesh


# ── per-image extraction ───────────────────────────────────────────────────────

def extract_face_features(img) -> np.ndarray:
    """
    img: PIL.Image (RGB) or np.ndarray (H, W, 3) uint8 RGB.
    Returns ndarray shape (FEAT_DIM,) — zeros if detection fails or mediapipe absent.
    """
    if not _HAVE_MP:
        return _ZERO_VEC.copy()

    if isinstance(img, Image.Image):
        img_pil = img.convert('RGB')
    else:
        img_pil = Image.fromarray(np.asarray(img, dtype=np.uint8), 'RGB')

    orig_w, orig_h = img_pil.size
    if max(orig_w, orig_h) < 64:
        return _ZERO_VEC.copy()

    # Try detection at several target sizes — MediaPipe has a sweet spot ~320-640px.
    # Use .copy() to ensure the array is writeable (MediaPipe requires this).
    mesh = _get_mesh()
    best_result = None
    best_np     = None
    tried       = set()

    for target in [480, orig_w, 320, 640, 240]:
        max_side = max(orig_w, orig_h)
        scale    = target / max_side
        new_w    = max(1, int(orig_w * scale))
        new_h    = max(1, int(orig_h * scale))
        key      = (new_w, new_h)
        if key in tried:
            continue
        tried.add(key)

        attempt = img_pil.resize((new_w, new_h), Image.LANCZOS) if key != (orig_w, orig_h) else img_pil
        arr     = np.array(attempt, dtype=np.uint8).copy()
        result  = mesh.process(arr)
        if result.multi_face_landmarks:
            best_result = result
            best_np     = arr
            break

    if best_result is None:
        return _ZERO_VEC.copy()

    img_np = best_np
    h, w   = img_np.shape[:2]

    lm  = best_result.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)

    if len(pts) < 478:          # iris landmarks require refine_landmarks=True
        return _ZERO_VEC.copy()

    # ── face dimensions ──────────────────────────────────────────────────────
    face_w = float(np.linalg.norm(pts[454] - pts[234])) + 1e-6   # cheek-to-cheek
    face_h = float(np.linalg.norm(pts[152] - pts[10]))  + 1e-6   # chin-to-forehead

    # ── normalised inter-eye distance ─────────────────────────────────────────
    # mediapipe: right iris centre = 468, left iris centre = 473
    eye_dist = float(np.linalg.norm(pts[473] - pts[468])) / face_w

    # ── Eye Aspect Ratio (vertical / horizontal diameter) ────────────────────
    def ear(p_outer, p_inner, p_top, p_bot):
        h_span = np.linalg.norm(pts[p_inner] - pts[p_outer]) + 1e-6
        v_span = np.linalg.norm(pts[p_top]   - pts[p_bot])
        return float(v_span / h_span)

    left_ear  = ear(33,  133, 159, 145)
    right_ear = ear(362, 263, 386, 374)

    # ── iris colour & texture ─────────────────────────────────────────────────
    def iris_stats(centre_pt, radius=7):
        cx, cy = int(centre_pt[0]), int(centre_pt[1])
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(3, np.float32), 0.0
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        mean_hsv = hsv.mean(axis=(0, 1)) / np.array([179.0, 255.0, 255.0])
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        variance = float(gray.var()) / (255.0 ** 2)
        return mean_hsv.astype(np.float32), variance

    r_hsv, r_var = iris_stats(pts[468])  # right iris
    l_hsv, l_var = iris_stats(pts[473])  # left  iris

    # ── jaw shape ratios ─────────────────────────────────────────────────────
    # pts[172]/[397] = mid-jaw,  pts[58]/[288] = lower jaw / chin area
    mid_jaw_w  = float(np.linalg.norm(pts[397] - pts[172])) / face_w
    chin_w     = float(np.linalg.norm(pts[288] - pts[58]))  / face_w

    # ── nose ─────────────────────────────────────────────────────────────────
    # pts[129]/[358] = nostril outer edges,  pts[4] = tip,  pts[168] = bridge
    nose_w     = float(np.linalg.norm(pts[358] - pts[129])) + 1e-6
    nose_h     = float(np.linalg.norm(pts[4]   - pts[168])) + 1e-6
    nose_ratio = nose_w / nose_h

    # ── lips ─────────────────────────────────────────────────────────────────
    # pts[61]/[291] = corners,  pts[13]/[14] = inner top/bottom
    lip_w      = float(np.linalg.norm(pts[291] - pts[61]))
    lip_h      = float(np.linalg.norm(pts[14]  - pts[13])) + 1e-6
    lip_w_norm = lip_w / face_w
    lip_h_norm = lip_h / face_h

    # ── hair colour (region above forehead landmark 10) ───────────────────────
    fore_y  = int(pts[10][1])
    hair_y0 = max(0, fore_y - int(0.45 * face_h))
    fx1     = max(0, int(pts[234][0]))
    fx2     = min(w, int(pts[454][0]))
    hair_crop = img_np[hair_y0:fore_y, fx1:fx2]
    if hair_crop.size > 0:
        bgr_h    = cv2.cvtColor(hair_crop, cv2.COLOR_RGB2BGR)
        hsv_h    = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hair_hsv = (hsv_h.mean(axis=(0, 1)) / np.array([179.0, 255.0, 255.0])).astype(np.float32)
    else:
        hair_hsv = np.zeros(3, np.float32)

    # ── skin tone (small cheek patches) ──────────────────────────────────────
    skin_patches = []
    for lm_idx in (234, 454):
        cx, cy = int(pts[lm_idx][0]), int(pts[lm_idx][1])
        r = 12
        patch = img_np[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
        if patch.size > 0:
            skin_patches.append(patch.reshape(-1, 3))
    if skin_patches:
        combined  = np.concatenate(skin_patches, axis=0).reshape(1, -1, 3)
        bgr_s     = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        lab_s     = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_lab  = (lab_s.mean(axis=(0, 1)) / 255.0).astype(np.float32)
    else:
        skin_lab = np.zeros(3, np.float32)

    # ── forehead proportion ───────────────────────────────────────────────────
    eye_y_mid      = float((pts[468][1] + pts[473][1]) / 2)
    forehead_ratio = (eye_y_mid - pts[10][1]) / face_h

    return np.array([
        # original 12
        eye_dist, left_ear, right_ear,
        *r_hsv, *l_hsv,
        r_var, l_var,
        face_w / face_h,
        # new 12 — shape, colour, proportions
        chin_w, mid_jaw_w,
        nose_ratio,
        lip_w_norm, lip_h_norm,
        *hair_hsv,
        *skin_lab,
        forehead_ratio,
    ], dtype=np.float32)


# ── disk-cached batch extraction ──────────────────────────────────────────────

def build_feature_cache(image_paths, cache_path="feature_cache.pkl",
                        progress_cb=None):
    """
    Pre-compute and cache features for a list of image paths.
    Loads any existing entries from cache_path, only computes missing ones.
    progress_cb(done: int, total: int) called after each newly-computed image.
    Returns dict {str path -> ndarray (FEAT_DIM,)}.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            loaded = pickle.load(f)
        # discard cache if feature dimension changed (e.g. after FEAT_DIM update)
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
