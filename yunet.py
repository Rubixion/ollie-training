"""
Face detection + ArcFace-style 5-point alignment with YuNet (OpenCV zoo, MIT licence), replacing InsightFace's
buffalo_l detector, whose weights are licensed for non-commercial research only.

faces(rgb)       -> [(box x1,y1,x2,y2, kps (5,2), score)], largest face first. kps order matches InsightFace:
                    image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
align(rgb, kps)  -> 112x112 crop warped onto the ArcFace template (same maths as insightface's norm_crop).

The server, export_for_hf.py (index) and eval_lfw_aligned.py all use this, so queries and the index are aligned
the same way. Needs opencv >= 4.8 (cv2.FaceDetectorYN with the 2023mar model).
"""
import os
import threading

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")
DET_MAX = 640       # detect on a copy no bigger than this (like InsightFace's det_size=640), align on the original
MIN_SCORE = 0.6

ARCFACE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)

_local = threading.local()  # FaceDetectorYN keeps per-size state: one per thread


def _detector():
    if not hasattr(_local, "det"):
        _local.det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), MIN_SCORE, 0.3, 5000)
    return _local.det


def faces(rgb):
    """RGB uint8 image -> list of (box, kps, score), largest face first."""
    h, w = rgb.shape[:2]
    s = min(1.0, DET_MAX / max(h, w))
    img = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    det = _detector()
    det.setInputSize((img.shape[1], img.shape[0]))
    _, rows = det.detect(np.ascontiguousarray(img[:, :, ::-1]))  # YuNet expects BGR
    out = []
    for r in (rows if rows is not None else []):
        x, y, bw, bh = r[:4] / s
        out.append((np.array([x, y, x + bw, y + bh]), r[4:14].reshape(5, 2).astype(np.float64) / s, float(r[14])))
    return sorted(out, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)


def _similarity(src, dst):
    """Least-squares similarity transform src -> dst (Umeyama), as a 2x3 matrix. Same result as
    skimage's SimilarityTransform.estimate, which insightface's norm_crop uses."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / s_c.var(0).sum()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def align(rgb, kps, size=112):
    """Warp the face onto the ArcFace template; works on RGB or BGR (channel-agnostic)."""
    M = _similarity(np.asarray(kps, dtype=np.float64), ARCFACE * (size / 112.0))
    return cv2.warpAffine(rgb, M, (size, size), borderValue=0.0)


if __name__ == "__main__":
    # self-check: the Umeyama fit recovers a known similarity transform exactly
    a = np.deg2rad(17)
    M = np.array([[1.7 * np.cos(a), -1.7 * np.sin(a), 12.0], [1.7 * np.sin(a), 1.7 * np.cos(a), -5.0]])
    src = ARCFACE + 3.0
    dst = src @ M[:, :2].T + M[:, 2]
    assert np.allclose(_similarity(src, dst), M, atol=1e-6), "similarity fit"
    assert np.allclose(align(np.zeros((200, 200, 3), np.uint8), ARCFACE).shape, (112, 112, 3))
    print("yunet.py self-check: ok")
