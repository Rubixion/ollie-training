"""Norms for the /face-symmetry-test tool (ollie-frontend/lib/symmetry.ts).

Runs the same MediaPipe Face Landmarker as the browser on straight-on photos in data/ (one per person), then:
  1. mirror pairs: each of the 468 mesh points' left/right partner, found on the average aligned face
     (a point's partner is whichever point sits nearest its mirror image), so no hand-typed index list;
  2. the asymmetry of every photo, measured exactly as symmetry.ts does (mirror the face, best rigid fit
     onto itself, RMS leftover distance in inter-eye units), overall and per region;
  3. quantiles of those numbers, so the tool can say "more symmetric than N% of faces".

    .venv\\Scripts\\python celeb_v2\\symmetry_norms.py   (from "neural network learning")
      -> ollie-frontend/lib/symmetry-norms.json
"""
import json, math, os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ollie-frontend", "lib", "symmetry-norms.json")
MAX_YAW, MAX_PITCH = 6, 12  # degrees; the browser rejects photos outside the same limits
LIMIT = 4000                # people; plenty for stable quantiles
N = 468                     # mesh points (the 10 iris points are left out)
EYES = ([33, 133], [362, 263])  # corners of each eye; their centres set position, roll and scale

C = vision.FaceLandmarksConnections
REGIONS = {
    "eyes": C.FACE_LANDMARKS_LEFT_EYE + C.FACE_LANDMARKS_RIGHT_EYE,
    "brows": C.FACE_LANDMARKS_LEFT_EYEBROW + C.FACE_LANDMARKS_RIGHT_EYEBROW,
    "mouth": C.FACE_LANDMARKS_LIPS,
    "jaw": C.FACE_LANDMARKS_FACE_OVAL,
}
NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 129, 168, 195, 197, 209, 218, 275, 278, 294, 326, 327, 344, 358, 429, 438, 440]


def pose(m):
    """4x4 face transform -> (yaw, pitch) in degrees, same formula as the browser and face_norms.py."""
    return math.degrees(math.atan2(m[0][2], m[2][2])), math.degrees(math.atan2(-m[1][2], math.hypot(m[0][2], m[2][2])))


def align(p):
    """Eye centres to (+-0.5, 0): removes position, roll and size."""
    l, r = p[EYES[0]].mean(0), p[EYES[1]].mean(0)
    v = r - l
    s = np.linalg.norm(v)
    c, si = v / s
    rot = np.array([[c, si], [-si, c]])
    return (p - (l + r) / 2) @ rot.T / s


def asymmetry(p, partner):
    """Per-point leftover distance after mirroring the face and fitting it back onto itself (rotation + shift).
    Same maths as asymmetry() in symmetry.ts."""
    q = p[partner] * [-1, 1]
    pc, qc = p - p.mean(0), q - q.mean(0)
    a = (qc[:, 0] * pc[:, 0] + qc[:, 1] * pc[:, 1]).sum()
    b = (qc[:, 0] * pc[:, 1] - qc[:, 1] * pc[:, 0]).sum()
    t = math.atan2(b, a)
    c, s = math.cos(t), math.sin(t)
    fit = qc @ np.array([[c, s], [-s, c]])
    return np.linalg.norm(fit - pc, axis=1) / 2  # each point counts both halves of a pair: halve to get per side


def main():
    lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(HERE, "models", "face_landmarker.task")),
        output_facial_transformation_matrixes=True, num_faces=1))
    faces = []
    dirs = sorted(d for d in os.listdir(os.path.join(HERE, "data")) if os.path.isdir(os.path.join(HERE, "data", d)))
    for n, person in enumerate(dirs):
        if len(faces) >= LIMIT:
            break
        folder = os.path.join(HERE, "data", person)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".jpg"):
                continue
            try:
                img = mp.Image.create_from_file(os.path.join(folder, f))
                res = lmk.detect(img)
            except Exception:
                continue
            if not res.face_landmarks:
                continue
            yaw, pitch = pose(res.facial_transformation_matrixes[0])
            p = np.array([[q.x * img.width, q.y * img.height] for q in res.face_landmarks[0][:N]])
            iod = np.linalg.norm(p[EYES[0]].mean(0) - p[EYES[1]].mean(0))
            if abs(yaw) < MAX_YAW and abs(pitch) < MAX_PITCH and iod >= 60:
                faces.append(align(p))
                break  # one photo per person
        if n % 500 == 0:
            print(f"{n}/{len(dirs)} people, {len(faces)} frontal faces", flush=True)

    faces = np.array(faces)
    avg = faces.mean(0)
    mirrored = avg * [-1, 1]
    partner = np.array([int(np.argmin(np.linalg.norm(avg - mirrored[i], axis=1))) for i in range(N)])
    mutual = (partner[partner] == np.arange(N)).mean()
    print(f"{len(faces)} faces; mirror pairs mutual for {mutual:.1%} of points")

    per_point = np.array([asymmetry(f, partner) for f in faces])  # faces x N
    overall = np.sqrt((per_point ** 2).mean(1))
    region_idx = {k: sorted({i for c in v for i in (c.start, c.end)}) for k, v in REGIONS.items()}
    region_idx["nose"] = NOSE
    regions = {k: np.sqrt((per_point[:, idx] ** 2).mean(1)) for k, idx in region_idx.items()}
    q = list(range(0, 101, 2))
    out = {
        "n": int(len(faces)),
        "maxYaw": MAX_YAW,
        "maxPitch": MAX_PITCH,
        "partner": partner.tolist(),
        "regions": region_idx,
        "quantiles": {"pct": q, "overall": np.percentile(overall, q).round(5).tolist(),
                      **{k: np.percentile(v, q).round(5).tolist() for k, v in regions.items()}},
    }
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print("median asymmetry", round(float(np.median(overall)), 4), {k: round(float(np.median(v)), 4) for k, v in regions.items()})
    # Self-check: a perfectly mirrored face scores ~0, and the average face is far more symmetric than any one face
    sym = (avg + avg[partner] * [-1, 1]) / 2
    assert asymmetry(sym, partner).max() < 1e-6 or mutual < 1, "mirror fit broken"
    assert np.sqrt((asymmetry(avg, partner) ** 2).mean()) < np.percentile(overall, 5), "average face should be near-perfectly symmetric"
    print("ok ->", OUT)


if __name__ == "__main__":
    main()
