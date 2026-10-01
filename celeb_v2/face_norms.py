"""Face-shape norms for the /style page: the mean and spread of the four face-outline ratios over
straight-on photos in data/, the same measurement as ollie-frontend/lib/style/face-shape.ts.

    .venv\\Scripts\\python face_norms.py      -> face_norms.json (paste mean/std into NORMS in face-shape.ts)

Also keeps each person's average ratios, for "celebrities with your face shape" later.
"""
import json, math, os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = os.path.dirname(os.path.abspath(__file__))
PER_PERSON = 3      # photos per person is plenty for population norms
MAX_TILT = 10       # degrees; the browser scan uses the same cut-off

PAIRS = [(10, 152), (21, 251), (58, 288), (150, 379)]  # length, forehead, jaw, chin (each / cheekbones 234-454)


def pose(m):
    """4x4 face transform -> (yaw, pitch) in degrees, same formula as the browser."""
    return math.degrees(math.atan2(m[0][2], m[2][2])), math.degrees(math.atan2(-m[1][2], math.hypot(m[0][2], m[2][2])))


def ratios(lm, w, h):
    d = lambda a, b: math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)
    cheek = d(234, 454)
    return [d(a, b) / cheek for a, b in PAIRS], cheek


def main():
    lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(HERE, "models", "face_landmarker.task")),
        output_facial_transformation_matrixes=True, num_faces=1))
    rows, people = [], {}
    dirs = sorted(os.listdir(os.path.join(HERE, "data")))
    for n, person in enumerate(dirs):
        folder = os.path.join(HERE, "data", person)
        if not os.path.isdir(folder):
            continue
        mine = []
        for f in sorted(os.listdir(folder)):
            if len(mine) >= PER_PERSON or not f.endswith(".jpg"):
                continue
            try:
                img = mp.Image.create_from_file(os.path.join(folder, f))
                res = lmk.detect(img)
            except Exception:
                continue
            if not res.face_landmarks:
                continue
            yaw, pitch = pose(res.facial_transformation_matrixes[0])
            r, cheek = ratios(res.face_landmarks[0], img.width, img.height)
            if abs(yaw) < MAX_TILT and abs(pitch) < MAX_TILT and cheek >= 100:
                mine.append(r)
        if mine:
            rows += mine
            people[person] = np.mean(mine, 0).round(4).tolist()
        if n % 500 == 0:
            print(f"{n}/{len(dirs)} people, {len(rows)} frontal photos", flush=True)

    a = np.array(rows)
    out = {"n": len(a), "mean": a.mean(0).round(4).tolist(), "std": a.std(0).round(4).tolist(), "people": people}
    json.dump(out, open(os.path.join(HERE, "face_norms.json"), "w"))
    print("n", out["n"], "mean", out["mean"], "std", out["std"])


if __name__ == "__main__":
    main()
