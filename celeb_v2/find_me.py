"""Find where YOU appear on social media: SerpApi finds candidate images, InsightFace confirms it's your face.

    set SERPAPI_KEY=...
    .venv\\Scripts\\python find_me.py me1.jpg me2.jpg --name "Liam Bradley" [--lens-url https://.../me.jpg]

--name   Google Images search for your name on social sites.
--lens-url  Google Lens reverse search (SerpApi only takes a public image URL, not a local file).
Nothing is saved: thumbnails are checked in memory and only links + scores are written to find_me.csv.
Only use it on photos of yourself.
"""
import argparse, csv, io, os, sys
from urllib.parse import urlparse

import numpy as np, requests
from PIL import Image, ImageOps

from collect import face_app

SOCIAL = ["instagram.com", "tiktok.com", "facebook.com", "x.com", "twitter.com", "linkedin.com",
          "youtube.com", "reddit.com", "pinterest.com", "threads.net", "snapchat.com", "tumblr.com"]
MATCH = 0.40  # ponytail: buffalo_l cosine; ~0.35-0.45 is the usual same-person cutoff, tune on your own hits


def embed_all(data):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return [f.normed_embedding for f in face_app().get(np.asarray(im)[:, :, ::-1].copy())]


def serp(**params):
    r = requests.get("https://serpapi.com/search.json", params={**params, "api_key": os.environ["SERPAPI_KEY"]}, timeout=60)
    r.raise_for_status()
    return r.json()


def candidates(name, lens_url, pages):
    if name:
        q = f'"{name}" (' + " OR ".join(f"site:{d}" for d in SOCIAL) + ")"
        for p in range(pages):
            for x in serp(engine="google_images", q=q, ijn=p).get("images_results", []):
                yield x.get("link"), x.get("thumbnail"), x.get("title", "")
    if lens_url:
        for x in serp(engine="google_lens", url=lens_url).get("visual_matches", []):
            yield x.get("link"), x.get("thumbnail"), x.get("title", "")


def is_social(link):
    host = urlparse(link or "").netloc.lower()
    return any(host == d or host.endswith("." + d) for d in SOCIAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("me", nargs="+", help="1+ clear photos of your face")
    ap.add_argument("--name")
    ap.add_argument("--lens-url")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--all-sites", action="store_true", help="don't restrict to social media domains")
    a = ap.parse_args()
    if not (a.name or a.lens_url):
        sys.exit("give --name and/or --lens-url")

    refs = []
    for p in a.me:
        faces = embed_all(open(p, "rb").read())
        if len(faces) != 1:
            sys.exit(f"{p}: need exactly one face, found {len(faces)}")
        refs.append(faces[0])
    me = np.mean(refs, 0)
    me /= np.linalg.norm(me)

    rows, seen = [], set()
    for link, thumb, title in candidates(a.name, a.lens_url, a.pages):
        if not thumb or link in seen or not (a.all_sites or is_social(link)):
            continue
        seen.add(link)
        try:
            faces = embed_all(requests.get(thumb, timeout=20).content)
        except Exception:
            continue
        if faces:
            score = float(max(f @ me for f in faces))
            rows.append((score, urlparse(link).netloc, link, title))
            print(f"{score:.2f} {'MATCH' if score >= MATCH else '     '} {link}")

    rows.sort(reverse=True)
    with open("find_me.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("score", "site", "link", "title"), *rows])
    hits = [r for r in rows if r[0] >= MATCH]
    print(f"\n{len(hits)} likely matches out of {len(rows)} images with faces -> find_me.csv")


if __name__ == "__main__":
    main()
