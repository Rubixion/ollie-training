"""
Collects verified, freely licensed photos for each person in celebs.json.

Source: Wikimedia Commons only, and only public domain / CC0 / CC BY / CC BY-SA files. Every kept
photo records author, license and source page in manifest.json, so the site can credit it
("<author>, <license>, via Wikimedia Commons"). A license covers the photographer's copyright
only, not the person's rights over their likeness.

Per person, in rank order:
  1. Candidates: their Wikidata photo, their Commons category (+ subcategories named after them),
     then a Commons search for their name. Dropped before download: disallowed licenses, files
     flagged for deletion, and non-photos by title/description (wax figures, statues, drawings,
     stickers, posters, AI images, ...).
  2. InsightFace checks each photo: one clear, large, near-frontal face (no group shots), fully in
     frame; in colour (no black & white, sepia or tinted prints); sharp; an adult.
  3. Identity (ArcFace): the Wikidata photo and the person's own Commons category define who they
     are, and a photo is kept only if it clearly matches. If those reference photos disagree, the
     person is skipped rather than guessed.
  4. Near-duplicates dropped; the best MAX_KEEP go to data/<Name>_<QID>/ with manifest.json and
     faces.npy (the ArcFace embeddings, same order as the manifest).

Wikimedia's robot policy is built in: Commons API one request at a time and at most 3 per second;
image downloads at most 2 at a time and under 25 Mbps; standard thumbnail size; identifying
User-Agent.

    .venv\\Scripts\\python collect.py --pilot 200 --keep-rejects   # people spread over the ranking
    .venv\\Scripts\\python collect.py                               # rank order until TARGET are ok
Resumable: anyone with a manifest.json is skipped (--redo redoes them).
"""
import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import html
import io
import json
import os
import queue
import re
import threading
import time
import unicodedata

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = "OllieCelebIndex/2.0 (https://ollie.ml)"
API = "https://commons.wikimedia.org/w/api.php"

TARGET = 5000          # people with >= MIN_KEEP verified photos
MIN_KEEP, MAX_KEEP = 5, 15
THUMB_W = 960          # a Wikimedia standard thumbnail width
MAX_TRUSTED, MAX_SEARCH = 80, 40   # candidates per person: own photo/depicts/category, name search
MAX_DOWNLOADS = 60     # per person; downloads are the bottleneck (robot policy: 2 at a time)

FACE_MIN_W = 80        # px, in the 960px thumbnail
DET_MIN = 0.75
SECOND_FACE_MAX = 0.40  # a second face bigger than this (area ratio) = group photo
YAW_MAX = 0.40          # nose offset from the eye midline, in eye-distances (0 = frontal)
CHROMA_MIN = 3.0        # mean colour in the inner face; true B&W is ~0
INDEP_MIN = 3.5         # colour not explained by brightness; B&W, sepia and tints are < 3
BLUR_MIN = 40.0         # Laplacian variance of the aligned 112px face
AGE_MIN = 14            # estimated age; rejects childhood photos
SIM_TRUSTED, SIM_SEARCH = 0.45, 0.50   # ArcFace cosine to the person's centroid
SIM_ANCHOR_MIN = 0.25   # and never far from their own Wikidata photo
DUP_SIM = 0.92          # same photo (crop/resize) or near-identical burst shot

# cc-by / cc-by-sa any version (+ jurisdiction/"migrated" suffix), cc0, public domain. NC/ND never match.
LICENSE_RE = re.compile(r"cc0|pd|pd-[a-z0-9-]+|cc-by(-sa)?-\d(\.\d)?(-[a-z]+)*")
BAD_TEMPLATES = ["Template:Deletion template tag", "Template:Delete", "Template:Copyvio",
                 "Template:Speedydelete", "Template:No permission since", "Template:No license since",
                 "Template:No source since"]
# Whole words only: "art" must not hit "Stewart", "comic" must not drop Comic-Con, "costume" not the Met Gala.
NON_PHOTO = re.compile(r"\b(" + "|".join([
    r"wax\w*", r"tussauds?", r"statues?", r"sculptures?", r"bust of", r"murals?", r"graffiti", r"paintings?",
    r"painted", r"drawings?", r"sketch(es)?", r"caricatures?", r"cartoons?", r"illustrations?", r"stickers?",
    r"posters?", r"billboards?", r"dolls?", r"figurines?", r"action figures?", r"cosplay\w*", r"impersonat\w*",
    r"look-?alikes?", r"fan ?art", r"ai[- ]generated", r"generated (by|with) ai", r"midjourney",
    r"stable diffusion", r"dall-?e", r"stamps?", r"banknotes?", r"coins?", r"masks?", r"signatures?", r"logos?",
    r"(album|book|magazine) covers?", r"cover art", r"renders?", r"rendering", r"3d models?", r"holograms?",
    r"puppets?", r"emojis?", r"oil on canvas", r"mannequins?", r"cardboard", r"cut-?outs?",
]) + r")\b")
# Subcategories that aren't portraits of the person (award shows, festivals, parliament etc. stay in).
SUBCAT_SKIP = re.compile(r"\b(" + "|".join([
    r"albums?", r"songs?", r"singles?", r"discography", r"signatures?", r"autographs?", r"family",
    r"families", r"children", r"logos?", r"merchandise", r"wax\w*", r"statues?", r"art", r"artworks?",
    r"drawings?", r"paintings?", r"caricatures?", r"fans?", r"covers?", r"characters?", r"roles?",
    r"impersonat\w*", r"star on", r"walk of fame", r"named after", r"related", r"things", r"murals?",
    r"graffiti", r"stamps?", r"audio", r"maps?", r"products?", r"perfumes?", r"fragrances?", r"clothing",
    r"dresses", r"outfits?",
]) + r")\b")

# Title words that usually mean a crowd, stage or far-away shot: downloaded last.
BUSY = re.compile(r"\b(concert|performing|performance|live|tour|stage|match|game|vs|versus|rally|crowd|"
                  r"cast|team|squad|family|fans|panel|group|band|and|with|meets|meeting|visit|ceremony|parade)\b")

EXTMETA = ("License|LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription|ObjectName|"
           "AttributionRequired|Restrictions")


# ── polite HTTP ────────────────────────────────────────────────────────────────

def session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    retry = Retry(total=8, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504),
                  respect_retry_after_header=True, allowed_methods=None)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    return s


API_SESSION, MEDIA_SESSION = session(), session()
_api_lock = threading.Lock()
_api_last = [0.0]


def api(**params):
    """One Commons API request at a time, at most ~3/s (robot policy: 1 concurrent, < 5/s; 200/min)."""
    params.update(action="query", format="json", formatversion=2)
    with _api_lock:
        wait = _api_last[0] + 0.34 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            r = API_SESSION.get(API, params=params, timeout=60)
        finally:
            _api_last[0] = time.monotonic()
    r.raise_for_status()
    return r.json()


class ByteRate:
    """Caps total download speed (robot policy: under 25 Mbps)."""

    def __init__(self, bytes_per_s):
        self.rate, self.allow, self.t, self.lock = bytes_per_s, bytes_per_s, time.monotonic(), threading.Lock()

    def take(self, n):
        with self.lock:
            now = time.monotonic()
            self.allow = min(self.rate, self.allow + (now - self.t) * self.rate) - n
            self.t = now
            wait = -self.allow / self.rate if self.allow < 0 else 0
        if wait:
            time.sleep(wait)


BANDWIDTH = ByteRate(2_500_000)            # 20 Mbps
DOWNLOADS = cf.ThreadPoolExecutor(2)       # robot policy: at most 2 media downloads at a time


def download(url):
    try:
        r = MEDIA_SESSION.get(url, timeout=60)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            return None
        BANDWIDTH.take(len(r.content))
        return r.content
    except requests.RequestException:
        return None


# ── candidates from Commons ────────────────────────────────────────────────────

def plain(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def file_info(pages, source):
    """Commons page records -> usable candidates (allowed license, not flagged, looks like a photo).
    A Wikidata photo under another free license is still returned, marked anchor_only: it helps
    identify the person but is never kept."""
    out, dropped = [], collections.Counter()
    for p in pages:
        info = (p.get("imageinfo") or [None])[0]
        if not info:
            continue
        meta = {k: (v or {}).get("value", "") for k, v in info.get("extmetadata", {}).items()}
        lic = str(meta.get("License", "")).lower().strip()
        short = plain(str(meta.get("LicenseShortName", "")))
        text = f"{p['title']} {plain(str(meta.get('ImageDescription', '')))} {plain(str(meta.get('ObjectName', '')))}".lower()
        licensed = bool(LICENSE_RE.fullmatch(lic)) or short.lower() in ("public domain", "cc0")
        if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
            dropped["not_a_photo_file"] += 1
        elif min(info.get("width", 0), info.get("height", 0)) < 300:
            dropped["too_small"] += 1
        elif p.get("templates"):
            dropped["flagged_for_deletion"] += 1
        elif NON_PHOTO.search(text):
            dropped["not_a_photo_by_title"] += 1
        elif not licensed and source != "wikidata_photo":
            dropped["license"] += 1
        else:
            author = plain(str(meta.get("Artist", ""))) or plain(str(meta.get("Credit", ""))) or "Unknown author"
            out.append({
                "title": p["title"], "source": source, "anchor_only": not licensed,
                "width": info.get("width", 0), "height": info.get("height", 0),
                "url": info.get("thumburl") or info["url"],
                "page": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{p['title'].replace(' ', '_')}",
                "author": author[:200], "license": short or lic, "license_url": str(meta.get("LicenseUrl", "")),
                "attribution_required": str(meta.get("AttributionRequired", "true")).lower() != "false",
                "restrictions": plain(str(meta.get("Restrictions", ""))),
            })
    return out, dropped


INFO = dict(prop="imageinfo|templates", iiprop="url|size|mime|extmetadata", iiurlwidth=THUMB_W,
            iiextmetadatafilter=EXTMETA, iiextmetadatalanguage="en",
            tltemplates="|".join(BAD_TEMPLATES), tllimit="max")


def pages_of(resp):
    return resp.get("query", {}).get("pages", [])


def gather(celeb):
    """All candidate photos for one person, most trustworthy first, plus pre-download drop counts."""
    cands, dropped, seen = [], collections.Counter(), set()

    def add(pages, source, cap):
        got, why = file_info([p for p in pages if p["title"] not in seen], source)
        dropped.update(why)
        for c in got[:cap]:
            seen.add(c["title"])
            cands.append(c)

    if celeb["images"]:
        add(pages_of(api(titles="|".join("File:" + f for f in celeb["images"][:5]), **INFO)), "wikidata_photo", 5)
    name_words = [w.lower() for w in re.findall(r"\w+", celeb["name"]) if len(w) > 2]
    for cat in celeb["commons_categories"][:2]:
        add(pages_of(api(generator="categorymembers", gcmtitle=f"Category:{cat}", gcmtype="file",
                         gcmlimit=50, **INFO)), "category", MAX_TRUSTED)
        if len(cands) >= 40:
            continue
        subcats = [m["title"] for m in api(list="categorymembers", cmtitle=f"Category:{cat}", cmtype="subcat",
                                           cmlimit=100).get("query", {}).get("categorymembers", [])]
        subcats = [s for s in subcats if all(w in s.lower() for w in name_words[-1:])
                   and not SUBCAT_SKIP.search(s.lower())]
        subcats.sort(key=lambda s: max(re.findall(r"(?:19|20)\d\d", s) or ["0"]), reverse=True)  # recent first
        for sub in subcats[:6]:
            if len(cands) >= MAX_TRUSTED:
                break
            add(pages_of(api(generator="categorymembers", gcmtitle=sub, gcmtype="file", gcmlimit=12, **INFO)),
                "subcategory", MAX_TRUSTED - len(cands))
    if len(cands) < 40:
        add(pages_of(api(generator="search", gsrsearch=f'"{celeb["name"]}" filetype:bitmap', gsrnamespace=6,
                         gsrlimit=50, **INFO)), "search", MAX_SEARCH)
    order = {"wikidata_photo": 0, "category": 1, "subcategory": 2, "search": 3}
    cands.sort(key=lambda c: order[c["source"]])
    return cands, dropped


# ── photo checks (InsightFace) ─────────────────────────────────────────────────

_app = None


def face_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "genderage"],
                            providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def indep_color(rgb):
    """RMS of Lab a*/b* left after predicting them from lightness: ~0 for B&W and for toned prints."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    L, ab = lab[:, :1] / 255.0, lab[:, 1:] - 128.0
    X = np.hstack([np.ones_like(L), L, L * L])
    coef, *_ = np.linalg.lstsq(X, ab, rcond=None)
    return float(np.sqrt(((ab - X @ coef) ** 2).sum(1).mean()))


def analyze(data):
    """-> (reject reason or None, details). details has the main face's embedding whenever a
    face was found, so even a rejected Wikidata photo can still anchor who the person is."""
    from insightface.app.common import Face
    from insightface.utils.face_align import norm_crop
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return "unreadable", None
    im.thumbnail((2000, 2000))  # thumbnails are 960 wide; only guards odd originals
    rgb = np.ascontiguousarray(np.asarray(im))
    bgr = rgb[:, :, ::-1].copy()
    app = face_app()
    boxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
    faces = [(b, k) for b, k in zip(boxes, kpss) if b[4] >= 0.5]
    if not faces:
        return "no_face", None
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    faces.sort(key=lambda f: area(f[0]), reverse=True)
    (box, kps) = faces[0]
    face = Face(bbox=box[:4], kps=kps, det_score=box[4])
    app.models["recognition"].get(bgr, face)
    app.models["genderage"].get(bgr, face)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    fw, fh = x2 - x1, y2 - y1
    second = area(faces[1][0]) / area(box) if len(faces) > 1 else 0.0
    d = {"emb": face.normed_embedding.astype(np.float32), "det_score": round(float(box[4]), 3),
         "face_width": round(fw), "age_estimate": int(face.age), "second_face": round(second, 2),
         "image": im, "anchor_ok": box[4] >= 0.6 and fw >= 50 and second <= SECOND_FACE_MAX}

    inside = (min(x2, w) - max(x1, 0)) * (min(y2, h) - max(y1, 0)) / max(fw * fh, 1)
    le, re_, nose = kps[0], kps[1], kps[2]
    eye_d = float(np.linalg.norm(re_ - le)) or 1.0
    yaw = abs(float(np.dot(nose - (le + re_) / 2, (re_ - le) / eye_d))) / eye_d
    xi, yi = int(max(x1, 0)), int(max(y1, 0))
    inner = rgb[int(yi + fh / 5):int(y2 - fh / 10), int(xi + fw / 5):int(x2 - fw / 5)]
    around = rgb[max(0, int(y1 - fh / 4)):min(h, int(y2 + fh / 2)), max(0, int(x1 - fw / 4)):min(w, int(x2 + fw / 4))]
    chroma = float(np.hypot(*(cv2.cvtColor(inner, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1:] - 128).transpose(2, 0, 1)).mean()) if inner.size else 0.0
    indep = indep_color(around) if around.size else 0.0
    d["colour"] = round(min(chroma, indep), 2)
    aligned = norm_crop(bgr, kps, image_size=112)
    d["sharpness"] = round(float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1)
    d["yaw"] = round(yaw, 2)

    if box[4] < DET_MIN:
        return "unclear_face", d
    if fw < FACE_MIN_W:
        return "small_face", d
    if second > SECOND_FACE_MAX:
        return "group_photo", d
    if inside < 0.9:
        return "face_cut_off", d
    if yaw > YAW_MAX or eye_d / max(fw, 1) < 0.25:
        return "not_frontal", d
    if chroma < CHROMA_MIN or indep < INDEP_MIN:
        return "black_and_white", d
    if d["sharpness"] < BLUR_MIN:
        return "blurry", d
    if face.age < AGE_MIN:
        return "child", d
    return None, d


# ── identity ───────────────────────────────────────────────────────────────────

def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def consensus(E, t=0.45):
    """Densest group of mutually matching faces -> (centroid, size, size of the next group)."""
    if len(E) < 3:
        return None, 0, 0
    S = E @ E.T
    nb = (S >= t).sum(1)
    i = int(np.argmax(nb + S.mean(1) * 1e-3))
    members = S[i] >= t
    if members.sum() < 3:
        return None, 0, 0
    c = unit(E[members].mean(0))
    rest = (E @ c) < t - 0.1
    second = int((S[np.ix_(rest, rest)] >= t).sum(1).max()) if rest.sum() >= 2 else 0
    return c, int(members.sum()), second


def identify(accepted, anchors):
    """Keep only photos of the right person. -> (kept, sims, reason if the person can't be trusted)."""
    if not accepted:
        return [], [], "no_usable_photos"
    E = np.stack([a["emb"] for a in accepted])
    trusted = np.array([a["source"] != "search" for a in accepted])
    thr = np.where(trusted, SIM_TRUSTED, SIM_SEARCH)
    anchor = unit(np.mean(anchors, 0)) if anchors else None
    cons, size, second = consensus(E[trusted])
    if cons is not None and second >= 0.5 * size:
        return [], [], "category_mixes_two_people"
    if anchor is not None and cons is not None and size >= 4 and float(anchor @ cons) < 0.35:
        return [], [], "wikidata_photo_disagrees_with_category"
    seed = anchor if anchor is not None else cons
    if seed is None:  # only search results: demand a strong, unambiguous majority
        cons, size, second = consensus(E)
        if cons is None or size < MIN_KEEP or second > 0.3 * size:
            return [], [], "no_reliable_reference"
        seed = cons
    c = seed
    for _ in range(3):
        m = (E @ c) >= thr
        if not m.any():
            break
        c = unit(E[m].sum(0) + (anchor if anchor is not None else 0))  # the Wikidata photo counts as one more vote
    sims = E @ c
    keep = sims >= thr
    if anchor is not None:
        keep &= (E @ anchor) >= SIM_ANCHOR_MIN
    return [a for a, k in zip(accepted, keep) if k], [float(s) for s, k in zip(sims, keep) if k], None


def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def best_unique(kept, sims):
    """Best photos first (match x face size x detection), skipping copies/crops of one already chosen.
    -> (chosen, number skipped as duplicates)"""
    order = sorted(range(len(kept)), key=lambda i: sims[i] * min(1.0, kept[i]["face_width"] / 200)
                   * kept[i]["det_score"], reverse=True)
    chosen, dups = [], 0
    for i in order:
        a = kept[i]
        a["hash"] = dhash(a["image"])
        if any(float(a["emb"] @ b["emb"]) >= DUP_SIM or int((a["hash"] != b["hash"]).sum()) <= 4 for b in chosen):
            dups += 1
            continue
        a["similarity"] = round(sims[i], 3)
        chosen.append(a)
        if len(chosen) == MAX_KEEP:
            break
    return chosen, dups


# ── per person ─────────────────────────────────────────────────────────────────

def slug(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "person"


def folder_of(celeb):
    return os.path.join(DATA, f"{slug(celeb['name'])}_{celeb['qid']}")


def process(celeb, cands, dropped, keep_rejects=False):
    t0 = time.time()
    rejects = collections.Counter(dropped)
    accepted, anchors = [], []
    futures = [DOWNLOADS.submit(download, c["url"]) for c in cands]
    for n, (c, fut) in enumerate(zip(cands, futures)):
        data = fut.result()
        if data is None:
            rejects["download_failed"] += 1
            continue
        reason, d = analyze(data)
        if d is not None and c["source"] == "wikidata_photo" and d["anchor_ok"]:
            anchors.append(d["emb"])
        if not reason and c["anchor_only"]:
            reason = "license_reference_only"
        if reason:
            rejects[reason] += 1
            if keep_rejects and d is not None:
                path = os.path.join(DATA, "_rejects", os.path.basename(folder_of(celeb)), f"{reason}__{n:03d}.jpg")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                small = d["image"].copy()
                small.thumbnail((320, 320))
                small.save(path, quality=85)
            continue
        accepted.append({**c, **d})
        if len(accepted) >= MAX_KEEP * 2 + 5:  # plenty to choose from; stop downloading
            for f in futures[n + 1:]:
                f.cancel()
            break

    kept, sims, problem = identify(accepted, anchors)
    chosen, dups = best_unique(kept, sims) if not problem else ([], 0)
    if not problem:
        rejects["wrong_person"] += len(accepted) - len(kept)
        rejects["duplicate"] += dups
    status = "ok" if len(chosen) >= MIN_KEEP else ("skipped" if problem else "too_few")

    folder = folder_of(celeb)
    os.makedirs(folder, exist_ok=True)
    images = []
    for i, a in enumerate(chosen, 1):
        name = f"{i:02d}.jpg"
        im = a["image"].copy()
        im.thumbnail((THUMB_W, THUMB_W))
        im.save(os.path.join(folder, name), quality=90)
        images.append({"file": name, **{k: a[k] for k in (
            "title", "page", "author", "license", "license_url", "attribution_required", "restrictions",
            "source", "similarity", "det_score", "face_width", "age_estimate", "colour", "sharpness", "yaw")},
            "credit": f"{a['author']}, {a['license']}, via Wikimedia Commons"})
    np.save(os.path.join(folder, "faces.npy"), np.stack([a["emb"] for a in chosen]) if chosen
            else np.zeros((0, 512), np.float32))
    manifest = {"qid": celeb["qid"], "name": celeb["name"], "rank": celeb["rank"], "status": status,
                "problem": problem, "kept": len(images), "candidates": len(cands),
                "anchor": bool(anchors), "rejects": dict(rejects), "images": images,
                "collected": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(folder, "manifest.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(os.path.join(folder, "manifest.json.tmp"), os.path.join(folder, "manifest.json"))
    top = ", ".join(f"{k} {v}" for k, v in rejects.most_common(3))
    print(f"[{celeb['rank']:5d}] {celeb['name'][:32]:32s} {status:8s} kept {len(images):2d}/{len(cands):3d} "
          f"{'anchor ' if anchors else 'no-anchor'} {problem or ''} | {top} | {time.time() - t0:.0f}s", flush=True)
    return status


def _check():
    """Licensing, title filters and the identity rules are the safety-critical logic; check them each run."""
    ok = lambda lic: bool(LICENSE_RE.fullmatch(lic))
    assert all(map(ok, ["cc-by-sa-4.0", "cc-by-2.0", "cc-by-sa-3.0-migrated", "cc-by-3.0-us", "cc0", "pd", "pd-usgov"]))
    assert not any(map(ok, ["cc-by-nc-sa-2.0", "cc-by-nd-4.0", "cc-by-nc-2.0", "gfdl", "fal", "cc-by-sa-4.0,gfdl"]))
    assert not NON_PHOTO.search("file:kristen stewart at comic-con 2019 by gage skidmore.jpg")
    assert not NON_PHOTO.search("zendaya at the costume institute gala; avatar premiere")
    assert NON_PHOTO.search("file:wax figure of taylor swift.jpg") and NON_PHOTO.search("madame tussauds london")
    assert not SUBCAT_SKIP.search("taylor swift at the 2023 mtv video music awards")
    assert not SUBCAT_SKIP.search("kristen stewart at the 2012 cannes film festival")
    assert SUBCAT_SKIP.search("taylor swift albums") and SUBCAT_SKIP.search("wax figures of taylor swift")

    rng = np.random.default_rng(0)
    p, q = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    like = lambda base, n: [unit(base + rng.normal(size=512) * 0.035) for _ in range(n)]  # cos ~0.6 to base
    photo = lambda emb, src: {"emb": emb.astype(np.float32), "source": src}
    person = [photo(e, "category") for e in like(p, 6)]
    impostors = [photo(e, "search") for e in like(q, 3)]
    kept, sims, problem = identify(person + impostors, like(p, 1))
    assert problem is None and len(kept) == 6 and all(k["source"] == "category" for k in kept)
    assert identify(person, like(q, 1))[2] == "wikidata_photo_disagrees_with_category"
    mixed = [photo(e, "category") for e in like(p, 4) + like(q, 4)]
    assert identify(mixed, [])[2] == "category_mixes_two_people"
    assert identify([photo(e, "search") for e in like(p, 2) + like(q, 2)], [])[2] == "no_reliable_reference"


def main():
    _check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, help="only N people, evenly spread over the first 6000 ranks")
    ap.add_argument("--keep-rejects", action="store_true", help="save rejected photos to data/_rejects for review")
    ap.add_argument("--redo", action="store_true", help="redo people that already have a manifest")
    ap.add_argument("--list", default=os.path.join(HERE, "celebs.json"), help="ranked list (default celebs.json)")
    args = ap.parse_args()

    with open(args.list, encoding="utf-8") as f:
        celebs = json.load(f)
    if args.pilot:
        step = min(6000, len(celebs)) / args.pilot
        celebs = [celebs[int(i * step)] for i in range(args.pilot)]
    os.makedirs(DATA, exist_ok=True)

    def done(c):
        path = os.path.join(folder_of(c), "manifest.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)["status"]

    ok = sum(done(c) == "ok" for c in celebs)
    todo = [c for c in celebs if args.redo or done(c) is None]
    print(f"{len(celebs)} people, {ok} already ok, {len(todo)} to collect", flush=True)
    face_app()

    work = queue.Queue(maxsize=3)

    def producer():  # Commons API lookups run ahead of downloads + face checks
        for c in todo:
            try:
                work.put((c, *gather(c)))
            except Exception as e:
                print(f"[{c['rank']:5d}] {c['name']}: lookup failed ({e}); will retry next run", flush=True)
            if stop.is_set():
                break
        work.put(None)

    stop = threading.Event()
    threading.Thread(target=producer, daemon=True).start()
    while (item := work.get()) is not None:
        c, cands, dropped = item
        try:
            ok += process(c, cands, dropped, args.keep_rejects) == "ok"
        except Exception as e:
            print(f"[{c['rank']:5d}] {c['name']}: failed ({type(e).__name__}: {e}); will retry next run", flush=True)
        if not args.pilot and ok >= TARGET:
            print(f"reached {TARGET} people with >= {MIN_KEEP} verified photos", flush=True)
            stop.set()
            break
    print(f"done: {ok} ok", flush=True)


if __name__ == "__main__":
    main()
