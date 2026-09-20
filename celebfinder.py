import os
import re
import csv
import time
import random
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "combined_player_names.txt"

OUTPUT_DIR = Path("headshots")
TEMP_DIR = Path("_temp_downloads")
RESULTS_FILE = Path("results.csv")
FAILED_FILE = Path("failed.txt")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# IMPORTANT:
# Wikimedia now uses a fixed set of thumbnail widths.
# 1920 is a standard Wikimedia thumbnail size.
THUMB_WIDTH = 1920

# Keep this conservative. Do not parallelize Wikimedia requests.
REQUEST_DELAY = 3.5

# Retries are only for temporary errors.
MAX_RETRIES = 6

# Put your real email here.
# Wikimedia asks automated clients to identify themselves.
CONTACT_EMAIL = "YOUR_EMAIL@example.com"

USER_AGENT = (
    f"CelebFinderBot/2.0 "
    f"(personal soccer headshot project; contact: {CONTACT_EMAIL}) "
    f"requests/{requests.__version__}"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Api-User-Agent": USER_AGENT,
    "Accept": "application/json",
}

DOWNLOAD_HEADERS = {
    "User-Agent": USER_AGENT,
    "Api-User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# HELPERS
# ============================================================

def clean_filename(name):
    """Make a safe Windows filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]


def clean_url(url):
    """
    Remove Wikimedia tracking query parameters.

    Wikimedia imageinfo responses can contain URLs such as:
    ...jpg?utm_source=commons.wikimedia.org&...
    The actual image URL does not need those parameters.
    """
    if not url:
        return url

    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def sleep_between_requests():
    time.sleep(REQUEST_DELAY + random.uniform(0.0, 1.0))


def read_players():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        players = [line.strip() for line in f if line.strip()]

    # Preserve order but remove duplicates.
    seen = set()
    result = []

    for player in players:
        key = player.casefold()
        if key not in seen:
            seen.add(key)
            result.append(player)

    return result


def load_successful_players():
    """
    Only successful downloads are considered completed.

    This is important: a temporary 403/429/error should NOT
    permanently prevent the player from being retried.
    """
    completed = set()

    if not RESULTS_FILE.exists():
        return completed

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                status = (row.get("status") or "").strip().lower()
                player = (row.get("player") or "").strip()

                if player and status in {
                    "downloaded",
                    "already_exists",
                }:
                    completed.add(player.casefold())

    except Exception as e:
        print(f"Warning: could not read {RESULTS_FILE}: {e}")

    return completed


def ensure_results_header():
    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        return

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "player",
            "status",
            "commons_file",
            "source_width",
            "source_height",
            "download_width",
            "download_height",
            "output_file",
        ])


def save_result(
    player,
    status,
    commons_file="",
    source_width="",
    source_height="",
    download_width="",
    download_height="",
    output_file="",
):
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            player,
            status,
            commons_file,
            source_width,
            source_height,
            download_width,
            download_height,
            output_file,
        ])


def append_failure(player, reason):
    with open(FAILED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{player}\t{reason}\n")


# ============================================================
# WIKIMEDIA API
# ============================================================

def api_get(params):
    """
    Call Wikimedia Commons API with exponential backoff.

    429:
        Respect Retry-After.

    403:
        Usually means the request/client is blocked or malformed.
        Do not hammer the server. Wait and retry a few times.

    5xx:
        Temporary server-side problem.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                COMMONS_API,
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 60
                else:
                    wait = min(300, 60 * (2 ** attempt))

                wait += random.uniform(1, 5)

                print(
                    f"  Wikimedia rate limit (429). "
                    f"Waiting {wait:.0f}s..."
                )

                time.sleep(wait)
                continue

            if response.status_code == 403:
                wait = min(300, 30 * (2 ** attempt))
                wait += random.uniform(1, 5)

                print(
                    f"  Wikimedia API 403. "
                    f"Waiting {wait:.0f}s before retry..."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(180, 10 * (2 ** attempt))
                wait += random.uniform(1, 3)

                print(
                    f"  Wikimedia server error "
                    f"{response.status_code}. "
                    f"Waiting {wait:.0f}s..."
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

        except requests.RequestException as e:
            wait = min(120, 10 * (2 ** attempt))
            wait += random.uniform(1, 3)

            print(
                f"  API connection error: {e}. "
                f"Waiting {wait:.0f}s..."
            )

            time.sleep(wait)

    return None


def search_commons(player):
    """
    Search Wikimedia Commons for image files.

    We ask the API for a standard 1920px thumbnail.
    Wikimedia currently recommends standard thumbnail widths.
    """

    params = {
        "action": "query",
        "format": "json",

        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": f'"{player}"',
        "gsrlimit": 10,

        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": THUMB_WIDTH,

        "maxlag": 5,
    }

    data = api_get(params)

    if not data:
        return []

    pages = data.get("query", {}).get("pages", {})

    results = []

    for page in pages.values():
        title = page.get("title", "")
        imageinfo = page.get("imageinfo", [])

        if not imageinfo:
            continue

        info = imageinfo[0]

        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            continue

        source_width = info.get("width")
        source_height = info.get("height")

        thumb_url = info.get("thumburl")

        if not thumb_url:
            continue

        results.append({
            "title": title,
            "thumburl": clean_url(thumb_url),
            "source_url": clean_url(info.get("url", "")),
            "source_width": source_width,
            "source_height": source_height,
            "thumb_width": info.get("thumbwidth"),
            "thumb_height": info.get("thumbheight"),
            "mime": mime,
        })

    return results


# ============================================================
# IMAGE DOWNLOAD
# ============================================================

def download_image(url, destination):
    """
    Download a Wikimedia thumbnail.

    The URL is cleaned and the request uses the same descriptive
    User-Agent as the API request.
    """

    url = clean_url(url)

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url,
                headers=DOWNLOAD_HEADERS,
                timeout=60,
                stream=True,
            )

            if response.status_code == 200:
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

                return True, ""

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 60
                else:
                    wait = min(300, 60 * (2 ** attempt))

                wait += random.uniform(1, 5)

                print(
                    f"  Download 429. Waiting {wait:.0f}s..."
                )

                time.sleep(wait)
                continue

            if response.status_code == 403:
                wait = min(300, 30 * (2 ** attempt))
                wait += random.uniform(1, 5)

                print(
                    f"  Download 403. Waiting {wait:.0f}s..."
                )

                time.sleep(wait)
                continue

            if 500 <= response.status_code < 600:
                wait = min(180, 10 * (2 ** attempt))
                wait += random.uniform(1, 3)

                print(
                    f"  Download {response.status_code}. "
                    f"Waiting {wait:.0f}s..."
                )

                time.sleep(wait)
                continue

            return False, f"HTTP {response.status_code}"

        except requests.RequestException as e:
            wait = min(120, 10 * (2 ** attempt))
            wait += random.uniform(1, 3)

            print(
                f"  Download error: {e}"
            )
            print(
                f"  Retrying in {wait:.0f}s..."
            )

            time.sleep(wait)

    return False, "maximum retries exceeded"


# ============================================================
# FACE DETECTION
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

PROFILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def detect_faces(image):
    """
    Detect frontal and profile faces.

    Returns a list of (x, y, w, h).
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(50, 50),
    )

    faces = list(faces)

    # Try profile faces if frontal detection did not find anything.
    if len(faces) == 0:
        profile_faces = PROFILE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(50, 50),
        )

        faces.extend(profile_faces)

    return faces


def choose_best_face(faces, image_shape):
    """
    Choose the largest detected face, preferring faces near
    the upper/central portion of the image.
    """

    if not faces:
        return None

    image_height, image_width = image_shape[:2]

    best = None
    best_score = -1

    for x, y, w, h in faces:
        area = w * h

        center_x = x + w / 2
        center_y = y + h / 2

        # Prefer faces closer to the image center horizontally.
        horizontal_penalty = abs(center_x - image_width / 2) / image_width

        # Faces higher in the frame are more likely to be portraits.
        vertical_bonus = 1.0 - min(center_y / image_height, 1.0)

        score = (
            area
            * (1.0 - 0.35 * horizontal_penalty)
            * (0.7 + 0.3 * vertical_bonus)
        )

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def crop_headshot(image, face):
    """
    Crop face + head + neck + shoulders.

    We intentionally include more area around the face so the
    result is a natural headshot instead of a tiny face crop.
    """

    x, y, w, h = face

    image_height, image_width = image.shape[:2]

    # Horizontal padding.
    left = int(w * 1.05)
    right = int(w * 1.05)

    # More space above the head.
    top = int(h * 0.90)

    # Include neck and shoulders.
    bottom = int(h * 1.90)

    x1 = max(0, x - left)
    y1 = max(0, y - top)
    x2 = min(image_width, x + w + right)
    y2 = min(image_height, y + h + bottom)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def save_headshot(image, output_file):
    """
    Save JPEG at high quality.
    """

    ok = cv2.imwrite(
        str(output_file),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            1,
        ],
    )

    return ok


# ============================================================
# MAIN
# ============================================================

def main():
    players = read_players()
    completed = load_successful_players()

    ensure_results_header()

    print(f"Found {len(players)} players.")
    print(
        f"Using standard Wikimedia {THUMB_WIDTH}px thumbnails "
        f"and cropping headshots."
    )
    print(
        f"Already completed: {len(completed)}"
    )
    print()

    for index, player in enumerate(players, start=1):

        # Skip only genuinely successful players.
        if player.casefold() in completed:
            print(
                f"[{index}/{len(players)}] "
                f"{player} - already completed"
            )
            continue

        print(
            f"[{index}/{len(players)}] Searching: {player}"
        )

        candidates = search_commons(player)

        if not candidates:
            print("  ❌ No usable Commons image found")

            save_result(
                player,
                "no_image",
            )

            append_failure(
                player,
                "no usable Wikimedia Commons image found",
            )

            sleep_between_requests()
            continue

        found_image = False

        for candidate_index, candidate in enumerate(candidates, start=1):

            title = candidate["title"]

            print(f"  Candidate {candidate_index}: {title}")
            print(
                f"  Source: "
                f"{candidate['source_width']} x "
                f"{candidate['source_height']}"
            )
            print(
                f"  Thumbnail: "
                f"{candidate['thumb_width']} x "
                f"{candidate['thumb_height']}"
            )

            temp_file = (
                TEMP_DIR
                / f"{clean_filename(player)}.jpg"
            )

            ok, error = download_image(
                candidate["thumburl"],
                temp_file,
            )

            if not ok:
                print(f"  ❌ Download failed: {error}")

                # Try the next Commons result rather than immediately
                # giving up on the player.
                continue

            image = cv2.imread(str(temp_file))

            if image is None:
                print("  ❌ OpenCV could not read image")
                continue

            download_height, download_width = image.shape[:2]

            print(
                f"  Downloaded: "
                f"{download_width} x {download_height}"
            )

            faces = detect_faces(image)

            if not faces:
                print("  ⚠️ No face detected")
                continue

            print(f"  Faces detected: {len(faces)}")

            best_face = choose_best_face(
                faces,
                image.shape,
            )

            if best_face is None:
                print("  ⚠️ Could not choose a face")
                continue

            crop = crop_headshot(
                image,
                best_face,
            )

            if crop is None:
                print("  ⚠️ Could not crop headshot")
                continue

            output_file = (
                OUTPUT_DIR
                / f"{clean_filename(player)}.jpg"
            )

            if not save_headshot(crop, output_file):
                print("  ❌ Could not save headshot")
                continue

            print(
                f"  ✅ Saved: {output_file}"
            )

            save_result(
                player,
                "downloaded",
                commons_file=title,
                source_width=candidate["source_width"],
                source_height=candidate["source_height"],
                download_width=download_width,
                download_height=download_height,
                output_file=str(output_file),
            )

            found_image = True

            try:
                temp_file.unlink()
            except OSError:
                pass

            break

        if not found_image:
            print(
                f"  ❌ No candidate produced a usable headshot"
            )

            save_result(
                player,
                "failed",
            )

            append_failure(
                player,
                "all Commons candidates failed or had no detectable face",
            )

            try:
                temp_file.unlink()
            except OSError:
                pass

        print()

        # Delay after every player.
        sleep_between_requests()

    print("Finished.")


if __name__ == "__main__":
    main()
