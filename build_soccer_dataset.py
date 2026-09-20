"""
Builds a face-image dataset for the soccer-player "who do you look like"
lookalike-match feature, covering the ~2,800 names in soccer_players.py.

Pipeline (matches the existing celebrity_scraper.py convention already
used to build the ~450 folders in celebrity_data/ for actors/musicians/
etc.): DuckDuckGo image search -> download -> mediapipe face filter ->
optional Groq vision verification.

NOTE ON IMAGE RIGHTS: this pulls whatever DuckDuckGo image search
returns, same as the rest of celebrity_data/ — it does not track
per-image license/attribution. That's fine for computing face
embeddings internally, but if the site will display the raw matched
photo (not just the player's name) to end users, swap in licensed
images (e.g. Wikimedia Commons CC-BY-SA/CC0 portraits, which do carry
attribution) for whichever players actually get shown.

This sandbox's network policy blocks general web egress (Wikipedia,
DuckDuckGo, etc. all rejected) so this script cannot run here — run it
somewhere with normal internet access:

    pip install ddgs requests mediapipe
    python3 build_soccer_dataset.py                    # scrape + face-filter
    GROQ_API_KEY=... python3 build_soccer_dataset.py --verify   # + AI verify

Resumable: re-running skips players who already have >= --n images
downloaded (scrape_celebrity's existing count_images short-circuit).
"""

import argparse
import os
import time

from soccer_players import SOCCER_PLAYERS
from celebrity_scraper import (
    scrape_soccer_players,
    clean_non_faces,
    ai_verify_all,
    SCRAPE_ROOT,
)

LOG_PATH = "soccer_scrape_progress.log"


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8,
                         help="images to try to keep per player (default 8)")
    parser.add_argument("--root", default=SCRAPE_ROOT,
                         help=f"output dataset root (default {SCRAPE_ROOT})")
    parser.add_argument("--verify", action="store_true",
                         help="run Groq AI verification after scraping (needs GROQ_API_KEY)")
    parser.add_argument("--skip-scrape", action="store_true",
                         help="only run face-filter/verify on already-downloaded images")
    args = parser.parse_args()

    total = len(SOCCER_PLAYERS)
    _log(f"Starting soccer dataset build: {total} players, "
         f"target {args.n} imgs/player, root={args.root}")

    if not args.skip_scrape:
        def _scrape_progress(name, done, total_):
            _log(f"scraped [{done}/{total_}] {name}")

        scrape_soccer_players(
            SOCCER_PLAYERS, n_per_player=args.n, root=args.root,
            progress_cb=_scrape_progress,
        )
    else:
        _log("--skip-scrape set: skipping download step")

    _log("Filtering out images with no detectable face...")

    def _face_progress(done, total_, kept, removed):
        if done % 50 == 0 or done == total_:
            _log(f"face-filter [{done}/{total_}] kept={kept} removed={removed}")

    kept, removed = clean_non_faces(root=args.root, progress_cb=_face_progress)
    _log(f"Face filter done: kept={kept} removed={removed}")

    if args.verify:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            _log("--verify requested but GROQ_API_KEY is not set; skipping AI verification")
        else:
            def _verify_progress(done, total_, kept_, removed_):
                if done % 50 == 0 or done == total_:
                    _log(f"ai-verify [{done}/{total_}] kept={kept_} removed={removed_}")

            v_kept, v_removed = ai_verify_all(
                root=args.root, api_key=api_key, progress_cb=_verify_progress,
            )
            _log(f"AI verify done: kept={v_kept} removed={v_removed}")

    _log("Done.")


if __name__ == "__main__":
    main()
