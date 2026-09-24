"""
Checks the collected dataset (data/) as a whole.

    .venv\\Scripts\\python audit.py summary            # counts: people, photos, licenses, reject reasons
    .venv\\Scripts\\python audit.py sheets [--rejects]  # contact sheets in sheets/ for eyeballing
    .venv\\Scripts\\python audit.py check [--fix]       # cross-person identity audit (see check())

check() catches what a per-person pass can't see: the same photo kept for two people, a photo that
looks more like someone else in the dataset than like its own person, and two entries that are
really the same person. --fix moves flagged photos to data/_quarantine and rewrites the manifests.
"""
import argparse
import collections
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SHEETS = os.path.join(HERE, "sheets")


def people(status=None):
    for d in sorted(os.listdir(DATA)):
        path = os.path.join(DATA, d, "manifest.json")
        if d.startswith("_") or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if status is None or m["status"] == status:
            yield os.path.join(DATA, d), m


def summary():
    statuses, reasons, licenses, sources, problems = (collections.Counter() for _ in range(5))
    n_img = 0
    for _, m in people():
        statuses[m["status"]] += 1
        reasons.update(m["rejects"])
        if m["problem"]:
            problems[m["problem"]] += 1
        if m["status"] == "ok":
            n_img += m["kept"]
            licenses.update(i["license"] for i in m["images"])
            sources.update(i["source"] for i in m["images"])
    print(f"people: {dict(statuses)}\nphotos in ok people: {n_img} "
          f"(avg {n_img / max(statuses['ok'], 1):.1f})\nskipped because: {dict(problems)}")
    print(f"licenses: {dict(licenses.most_common(12))}\nsources: {dict(sources)}")
    print(f"rejects: {dict(reasons.most_common())}")


def sheets(rejects=False, per_page=10, cell=112):
    """One row per person (name + all kept photos), 10 people per page; --rejects shows rejected ones."""
    os.makedirs(SHEETS, exist_ok=True)
    rows = []
    for folder, m in people():
        if rejects:
            rdir = os.path.join(DATA, "_rejects", os.path.basename(folder))
            files = sorted(os.path.join(rdir, f) for f in os.listdir(rdir)) if os.path.isdir(rdir) else []
            labels = [os.path.basename(f).split("__")[0] for f in files]
        else:
            files = [os.path.join(folder, i["file"]) for i in m["images"]]
            labels = [f"{i['similarity']:.2f} {i['source'][:3]}" for i in m["images"]]
        rows.append((f"#{m['rank']} {m['name']} [{m['status']}]", files, labels))
    cols = max((len(r[1]) for r in rows), default=1)
    cols = min(cols, 24 if rejects else 15)
    for page in range(0, len(rows), per_page):
        chunk = rows[page:page + per_page]
        sheet = Image.new("RGB", (cols * cell, len(chunk) * (cell + 30)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, (title, files, labels) in enumerate(chunk):
            y = r * (cell + 30)
            draw.text((4, y + 2), title, fill="black")
            for c, (f, lab) in enumerate(list(zip(files, labels))[:cols]):
                im = ImageOps.fit(Image.open(f).convert("RGB"), (cell, cell))
                sheet.paste(im, (c * cell, y + 16))
                draw.text((c * cell + 2, y + 16 + cell - 12), lab[:18], fill="yellow")
        out = os.path.join(SHEETS, f"{'rejects' if rejects else 'kept'}_{page // per_page + 1:03d}.jpg")
        sheet.save(out, quality=85)
        print(out)


def check(fix=False):
    """Flags: (1) one Commons file kept for two people, (2) a photo closer to another person's centroid
    than to its own (or close to both), (3) two people whose centroids are nearly identical."""
    folders, names, embs, cents = [], [], [], []
    for folder, m in people("ok"):
        e = np.load(os.path.join(folder, "faces.npy"))
        folders.append((folder, m))
        names.append(m["name"])
        embs.append(e)
        c = e.sum(0)
        cents.append(c / np.linalg.norm(c))
    C = np.stack(cents)
    flags = collections.defaultdict(list)  # folder -> [(file, why)]

    by_title = collections.defaultdict(list)
    for (folder, m) in folders:
        for img in m["images"]:
            by_title[img["title"]].append((folder, img["file"]))
    for title, where in by_title.items():
        if len(where) > 1:
            for folder, file in where:
                flags[folder].append((file, f"same photo kept for {len(where)} people: {title}"))

    for p, ((folder, m), E) in enumerate(zip(folders, embs)):
        own = E @ C[p]
        others = E @ C.T
        others[:, p] = -1
        best = others.argmax(1)
        for i, img in enumerate(m["images"]):
            o = others[i, best[i]]
            if o >= own[i] - 0.05 or o >= 0.45:
                flags[folder].append((img["file"], f"looks like {names[best[i]]} ({o:.2f}) vs own ({own[i]:.2f})"))

    S = C @ C.T
    np.fill_diagonal(S, -1)
    twins = [(names[a], names[b], round(float(S[a, b]), 2)) for a, b in zip(*np.where(np.triu(S) >= 0.6))]

    n = sum(len(v) for v in flags.values())
    print(f"{len(folders)} people checked; {n} photos flagged in {len(flags)} people; "
          f"{len(twins)} near-identical pairs")
    for folder, items in sorted(flags.items()):
        for file, why in items:
            print(f"  {os.path.basename(folder)}/{file}: {why}")
    for a, b, s in twins:
        print(f"  same person? {a} <-> {b} ({s})")
    with open(os.path.join(HERE, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump({"flags": {os.path.basename(k): v for k, v in flags.items()}, "twins": twins}, f,
                  ensure_ascii=False, indent=1)
    if fix:
        quarantine(flags)


def quarantine(flags):
    """Moves flagged photos out and rewrites manifest + faces.npy; people left below MIN_KEEP become too_few."""
    from collect import MIN_KEEP
    for folder, items in flags.items():
        bad = {file for file, _ in items}
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        E = np.load(os.path.join(folder, "faces.npy"))
        keep = [i for i, img in enumerate(m["images"]) if img["file"] not in bad]
        qdir = os.path.join(DATA, "_quarantine", os.path.basename(folder))
        os.makedirs(qdir, exist_ok=True)
        for img in m["images"]:
            if img["file"] in bad:
                shutil.move(os.path.join(folder, img["file"]), os.path.join(qdir, img["file"]))
        m["images"] = [m["images"][i] for i in keep]
        m["kept"] = len(keep)
        m["rejects"]["audit_flagged"] = m["rejects"].get("audit_flagged", 0) + len(bad)
        if m["kept"] < MIN_KEEP:
            m["status"] = "too_few"
        np.save(os.path.join(folder, "faces.npy"), E[keep] if keep else np.zeros((0, 512), np.float32))
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
    print(f"quarantined photos from {len(flags)} people into data/_quarantine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "sheets", "check"])
    ap.add_argument("--rejects", action="store_true")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    {"summary": summary, "sheets": lambda: sheets(a.rejects), "check": lambda: check(a.fix)}[a.cmd]()
