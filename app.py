"""
Gradio UI for LFW face verification.

Tabs:
  Train       — start/stop background training, watch live log
  Scrape Data — download celebrity images from Google for training
  Identify    — upload a face, find closest match in LFW
  Feedback    — label random celebrity pairs to grow the training set
"""

import os
import csv
import time
import random
import threading
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torch.utils.data import DataLoader
import kagglehub

from lfw_pytorch import (
    SphereFaceNet, SiameseNet, FacePairDataset, MarginCosineProduct,
    train_transform, test_transform,
    DEVICE, EMBEDDING_SIZE, IMAGE_SIZE,
    load_csv_pairs, load_pairs_csv, generate_pairs_from_filesystem, generate_pairs_from_flat_dir,
    get_names_from_csv, k_fold_eval,
    scan_ms1mv2, sample_ms1mv2_epoch_pairs, MS1MV2Dataset,
)
from face_features import build_feature_cache, extract_face_features, FEAT_DIM, _ZERO_VEC, using_gpu
from celebrity_scraper import (
    scrape_all, get_scraped_pairs, count_images, clean_non_faces,
    ai_verify_all, CELEBRITIES, SCRAPE_ROOT,
)

APP_CHECKPOINT = "app_checkpoint.pt"
APP_BEST       = "app_best.pt"
FEEDBACK_CSV   = "feedback_pairs.csv"
EMBED_CACHE         = "embed_cache.npz"
EMBED_CACHE_BEST    = "embed_cache_best.npz"
EMBED_CACHE_CKPT    = "embed_cache_compare.npz"
FEAT_CACHE     = "feature_cache.pkl"
VGG_PATH_FILE    = "vggface2_path.txt"    # persists the kagglehub download location
MS1MV2_PATH_FILE = "ms1mv2_path.txt"     # path to MS1MV2 112x112 dataset
CELEBA_PATH_FILE = "celeba_path.txt"     # path to CelebA aligned faces
MARGIN         = 2.0


# ── contrastive loss ──────────────────────────────────────────────────────────

def contrastive_loss(dist, label, margin=MARGIN):
    """label=1 → same person (push dist toward 0), label=0 → different."""
    return (label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)).mean()


# ── shared state ──────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_train_thread  = None
_stop_event    = threading.Event()
_log_lines     = []
_scrape_thread = None
_scrape_log    = []
_dataset_path  = None
_embed_index   = None          # (names, paths, embeddings_array)
_feat_cache    = None          # dict: path -> np.array(FEAT_DIM)
_feedback_cur  = {'p1': None, 'p2': None}
_compare_log: list = []
_compare_thread = None
_embed_index_best = None
_embed_index_ckpt = None


def _get_dataset():
    global _dataset_path
    if _dataset_path is None:
        cached = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub",
                              "datasets", "jessicali9530", "lfw-dataset", "versions", "4")
        if os.path.isdir(cached):
            _dataset_path = cached
        else:
            _dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    return _dataset_path


def _lfw_root():
    return os.path.join(_get_dataset(), "lfw-deepfunneled", "lfw-deepfunneled")


def _vgg_root():
    """Return the VGGFace2 train directory, or None if not downloaded yet."""
    if not os.path.exists(VGG_PATH_FILE):
        return None
    base = open(VGG_PATH_FILE).read().strip()
    if not os.path.isdir(base):
        return None
    for candidate in [os.path.join(base, 'train'), base]:
        if os.path.isdir(candidate):
            subdirs = [d for d in os.listdir(candidate)
                       if os.path.isdir(os.path.join(candidate, d))]
            if len(subdirs) > 10:
                return candidate
    return None


def _vgg_status():
    root = _vgg_root()
    if root is None:
        return "VGGFace2: not downloaded"
    identities = sum(1 for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    images = sum(
        len([f for f in os.listdir(os.path.join(root, d))
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    return f"VGGFace2: {images:,} images across {identities:,} identities  ({root})"


def download_vggface2():
    """Generator — downloads VGGFace2 from Kaggle and reports progress."""
    yield "Searching Kaggle for VGGFace2 dataset..."
    KAGGLE_IDS = ["hearfool/vggface2", "vasukipatel/face-recognition-dataset"]
    base = None
    for kid in KAGGLE_IDS:
        try:
            yield f"Trying {kid} ..."
            base = kagglehub.dataset_download(kid)
            yield f"Downloaded to: {base}"
            break
        except Exception as e:
            yield f"  {kid} failed: {e}"

    if base is None:
        yield ("Could not find VGGFace2 on Kaggle automatically.\n"
               "Go to kaggle.com, search 'VGGFace2', download manually,\n"
               "then paste the path into vggface2_path.txt in this folder.")
        return

    with open(VGG_PATH_FILE, 'w') as f:
        f.write(base)

    yield _vgg_status()


# ── MS1MV2 ────────────────────────────────────────────────────────────────────

def _ms1mv2_root():
    """Return the MS1MV2 training directory, or None if not set up."""
    if not os.path.exists(MS1MV2_PATH_FILE):
        return None
    base = open(MS1MV2_PATH_FILE).read().strip()
    if not os.path.isdir(base):
        return None
    for candidate in [base, os.path.join(base, 'train'), os.path.join(base, 'images')]:
        if not os.path.isdir(candidate):
            continue
        # MS1MV2 has 85k+ identity subdirs
        n = sum(1 for e in os.scandir(candidate) if e.is_dir())
        if n > 1000:
            return candidate
    return None


def _ms1mv2_status():
    root = _ms1mv2_root()
    if root is None:
        return "MS1MV2: not set up  —  click button below for instructions"
    n = sum(1 for e in os.scandir(root) if e.is_dir())
    return f"MS1MV2: {n:,} identities  ({root})"


def download_ms1mv2():
    """
    Extract MS1MV2 from the already-downloaded archive, resuming where extraction
    was interrupted.  Falls back to kagglehub download if archive not present.
    """
    import zipfile

    ARCHIVE = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
        "yakhyokhuja", "ms1m-arcface-dataset", "1.archive"
    )
    DEST = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
        "yakhyokhuja", "ms1m-arcface-dataset", "versions", "1"
    )
    FINAL = os.path.join(DEST, "ms1m-arcface")

    # ── if archive already on disk, extract from it directly ─────────────────
    if os.path.exists(ARCHIVE):
        yield f"Archive found ({os.path.getsize(ARCHIVE)//1_073_741_824:.1f} GB) — extracting..."
        os.makedirs(FINAL, exist_ok=True)
        existing = set(os.listdir(FINAL))
        yield f"Already extracted: {len(existing):,} identity folders — resuming..."

        try:
            with zipfile.ZipFile(ARCHIVE) as zf:
                # Group entries by identity folder name to skip whole folders at once
                entries_by_id = {}
                for name in zf.namelist():
                    parts = name.split('/')
                    if len(parts) >= 3:                # ms1m-arcface/<id>/<file>
                        entries_by_id.setdefault(parts[1], []).append(name)

                total_ids  = len(entries_by_id)
                done_ids   = 0
                new_ids    = 0
                for identity, entries in entries_by_id.items():
                    if identity in existing:
                        done_ids += 1
                        continue
                    for entry in entries:
                        zf.extract(entry, DEST)
                    existing.add(identity)
                    done_ids += 1
                    new_ids  += 1
                    if new_ids % 500 == 0 or done_ids == total_ids:
                        yield f"  {done_ids:,}/{total_ids:,} identities extracted..."

        except Exception as e:
            yield f"Extraction error: {e}"
            return

        with open(MS1MV2_PATH_FILE, 'w') as f:
            f.write(FINAL)
        yield f"Done — {len(os.listdir(FINAL)):,} identities at {FINAL}"
        yield _ms1mv2_status()
        return

    # ── archive not present — try kagglehub download ──────────────────────────
    yield "Archive not found locally — trying Kaggle download (15 GB)..."
    KAGGLE_IDS = ["yakhyokhuja/ms1m-arcface-dataset"]
    base = None
    for kid in KAGGLE_IDS:
        try:
            yield f"Downloading {kid} ..."
            base = kagglehub.dataset_download(kid)
            yield f"Downloaded to: {base}"
            break
        except Exception as e:
            yield f"  {kid}: {e}"

    if base is None:
        yield (
            "Download failed.\n\n"
            "Manual setup:\n"
            "1. Download from Kaggle: yakhyokhuja/ms1m-arcface-dataset\n"
            "2. Extract so each identity is a subfolder under a single root\n"
            f"3. Write that root path into ms1mv2_path.txt"
        )
        return

    # kagglehub extracted it — find the identity folder root
    for candidate in [base, os.path.join(base, "ms1m-arcface")]:
        if os.path.isdir(candidate) and any(
            os.path.isdir(os.path.join(candidate, d)) for d in os.listdir(candidate)
        ):
            with open(MS1MV2_PATH_FILE, 'w') as f:
                f.write(candidate)
            yield f"Path saved: {candidate}"
            yield _ms1mv2_status()
            return

    with open(MS1MV2_PATH_FILE, 'w') as f:
        f.write(base)
    yield _ms1mv2_status()


# ── CelebA ────────────────────────────────────────────────────────────────────

def _celeba_root():
    """Return the CelebA aligned image directory, or None if not available."""
    if not os.path.exists(CELEBA_PATH_FILE):
        return None
    base = open(CELEBA_PATH_FILE).read().strip()
    for candidate in [
        os.path.join(base, 'img_align_celeba', 'img_align_celeba'),
        os.path.join(base, 'img_align_celeba'),
        os.path.join(base, 'celeba', 'img_align_celeba'),
        base,
    ]:
        if os.path.isdir(candidate):
            # Verify it contains jpg images
            try:
                sample = next((f for f in os.listdir(candidate) if f.endswith('.jpg')), None)
                if sample:
                    return candidate
            except Exception:
                pass
    return None


def _celeba_identity_file():
    """Return path to CelebA identity annotation file, or None."""
    if not os.path.exists(CELEBA_PATH_FILE):
        return None
    base = open(CELEBA_PATH_FILE).read().strip()
    for candidate in [
        os.path.join(base, 'Anno', 'identity_CelebA.txt'),
        os.path.join(base, 'identity_CelebA.txt'),
        os.path.join(base, 'celeba', 'Anno', 'identity_CelebA.txt'),
        os.path.join(base, 'list_identity_celeba.txt'),
        # jessicali9530 dataset puts it alongside the CSVs
        os.path.join(base, 'list_attr_celeba.csv').replace('list_attr_celeba.csv', 'identity_CelebA.txt'),
    ]:
        if os.path.exists(candidate):
            return candidate
    # also scan parent directory (kagglehub sometimes extracts alongside base)
    parent = os.path.dirname(base)
    for candidate in [
        os.path.join(parent, 'identity_CelebA.txt'),
        os.path.join(parent, 'list_identity_celeba.txt'),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def _celeba_status():
    root = _celeba_root()
    if root is None:
        return "CelebA: not downloaded"
    try:
        n_imgs = sum(1 for f in os.listdir(root) if f.endswith('.jpg'))
    except Exception:
        n_imgs = 0
    id_info = ""
    id_file = _celeba_identity_file()
    if id_file:
        try:
            ids = {line.strip().split()[1] for line in open(id_file)
                   if len(line.strip().split()) == 2}
            id_info = f"  |  {len(ids):,} identities"
        except Exception:
            pass
    return f"CelebA: {n_imgs:,} images{id_info}  ({root})"


def download_celeba():
    """
    Ensures CelebA images + identity labels are present.
    Images come from jessicali9530/celeba-dataset (already downloaded).
    Identity labels (identity_CelebA.txt) come from a separate Kaggle source
    if they are not already on disk.
    """
    base = open(CELEBA_PATH_FILE).read().strip() if os.path.exists(CELEBA_PATH_FILE) else None

    # ── step 1: images ────────────────────────────────────────────────────────
    if _celeba_root() is None:
        yield "Downloading CelebA images (~1.5 GB)..."
        try:
            base = kagglehub.dataset_download("jessicali9530/celeba-dataset")
            with open(CELEBA_PATH_FILE, 'w') as f:
                f.write(base)
            yield f"Images downloaded to: {base}"
        except Exception as e:
            yield f"Image download failed: {e}"
            return
    else:
        yield f"Images already present: {_celeba_root()}"

    # ── step 2: identity file ─────────────────────────────────────────────────
    if _celeba_identity_file() is not None:
        yield f"Identity file already present: {_celeba_identity_file()}"
    else:
        yield "identity_CelebA.txt not found — fetching from Kaggle..."
        identity_dest = os.path.join(base or open(CELEBA_PATH_FILE).read().strip(), 'identity_CelebA.txt')
        # Try several Kaggle datasets known to carry the CelebA identity annotations
        IDENTITY_SOURCES = [
            "nroman/celeba",
            "jessicali9530/celeba-dataset",
        ]
        got_identity = False
        for kid in IDENTITY_SOURCES:
            try:
                yield f"  Trying {kid} ..."
                id_base = kagglehub.dataset_download(kid)
                # search recursively for the identity file
                for root_d, _, files in os.walk(id_base):
                    for fn in files:
                        if fn.lower() in ('identity_celeba.txt', 'list_identity_celeba.txt'):
                            import shutil
                            shutil.copy(os.path.join(root_d, fn), identity_dest)
                            yield f"  Copied {fn} → {identity_dest}"
                            got_identity = True
                            break
                    if got_identity:
                        break
            except Exception as e:
                yield f"  {kid}: {e}"
            if got_identity:
                break

        if not got_identity:
            yield (
                "Could not fetch identity_CelebA.txt automatically.\n"
                "Download it manually from the official CelebA Google Drive:\n"
                "  drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs\n"
                f"Place it at: {identity_dest}\n"
                "CelebA images are still usable — search will show individual images."
            )

    yield _celeba_status()


def _load_model(path=None):
    model = SphereFaceNet(EMBEDDING_SIZE).to(DEVICE)
    src = path or (APP_BEST if os.path.exists(APP_BEST) else
                   (APP_CHECKPOINT if os.path.exists(APP_CHECKPOINT) else None))
    if src:
        try:
            state = torch.load(src, map_location=DEVICE, weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError:
                model.load_state_dict(state, strict=False)
        except RuntimeError:
            # checkpoint is from a different architecture — ignore it
            pass
    model.eval()
    return model


def _pil_square(img, size):
    """Resize PIL image to a square by resizing the short edge then center-cropping."""
    w, h   = img.size
    scale  = size / min(w, h)
    nw, nh = max(size, int(w * scale)), max(size, int(h * scale))
    img    = img.resize((nw, nh), Image.LANCZOS)
    left   = (nw - size) // 2
    top    = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def _get_feat_cache():
    global _feat_cache
    if _feat_cache is None and os.path.exists(FEAT_CACHE):
        import pickle
        with open(FEAT_CACHE, 'rb') as f:
            raw = pickle.load(f)
        # Discard entries with wrong dimension (cache built with old FEAT_DIM)
        _feat_cache = {k: v for k, v in raw.items() if len(v) == FEAT_DIM}
    return _feat_cache or {}


# ── feature diagnostic ────────────────────────────────────────────────────────

def _describe_features(feats):
    if np.all(feats == 0):
        return (
            "Face detection:  FAILED\n"
            "                 No face found — CNN embedding only.\n"
            "                 Skin, hair, face shape NOT compared.\n\n"
            "Tip: try a clearer, front-facing photo with good lighting."
        )

    detector = "InsightFace GPU" if using_gpu() else "MediaPipe CPU"
    lines = [f"Face detection:  OK  ({detector})\n"]

    # Skin tone
    L    = feats[20] * 255
    bval = feats[22] * 255 - 128
    skin_d = ("very light" if L>185 else "light" if L>155 else
              "medium" if L>120 else "medium-dark" if L>85 else "dark")
    warm   = "warm" if bval>8 else ("cool" if bval<-5 else "neutral")
    lines.append(f"Skin tone:       {skin_d}  (L={L:.0f}, {warm})")

    # Hair color
    H = feats[17]*179; S = feats[18]*255; V = feats[19]*255
    hair_d = ("black"          if V < 40 else
              "white/silver"   if S < 25 and V > 180 else
              "grey"           if S < 25 else
              "red/auburn"     if H < 15 or H > 165 else
              "dark brown"     if H < 30 else
              "blonde"         if H < 60 else "brown")
    lines.append(f"Hair color:      {hair_d}  (H={H:.0f}°, S={S:.0f}, V={V:.0f})")

    # Hair length
    hl   = feats[27]
    hl_d = "long" if hl > 0.5 else ("medium" if hl > 0.2 else "short")
    lines.append(f"Hair length:     {hl_d}  (score={hl:.2f})")

    # Eye color
    iH = ((feats[3]+feats[6])/2)*179; iS = ((feats[4]+feats[7])/2)*255
    eye_d = ("dark brown/black" if iS<25 else
             "brown"            if iH<20 or iH>150 else
             "hazel/amber"      if iH<45 else
             "green"            if iH<90 else "blue")
    lines.append(f"Eye color:       {eye_d}  (H={iH:.0f}°, S={iS:.0f})")

    # Face shape
    ratio   = feats[11]
    shape_d = ("round/wide" if ratio>0.90 else "oval" if ratio>0.75 else "narrow/long")
    lines.append(f"Face shape:      {shape_d}  (w/h={ratio:.2f})")

    # Facial proportions
    enr = feats[24]; nmr = feats[25]; mcr = feats[26]
    lines.append(f"Proportions:     eye→nose {enr:.2f}  nose→mouth {nmr:.2f}  mouth→chin {mcr:.2f}")

    # Nose
    nose  = feats[14]
    nose_d = "broad" if nose>1.4 else ("medium" if nose>0.8 else "narrow")
    lines.append(f"Nose:            {nose_d}  (w/h={nose:.2f})")

    # Jaw
    chin = feats[12]; mid = feats[13]
    jaw_d = ("square"          if chin>mid*0.97 else
             "tapered/V-shape" if chin<mid*0.87 else "slightly tapered")
    if chin > 0 or mid > 0:
        lines.append(f"Jaw shape:       {jaw_d}  (chin={chin:.2f}, mid={mid:.2f})")

    # Eye spacing
    ed  = feats[0]
    esp = "wide-set" if ed>0.45 else ("normal" if ed>0.32 else "close-set")
    lines.append(f"Eye spacing:     {esp}  ({ed:.3f})")

    # Lips
    lw = feats[15]; lh = feats[16]
    lip_d = "full/wide" if lw>0.50 else ("medium" if lw>0.38 else "narrow")
    lines.append(f"Lips:            {lip_d}  (w={lw:.2f}, h={lh:.3f})")

    # Forehead
    fore  = feats[23]
    fore_d = "high" if fore>0.28 else ("medium" if fore>0.18 else "low")
    lines.append(f"Forehead:        {fore_d}  ({fore:.3f})")

    # Age / gender (InsightFace only)
    age_n = feats[28]; gen = feats[29]
    if age_n > 0:
        age_yrs = int(age_n * 100)
        gen_s   = "male" if gen > 0.5 else "female"
        lines.append(f"Age / gender:    ~{age_yrs} yrs  ({gen_s})")

    lines.append(f"\nFeatures:        32-dim  |  detector: {detector}")
    return "\n".join(lines)


# ── TRAINING TAB ──────────────────────────────────────────────────────────────

def _train_worker(start_fresh=False):
    try:
        dp = _get_dataset()

        def log(msg):
            with _lock:
                _log_lines.append(msg)

        torch.backends.cudnn.benchmark = True

        # ── LFW test set (evaluation only, never trained on) ─────────────────
        log("Loading LFW test pairs (all 6,000 — reference protocol)...")
        test_pairs = load_pairs_csv(dp, exclude_people=set())
        if not test_pairs:
            test_pairs = load_csv_pairs(dp, "matchpairsDevTest.csv", "mismatchpairsDevTest.csv")
            log(f"  pairs.csv not found — using DevTest ({len(test_pairs)} pairs)")
        else:
            log(f"  {len(test_pairs)} pairs loaded for 10-fold CV")

        test_people = set()
        for p1, p2, _ in test_pairs:
            test_people.add(os.path.basename(os.path.dirname(p1)))
            test_people.add(os.path.basename(os.path.dirname(p2)))
        log(f"  Protecting {len(test_people)} LFW identities from training")

        # test loader fixed for all epochs — no features needed (pure CNN embeddings)
        test_loader = DataLoader(
            FacePairDataset(test_pairs, test_transform, feature_cache=None),
            batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        # ── MS1MV2 — sole training source ────────────────────────────────────
        ms1mv2_root = _ms1mv2_root()
        if not ms1mv2_root:
            log("ERROR: MS1MV2 not set up. Use the Download button in the Training tab first.")
            return

        log("Scanning MS1MV2 identities (one-time, ~30 s for 85k folders)...")
        ms1mv2_persons = scan_ms1mv2(ms1mv2_root, exclude_people=test_people)
        n_ids  = len(ms1mv2_persons)
        n_imgs = sum(len(v) for v in ms1mv2_persons.values())
        log(f"  {n_ids:,} identities  |  {n_imgs:,} images")

        # Build the full flat sample list once — all images, every epoch (matches reference)
        log("  Building sample list (all images)...")
        all_samples = []
        for class_idx, (_, imgs) in enumerate(ms1mv2_persons.items()):
            for p in imgs:
                all_samples.append((p, class_idx))
        log(f"  {len(all_samples):,} images per epoch  ({len(all_samples)//512:,} batches at bs=512)\n")

        # ── model + CosFace head ──────────────────────────────────────────────
        global _feat_cache, _embed_index
        model        = SphereFaceNet(EMBEDDING_SIZE).to(DEVICE)
        cosface_head = MarginCosineProduct(EMBEDDING_SIZE, n_ids).to(DEVICE)
        criterion    = nn.CrossEntropyLoss()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"Model: SphereFaceNet  ({n_params:.1f}M params)")

        # SGD lr=0.1 — matches reference exactly
        optimizer = optim.SGD(
            [{'params': model.parameters()}, {'params': cosface_head.parameters()}],
            lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)

        # AMP — uses FP16 Tensor Cores on CUDA (4060 Ti: 88 TFLOPS vs 22 TFLOPS FP32)
        use_amp = DEVICE.type == 'cuda'
        scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
        log(f"Optimizer: SGD lr=0.1 momentum=0.9 wd=5e-4  |  MultiStepLR [10,20,25]×0.1"
            f"  |  AMP {'ON' if use_amp else 'OFF'}")

        start_epoch = 0
        best_acc    = 0.0
        threshold   = 0.5

        if start_fresh:
            log("Starting from scratch — existing checkpoint ignored.\n")
        elif os.path.exists(APP_CHECKPOINT):
            ckpt = torch.load(APP_CHECKPOINT, map_location=DEVICE, weights_only=False)
            try:
                model.load_state_dict(ckpt['model'], strict=True)
            except RuntimeError:
                try:
                    model.load_state_dict(ckpt['model'], strict=False)
                    log("Architecture changed — backbone loaded, head layers start fresh.")
                except RuntimeError:
                    log("Checkpoint incompatible — starting fresh.")
                    ckpt = None
            if ckpt is not None:
                if 'cosface_head' in ckpt and ckpt.get('n_ids') == n_ids:
                    try:
                        cosface_head.load_state_dict(ckpt['cosface_head'])
                    except RuntimeError:
                        pass
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
                if 'scaler' in ckpt:
                    try:
                        scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        pass
                start_epoch = ckpt['epoch'] + 1
                best_acc    = ckpt['best_acc']
                threshold   = ckpt.get('threshold', 0.5)
                log(f"Resumed from epoch {ckpt['epoch']}  (best LFW: {best_acc*100:.1f}%)\n")
        else:
            log("No checkpoint — starting fresh.\n")

        train_loader = DataLoader(
            MS1MV2Dataset(all_samples, train_transform),
            batch_size=512, shuffle=True, num_workers=6, pin_memory=True,
            persistent_workers=True, prefetch_factor=4)

        for epoch in range(start_epoch, 35):
            if _stop_event.is_set():
                log("Stopped by user.")
                break

            # ── train (CosFace + AMP) ─────────────────────────────────────────
            model.train()
            cosface_head.train()
            tr_correct = tr_total = 0
            tr_loss_sum = 0.0
            tr_batches  = 0
            n_batches   = len(train_loader)
            epoch_start = time.time()
            for imgs, targets in train_loader:
                imgs    = imgs.to(DEVICE, non_blocking=True)
                targets = targets.long().to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                    embs = model.get_embedding(imgs)
                # CosFace head in FP32 — s=64 overflows FP16 on 85K classes
                logits = cosface_head(embs.float(), targets)
                loss   = criterion(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cosface_head.parameters()), 5.0)
                scaler.step(optimizer)
                scaler.update()

                tr_correct    += (logits.argmax(1) == targets).sum().item()
                tr_total      += len(targets)
                tr_loss_sum   += loss.item()
                tr_batches    += 1

                if tr_batches == 1:
                    log(f"  first batch loss: {loss.item():.3f}  scaler: {scaler.get_scale():.0f}")

                if tr_batches % 500 == 0 or tr_batches == n_batches:
                    elapsed   = time.time() - epoch_start
                    eta_s     = elapsed / tr_batches * (n_batches - tr_batches)
                    avg_l     = tr_loss_sum / tr_batches
                    tr_acc_so = tr_correct / tr_total * 100 if tr_total else 0
                    log(f"  [{tr_batches:5d}/{n_batches}]  "
                        f"loss {avg_l:.3f}  acc {tr_acc_so:.2f}%  "
                        f"elapsed {elapsed/60:.1f}m  eta {eta_s/60:.1f}m")

            # ── eval (LFW 10-fold CV) ─────────────────────────────────────────
            model.eval()
            all_dists, all_labels_np = [], []
            with torch.no_grad():
                for img1, img2, labels in test_loader:
                    img1 = img1.to(DEVICE, non_blocking=True)
                    img2 = img2.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                        e1 = model.get_embedding(img1)
                        e2 = model.get_embedding(img2)
                    dist = F.pairwise_distance(e1.float(), e2.float())
                    all_dists.extend(dist.cpu().numpy())
                    all_labels_np.extend(labels.numpy())

            all_dists     = np.array(all_dists)
            all_labels_np = np.array(all_labels_np)
            te_acc, te_std, threshold = k_fold_eval(all_dists, all_labels_np)
            tr_acc = tr_correct / tr_total
            scheduler.step()

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(model.state_dict(), APP_BEST)
                if os.path.exists(EMBED_CACHE):
                    os.remove(EMBED_CACHE)
                _embed_index = None
                log(f"  ↑ New best — saved app_best.pt")

            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'cosface_head': cosface_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc, 'threshold': threshold,
                'n_ids': n_ids, 'optimizer_type': 'SGD',
            }, APP_CHECKPOINT)

            lr = optimizer.param_groups[0]['lr']
            avg_loss = tr_loss_sum / tr_batches if tr_batches else float('nan')
            log(f"Epoch {epoch:2d}/34  loss {avg_loss:.3f}  train {tr_acc*100:.1f}%  "
                f"LFW {te_acc*100:.2f}%±{te_std*100:.2f}%  best {best_acc*100:.2f}%  "
                f"thr {threshold:.3f}  lr={lr:.2e}")

        log(f"\nDone. Best LFW accuracy: {best_acc*100:.2f}%")

    except Exception as exc:
        with _lock:
            _log_lines.append(f"ERROR: {exc}")
        raise


def start_training(start_fresh=False):
    global _train_thread
    if _train_thread and _train_thread.is_alive():
        return "Already training — check the log."
    _stop_event.clear()
    with _lock:
        _log_lines.clear()
    _train_thread = threading.Thread(target=_train_worker, args=(start_fresh,), daemon=True)
    _train_thread.start()
    return "Training started."


def stop_training():
    _stop_event.set()
    return "Stop signal sent — will pause after this epoch."


def get_log():
    with _lock:
        return "\n".join(_log_lines[-40:])


# ── SCRAPE TAB ────────────────────────────────────────────────────────────────

def _scrape_worker(n_per_celeb, groq_key=None):
    def log(msg):
        with _lock:
            _scrape_log.append(msg)

    log(f"Scraping {len(CELEBRITIES)} celebrities × {n_per_celeb} images each...")
    log("Using DuckDuckGo image search. May take 10–20 min.\n")

    def progress_cb(name, done, total):
        log(f"[{done}/{total}] {name}")

    try:
        scrape_all(n_per_celebrity=n_per_celeb, progress_cb=progress_cb)

        log("\nRemoving non-face images (cartoons, coloring pages, etc.)...")
        def clean_progress(done, total, kept, removed):
            if done % 100 == 0 or done == total:
                log(f"  Checked {done}/{total}  —  kept {kept}, removed {removed}")
        kept, removed = clean_non_faces(progress_cb=clean_progress)
        log(f"Cleanup done: {removed} junk images removed, {kept} face images kept.")

        total_imgs = sum(
            count_images(os.path.join(SCRAPE_ROOT, d))
            for d in os.listdir(SCRAPE_ROOT)
            if os.path.isdir(os.path.join(SCRAPE_ROOT, d))
        ) if os.path.isdir(SCRAPE_ROOT) else 0
        log(f"\nDone! {total_imgs} clean face images in {SCRAPE_ROOT}/")

        if groq_key:
            log("\nRunning AI verification (Groq) to remove wrong celebrities...")
            def ai_cb(done, total, kept, removed):
                if done % 50 == 0 or done == total:
                    log(f"  AI check: {done}/{total} — kept {kept}, removed {removed}")
            try:
                kept, removed = ai_verify_all(api_key=groq_key, progress_cb=ai_cb)
                log(f"AI verification done: {removed} wrong images removed, {kept} kept.")
            except RuntimeError as e:
                log(f"AI verification failed: {e}")

        log("Click 'Start Training' on the Train tab to use the new data.")
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
    except Exception as exc:
        log(f"ERROR: {exc}")


def start_scraping(n_per_celeb, groq_key):
    global _scrape_thread
    if _scrape_thread and _scrape_thread.is_alive():
        return "Already scraping — check the log below."
    with _lock:
        _scrape_log.clear()
    _scrape_thread = threading.Thread(
        target=_scrape_worker, args=(int(n_per_celeb), groq_key.strip() or None), daemon=True)
    _scrape_thread.start()
    return f"Scraping started ({len(CELEBRITIES)} celebrities)."


def start_ai_verify(groq_key):
    global _scrape_thread
    if _scrape_thread and _scrape_thread.is_alive():
        return "Scraping in progress — wait for it to finish first."
    key = groq_key.strip()
    if not key:
        return "Paste your Groq API key first (free at console.groq.com)."
    with _lock:
        _scrape_log.clear()
        _scrape_log.append("Running AI verification on all downloaded images...")
        _scrape_log.append("This checks every image with Groq vision — may take a while.\n")
    def _worker():
        def cb(done, total, kept, removed):
            if done % 50 == 0 or done == total:
                with _lock:
                    _scrape_log.append(
                        f"  {done}/{total} checked — kept {kept}, removed {removed}")
        try:
            kept, removed = ai_verify_all(api_key=key, progress_cb=cb)
            with _lock:
                _scrape_log.append(f"\nDone. Kept {kept} images, removed {removed} wrong/unclear ones.")
        except RuntimeError as e:
            with _lock:
                _scrape_log.append(f"\nERROR: {e}")
    _scrape_thread = threading.Thread(target=_worker, daemon=True)
    _scrape_thread.start()
    return "AI verification started — click Refresh Log to watch progress."


def get_scrape_log():
    with _lock:
        return "\n".join(_scrape_log[-50:])


def get_scrape_status():
    if not os.path.isdir(SCRAPE_ROOT):
        return "No scraped data yet."
    n_celebs = sum(
        1 for d in os.listdir(SCRAPE_ROOT)
        if os.path.isdir(os.path.join(SCRAPE_ROOT, d)))
    total_imgs = sum(
        count_images(os.path.join(SCRAPE_ROOT, d))
        for d in os.listdir(SCRAPE_ROOT)
        if os.path.isdir(os.path.join(SCRAPE_ROOT, d)))
    return f"{total_imgs} images across {n_celebs} celebrities stored in {SCRAPE_ROOT}/"


# ── IDENTIFY TAB ──────────────────────────────────────────────────────────────

def _build_faiss(embeddings: np.ndarray):
    """Build an exact L2 FAISS index from a float32 (N, D) array."""
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    return idx


def _build_embed_index(status_cb=None):
    global _embed_index
    if _embed_index is not None:
        return _embed_index

    if os.path.exists(EMBED_CACHE):
        d = np.load(EMBED_CACHE, allow_pickle=True)
        cached_size = int(d['image_size']) if 'image_size' in d else 96
        if cached_size != IMAGE_SIZE:
            if status_cb:
                status_cb(f"Cache built at {cached_size}px, model now uses {IMAGE_SIZE}px — rebuilding...")
            os.remove(EMBED_CACHE)
        else:
            if status_cb:
                status_cb("Loading cached embedding index...")
            names = d['names'].tolist()
            paths = d['paths'].tolist()
            embs  = d['embeddings'].astype(np.float32)
            raw   = d['features'] if 'features' in d else None
            feats = (raw if raw is not None and raw.shape[1] == FEAT_DIM
                     else np.zeros((len(names), FEAT_DIM), dtype=np.float32))
            fidx  = _build_faiss(embs)
            _embed_index = (names, paths, fidx, feats)
            if status_cb:
                status_cb(f"Index loaded: {len(names)} images  (FAISS ready).")
            return _embed_index

    model      = _load_model()
    feat_cache = _get_feat_cache()
    IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.webp'}

    all_names, all_paths = [], []

    root = _lfw_root()
    if os.path.isdir(root):
        for person in sorted(os.listdir(root)):
            folder = os.path.join(root, person)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.jpg'):
                    all_names.append(person)
                    all_paths.append(os.path.join(folder, fname))

    if os.path.isdir(SCRAPE_ROOT):
        for celeb in sorted(os.listdir(SCRAPE_ROOT)):
            folder = os.path.join(SCRAPE_ROOT, celeb)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    all_names.append(celeb)
                    all_paths.append(os.path.join(folder, fname))

    # CelebA — 10,177 celebrity identities for Ollie's lookalike search
    celeba_dir = _celeba_root()
    celeba_id_file = _celeba_identity_file()
    if celeba_dir:
        if status_cb:
            status_cb("Adding CelebA celebrity faces to search index...")
        if celeba_id_file:
            # Group images by identity, sample up to 5 per identity to keep index manageable
            img_to_id = {}
            with open(celeba_id_file) as _f:
                for _line in _f:
                    _parts = _line.strip().split()
                    if len(_parts) == 2:
                        img_to_id[_parts[0]] = f"celeb_{_parts[1]}"
            id_to_imgs = defaultdict(list)
            for _fname, _identity in img_to_id.items():
                id_to_imgs[_identity].append(_fname)
            for _identity, _fnames in id_to_imgs.items():
                sampled = random.sample(_fnames, min(5, len(_fnames)))
                for _fname in sampled:
                    _fpath = os.path.join(celeba_dir, _fname)
                    if os.path.exists(_fpath):
                        all_names.append(_identity)
                        all_paths.append(_fpath)
        else:
            # No identity file — add all images
            for _fname in sorted(os.listdir(celeba_dir)):
                if _fname.endswith('.jpg'):
                    all_names.append('celeba')
                    all_paths.append(os.path.join(celeba_dir, _fname))

    total = len(all_paths)
    if status_cb:
        status_cb(f"Embedding {total} images — this takes ~1-3 min on GPU...")

    all_embs  = []
    all_feats = []
    for i in range(0, total, 64):
        batch_paths = all_paths[i:i+64]
        imgs, batch_feats = [], []
        for p in batch_paths:
            try:
                imgs.append(test_transform(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(test_transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))))
            batch_feats.append(feat_cache.get(p, _ZERO_VEC))

        feat_arr = np.stack(batch_feats).astype(np.float32)
        imgs_t   = torch.stack(imgs).to(DEVICE)
        feats_t  = torch.tensor(feat_arr).to(DEVICE)
        with torch.no_grad():
            all_embs.append(model.get_embedding(imgs_t, feats_t).cpu().numpy())
        all_feats.append(feat_arr)

        if status_cb and (i // 64) % 10 == 0:
            status_cb(f"  Embedded {min(i+64, total)}/{total}...")

    embeddings   = np.concatenate(all_embs,  axis=0).astype(np.float32)
    features_arr = np.concatenate(all_feats, axis=0)
    np.savez(EMBED_CACHE,
             names=np.array(all_names, dtype=object),
             paths=np.array(all_paths, dtype=object),
             embeddings=embeddings,
             features=features_arr,
             image_size=np.array(IMAGE_SIZE))
    fidx = _build_faiss(embeddings)
    _embed_index = (all_names, all_paths, fidx, features_arr)
    if status_cb:
        status_cb(f"Index built: {total} images  (FAISS ready).")
    return _embed_index


def build_feature_index():
    """
    Generator. Runs parallel MediaPipe extraction on every index image that
    currently has zero features, then updates embed_cache.npz in-place and
    resets the in-memory index so the next search uses real features.
    """
    global _embed_index

    if not os.path.exists(EMBED_CACHE):
        yield "No embed cache found — run a search first to build the embedding index."; return

    yield "Loading embed cache..."
    d     = np.load(EMBED_CACHE, allow_pickle=True)
    paths = d['paths'].tolist()
    feats = d['features'].copy() if 'features' in d else np.zeros((len(paths), FEAT_DIM), dtype=np.float32)

    missing = [i for i, f in enumerate(feats) if np.all(f == 0)]
    total   = len(missing)
    if total == 0:
        yield f"All {len(paths)} images already have features — nothing to do."; return

    if using_gpu():
        yield f"Extracting features for {total}/{len(paths)} images using InsightFace GPU..."
        done = 0
        for idx in missing:
            try:
                feats[idx] = extract_face_features(Image.open(paths[idx]).convert('RGB'))
            except Exception:
                pass
            done += 1
            if done % 500 == 0 or done == total:
                yield f"  {done}/{total} features extracted  (GPU)..."
    else:
        workers = min(8, (os.cpu_count() or 4))
        yield f"Extracting features for {total}/{len(paths)} images using {workers} CPU workers..."
        done = 0
        lock = threading.Lock()

        def _extract(idx):
            try:
                return idx, extract_face_features(Image.open(paths[idx]).convert('RGB'))
            except Exception:
                return idx, _ZERO_VEC.copy()

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_extract, i): i for i in missing}
            for fut in as_completed(futures):
                i2, feat = fut.result()
                with lock:
                    feats[i2] = feat
                    done += 1
                    if done % 500 == 0 or done == total:
                        yield f"  {done}/{total} features extracted  (CPU ×{workers})..."

    yield "Saving updated features to embed cache..."
    np.savez(EMBED_CACHE,
             names=d['names'],
             paths=d['paths'],
             embeddings=d['embeddings'],
             features=feats)

    _embed_index = None  # force reload on next search
    nonzero = int(np.any(feats != 0, axis=1).sum())
    yield f"Done. {nonzero}/{len(paths)} images now have face features. Re-ranking will be much more accurate."


def find_dataset_matches(image, mode="CNN + Features"):
    """Generator — yields (diag_text, gallery_items)."""
    if image is None:
        yield "Upload a face photo first.", []
        return

    if not os.path.exists(APP_BEST) and not os.path.exists(APP_CHECKPOINT):
        yield "No trained model yet — click Start Training first.", []
        return

    yield "Extracting face features...", []

    model   = _load_model()
    img_pil = Image.fromarray(image).convert('RGB')
    q_feats = extract_face_features(img_pil)

    # CNN embedding (skip for Features Only)
    q_emb = None
    if mode != "Features Only":
        img_t    = test_transform(img_pil).unsqueeze(0).to(DEVICE)
        q_feat_t = torch.tensor(q_feats).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_emb = model.get_embedding(img_t, q_feat_t).cpu().numpy()[0]

    diag_lines = ["── Uploaded image analysis ──────────────────────────\n"]
    diag_lines.append(_describe_features(q_feats))
    yield "\n".join(diag_lines) + "\n\nBuilding search index...", []

    # Stream index-build progress into the diagnostic box
    log_buf = list(diag_lines)

    def index_progress(msg):
        log_buf.append(msg)

    names, paths, faiss_idx, index_features = _build_embed_index(status_cb=index_progress)

    # Show index progress in the textbox
    yield "\n".join(log_buf), []

    if len(names) == 0:
        yield "\n".join(log_buf) + "\n\nNo images in embed index.", []
        return

    q_has_feats = not np.all(q_feats == 0)
    scored = []

    if mode == "Features Only":
        # ── pure feature distance search ─────────────────────────────────────
        if not q_has_feats:
            yield "\n".join(log_buf) + "\n\nFace detection failed — can't do Features Only search.", []
            return
        has_feats = np.any(index_features != 0, axis=1)
        if not has_feats.any():
            yield "\n".join(log_buf) + "\n\nNo feature index built yet — click Build Feature Index first.", []
            return
        # Weighted Euclidean distance
        W = np.ones(FEAT_DIM, dtype=np.float32)
        W[20]=5.0; W[17]=2.0; W[11]=1.5; W[28]=3.0; W[29]=4.0
        diffs     = index_features - q_feats[np.newaxis, :]
        feat_dist = np.sqrt((diffs**2 * W[np.newaxis,:]).sum(axis=1))
        feat_dist[~has_feats] = np.inf
        top200_idx = np.argsort(feat_dist)[:200]
        for abs_i in top200_idx:
            if feat_dist[abs_i] == np.inf: continue
            scored.append((float(feat_dist[abs_i]), abs_i, float(feat_dist[abs_i])))
        log_buf.append(f"\nSearch mode: Features Only  ({int(has_feats.sum())} images with features)")

    else:
        # ── FAISS CNN search ──────────────────────────────────────────────────
        q_emb_f = np.ascontiguousarray(q_emb.reshape(1,-1), dtype=np.float32)
        D, I    = faiss_idx.search(q_emb_f, 200)
        top200_idx  = I[0]
        top200_dist = np.sqrt(np.maximum(D[0], 0.0))

        use_feats = (mode == "CNN + Features") and q_has_feats
        for abs_i, embed_dist in zip(top200_idx, top200_dist):
            if abs_i < 0: continue
            embed_dist = float(embed_dist)
            penalty    = 0.0
            if use_feats:
                c_feats = index_features[abs_i]
                c_has   = not np.all(c_feats == 0)
                if c_has:
                    skin_diff  = abs(float(q_feats[20]) - float(c_feats[20]))
                    hh         = abs(float(q_feats[17]) - float(c_feats[17]))
                    hair_diff  = min(hh, 1.0 - hh)
                    shape_diff = abs(float(q_feats[11]) - float(c_feats[11]))
                    penalty    = 5.0*skin_diff + 2.0*hair_diff + shape_diff
                    # age penalty (if both have age)
                    if q_feats[28] > 0 and c_feats[28] > 0:
                        penalty += 3.0 * abs(float(q_feats[28]) - float(c_feats[28]))
                    # gender penalty
                    if q_feats[29] > 0 and c_feats[29] > 0:
                        penalty += 4.0 * abs(float(q_feats[29]) - float(c_feats[29]))
                else:
                    penalty = 0.30
            scored.append((embed_dist + penalty, abs_i, embed_dist))

        if mode == "CNN Only":
            log_buf.append("\nSearch mode: CNN Only  (no feature re-ranking)")
        elif use_feats:
            log_buf.append("\nSearch mode: CNN + Features  "
                           "(skin×5, hair×2, shape×1.5, age×3, gender×4)")
        else:
            log_buf.append("\nSearch mode: CNN + Features  (OFF — no face detected)")

    scored.sort(key=lambda x: x[0])

    gallery_items = []
    seen_names    = {}

    for combined, abs_i, embed_dist in scored:
        display = names[abs_i].replace('_', ' ')
        sim_pct = max(0.0, (1.0 - embed_dist / MARGIN)) * 100

        seen_names[display] = seen_names.get(display, 0) + 1
        if seen_names[display] > 2:
            continue

        try:
            img_match = _pil_square(Image.open(paths[abs_i]).convert('RGB'), 160)
            caption   = f"{display}  ({sim_pct:.0f}%)"
            gallery_items.append((img_match, caption))
        except Exception:
            continue

        if len(gallery_items) >= 12:
            break

    log_buf.append("\n── Closest matches ──────────────────────────────────")
    if gallery_items:
        for item in gallery_items[:5]:
            log_buf.append(f"  {item[1]}")
    else:
        log_buf.append("  No matches found.")
    log_buf.append(f"\nShowing {len(gallery_items)} images from dataset "
                   f"({len(names)} total indexed).")

    if np.all(q_feats == 0):
        log_buf.append(
            "\nWARNING: No face landmarks detected — only CNN used.\n"
            "         Matches may not respect skin tone / face shape.\n"
            "         Try a clearer, front-facing photo."
        )

    yield "\n".join(log_buf), gallery_items


# ── FEEDBACK TAB ──────────────────────────────────────────────────────────────

def _feedback_count():
    if not os.path.exists(FEEDBACK_CSV):
        return 0
    with open(FEEDBACK_CSV) as f:
        return sum(1 for _ in f)


def _celebrity_image_index():
    """Return {name: [path, ...]} for all scraped celebrities with >= 2 images."""
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
    result = {}
    if not os.path.isdir(SCRAPE_ROOT):
        return result
    for name in sorted(os.listdir(SCRAPE_ROOT)):
        folder = os.path.join(SCRAPE_ROOT, name)
        if not os.path.isdir(folder):
            continue
        imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if len(imgs) >= 2:
            result[name] = imgs
    return result


def load_feedback_pair():
    """
    Loads a pair from scraped celebrity images when available (falls back to LFW).
    Samples uniformly by celebrity so no single person dominates the session.
    Shows the celebrity names so the user is verifying downloads, not guessing.
    """
    celeb_index = _celebrity_image_index()

    if celeb_index:
        people = list(celeb_index.keys())
        same   = random.random() > 0.5

        if same:
            # Pick one celebrity at random — uniform over celebrities, not images
            person = random.choice(people)
            p1, p2 = random.sample(celeb_index[person], 2)
            hint   = f"Both should be {person.replace('_', ' ')} — correct?"
        else:
            per1, per2 = random.sample(people, 2)
            p1 = random.choice(celeb_index[per1])
            p2 = random.choice(celeb_index[per2])
            hint = (f"{per1.replace('_', ' ')}  vs  {per2.replace('_', ' ')}"
                    " — are these really different people?")
    else:
        # Fallback: LFW pairs (no scraped data yet)
        root   = _lfw_root()
        people = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        same   = random.random() > 0.5
        hint   = "LFW pair (scrape celebrities for better feedback)"

        if same:
            cands  = [p for p in people if len(os.listdir(os.path.join(root, p))) >= 2]
            person = random.choice(cands)
            imgs   = [f for f in os.listdir(os.path.join(root, person)) if f.endswith('.jpg')]
            f1, f2 = random.sample(imgs, 2)
            p1 = os.path.join(root, person, f1)
            p2 = os.path.join(root, person, f2)
        else:
            per1, per2 = random.sample(people, 2)
            imgs1 = [f for f in os.listdir(os.path.join(root, per1)) if f.endswith('.jpg')]
            imgs2 = [f for f in os.listdir(os.path.join(root, per2)) if f.endswith('.jpg')]
            p1 = os.path.join(root, per1, random.choice(imgs1))
            p2 = os.path.join(root, per2, random.choice(imgs2))

    _feedback_cur['p1'] = p1
    _feedback_cur['p2'] = p2
    img1  = _pil_square(Image.open(p1).convert('RGB'), 220)
    img2  = _pil_square(Image.open(p2).convert('RGB'), 220)
    count = _feedback_count()
    return img1, img2, f"{count} labelled  |  {hint}"


def _submit_feedback(label: float):
    p1, p2 = _feedback_cur.get('p1'), _feedback_cur.get('p2')
    if p1 and p2:
        with open(FEEDBACK_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([p1, p2, label])
    return load_feedback_pair()


def label_same():
    return _submit_feedback(1.0)


def label_diff():
    return _submit_feedback(0.0)


# ── COMPARE MODELS TAB ────────────────────────────────────────────────────────

def _load_model_file(path: str):
    """Load a SphereFaceNet from a path — handles both bare state-dict and full checkpoint."""
    m = SphereFaceNet(EMBEDDING_SIZE).to(DEVICE)
    raw = torch.load(path, map_location=DEVICE, weights_only=False)
    state = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    try:
        m.load_state_dict(state, strict=True)
    except RuntimeError:
        try:
            m.load_state_dict(state, strict=False)
        except RuntimeError:
            pass  # incompatible checkpoint (e.g. old 256-dim) — return fresh model
    m.eval()
    return m


def _search_with_index(index_tuple, q_emb, q_feats):
    """FAISS search + feature re-ranking. Returns list of (pil_img, caption) tuples."""
    names, paths, fidx, index_features = index_tuple
    q_emb_f = np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32)
    D, I    = fidx.search(q_emb_f, 200)
    top_idx  = I[0]
    top_dist = np.sqrt(np.maximum(D[0], 0.0))

    q_has_feats = not np.all(q_feats == 0)
    scored = []
    for abs_i, embed_dist in zip(top_idx, top_dist):
        if abs_i < 0:
            continue
        penalty = 0.0
        if q_has_feats:
            c = index_features[abs_i]
            if not np.all(c == 0):
                skin_diff  = abs(float(q_feats[20]) - float(c[20]))
                hh         = abs(float(q_feats[17]) - float(c[17]))
                hair_diff  = min(hh, 1.0 - hh)
                shape_diff = abs(float(q_feats[11]) - float(c[11]))
                penalty    = 5.0*skin_diff + 2.0*hair_diff + shape_diff
                if q_feats[28] > 0 and c[28] > 0:
                    penalty += 3.0 * abs(float(q_feats[28]) - float(c[28]))
                if q_feats[29] > 0 and c[29] > 0:
                    penalty += 4.0 * abs(float(q_feats[29]) - float(c[29]))
            else:
                penalty = 0.30
        scored.append((float(embed_dist) + penalty, abs_i, float(embed_dist)))

    scored.sort(key=lambda x: x[0])
    gallery, seen = [], {}
    for _, abs_i, embed_dist in scored:
        display = names[abs_i].replace('_', ' ')
        seen[display] = seen.get(display, 0) + 1
        if seen[display] > 2:
            continue
        sim_pct = max(0.0, (1.0 - embed_dist / MARGIN)) * 100
        try:
            gallery.append((_pil_square(Image.open(paths[abs_i]).convert('RGB'), 160),
                            f"{display}  ({sim_pct:.0f}%)"))
        except Exception:
            continue
        if len(gallery) >= 10:
            break
    return gallery


def _compare_build_worker():
    global _embed_index_best, _embed_index_ckpt

    def log(msg):
        with _lock:
            _compare_log.append(msg)

    try:
        if not os.path.exists(APP_BEST):
            log("ERROR: app_best.pt not found."); return
        if not os.path.exists(APP_CHECKPOINT):
            log("ERROR: app_checkpoint.pt not found."); return

        log("Loading Best model (app_best.pt)...")
        model_best = _load_model_file(APP_BEST)
        log("Loading Checkpoint model (app_checkpoint.pt)...")
        model_ckpt = _load_model_file(APP_CHECKPOINT)

        feat_cache = _get_feat_cache()
        IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.webp'}
        all_names, all_paths = [], []

        root = _lfw_root()
        if os.path.isdir(root):
            for person in sorted(os.listdir(root)):
                folder = os.path.join(root, person)
                if not os.path.isdir(folder):
                    continue
                for fname in sorted(os.listdir(folder)):
                    if fname.endswith('.jpg'):
                        all_names.append(person)
                        all_paths.append(os.path.join(folder, fname))

        if os.path.isdir(SCRAPE_ROOT):
            for celeb in sorted(os.listdir(SCRAPE_ROOT)):
                folder = os.path.join(SCRAPE_ROOT, celeb)
                if not os.path.isdir(folder):
                    continue
                for fname in sorted(os.listdir(folder)):
                    if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                        all_names.append(celeb)
                        all_paths.append(os.path.join(folder, fname))

        total = len(all_paths)
        log(f"Embedding {total} images through both models in one pass — ~2-4 min...")

        all_embs_best, all_embs_ckpt, all_feats = [], [], []
        for i in range(0, total, 64):
            batch_paths = all_paths[i:i+64]
            imgs, batch_feats = [], []
            for p in batch_paths:
                try:
                    imgs.append(test_transform(Image.open(p).convert('RGB')))
                except Exception:
                    imgs.append(test_transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))))
                batch_feats.append(feat_cache.get(p, _ZERO_VEC))

            feat_arr = np.stack(batch_feats).astype(np.float32)
            imgs_t   = torch.stack(imgs).to(DEVICE)
            feats_t  = torch.tensor(feat_arr).to(DEVICE)
            with torch.no_grad():
                all_embs_best.append(model_best.get_embedding(imgs_t, feats_t).cpu().numpy())
                all_embs_ckpt.append(model_ckpt.get_embedding(imgs_t, feats_t).cpu().numpy())
            all_feats.append(feat_arr)

            if i % (64 * 20) == 0 and i > 0:
                log(f"  {min(i + 64, total)}/{total} embedded...")

        embs_best = np.concatenate(all_embs_best, axis=0).astype(np.float32)
        embs_ckpt = np.concatenate(all_embs_ckpt, axis=0).astype(np.float32)
        feats_arr = np.concatenate(all_feats, axis=0)
        names_arr = np.array(all_names, dtype=object)
        paths_arr = np.array(all_paths, dtype=object)

        log("Saving caches and building FAISS indices...")
        np.savez(EMBED_CACHE_BEST, names=names_arr, paths=paths_arr,
                 embeddings=embs_best, features=feats_arr)
        np.savez(EMBED_CACHE_CKPT, names=names_arr, paths=paths_arr,
                 embeddings=embs_ckpt, features=feats_arr)

        _embed_index_best = (all_names, all_paths, _build_faiss(embs_best), feats_arr)
        _embed_index_ckpt = (all_names, all_paths, _build_faiss(embs_ckpt), feats_arr)
        log(f"Done — {total} images indexed. Ready to compare.")

    except Exception as exc:
        with _lock:
            _compare_log.append(f"ERROR: {exc}")
        raise


def start_compare_build():
    global _compare_thread, _embed_index_best, _embed_index_ckpt
    if _train_thread and _train_thread.is_alive():
        return "Stop training first before building the comparison index."
    if _compare_thread and _compare_thread.is_alive():
        return "Already building — check the log."
    _embed_index_best = None
    _embed_index_ckpt = None
    with _lock:
        _compare_log.clear()
    _compare_thread = threading.Thread(target=_compare_build_worker, daemon=True)
    _compare_thread.start()
    return "Building comparison indices (both models, one pass)..."


def get_compare_log():
    with _lock:
        return "\n".join(_compare_log[-30:])


def _load_compare_cache(cache_file):
    d     = np.load(cache_file, allow_pickle=True)
    embs  = d['embeddings'].astype(np.float32)
    feats = d['features'] if 'features' in d else np.zeros((len(d['names']), FEAT_DIM), dtype=np.float32)
    return d['names'].tolist(), d['paths'].tolist(), _build_faiss(embs), feats


def run_compare_search(image):
    global _embed_index_best, _embed_index_ckpt

    if image is None:
        return "Upload a photo first.", [], []

    # Lazy-load caches from disk if available
    if _embed_index_best is None and os.path.exists(EMBED_CACHE_BEST):
        _embed_index_best = _load_compare_cache(EMBED_CACHE_BEST)
    if _embed_index_ckpt is None and os.path.exists(EMBED_CACHE_CKPT):
        _embed_index_ckpt = _load_compare_cache(EMBED_CACHE_CKPT)

    if _embed_index_best is None or _embed_index_ckpt is None:
        return "Click 'Build Comparison Index' first.", [], []

    if not os.path.exists(APP_BEST) or not os.path.exists(APP_CHECKPOINT):
        return "Both app_best.pt and app_checkpoint.pt must exist.", [], []

    img_pil  = Image.fromarray(image).convert('RGB')
    q_feats  = extract_face_features(img_pil)
    img_t    = test_transform(img_pil).unsqueeze(0).to(DEVICE)
    feats_t  = torch.tensor(q_feats).unsqueeze(0).to(DEVICE)

    model_best = _load_model_file(APP_BEST)
    model_ckpt = _load_model_file(APP_CHECKPOINT)

    with torch.no_grad():
        q_emb_best = model_best.get_embedding(img_t, feats_t).cpu().numpy()[0]
        q_emb_ckpt = model_ckpt.get_embedding(img_t, feats_t).cpu().numpy()[0]

    gallery_best = _search_with_index(_embed_index_best, q_emb_best, q_feats)
    gallery_ckpt = _search_with_index(_embed_index_ckpt, q_emb_ckpt, q_feats)
    return "Done.", gallery_best, gallery_ckpt


def search_and_compare(image, mode):
    """
    Generator — extracts face features, then searches the dataset with both
    app_best.pt and app_checkpoint.pt side by side.
    Requires the comparison index to be built first (start_compare_build).
    """
    if image is None:
        yield "Upload a face photo first.", [], []
        return

    missing = [f for f in [APP_BEST, APP_CHECKPOINT] if not os.path.exists(f)]
    if missing:
        yield f"Missing: {', '.join(missing)} — train the model first.", [], []
        return

    img_pil = Image.fromarray(image).convert('RGB')
    q_feats = extract_face_features(img_pil)
    q_feats_search = _ZERO_VEC.copy() if mode == "CNN Only" else q_feats

    diag = "── Feature Analysis ──────────────────────────────────────\n"
    diag += _describe_features(q_feats)
    yield diag + "\n\nLoading index...", [], []

    global _embed_index_best, _embed_index_ckpt
    if _embed_index_best is None and os.path.exists(EMBED_CACHE_BEST):
        _embed_index_best = _load_compare_cache(EMBED_CACHE_BEST)
    if _embed_index_ckpt is None and os.path.exists(EMBED_CACHE_CKPT):
        _embed_index_ckpt = _load_compare_cache(EMBED_CACHE_CKPT)

    if _embed_index_best is None or _embed_index_ckpt is None:
        yield (diag + "\n\nNo index yet — click **Build Index** first.\n"
               "This embeds all dataset images through both models (~2–4 min)."), [], []
        return

    img_t   = test_transform(img_pil).unsqueeze(0).to(DEVICE)
    feats_t = torch.tensor(q_feats).unsqueeze(0).to(DEVICE)

    model_best = _load_model_file(APP_BEST)
    model_ckpt = _load_model_file(APP_CHECKPOINT)

    with torch.no_grad():
        q_emb_best = model_best.get_embedding(img_t, feats_t).cpu().numpy()[0]
        q_emb_ckpt = model_ckpt.get_embedding(img_t, feats_t).cpu().numpy()[0]

    g_best = _search_with_index(_embed_index_best, q_emb_best, q_feats_search)
    g_ckpt = _search_with_index(_embed_index_ckpt, q_emb_ckpt, q_feats_search)

    mode_note = {"CNN + Features": "CNN + feature re-ranking",
                 "CNN Only":       "CNN only"}.get(mode, mode)
    diag += f"\n\nMode: {mode_note}  |  Best: {len(g_best)} matches  |  Checkpoint: {len(g_ckpt)} matches"
    yield diag, g_best, g_ckpt


# ── GRADIO UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Face Verification") as app:
    gr.Markdown("# Face Verification — Siamese Network")

    with gr.Tabs():

        # ── TAB 1: TRAIN ──────────────────────────────────────────────────────
        with gr.Tab("Train"):
            gr.Markdown(
                "Trains with **contrastive loss** + **geometric face features** "
                "(inter-eye distance, iris colour & texture, eye aspect ratio).  \n"
                "Input size: **112×112** (MS1MV2 native). "
                "Uses LFW + MS1MV2 + VGGFace2 + scraped celebrities + feedback labels.  \n"
                "With MS1MV2/VGGFace2: switches to **SGD + MultiStepLR** (reference protocol). "
                "**6,000-pair LFW 10-fold CV** used for validation every epoch (reference protocol)."
            )
            chk_fresh = gr.Checkbox(label="Start from scratch (ignore existing checkpoint)",
                                    value=False)
            with gr.Row():
                btn_start   = gr.Button("Start Training", variant="primary")
                btn_stop    = gr.Button("Stop Training",  variant="stop")
                btn_refresh = gr.Button("Refresh Log")
            status_box = gr.Textbox(label="Status", lines=1, interactive=False)
            log_box    = gr.Textbox(label="Training Log", lines=22, interactive=False)

            btn_start.click(start_training, inputs=chk_fresh, outputs=status_box)
            btn_stop.click(stop_training,   outputs=status_box)
            btn_refresh.click(get_log,      outputs=log_box)

            gr.Markdown("---")
            gr.Markdown(
                "**VGGFace2** — 3.3M images across 9,131 celebrities, purpose-built for face recognition.  \n"
                "Download once, then restart training — pairs are added automatically.  \n"
                "*(Requires Kaggle API key in `~/.kaggle/kaggle.json`)*"
            )
            vgg_status_box = gr.Textbox(label="VGGFace2 Status",
                                        value=_vgg_status(), lines=1, interactive=False)
            btn_vgg = gr.Button("Download VGGFace2", variant="secondary")
            vgg_log = gr.Textbox(label="Download Log", lines=5, interactive=False)

            btn_vgg.click(download_vggface2, inputs=None,
                          outputs=vgg_log, show_progress="hidden")
            btn_vgg.click(lambda: _vgg_status(), outputs=vgg_status_box)

            gr.Markdown("---")
            gr.Markdown(
                "**MS1MV2 (MS1M-ArcFace)** — 5.8M images across 85,742 identities at 112×112.  \n"
                "The gold standard for face recognition training (InsightFace dataset).  \n"
                "When present: switches to **SGD + MultiStepLR** (reference training protocol).  \n"
                "*(Large download ~35 GB — see button for Kaggle / manual instructions.)*"
            )
            ms1mv2_status_box = gr.Textbox(label="MS1MV2 Status",
                                           value=_ms1mv2_status(), lines=1, interactive=False)
            btn_ms1mv2 = gr.Button("Download MS1MV2 / Setup Instructions", variant="secondary")
            ms1mv2_log = gr.Textbox(label="MS1MV2 Log", lines=7, interactive=False)

            btn_ms1mv2.click(download_ms1mv2, inputs=None,
                             outputs=ms1mv2_log, show_progress="hidden")
            btn_ms1mv2.click(lambda: _ms1mv2_status(), outputs=ms1mv2_status_box)

        # ── TAB 2: SEARCH ─────────────────────────────────────────────────────
        with gr.Tab("Search"):
            gr.Markdown(
                "Upload a face — searches with **both models** side by side.  \n"
                "**Step 1:** Click *Build Index* once (~2–4 min) to index all images.  \n"
                "**Step 2:** Upload a photo and click *Search*."
            )
            srch_img  = gr.Image(label="Upload Face", type="numpy")
            srch_mode = gr.Radio(
                ["CNN + Features", "CNN Only"],
                value="CNN + Features", label="Search Mode")
            with gr.Row():
                btn_search    = gr.Button("Search",      variant="primary")
                btn_bld_idx   = gr.Button("Build Index", variant="secondary")
                btn_bld_log_r = gr.Button("Refresh Log")
            srch_diag   = gr.Textbox(label="Feature Analysis & Log", lines=14, interactive=False)
            bld_log_box = gr.Textbox(label="Index Build Log",         lines=4,  interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Best Model  (`app_best.pt`)")
                    gallery_best_s = gr.Gallery(label="Best model", columns=4, height=380)
                with gr.Column():
                    gr.Markdown("### Current Checkpoint  (`app_checkpoint.pt`)")
                    gallery_ckpt_s = gr.Gallery(label="Checkpoint", columns=4, height=380)

            gr.Markdown("---")
            gr.Markdown(
                "**Build Feature Index** — runs face detection on every image so "
                "skin/hair/age re-ranking has real data. Run once after building the search index."
            )
            btn_feat_s = gr.Button("Build Feature Index", variant="secondary")
            feat_log_s = gr.Textbox(label="Feature Index Log", lines=4, interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                "**CelebA** — 10,177 celebrity identities (202k images). "
                "Download to add them to the search index."
            )
            celeba_status_s = gr.Textbox(label="CelebA Status",
                                         value=_celeba_status(), lines=1, interactive=False)
            btn_celeba_s = gr.Button("Download CelebA", variant="secondary")
            celeba_log_s = gr.Textbox(label="CelebA Log", lines=3, interactive=False)

            btn_search.click(search_and_compare,
                             inputs=[srch_img, srch_mode],
                             outputs=[srch_diag, gallery_best_s, gallery_ckpt_s],
                             show_progress="hidden")
            btn_bld_idx.click(start_compare_build, outputs=bld_log_box)
            btn_bld_log_r.click(get_compare_log,   outputs=bld_log_box)
            btn_feat_s.click(build_feature_index,  outputs=feat_log_s, show_progress="hidden")
            btn_celeba_s.click(download_celeba,    outputs=celeba_log_s, show_progress="hidden")
            btn_celeba_s.click(lambda: _celeba_status(), outputs=celeba_status_s)

        # ── SCRAPE DATA tab removed from UI — code kept in celebrity_scraper.py ──
        # Use celebrity_scraper.py directly via CLI if scraping is needed again.
        # The scraped images in celebrity_data/ are still picked up by training.

        # ── FEEDBACK tab removed from UI — code kept below ────────────────────
        # feedback_pairs.csv is still loaded during training automatically.
        # Call load_feedback_pair / label_same / label_diff directly if needed.


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft())
