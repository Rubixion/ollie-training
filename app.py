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
import random
import threading
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torch.utils.data import DataLoader
import kagglehub

from lfw_pytorch import (
    SiameseNet, FacePairDataset,
    train_transform, test_transform,
    DEVICE, EMBEDDING_SIZE, IMAGE_SIZE,
    load_csv_pairs, generate_pairs_from_filesystem,
    get_names_from_csv,
)
from face_features import build_feature_cache, extract_face_features, FEAT_DIM, _ZERO_VEC
from celebrity_scraper import (
    scrape_all, get_scraped_pairs, count_images, clean_non_faces,
    ai_verify_all, CELEBRITIES, SCRAPE_ROOT,
)

APP_CHECKPOINT = "app_checkpoint.pt"
APP_BEST       = "app_best.pt"
FEEDBACK_CSV   = "feedback_pairs.csv"
EMBED_CACHE    = "embed_cache.npz"
FEAT_CACHE     = "feature_cache.pkl"
MARGIN         = 2.0   # max L2 dist on unit sphere = 2; keeps loss non-zero from init


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


def _get_dataset():
    global _dataset_path
    if _dataset_path is None:
        _dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    return _dataset_path


def _lfw_root():
    return os.path.join(_get_dataset(), "lfw-deepfunneled", "lfw-deepfunneled")


def _load_model(path=None):
    model = SiameseNet(embedding_size=EMBEDDING_SIZE, dropout=0.25, feat_dim=FEAT_DIM).to(DEVICE)
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
    """Return the in-memory feature cache, loading from disk if needed."""
    global _feat_cache
    if _feat_cache is None and os.path.exists(FEAT_CACHE):
        import pickle
        with open(FEAT_CACHE, 'rb') as f:
            _feat_cache = pickle.load(f)
    return _feat_cache or {}


# ── feature diagnostic ────────────────────────────────────────────────────────

def _describe_features(feats):
    """Human-readable breakdown of geometric features extracted from a face."""
    if np.all(feats == 0):
        return (
            "Face detection:  FAILED — MediaPipe found no face in the image.\n"
            "                 Using CNN visual embedding only.\n"
            "                 Skin color, hair color, face shape are NOT being compared.\n\n"
            "Tip: try a clearer, front-facing photo with good lighting."
        )

    lines = ["Face detection:  OK (MediaPipe landmarks found)\n"]

    # Skin tone — LAB L channel stored as L/255; cv2 L range is 0-255
    L = feats[20] * 255
    if L > 185:   skin_d = "very light"
    elif L > 155: skin_d = "light"
    elif L > 120: skin_d = "medium"
    elif L > 85:  skin_d = "medium-dark"
    else:         skin_d = "dark"
    b_val = feats[22] * 255 - 128  # b > 0 = yellow/warm, b < 0 = blue/cool
    warm = "warm" if b_val > 8 else ("cool" if b_val < -5 else "neutral")
    lines.append(f"Skin tone:       {skin_d}  (L={L:.0f}, {warm} undertone)")

    # Hair color — HSV stored as H/179, S/255, V/255
    H = feats[17] * 179; S = feats[18] * 255; V = feats[19] * 255
    if V < 40:
        hair_d = "black"
    elif S < 25:
        hair_d = "white / silver" if V > 180 else "grey"
    elif H < 15 or H > 165:
        hair_d = "red / auburn"
    elif H < 30:
        hair_d = "dark brown"
    elif H < 60:
        hair_d = "blonde"
    else:
        hair_d = "brown"
    lines.append(f"Hair color:      {hair_d}  (H={H:.0f}°, S={S:.0f}, V={V:.0f})")

    # Eye color — average of both iris HSV (feats[3:6] right, feats[6:9] left)
    iH = ((feats[3] + feats[6]) / 2) * 179
    iS = ((feats[4] + feats[7]) / 2) * 255
    iV = ((feats[5] + feats[8]) / 2) * 255
    if iS < 25 or iV < 40:
        eye_d = "dark brown / black"
    elif iH < 20 or iH > 150:
        eye_d = "brown"
    elif iH < 45:
        eye_d = "hazel / amber"
    elif iH < 90:
        eye_d = "green"
    else:
        eye_d = "blue"
    lines.append(f"Eye color:       {eye_d}  (H={iH:.0f}°, S={iS:.0f})")

    # Face shape — w/h ratio (feats[11])
    ratio = feats[11]
    shape_d = ("round / wide" if ratio > 0.90
               else "oval" if ratio > 0.75
               else "narrow / long")
    lines.append(f"Face shape:      {shape_d}  (w/h={ratio:.2f})")

    # Nose (feats[14]: nose_w / nose_h)
    nose = feats[14]
    nose_d = "broad" if nose > 1.4 else ("medium" if nose > 0.8 else "narrow")
    lines.append(f"Nose:            {nose_d}  (w/h ratio={nose:.2f})")

    # Jaw shape (feats[12] chin_w, feats[13] mid_jaw_w, both / face_w)
    chin = feats[12]; mid = feats[13]
    jaw_d = ("square" if chin > mid * 0.97
             else "tapered / V-shape" if chin < mid * 0.87
             else "slightly tapered")
    lines.append(f"Jaw shape:       {jaw_d}  (chin={chin:.2f}, mid-jaw={mid:.2f})")

    # Eye spacing (feats[0]: eye_dist / face_w)
    ed = feats[0]
    esp = "wide-set" if ed > 0.45 else ("normal" if ed > 0.32 else "close-set")
    lines.append(f"Eye spacing:     {esp}  ({ed:.3f})")

    # Lips (feats[15] lip_w / face_w, feats[16] lip_h / face_h)
    lip_w = feats[15]; lip_h = feats[16]
    lip_d = "full / wide" if lip_w > 0.50 else ("medium" if lip_w > 0.38 else "narrow")
    lines.append(f"Lips:            {lip_d}  (width={lip_w:.2f}, height={lip_h:.3f})")

    # Forehead (feats[23])
    fore = feats[23]
    fore_d = "high" if fore > 0.28 else ("medium" if fore > 0.18 else "low")
    lines.append(f"Forehead:        {fore_d}  ({fore:.3f})")

    lines.append("")
    lines.append("Active modules:  CNN backbone  +  24-dim geometric features")
    lines.append("                 (skin tone, hair, eyes, face shape, jaw,")
    lines.append("                  nose, lips, forehead all fused into embedding)")

    return "\n".join(lines)


# ── TRAINING TAB ──────────────────────────────────────────────────────────────

def _train_worker():
    try:
        dp = _get_dataset()

        def log(msg):
            with _lock:
                _log_lines.append(msg)

        log("Loading test pairs...")
        test_pairs  = load_csv_pairs(dp, "matchpairsDevTest.csv", "mismatchpairsDevTest.csv")
        test_people = get_names_from_csv(dp, "peopleDevTest.csv")

        log("Loading LFW train pairs...")
        official  = load_csv_pairs(dp, "matchpairsDevTrain.csv", "mismatchpairsDevTrain.csv")
        generated = generate_pairs_from_filesystem(dp, exclude_people=test_people, max_pos=5000)

        celeb_pairs = get_scraped_pairs(SCRAPE_ROOT)
        if celeb_pairs:
            log(f"Using {len(celeb_pairs)} scraped celebrity pairs.")

        feedback = []
        if os.path.exists(FEEDBACK_CSV):
            with open(FEEDBACK_CSV, newline='') as f:
                for row in csv.reader(f):
                    if len(row) == 3 and os.path.exists(row[0]) and os.path.exists(row[1]):
                        feedback.append((row[0], row[1], float(row[2])))
            if feedback:
                log(f"Loaded {len(feedback)} human feedback pairs.")

        train_pairs = official + generated + celeb_pairs + feedback
        random.shuffle(train_pairs)
        log(f"Train: {len(train_pairs)} pairs  |  Test: {len(test_pairs)} pairs\n")

        # ── build geometric feature cache ─────────────────────────────────────
        all_paths = list({p for pair in train_pairs + test_pairs for p in pair[:2]})
        log(f"Building feature cache for {len(all_paths)} images...")

        def feat_progress(done, total):
            if done % 500 == 0 or done == total:
                log(f"  Features: {done}/{total}")

        global _feat_cache, _embed_index
        _feat_cache = build_feature_cache(all_paths, FEAT_CACHE, progress_cb=feat_progress)
        log(f"Feature cache ready ({len(_feat_cache)} entries).\n")

        train_loader = DataLoader(
            FacePairDataset(train_pairs, train_transform, feature_cache=_feat_cache),
            batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(
            FacePairDataset(test_pairs, test_transform, feature_cache=_feat_cache),
            batch_size=32, shuffle=False, num_workers=0)

        model     = SiameseNet(embedding_size=EMBEDDING_SIZE, dropout=0.25, feat_dim=FEAT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
        start_epoch = 0
        best_acc    = 0.0
        threshold   = 0.5

        if os.path.exists(APP_CHECKPOINT):
            ckpt       = torch.load(APP_CHECKPOINT, map_location=DEVICE, weights_only=False)
            compatible = True
            try:
                model.load_state_dict(ckpt['model'], strict=True)
            except RuntimeError:
                try:
                    model.load_state_dict(ckpt['model'], strict=False)
                    log("Architecture changed — backbone loaded, new layers start fresh.")
                except RuntimeError:
                    compatible = False
                    log("Old checkpoint is incompatible with new architecture — starting fresh.")
            if compatible:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
                start_epoch = ckpt['epoch'] + 1
                best_acc    = ckpt['best_acc']
                threshold   = ckpt.get('threshold', 0.5)
                log(f"Resumed from epoch {ckpt['epoch']}  (best: {best_acc*100:.1f}%)\n")
            else:
                log("Starting fresh.\n")
        else:
            log("No checkpoint — starting fresh.\n")

        for epoch in range(start_epoch, 200):
            if _stop_event.is_set():
                log("Stopped by user.")
                break

            # ── train ────────────────────────────────────────────────────────
            model.train()
            tr_loss = tr_correct = tr_total = 0
            for img1, img2, labels, f1, f2 in train_loader:
                img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                labels     = labels.to(DEVICE)
                f1, f2     = f1.to(DEVICE), f2.to(DEVICE)
                e1   = model.get_embedding(img1, f1)
                e2   = model.get_embedding(img2, f2)
                dist = F.pairwise_distance(e1, e2)
                loss = contrastive_loss(dist, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss    += loss.item() * len(labels)
                tr_correct += ((dist < threshold) == labels.bool()).sum().item()
                tr_total   += len(labels)

            # ── eval ─────────────────────────────────────────────────────────
            model.eval()
            te_loss = te_total = 0
            all_dists, all_labels_np = [], []
            with torch.no_grad():
                for img1, img2, labels, f1, f2 in test_loader:
                    img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                    labels     = labels.to(DEVICE)
                    f1, f2     = f1.to(DEVICE), f2.to(DEVICE)
                    e1   = model.get_embedding(img1, f1)
                    e2   = model.get_embedding(img2, f2)
                    dist = F.pairwise_distance(e1, e2)
                    loss = contrastive_loss(dist, labels)
                    te_loss       += loss.item() * len(labels)
                    te_total      += len(labels)
                    all_dists.extend(dist.cpu().numpy())
                    all_labels_np.extend(labels.cpu().numpy())

            all_dists     = np.array(all_dists)
            all_labels_np = np.array(all_labels_np)
            best_thr, best_thr_acc = threshold, 0.0
            for thr in np.linspace(0.05, 1.95, 60):
                acc = np.mean((all_dists < thr) == all_labels_np.astype(bool))
                if acc > best_thr_acc:
                    best_thr_acc, best_thr = acc, thr
            threshold = best_thr

            tr_acc = tr_correct / tr_total
            te_acc = best_thr_acc
            scheduler.step(te_loss / te_total)

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(model.state_dict(), APP_BEST)
                if os.path.exists(EMBED_CACHE):
                    os.remove(EMBED_CACHE)
                _embed_index = None

            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc, 'threshold': threshold,
            }, APP_CHECKPOINT)

            lr = optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch:3d}  train {tr_acc*100:.1f}%  "
                f"test {te_acc*100:.1f}%  best {best_acc*100:.1f}%  "
                f"thr {threshold:.3f}  lr={lr:.2e}")

        log(f"\nDone. Best test accuracy: {best_acc*100:.1f}%")

    except Exception as exc:
        with _lock:
            _log_lines.append(f"ERROR: {exc}")
        raise


def start_training():
    global _train_thread
    if _train_thread and _train_thread.is_alive():
        return "Already training — check the log."
    _stop_event.clear()
    with _lock:
        _log_lines.clear()
    _train_thread = threading.Thread(target=_train_worker, daemon=True)
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

def _build_embed_index(status_cb=None):
    global _embed_index
    if _embed_index is not None:
        return _embed_index

    if os.path.exists(EMBED_CACHE):
        if status_cb:
            status_cb("Loading cached embedding index...")
        d = np.load(EMBED_CACHE, allow_pickle=True)
        feats = d['features'] if 'features' in d else np.zeros((len(d['names']), FEAT_DIM), dtype=np.float32)
        _embed_index = (d['names'].tolist(), d['paths'].tolist(), d['embeddings'], feats)
        if status_cb:
            status_cb(f"Index loaded: {len(_embed_index[0])} images.")
        return _embed_index

    model      = _load_model()
    feat_cache = _get_feat_cache()
    IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.webp'}

    all_names, all_paths = [], []

    # LFW images
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

    # Scraped celebrity images
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
    if status_cb:
        status_cb(f"Embedding {total} images (LFW + celebrities) — this takes ~1-3 min on GPU...")

    all_embs  = []
    all_feats = []
    for i in range(0, total, 64):
        batch_paths  = all_paths[i:i+64]
        imgs         = []
        batch_feats  = []
        for p in batch_paths:
            try:
                imgs.append(test_transform(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(test_transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))))
            # Use training cache — covers most images; zeros for the rest
            batch_feats.append(feat_cache.get(p, _ZERO_VEC))

        feat_arr = np.stack(batch_feats).astype(np.float32)
        imgs_t   = torch.stack(imgs).to(DEVICE)
        feats_t  = torch.tensor(feat_arr).to(DEVICE)
        with torch.no_grad():
            all_embs.append(model.get_embedding(imgs_t, feats_t).cpu().numpy())
        all_feats.append(feat_arr)

        if status_cb and (i // 64) % 10 == 0:
            status_cb(f"  Embedded {min(i + 64, total)}/{total} images...")

    embeddings   = np.concatenate(all_embs,  axis=0)
    features_arr = np.concatenate(all_feats, axis=0)
    np.savez(EMBED_CACHE,
             names=np.array(all_names, dtype=object),
             paths=np.array(all_paths, dtype=object),
             embeddings=embeddings,
             features=features_arr)
    _embed_index = (all_names, all_paths, embeddings, features_arr)
    if status_cb:
        status_cb(f"Index built: {total} images. Saved to cache.")
    return _embed_index


def find_dataset_matches(image):
    """
    Upload a face → embed it → find the closest matching images in the dataset.
    Generator: yields progress updates so the UI stays live during index build.
    """
    if image is None:
        yield "Upload a face photo first.", []
        return

    if not os.path.exists(APP_BEST) and not os.path.exists(APP_CHECKPOINT):
        yield "No trained model yet — click Start Training first.", []
        return

    yield "Extracting face features...", []

    model    = _load_model()
    img_pil  = Image.fromarray(image).convert('RGB')
    img_t    = test_transform(img_pil).unsqueeze(0).to(DEVICE)
    q_feats  = extract_face_features(img_pil)
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

    names, paths, embeddings, index_features = _build_embed_index(status_cb=index_progress)

    # Show index progress in the textbox
    yield "\n".join(log_buf), []

    if len(names) == 0:
        yield "\n".join(log_buf) + "\n\nNo images in embed index.", []
        return

    dists    = np.sqrt(((embeddings - q_emb) ** 2).sum(axis=1))
    top200   = np.argsort(dists)[:200]

    # ── feature re-ranking ────────────────────────────────────────────────────
    # Re-score top-200 by embedding + explicit demographic feature penalties.
    # index_features[i] is guaranteed to exist for every index entry.
    q_has_feats = not np.all(q_feats == 0)

    scored = []
    for i in top200:
        embed_dist = float(dists[i])

        if q_has_feats:
            c_feats = index_features[i]
            c_has   = not np.all(c_feats == 0)
            if c_has:
                # Skin tone L (feats[20]) — most important, weight 5×
                skin_diff  = abs(float(q_feats[20]) - float(c_feats[20]))
                # Hair hue circular distance, weight 2×
                hh         = abs(float(q_feats[17]) - float(c_feats[17]))
                hair_diff  = min(hh, 1.0 - hh)
                # Face shape w/h, weight 1×
                shape_diff = abs(float(q_feats[11]) - float(c_feats[11]))
                penalty    = 5.0 * skin_diff + 2.0 * hair_diff + shape_diff
            else:
                penalty = 0.30   # no landmarks on candidate — deprioritise
        else:
            penalty = 0.0        # no query features — pure CNN

        scored.append((embed_dist + penalty, i, embed_dist))

    scored.sort(key=lambda x: x[0])

    reranked = q_has_feats
    if reranked:
        log_buf.append("\nFeature re-ranking: ON  "
                       "(skin tone ×5, hair hue ×1.5, face shape ×1)")
    else:
        log_buf.append("\nFeature re-ranking: OFF  (no face landmarks on query)")

    gallery_items = []
    seen_names    = {}

    for combined, i, embed_dist in scored:
        display = names[i].replace('_', ' ')
        sim_pct = max(0.0, (1.0 - embed_dist / MARGIN)) * 100

        seen_names[display] = seen_names.get(display, 0) + 1
        if seen_names[display] > 2:
            continue

        try:
            img_match = _pil_square(Image.open(paths[i]).convert('RGB'), 160)
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


# ── GRADIO UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Face Verification") as app:
    gr.Markdown("# Face Verification — Siamese Network")

    with gr.Tabs():

        # ── TAB 1: TRAIN ──────────────────────────────────────────────────────
        with gr.Tab("Train"):
            gr.Markdown(
                "Trains with **contrastive loss** + **geometric face features** "
                "(inter-eye distance, iris colour & texture, eye aspect ratio).  \n"
                "Uses LFW + scraped celebrity data + your feedback labels."
            )
            with gr.Row():
                btn_start   = gr.Button("Start Training", variant="primary")
                btn_stop    = gr.Button("Stop Training",  variant="stop")
                btn_refresh = gr.Button("Refresh Log")
            status_box = gr.Textbox(label="Status", lines=1, interactive=False)
            log_box    = gr.Textbox(label="Training Log", lines=22, interactive=False)

            btn_start.click(start_training, outputs=status_box)
            btn_stop.click(stop_training,   outputs=status_box)
            btn_refresh.click(get_log,      outputs=log_box)

        # ── TAB 2: SCRAPE DATA ────────────────────────────────────────────────
        with gr.Tab("Scrape Data"):
            gr.Markdown(
                "Download celebrity face images to supplement LFW training data.  \n"
                "Covers actors, musicians, athletes, K-pop, Bollywood, and more (~450 celebrities).  \n"
                "**Existing images are never deleted on restart** — scraping resumes where it left off."
            )
            scrape_status_box = gr.Textbox(label="Stored so far", lines=1, interactive=False)
            with gr.Row():
                n_slider   = gr.Slider(10, 50, value=25, step=5, label="Images per celebrity")
                btn_scrape = gr.Button("Start Scraping", variant="primary")
                btn_scrape_r = gr.Button("Refresh Log")
            gr.Markdown(
                "**Optional: AI Verification** — paste a free [Groq API key](https://console.groq.com) "
                "to have a vision model delete wrong/non-celebrity images automatically."
            )
            with gr.Row():
                groq_key_box = gr.Textbox(
                    label="Groq API Key (optional, free at console.groq.com)",
                    placeholder="gsk_...", type="password", scale=3)
                btn_ai_verify = gr.Button("AI Verify Existing Images", scale=1)
            scrape_log_box = gr.Textbox(label="Scrape Log", lines=20, interactive=False)

            btn_scrape.click(start_scraping, inputs=[n_slider, groq_key_box], outputs=scrape_status_box)
            btn_scrape_r.click(get_scrape_log, outputs=scrape_log_box)
            btn_ai_verify.click(start_ai_verify, inputs=groq_key_box, outputs=scrape_status_box)
            app.load(get_scrape_status, outputs=scrape_status_box)

        # ── TAB 3: FIND IN DATASET ───────────────────────────────────────────
        with gr.Tab("Find in Dataset"):
            gr.Markdown(
                "Upload a face — the model finds the closest matching images "
                "from your **own dataset** (LFW + scraped celebrities).  \n"
                "The log shows what features were detected: skin tone, hair color, "
                "eye color, face shape, and whether they are active.  \n"
                "*(First click builds an embedding index — ~1–2 min the first time.)*"
            )
            img_in   = gr.Image(label="Upload Face", type="numpy")
            btn_id   = gr.Button("Find Matches", variant="primary")
            diag_box = gr.Textbox(label="Feature Diagnostic & Results Log",
                                  lines=20, interactive=False)
            gallery  = gr.Gallery(label="Closest matches in dataset",
                                  columns=4, height=420)

            btn_id.click(find_dataset_matches, inputs=img_in,
                         outputs=[diag_box, gallery],
                         show_progress="hidden")

        # ── TAB 4: FEEDBACK ───────────────────────────────────────────────────
        with gr.Tab("Give Feedback"):
            gr.Markdown(
                "Verify scraped celebrity images are actually correct.  \n"
                "Google Images sometimes downloads wrong photos — your labels fix that noise.  \n"
                "**Scrape Data first**, then label here. Saves to `feedback_pairs.csv`."
            )
            btn_load = gr.Button("Load Pair", variant="primary")
            with gr.Row():
                fb1 = gr.Image(label="Face A", interactive=False)
                fb2 = gr.Image(label="Face B", interactive=False)
            fb_count = gr.Textbox(label="", lines=1, interactive=False)
            with gr.Row():
                btn_same = gr.Button("Same Person",      variant="primary")
                btn_diff = gr.Button("Different People", variant="secondary")

            btn_load.click(load_feedback_pair, outputs=[fb1, fb2, fb_count])
            btn_same.click(label_same,         outputs=[fb1, fb2, fb_count])
            btn_diff.click(label_diff,         outputs=[fb1, fb2, fb_count])


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft())
