"""
LFW face verification — full dataset Siamese network in PyTorch.

Training data comes from two sources combined:
  1. Official matchpairsDevTrain / mismatchpairsDevTrain CSVs
  2. Auto-generated pairs from every person in lfw-deepfunneled/
     (people who appear in the DevTest set are excluded to prevent leakage)

Test data: official matchpairsDevTest / mismatchpairsDevTest CSVs only.
"""

import os
import csv
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE     = 112   # MS1MV2 native alignment size (112×112 pre-cropped faces)
EMBEDDING_SIZE = 512   # exported so app.py stays in sync


# ── transforms ────────────────────────────────────────────────────────────────

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

test_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── margin cosine loss (CosFace) — for classification-based pre-training ──────

class MarginCosineProduct(nn.Module):
    """CosFace: Large Margin Cosine Loss.
    s=64 matches InsightFace training configs (sharper softmax, stronger gradients).
    m=0.40 from the original CosFace paper.
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.40):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, embeddings, labels):
        cosine  = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        one_hot = F.one_hot(labels.long(), num_classes=self.weight.size(0)).float()
        return self.s * (cosine - one_hot * self.m)


# ── dataset ───────────────────────────────────────────────────────────────────

class FacePairDataset(Dataset):
    """
    pairs: list of (path1, path2, label).
    feature_cache: optional dict {path -> np.ndarray (FEAT_DIM,)}.
      When provided, __getitem__ returns (img1, img2, label, feat1, feat2).
      When None, returns (img1, img2, label) — backward-compatible.
    """
    def __init__(self, pairs, transform, feature_cache=None):
        self.pairs         = pairs
        self.transform     = transform
        self.feature_cache = feature_cache
        # infer zero-vector dimension from cache; default to 12 for compat
        sample = next(iter(feature_cache.values()), None) if feature_cache else None
        self._zero = np.zeros(len(sample) if sample is not None else 12, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1    = self._open(p1)
        img2    = self._open(p2)
        label_t = torch.tensor(label, dtype=torch.float32)
        if self.feature_cache is not None:
            f1 = torch.tensor(self.feature_cache.get(p1, self._zero), dtype=torch.float32)
            f2 = torch.tensor(self.feature_cache.get(p2, self._zero), dtype=torch.float32)
            return img1, img2, label_t, f1, f2
        return img1, img2, label_t

    def _open(self, path):
        try:
            return self.transform(Image.open(path).convert('RGB'))
        except Exception:
            # corrupt / truncated file — return a black placeholder
            return self.transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)))


class MS1MV2Dataset(Dataset):
    """Single-image identity classification dataset used for CosFace training."""
    def __init__(self, samples, transform):
        self.samples   = samples   # list of (path, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = self.transform(Image.open(path).convert('RGB'))
        except Exception:
            img = self.transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)))
        return img, label


# ── model ──────────────────────────────────────────────────────────────────────

# ── SphereFace (sphere20) ──────────────────────────────────────────────────────

class _SFBlock(nn.Module):
    """sphere20 residual block — identity skip, PReLU, no BatchNorm."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return x + self.block(x)


def _sf_stage(in_ch, out_ch, n_blocks):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True),
              nn.PReLU(out_ch)]
    for _ in range(n_blocks):
        layers.append(_SFBlock(out_ch))
    return nn.Sequential(*layers)


class SphereFaceNet(nn.Module):
    """
    sphere20 backbone: 4 stride-2 stages + PReLU, preserves 7×7 spatial map,
    flattens to 25,088 before FC.  Matches the reference train.py architecture.
    Input: 112×112 pre-aligned faces.  Output: L2-normalised 512-d embedding.
    The optional `feats` argument is accepted but ignored — geometric features
    are used for search re-ranking outside the model.
    """
    def __init__(self, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.layer1 = _sf_stage(3,   64,  1)   # 112→56
        self.layer2 = _sf_stage(64,  128, 2)   # 56→28
        self.layer3 = _sf_stage(128, 256, 4)   # 28→14
        self.layer4 = _sf_stage(256, 512, 1)   # 14→7
        self.fc     = nn.Linear(512 * 7 * 7, embedding_size, bias=True)
        self.bn     = nn.BatchNorm1d(embedding_size)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_embedding(self, x, feats=None):   # feats ignored — pure CNN path
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.normalize(self.bn(self.fc(torch.flatten(x, 1))), p=2, dim=1)

    def forward(self, x):
        return self.get_embedding(x)


# ── legacy Siamese backbone (kept for reference) ───────────────────────────────

class _ResBlock(nn.Module):
    """Two 3×3 convs with a residual skip. stride=2 halves spatial dimensions."""
    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
        ) if (ch_in != ch_out or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, dropout=0.4, feat_dim=0):
        """
        Residual backbone (96×96 → 512-d) fused with optional geometric face
        features (hair colour, skin tone, jaw shape, iris colour, EAR, etc.)
        before the similarity head.
        """
        super().__init__()
        self.feat_dim = feat_dim

        # Stem + 3 residual stages: 96 → 48 → 24 → 12 → pool → 512-d
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _ResBlock(64,  128, stride=2),   # 48×48
            _ResBlock(128, 128),
            _ResBlock(128, 256, stride=2),   # 24×24
            _ResBlock(256, 256),
            _ResBlock(256, 512, stride=2),   # 12×12
            _ResBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_size),
        )

        if feat_dim > 0:
            self.feat_encoder = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
            )
            combined_dim = embedding_size + 64
        else:
            combined_dim = embedding_size

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def get_embedding(self, x, feats=None):
        cnn_emb = F.normalize(self.embedder(self.backbone(x)), p=2, dim=1)
        if feats is not None and self.feat_dim > 0:
            combined = torch.cat([cnn_emb, self.feat_encoder(feats)], dim=1)
            return F.normalize(combined, p=2, dim=1)
        return cnn_emb

    def forward(self, x1, x2, f1=None, f2=None):
        e1  = self.get_embedding(x1, f1)
        e2  = self.get_embedding(x2, f2)
        sim = self.head(torch.abs(e1 - e2)).squeeze(1)
        return sim, e1, e2


# ── data helpers ──────────────────────────────────────────────────────────────

def find_file(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def lfw_img_path(dataset_path, name, imagenum):
    return os.path.join(
        dataset_path, "lfw-deepfunneled", "lfw-deepfunneled",
        name, f"{name}_{int(imagenum):04d}.jpg"
    )


def load_csv_pairs(dataset_path, match_csv, mismatch_csv):
    """Load pairs from the official DevTrain or DevTest CSV files."""
    pairs = []

    match_path = find_file(dataset_path, match_csv)
    with open(match_path, newline='') as f:
        for row in csv.DictReader(f):
            p1 = lfw_img_path(dataset_path, row['name'], row['imagenum1'])
            p2 = lfw_img_path(dataset_path, row['name'], row['imagenum2'])
            if os.path.exists(p1) and os.path.exists(p2):
                pairs.append((p1, p2, 1.0))

    mismatch_path = find_file(dataset_path, mismatch_csv)
    with open(mismatch_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header (has duplicate 'name' columns)
        for cols in reader:
            p1 = lfw_img_path(dataset_path, cols[0], cols[1])
            p2 = lfw_img_path(dataset_path, cols[2], cols[3])
            if os.path.exists(p1) and os.path.exists(p2):
                pairs.append((p1, p2, 0.0))

    return pairs


def load_pairs_csv(dataset_path, exclude_people=None):
    """Load the full 6,000-pair LFW benchmark (pairs.csv), excluding test-set people."""
    path = find_file(dataset_path, "pairs.csv")
    if not path or not os.path.exists(path):
        return []
    exclude = set(exclude_people or [])
    pairs = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            non_empty = [r.strip() for r in row if r.strip()]
            if len(non_empty) == 3:          # matched: name, num1, num2
                name, n1, n2 = non_empty
                if name in exclude:
                    continue
                p1 = lfw_img_path(dataset_path, name, n1)
                p2 = lfw_img_path(dataset_path, name, n2)
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 1.0))
            elif len(non_empty) == 4:        # mismatched: name1, num1, name2, num2
                name1, n1, name2, n2 = non_empty
                if name1 in exclude or name2 in exclude:
                    continue
                p1 = lfw_img_path(dataset_path, name1, n1)
                p2 = lfw_img_path(dataset_path, name2, n2)
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 0.0))
    return pairs


def get_names_from_csv(dataset_path, csv_name):
    path = find_file(dataset_path, csv_name)
    if path is None:
        return set()
    with open(path, newline='') as f:
        return {row['name'] for row in csv.DictReader(f)}


def generate_pairs_from_filesystem(dataset_path, exclude_people, max_pos=3000):
    """
    Walk every person folder in lfw-deepfunneled.
    - Positive pairs: all image combos for the same person (capped per person)
    - Negative pairs: random cross-person pairs, equal in number to positives

    People in exclude_people are skipped (test-set leakage prevention).
    """
    lfw_root = os.path.join(dataset_path, "lfw-deepfunneled", "lfw-deepfunneled")

    # Build { person_name: [image_path, ...] } for everyone with ≥2 images
    person_images = {}
    for name in sorted(os.listdir(lfw_root)):
        if name in exclude_people:
            continue
        folder = os.path.join(lfw_root, name)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder) if f.endswith('.jpg')
        )
        if len(imgs) >= 2:
            person_images[name] = imgs

    print(f"  Found {len(person_images)} people with ≥2 images (test people excluded)")

    # Positive pairs — cap per person so no single celebrity dominates
    all_positive = []
    for name, imgs in person_images.items():
        combos = list(itertools.combinations(imgs, 2))
        if len(combos) > 10:          # at most 10 pairs per person
            combos = random.sample(combos, 10)
        all_positive.extend((p1, p2, 1.0) for p1, p2 in combos)

    random.shuffle(all_positive)
    all_positive = all_positive[:max_pos]

    # Negative pairs — random cross-person sampling
    all_imgs_flat = [(name, p) for name, imgs in person_images.items() for p in imgs]
    negative = []
    attempts = 0
    while len(negative) < len(all_positive) and attempts < len(all_positive) * 20:
        (n1, p1), (n2, p2) = random.sample(all_imgs_flat, 2)
        if n1 != n2:
            negative.append((p1, p2, 0.0))
        attempts += 1

    return all_positive + negative


def generate_pairs_from_flat_dir(root, exclude_people, max_pos=50000):
    """
    Like generate_pairs_from_filesystem but for datasets where identity folders
    sit directly under `root` (e.g. VGGFace2: root/n000001/, root/n000002/, ...).
    Also accepts .png and .jpeg in addition to .jpg.
    """
    person_images = {}
    for name in sorted(os.listdir(root)):
        if name in exclude_people:
            continue
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if len(imgs) >= 2:
            person_images[name] = imgs

    print(f"  Found {len(person_images)} identities with ≥2 images (test people excluded)")

    all_positive = []
    for name, imgs in person_images.items():
        combos = list(itertools.combinations(imgs, 2))
        if len(combos) > 10:
            combos = random.sample(combos, 10)
        all_positive.extend((p1, p2, 1.0) for p1, p2 in combos)

    random.shuffle(all_positive)
    all_positive = all_positive[:max_pos]

    all_imgs_flat = [(name, p) for name, imgs in person_images.items() for p in imgs]
    negative = []
    attempts = 0
    while len(negative) < len(all_positive) and attempts < len(all_positive) * 20:
        (n1, p1), (n2, p2) = random.sample(all_imgs_flat, 2)
        if n1 != n2:
            negative.append((p1, p2, 0.0))
        attempts += 1

    return all_positive + negative


def scan_ms1mv2(root, exclude_people=None):
    """
    Pre-scan the MS1MV2 directory once.  Returns {identity: [image_paths]}.
    Call once before the epoch loop; pass the result to sample_ms1mv2_epoch_pairs.
    """
    if exclude_people is None:
        exclude_people = set()
    person_images = {}
    for name in sorted(os.listdir(root)):
        if name in exclude_people:
            continue
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        imgs = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(imgs) >= 2:
            person_images[name] = imgs
    return person_images


def sample_ms1mv2_epoch_pairs(person_images, pairs_per_identity=5):
    """
    Sample a fresh set of pairs from the pre-scanned MS1MV2 dict for one epoch.
    Every identity is included every epoch; pairs differ each call so the model
    sees new combinations over time.  Returns ~2 * identities * pairs_per_identity pairs.
    """
    names = list(person_images.keys())
    positive = []
    for name, imgs in person_images.items():
        k = min(pairs_per_identity, len(imgs) // 2)
        if k == 0:
            continue
        pool = random.sample(imgs, min(k * 2, len(imgs)))
        for i in range(0, len(pool) - 1, 2):
            positive.append((pool[i], pool[i + 1], 1.0))

    n_neg = len(positive)
    negative = []
    attempts = 0
    while len(negative) < n_neg and attempts < n_neg * 10:
        n1, n2 = random.sample(names, 2)
        negative.append((
            random.choice(person_images[n1]),
            random.choice(person_images[n2]),
            0.0,
        ))
        attempts += 1

    pairs = positive + negative
    random.shuffle(pairs)
    return pairs


# ── 10-fold cross-validation evaluation (reference protocol) ──────────────────

def k_fold_eval(all_dists, all_labels, n_folds=10):
    """
    10-fold cross-validation on LFW pairs — matches the reference evaluate.py.
    Uses 9 folds to find the best threshold, evaluates on the remaining fold,
    repeats 10 times. Returns (mean_acc, std_acc, mean_threshold).
    """
    n          = len(all_dists)
    fold_size  = n // n_folds
    thresholds = np.linspace(0.05, 1.95, 200)
    accs, thrs = [], []

    for fold in range(n_folds):
        test_idx  = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_idx = [i for i in range(n) if i not in test_idx]

        # best threshold on the 9 train folds
        best_thr, best_acc = 0.5, 0.0
        for thr in thresholds:
            acc = np.mean(
                (all_dists[train_idx] < thr) == all_labels[train_idx].astype(bool))
            if acc > best_acc:
                best_acc, best_thr = acc, thr

        # accuracy on the held-out fold
        acc = np.mean(
            (all_dists[test_idx] < best_thr) == all_labels[test_idx].astype(bool))
        accs.append(acc)
        thrs.append(best_thr)

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


# ── training / evaluation ─────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, training):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)

        with torch.set_grad_enabled(training):
            preds, _, _ = model(img1, img2)
            loss = criterion(preds, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += ((preds >= 0.5) == labels.bool()).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print(f"Device: {DEVICE}\n")

    dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    print(f"Dataset: {dataset_path}\n")

    # ── test set (fixed — never used for training) ────────────────────────────
    print("Loading test pairs from DevTest CSVs...")
    test_pairs = load_csv_pairs(dataset_path,
                                "matchpairsDevTest.csv",
                                "mismatchpairsDevTest.csv")
    pos_te = sum(1 for *_, l in test_pairs if l == 1.0)
    print(f"  {pos_te} positive + {len(test_pairs)-pos_te} negative = {len(test_pairs)} test pairs")

    # People who appear in the test set — exclude from training pair generation
    test_people = get_names_from_csv(dataset_path, "peopleDevTest.csv")
    print(f"  Protecting {len(test_people)} test-set people from training data\n")

    # ── training set ──────────────────────────────────────────────────────────
    print("Loading official DevTrain pairs...")
    official = load_csv_pairs(dataset_path,
                              "matchpairsDevTrain.csv",
                              "mismatchpairsDevTrain.csv")
    pos_off = sum(1 for *_, l in official if l == 1.0)
    print(f"  {pos_off} positive + {len(official)-pos_off} negative = {len(official)} pairs")

    print("Generating additional pairs from full filesystem...")
    generated = generate_pairs_from_filesystem(dataset_path,
                                               exclude_people=test_people,
                                               max_pos=5000)
    pos_gen = sum(1 for *_, l in generated if l == 1.0)
    print(f"  {pos_gen} positive + {len(generated)-pos_gen} negative = {len(generated)} pairs")

    train_pairs = official + generated
    random.shuffle(train_pairs)
    pos_tr = sum(1 for *_, l in train_pairs if l == 1.0)
    print(f"\nTotal training: {pos_tr} pos + {len(train_pairs)-pos_tr} neg = {len(train_pairs)} pairs")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(FacePairDataset(train_pairs, train_transform),
                              batch_size=32, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(FacePairDataset(test_pairs,  test_transform),
                              batch_size=32, shuffle=False, num_workers=0)

    # ── model setup ───────────────────────────────────────────────────────────
    model     = SiameseNet(embedding_size=EMBEDDING_SIZE, dropout=0.25).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── resume from checkpoint if one exists ─────────────────────────────────
    CHECKPOINT = "checkpoint.pt"
    start_epoch   = 0
    best_test_acc = 0.0
    train_accs, test_accs, epoch_log = [], [], []

    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_test_acc = ckpt['best_test_acc']
        train_accs    = ckpt['train_accs']
        test_accs     = ckpt['test_accs']
        epoch_log     = ckpt['epoch_log']
        print(f"Resumed from epoch {ckpt['epoch']}  (best test acc so far: {best_test_acc*100:.1f}%)\n")
    else:
        print("No checkpoint found — starting fresh\n")

    # ── training loop ─────────────────────────────────────────────────────────
    print("Training...\n")

    for epoch in range(start_epoch, 150):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        te_loss, te_acc = run_epoch(model, test_loader,  criterion, optimizer, training=False)
        scheduler.step(te_loss)

        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        epoch_log.append(epoch)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(), "best_model.pt")

        # save full checkpoint after every epoch
        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_test_acc': best_test_acc,
            'train_accs':    train_accs,
            'test_accs':     test_accs,
            'epoch_log':     epoch_log,
        }, CHECKPOINT)

        if epoch % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}  "
                  f"train {tr_acc*100:.1f}% (loss {tr_loss:.4f})  |  "
                  f"test {te_acc*100:.1f}% (loss {te_loss:.4f})  lr={lr:.6f}")

    print(f"\nBest test accuracy: {best_test_acc*100:.1f}%")

    # ── training curve ────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_log, [a*100 for a in train_accs], label='Train')
    plt.plot(epoch_log, [a*100 for a in test_accs],  label='Test')
    plt.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='random chance')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('LFW Face Verification — PyTorch Siamese Network')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # ── sample predictions (scrollable) ──────────────────────────────────────────
    from matplotlib.widgets import Button as MplButton

    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE, weights_only=False))
    model.eval()

    # pre-compute all test predictions so page flipping is instant
    all_img1, all_img2, all_labels, all_scores = [], [], [], []
    with torch.no_grad():
        for img1, img2, labels in DataLoader(
                FacePairDataset(test_pairs, test_transform), batch_size=32):
            preds, _, _ = model(img1.to(DEVICE), img2.to(DEVICE))
            all_img1.extend(img1)
            all_img2.extend(img2)
            all_labels.extend(labels.numpy())
            all_scores.extend(preds.cpu().numpy())

    n_total = len(all_scores)
    PAGE    = 8
    n_pages = (n_total + PAGE - 1) // PAGE
    state   = {'page': 0}

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def to_display(t):
        return np.clip(t.permute(1, 2, 0).numpy() * std + mean, 0, 1)

    fig, axes = plt.subplots(PAGE, 2, figsize=(5, PAGE * 2 + 1))
    axes = np.array(axes).reshape(PAGE, 2)
    plt.subplots_adjust(bottom=0.08)

    ax_prev  = fig.add_axes([0.20, 0.01, 0.15, 0.04])
    ax_next  = fig.add_axes([0.65, 0.01, 0.15, 0.04])
    btn_prev = MplButton(ax_prev, '← Prev')
    btn_next = MplButton(ax_next, 'Next →')

    def draw_page(page):
        start = page * PAGE
        for i, row in enumerate(axes):
            for ax in row:
                ax.clear(); ax.axis('off')
            idx = start + i
            if idx >= n_total:
                continue
            score     = all_scores[idx]
            pred_same = score >= 0.5
            true_same = all_labels[idx] == 1.0
            correct   = pred_same == true_same
            row[0].imshow(to_display(all_img1[idx]))
            row[1].imshow(to_display(all_img2[idx]))
            row[1].set_title(
                f"pred:{'same' if pred_same else 'diff'}  "
                f"true:{'same' if true_same else 'diff'}\n({score:.2f})",
                color='green' if correct else 'red', fontsize=8)
        fig.suptitle(
            f"Test samples — page {page+1}/{n_pages}  (green=correct  red=wrong)",
            fontsize=11)
        fig.canvas.draw_idle()

    def on_prev(_):
        if state['page'] > 0:
            state['page'] -= 1
            draw_page(state['page'])

    def on_next(_):
        if state['page'] < n_pages - 1:
            state['page'] += 1
            draw_page(state['page'])

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    draw_page(0)
    plt.show()


if __name__ == "__main__":
    main()
