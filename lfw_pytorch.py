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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 64   # 64×64 colour — fast on CPU, much better than 32×32 grayscale


# ── transforms ────────────────────────────────────────────────────────────────

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── dataset ───────────────────────────────────────────────────────────────────

class FacePairDataset(Dataset):
    """
    pairs: list of (path1, path2, label)
    Images are loaded from disk on demand — no RAM spike at startup.
    """
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self.transform(Image.open(p1).convert('RGB'))
        img2 = self.transform(Image.open(p2).convert('RGB'))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ── model ──────────────────────────────────────────────────────────────────────

def conv_block(in_c, out_c, dropout=0.15):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(dropout),
    )


class SiameseNet(nn.Module):
    def __init__(self, embedding_size=128, dropout=0.4):
        super().__init__()
        # 64 → 32 → 16 → 8, then global avg pool → 1×1
        self.backbone = nn.Sequential(
            conv_block(3,   32,  0.10),
            conv_block(32,  64,  0.15),
            conv_block(64, 128,  0.20),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_size),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )

    def get_embedding(self, x):
        return self.embedder(self.backbone(x))

    def forward(self, x1, x2):
        e1 = self.get_embedding(x1)
        e2 = self.get_embedding(x2)
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
    model     = SiameseNet(embedding_size=128, dropout=0.25).to(DEVICE)
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
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
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

    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
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
