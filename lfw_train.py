import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from PIL import Image
from siamese_network import SiameseNetwork, IMAGE_SIZE


# ── dataset helpers ───────────────────────────────────────────────────────────

def download_dataset():
    print("Downloading LFW dataset (cached after first run)...")
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    print(f"Dataset path: {path}\n")
    return path


def find_file(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def load_lfw_image(dataset_path, name, imagenum):
    img_dir = os.path.join(dataset_path, "lfw-deepfunneled", "lfw-deepfunneled", name)
    img_path = os.path.join(img_dir, f"{name}_{int(imagenum):04d}.jpg")
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert('L')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def load_pairs(dataset_path, match_csv_name, mismatch_csv_name, max_each=None):
    """
    match CSV columns:    name, imagenum1, imagenum2
    mismatch CSV columns: name, imagenum1, name, imagenum2  (duplicate 'name' headers)
    Returns x1, x2, y arrays.
    """
    match_path = find_file(dataset_path, match_csv_name)
    mismatch_path = find_file(dataset_path, mismatch_csv_name)

    if match_path is None or mismatch_path is None:
        raise FileNotFoundError(
            f"Could not find {match_csv_name} or {mismatch_csv_name} inside {dataset_path}\n"
            f"Files found: {[f for _, _, fs in os.walk(dataset_path) for f in fs if f.endswith('.csv')]}"
        )

    x1, x2, y = [], [], []

    # positive pairs (same person)
    with open(match_path, newline='') as f:
        for row in csv.DictReader(f):
            i1 = load_lfw_image(dataset_path, row['name'], row['imagenum1'])
            i2 = load_lfw_image(dataset_path, row['name'], row['imagenum2'])
            if i1 is not None and i2 is not None:
                x1.append(i1); x2.append(i2); y.append(1.0)
            if max_each and len(x1) >= max_each:
                break

    # negative pairs — mismatch CSV has two columns both named 'name',
    # so DictReader can't distinguish them; read by column index instead
    target = len(x1)
    neg = 0
    with open(mismatch_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for cols in reader:
            # cols: [name1, imagenum1, name2, imagenum2]
            i1 = load_lfw_image(dataset_path, cols[0], cols[1])
            i2 = load_lfw_image(dataset_path, cols[2], cols[3])
            if i1 is not None and i2 is not None:
                x1.append(i1); x2.append(i2); y.append(0.0)
                neg += 1
            if neg >= target:
                break

    return (np.array(x1), np.array(x2),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ── training helpers ──────────────────────────────────────────────────────────

def train_epoch(net, x1, x2, y, learning_rate, batch_size=32):
    N = len(y)
    idx = np.random.permutation(N)
    x1, x2, y = x1[idx], x2[idx], y[idx]

    total_loss = 0.0
    n_batches = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx1, bx2, by = x1[start:end], x2[start:end], y[start:end]
        B = len(by)

        embeddings = net.backbone.forward(np.concatenate([bx1, bx2]))
        emb1, emb2 = embeddings[:B], embeddings[B:]
        diff = np.abs(emb1 - emb2)
        pred = net.head.forward(diff)

        eps = 1e-8
        loss = -np.mean(by * np.log(pred + eps) + (1 - by) * np.log(1 - pred + eps))
        total_loss += loss

        d_pred = -(by / (pred + eps) - (1 - by) / (1 - pred + eps)) / B
        d_diff = net.head.backward(d_pred, learning_rate)

        sign = np.sign(emb1 - emb2)
        d_emb = np.concatenate([d_diff * sign, d_diff * -sign])

        grad = d_emb
        for layer in reversed(net.backbone.layers):
            grad = layer.backward(grad, learning_rate)

        n_batches += 1

    return total_loss / n_batches


def evaluate(net, x1, x2, y, batch_size=32):
    N = len(y)
    preds = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx1, bx2 = x1[start:end], x2[start:end]
        B = len(bx1)
        emb = net.backbone.forward(np.concatenate([bx1, bx2]))
        pred = net.head.forward(np.abs(emb[:B] - emb[B:]))
        preds.append(pred)
    preds = np.concatenate(preds)
    return float(np.mean((preds >= 0.5) == y))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    dataset_path = download_dataset()

    # Load data
    # max_each limits pairs per class — lower = faster training, raise for better accuracy
    print("Loading training pairs...")
    x1_tr, x2_tr, y_tr = load_pairs(
        dataset_path,
        "matchpairsDevTrain.csv", "mismatchpairsDevTrain.csv",
        max_each=300
    )
    print(f"  {int(np.sum(y_tr))} positive  {int(np.sum(y_tr == 0))} negative  ({len(y_tr)} total)")

    print("Loading test pairs...")
    x1_te, x2_te, y_te = load_pairs(
        dataset_path,
        "matchpairsDevTest.csv", "mismatchpairsDevTest.csv",
        max_each=150
    )
    print(f"  {int(np.sum(y_te))} positive  {int(np.sum(y_te == 0))} negative  ({len(y_te)} total)")

    # Train
    net = SiameseNetwork(embedding_size=32)
    print("\nTraining  (this will take a few minutes — pure NumPy is slow)\n")

    train_accs, test_accs, epoch_log = [], [], []
    epochs = 50

    for epoch in range(epochs):
        loss = train_epoch(net, x1_tr, x2_tr, y_tr, learning_rate=0.001)
        if epoch % 5 == 0:
            tr_acc = evaluate(net, x1_tr, x2_tr, y_tr)
            te_acc = evaluate(net, x1_te, x2_te, y_te)
            train_accs.append(tr_acc); test_accs.append(te_acc); epoch_log.append(epoch)
            print(f"Epoch {epoch:3d}  Loss: {loss:.4f}  "
                  f"Train: {tr_acc*100:.1f}%  Test: {te_acc*100:.1f}%")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_log, [a * 100 for a in train_accs], label='Train')
    plt.plot(epoch_log, [a * 100 for a in test_accs], label='Test')
    plt.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='random chance')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('LFW Face Verification')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Show 8 example test predictions
    n_show = 8
    fig, axes = plt.subplots(n_show, 2, figsize=(4, n_show * 2 + 0.5))
    axes = np.array(axes).reshape(n_show, 2)
    fig.suptitle("Test samples  (green=correct  red=wrong)", fontsize=11)
    for i in range(n_show):
        pred, _, _ = net.forward(x1_te[i:i+1], x2_te[i:i+1])
        score = float(pred[0, 0])
        predicted_same = score >= 0.5
        actual_same = y_te[i, 0] == 1.0
        correct = predicted_same == actual_same
        pred_label = "same" if predicted_same else "diff"
        true_label = "same" if actual_same else "diff"

        axes[i, 0].imshow(x1_te[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(x2_te[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"pred:{pred_label}  true:{true_label}\n({score:.2f})",
            color="green" if correct else "red", fontsize=8
        )
        axes[i, 1].axis('off')
    plt.tight_layout(); plt.show()

    final_test_acc = evaluate(net, x1_te, x2_te, y_te)
    print(f"\nFinal test accuracy: {final_test_acc*100:.1f}%")


if __name__ == "__main__":
    main()
