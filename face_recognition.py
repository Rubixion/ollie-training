import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations
from siamese_network import SiameseNetwork, IMAGE_SIZE


def load_images(folder="images"):
    """
    Reads every image whose filename matches  <letters><digits>.<ext>.
    e.g.  a1.jpg  b2.png  person1.jpg
    Returns  { person_id: [(filename_stem, numpy_array), ...] }
    Arrays are grayscale, shape (32, 32), values in [0, 1].
    """
    people = {}
    for fname in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        m = re.match(r'^([a-zA-Z]+)(\d+)$', stem)
        if not m:
            continue
        person_id = m.group(1).lower()
        img = Image.open(os.path.join(folder, fname)).convert('L')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        people.setdefault(person_id, []).append((stem, arr))
    return people


def make_pairs(people):
    """
    Builds every positive pair (same person, label=1) and
    every negative pair (different people, label=0).
    """
    x1, x2, y, names = [], [], [], []
    ids = sorted(people.keys())

    for pid in ids:
        for (n1, i1), (n2, i2) in combinations(people[pid], 2):
            x1.append(i1); x2.append(i2); y.append(1.0)
            names.append((n1, n2))

    for pid1, pid2 in combinations(ids, 2):
        for n1, i1 in people[pid1]:
            for n2, i2 in people[pid2]:
                x1.append(i1); x2.append(i2); y.append(0.0)
                names.append((n1, n2))

    return (np.array(x1), np.array(x2),
            np.array(y, dtype=np.float32).reshape(-1, 1),
            names)


def show_loaded_images(people):
    ids = sorted(people.keys())
    cols = max(len(v) for v in people.values())
    rows = len(ids)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.5))
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle("Loaded face images", fontsize=14)
    for r, pid in enumerate(ids):
        for c, (name, img) in enumerate(people[pid]):
            axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[r, c].set_title(name)
            axes[r, c].axis('off')
        for c in range(len(people[pid]), cols):
            axes[r, c].axis('off')
    plt.tight_layout()
    plt.show()


def show_match_results(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(4, n * 2 + 0.5))
    axes = np.array(axes).reshape(n, 2)
    fig.suptitle("Best matches  (green = correct,  red = wrong)", fontsize=12)
    for i, (name, img, mname, mimg, score, ok) in enumerate(results):
        axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(name, fontsize=9)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mimg, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"-> {mname}  ({score:.2f})",
                              color="green" if ok else "red", fontsize=9)
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)

    # ── load images ──────────────────────────────────────────────────────────
    if not os.path.isdir("images"):
        os.makedirs("images")

    people = load_images("images")
    if not people:
        print("No images found in images/")
        print("Add files named:  a1.jpg  a2.jpg  b1.jpg  b2.jpg  c1.jpg ... etc.")
        print("One letter = one person.  Two images per person minimum.")
        return

    print(f"Found {len(people)} people: {sorted(people.keys())}")
    for pid, imgs in sorted(people.items()):
        print(f"  {pid}: {[n for n, _ in imgs]}")
    show_loaded_images(people)

    # ── build training pairs ──────────────────────────────────────────────────
    x1, x2, y, pair_names = make_pairs(people)
    pos = int(np.sum(y))
    print(f"\nPairs: {pos} positive (same person), {len(y) - pos} negative (different)")

    # ── train ─────────────────────────────────────────────────────────────────
    net = SiameseNetwork(embedding_size=32)
    print("\nTraining...\n")
    net.train(x1, x2, y, learning_rate=0.001, epochs=500)

    # ── matching test ─────────────────────────────────────────────────────────
    # For each image: which other image does the network think is the same person?
    print("\n--- Identity matching ---")
    all_names, all_imgs = [], []
    for pid in sorted(people.keys()):
        for name, img in people[pid]:
            all_names.append(name)
            all_imgs.append(img)

    results = []
    for i, (name, img) in enumerate(zip(all_names, all_imgs)):
        scores = []
        for j, other_img in enumerate(all_imgs):
            if i == j:
                scores.append(-1.0)
                continue
            pred, _, _ = net.forward(img[np.newaxis], other_img[np.newaxis])
            scores.append(float(pred[0, 0]))

        best = int(np.argmax(scores))
        match_name = all_names[best]
        score = scores[best]
        # "correct" = first letters match (same person group)
        correct = re.match(r'^([a-zA-Z]+)', name).group(1).lower() == \
                  re.match(r'^([a-zA-Z]+)', match_name).group(1).lower()
        status = "CORRECT" if correct else "WRONG"
        print(f"  {name:6s}  ->  {match_name:6s}  score={score:.3f}  {status}")
        results.append((name, img, match_name, all_imgs[best], score, correct))

    n_correct = sum(1 for *_, ok in results if ok)
    print(f"\nMatching accuracy: {n_correct}/{len(results)}  ({n_correct/len(results)*100:.0f}%)")
    show_match_results(results)


if __name__ == "__main__":
    main()
