# Face Verification — Siamese Network

A from-scratch face verification system built in PyTorch with a Gradio web UI.
Upload any face photo to find who it looks most like inside your own dataset.

---

## What it does

- **Trains** a Siamese neural network on the LFW dataset + scraped celebrity images
- **Verifies** whether two face photos are the same person using contrastive loss
- **Identifies** the closest matching faces in your dataset (LFW + celebrities)
- **Matches** using both CNN visual features AND geometric face features (skin tone, hair, eye colour, face shape, jaw, nose, lips) so results respect demographics
- **Learns from feedback** — you can label pairs to improve training data quality

---

## Architecture

### Model: `SiameseNet` (lfw_pytorch.py)
- **Backbone**: Residual CNN — 3 → 64 → 128 → 256 → 512 channels, 6 ResBlocks, AdaptiveAvgPool → 512-d
- **Embedder**: Linear 512 → 256, BN1d, ReLU, Dropout, L2-normalised → **256-d embedding**
- **Geometric encoder**: MediaPipe 24-dim features → Linear 24 → 64 → 64
- **Head**: Combined (256+64) → 64 → 1, Sigmoid — used for training loss
- **Matching**: L2 distance on the combined embedding at inference
- **Loss**: Contrastive loss, margin=2.0 — pushes same-person pairs toward 0, different-person pairs apart

### Geometric features: `face_features.py` (24 dimensions)
| Feature | Description |
|---------|-------------|
| Eye distance | Inter-iris distance / face width |
| EAR ×2 | Eye aspect ratio (left + right) |
| Iris HSV ×2 | Hue/saturation/value of each iris |
| Iris variance ×2 | Texture variance of each iris |
| Face ratio | Width / height |
| Chin + mid-jaw width | Jaw shape ratios |
| Nose ratio | Width / height |
| Lip width + height | Normalised to face size |
| Hair HSV | Mean colour of region above forehead |
| Skin LAB | Mean colour of cheek patches |
| Forehead ratio | Eye-to-forehead height / face height |

---

## Setup

### Requirements
- Python 3.10+
- CUDA GPU recommended (NVIDIA — tested on GTX 1650)
- ~5 GB free disk space for the LFW dataset

### Install

```bash
# Clone the repo and create a virtual environment
git clone <your-repo-url>
cd "neural network learning"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install gradio kagglehub mediapipe==0.10.9 opencv-python pillow numpy ddgs groq
```

> **GPU note:** If `torch.cuda.is_available()` returns `False`, uninstall and reinstall PyTorch from the cu124 index above. CPU-only PyTorch will train ~10× slower.

> **MediaPipe note:** Must be version `0.10.9` — newer versions (0.10.14+) removed the `solutions` API that face mesh requires.

### Kaggle dataset (LFW)
The LFW dataset downloads automatically via `kagglehub` on first run. You need a Kaggle account and API key set up:

```bash
pip install kaggle
# Place your kaggle.json in C:\Users\<you>\.kaggle\kaggle.json
```

---

## Running

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`.

---

## Using the app

### Tab 1 — Train

Trains the Siamese network. Uses:
- Official LFW DevTrain pairs (CSV)
- Auto-generated pairs from every person in the LFW filesystem
- Any scraped celebrity images you've downloaded
- Any human feedback pairs you've labelled

Click **Start Training** — training runs in the background and logs every epoch:
```
Epoch   0  train 61.2%  test 63.8%  best 63.8%  thr 0.412  lr=1.00e-03
Epoch   1  train 64.5%  test 66.1%  best 66.1%  thr 0.447  lr=1.00e-03
```

Click **Refresh Log** to update. Click **Stop Training** to pause (resumes from checkpoint next time).

**With a GTX 1650:** ~3–6 min per epoch. The model saves `app_checkpoint.pt` every epoch and `app_best.pt` when a new best test accuracy is reached. **Never delete these files.**

### Tab 2 — Scrape Data

Downloads celebrity face images (~450 celebrities: actors, musicians, athletes, K-pop, Bollywood, sports) to `celebrity_data/`.

1. Set the slider (images per celebrity, default 25)
2. Optionally paste a free [Groq API key](https://console.groq.com) for AI verification — it uses `llama-4-scout` vision to delete wrong/non-celebrity images automatically
3. Click **Start Scraping** → watch the log
4. Scraping is resumable — existing images are never re-downloaded

After scraping, click **AI Verify Existing Images** at any time to re-check what you have.

### Tab 3 — Find in Dataset

Upload any face photo to find the closest matches in your dataset.

The **Feature Diagnostic** box shows exactly what was detected:
```
Face detection:  OK (MediaPipe landmarks found)

Skin tone:       light  (L=185, warm undertone)
Hair color:      dark brown  (H=24°, S=110, V=80)
Eye color:       hazel / amber  (H=30°)
Face shape:      oval  (w/h=0.82)
Nose:            medium  (w/h ratio=1.02)
Jaw shape:       slightly tapered  (chin=0.84, mid-jaw=0.79)
Eye spacing:     normal  (0.38)

Feature re-ranking: ON  (skin tone ×5, hair hue ×2, face shape ×1)
```

**If face detection FAILED:** MediaPipe couldn't find landmarks in the image. Only the CNN embedding is used, which may not respect skin tone or face shape. Try a clearer, front-facing photo with good lighting.

The gallery below shows the 12 closest matching images from your dataset. The first time you click this it builds an embedding index of all your images (~2 min) — after that it's instant.

### Tab 4 — Give Feedback

Shows random pairs from your scraped celebrity data. Click:
- **Same Person** — both images are the same celebrity (label=1)
- **Different People** — they are different people (label=0)

Labels are saved to `feedback_pairs.csv` and used in the next training run. This is especially useful for cleaning up noisy scrape results (wrong celebrities, group photos, etc.).

---

## How results improve over time

| What you do | Effect |
|---|---|
| Train more epochs | CNN embedding gets better at separating identities |
| Scrape more celebrities | More diverse training data |
| AI verify scraped images | Removes noisy/wrong labels that hurt training |
| Give feedback labels | Corrects specific bad pairs that confuse the model |
| Rebuild embed cache (delete `embed_cache.npz`) | Picks up improved embeddings after retraining |

The **Feature re-ranking** (skin tone, hair, face shape) is an explicit override that works regardless of training quality. It re-scores the top 200 CNN matches by demographic similarity, so even an undertrained model won't show wrong skin tones.

---

## Teaching Claude about this codebase with Graphify

This repo uses [graphify](https://github.com/safishamsi/graphifyy) to build a knowledge graph of the codebase. The graph lets Claude Code answer questions about architecture, data flow, and relationships without reading every file from scratch.

**Current graph:** 551 nodes · 852 edges · 47 communities — covers the full stack (Python ML backend + Next.js frontend). Last updated 2026-06-07.

### Build the graph (first time or after big changes)

In Claude Code, type:
```
/graphify
```

### Update after code changes (fast, no API cost)

```
/graphify . --update
```

Re-extracts only changed files via AST — free and takes a few seconds.

### Query the graph

```
/graphify query "how does skin tone affect matching"
/graphify query "what happens when training starts"
/graphify query "how do blog posts link to the match page"
/graphify path "FacePairDataset" "SiameseNet"
/graphify explain "contrastive loss"
```

Returns a scoped subgraph — much faster than reading files one by one.

### Community map (47 communities)

**Python ML backend**
| Community | What's in it |
|---|---|
| Training Pipeline | `lfw_pytorch.py` — FacePairDataset, training loop, contrastive loss |
| Siamese Network Architecture | `lfw_pytorch.py` — ResBlock, SiameseNet, get_embedding |
| FAISS Search & Embeddings | `app.py` — build_faiss, build_embed_index, find_matches |
| Face Feature Extraction | `face_features.py` — InsightFace GPU / MediaPipe CPU, 32-dim vector |
| Gradio App & Dataset | `app.py` — all 4 tabs, feedback loop, dataset management |
| Celebrity Scraper Pipeline | `celebrity_scraper.py` — DuckDuckGo scraper, Groq llama-4-scout verify |
| VGGFace2 Data Pipeline | `app.py` — kagglehub download, vgg_root management |
| LFW Evaluation | `lfw_train.py` — standalone evaluation script |
| Face Recognition Baseline | `face_recognition.py` — original from-scratch version |
| Neural Network Layers | `activation_function.py`, `conv_layer.py`, `network.py` — manual backprop |

**Next.js frontend (`ollie-frontend/`)**
| Community | What's in it |
|---|---|
| AI Explainer Page | `app/ai/page.tsx` — NeuralDeepViz, 8 sections (01–08 incl. blog links) |
| Blog Post Dynamic Route | `app/blog/[slug]/page.tsx` — SEO metadata, JSON-LD, CTAs to /match + /ai |
| Blog Infrastructure | `app/blog/layout.tsx`, `app/blog/page.tsx` — listing + category filter |
| Homepage & Navigation | `app/page.tsx`, `components/nav.tsx` |
| App Layout & Auth | `app/layout.tsx`, `components/auth-modal.tsx`, `auth-provider.tsx` |
| Auth UI & shadcn Components | `components/sign-in.tsx`, `components/ui/*` — shadcn primitives |
| WebGL Visual Effects | `components/effects/` — Three.js particles, entropy, face-feature-viz |
| API Routes | `app/api/search/route.ts`, `app/api/feedback/route.ts` → Gradio port 7860 |
| Design System | `DESIGN.md` — color tokens, Geist typography, mesh gradient, elevation |
| 8-Bit UI & Changelog | `components/8bit-team2.tsx`, `components/ui/8bit-*` |

### God nodes (most connected — change carefully)

| Node | Edges | Why it matters |
|---|---|---|
| `Nav()` | 19 | Used by every page |
| `Footer()` | 16 | Used by every page |
| `extract_face_features()` | 15 | Core of all search + training paths |
| `_train_worker()` | 14 | Orchestrates the full training loop |
| `SiameseNet` | 13 | The model architecture itself |
| `FacePairDataset` | 13 | Dataset used by all training code |

---

## File structure

```
neural network learning/
├── app.py                  # Gradio UI — all 4 tabs
├── lfw_pytorch.py          # SiameseNet, FacePairDataset, training loop
├── face_features.py        # MediaPipe 24-dim geometric feature extraction
├── celebrity_scraper.py    # DuckDuckGo scraper + Groq AI verification
│
├── # From-scratch implementations (educational — not used by app.py)
├── siamese_network.py      # Siamese network with manual backprop
├── face_recognition.py     # Face recognition with from-scratch layers
├── network.py / network_layer.py
├── activation_function.py  # Relu, Sigmoid
├── conv_layer.py           # Convolution layer
├── pool_layer.py           # MaxPool layer
├── flatten_layer.py
├── main.py / main_cnn.py / lfw_train.py
│
├── CLAUDE.md               # Instructions for Claude Code (never delete trained models!)
├── .gitignore
│
├── # Generated — not committed to git
├── app_checkpoint.pt       # Latest training checkpoint (NEVER DELETE)
├── app_best.pt             # Best model by test accuracy (NEVER DELETE)
├── feature_cache.pkl       # Cached MediaPipe features for all training images
├── embed_cache.npz         # Cached embeddings + features for all index images
├── feedback_pairs.csv      # Human-labelled pairs from the Feedback tab
├── celebrity_data/         # Scraped celebrity images (~450 celebrities)
└── graphify-out/           # Knowledge graph (rebuild with /graphify)
```

---

## Backing up your trained models

The `.pt` files are excluded from git (too large). Back them up manually:

```bash
copy app_best.pt       backups\app_best_epoch50.pt
copy app_checkpoint.pt backups\app_checkpoint_epoch50.pt
```

Or use Git LFS if you want to track them in git:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```
