# Graph Report - .  (2026-06-05)

## Corpus Check
- Large corpus: 4454 files · ~40,790,474 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 148 nodes · 281 edges · 13 communities (10 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Gradio UI App Frontend|Gradio UI App Frontend]]
- [[_COMMUNITY_Siamese Training Pipeline|Siamese Training Pipeline]]
- [[_COMMUNITY_Celebrity Scraping & Groq|Celebrity Scraping & Groq]]
- [[_COMMUNITY_Scratch Activation Functions|Scratch Activation Functions]]
- [[_COMMUNITY_Face Feature Extraction|Face Feature Extraction]]
- [[_COMMUNITY_LFW Dataset Training|LFW Dataset Training]]
- [[_COMMUNITY_Scratch NN Layers|Scratch NN Layers]]
- [[_COMMUNITY_Face Recognition (Legacy)|Face Recognition (Legacy)]]
- [[_COMMUNITY_CNN Pooling Layer|CNN Pooling Layer]]
- [[_COMMUNITY_Convolutional Layer|Convolutional Layer]]
- [[_COMMUNITY_ResNet Backbone|ResNet Backbone]]
- [[_COMMUNITY_Flatten Layer & Scaffold|Flatten Layer & Scaffold]]
- [[_COMMUNITY_Scratch Network Core|Scratch Network Core]]

## God Nodes (most connected - your core abstractions)
1. `SiameseNetwork` - 16 edges
2. `Network` - 12 edges
3. `FacePairDataset` - 11 edges
4. `Layer` - 11 edges
5. `Relu` - 10 edges
6. `Sigmoid` - 10 edges
7. `_train_worker()` - 10 edges
8. `ConvLayer` - 10 edges
9. `SiameseNet` - 9 edges
10. `main()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `SiameseNetwork` --uses--> `Relu`  [INFERRED]
  siamese_network.py → activation_function.py
- `SiameseNetwork` --uses--> `Sigmoid`  [INFERRED]
  siamese_network.py → activation_function.py
- `float` --uses--> `FacePairDataset`  [INFERRED]
  app.py → lfw_pytorch.py
- `float` --uses--> `SiameseNet`  [INFERRED]
  app.py → lfw_pytorch.py
- `SiameseNetwork` --uses--> `ConvLayer`  [INFERRED]
  siamese_network.py → conv_layer.py

## Import Cycles
- None detected.

## Communities (13 total, 3 thin omitted)

### Community 0 - "Gradio UI App Frontend"
Cohesion: 0.12
Nodes (20): _build_embed_index(), _celebrity_image_index(), _feedback_count(), find_online_matches(), _get_dataset(), _get_feat_cache(), label_diff(), label_same() (+12 more)

### Community 1 - "Siamese Training Pipeline"
Cohesion: 0.14
Nodes (17): contrastive_loss(), label=1 → same person (push dist toward 0), label=0 → different., _train_worker(), Dataset, FacePairDataset, find_file(), generate_pairs_from_filesystem(), get_names_from_csv() (+9 more)

### Community 2 - "Celebrity Scraping & Groq"
Cohesion: 0.13
Nodes (20): get_scrape_status(), _scrape_worker(), start_ai_verify(), ai_verify_all(), count_images(), get_scraped_pairs(), _groq_client(), groq_identify_face() (+12 more)

### Community 3 - "Scratch Activation Functions"
Cohesion: 0.25
Nodes (3): Relu, Sigmoid, main()

### Community 4 - "Face Feature Extraction"
Cohesion: 0.24
Nodes (9): clean_non_faces(), Delete downloaded images that don't contain a detectable human face.     Uses me, build_feature_cache(), extract_face_features(), _get_mesh(), Geometric and colour features extracted from face images via MediaPipe FaceMesh., Pre-compute and cache features for a list of image paths.     Loads any existing, img: PIL.Image (RGB) or np.ndarray (H, W, 3) uint8 RGB.     Returns ndarray shap (+1 more)

### Community 5 - "LFW Dataset Training"
Cohesion: 0.39
Nodes (8): download_dataset(), evaluate(), find_file(), load_lfw_image(), load_pairs(), main(), match CSV columns:    name, imagenum1, imagenum2     mismatch CSV columns: name,, train_epoch()

### Community 6 - "Scratch NN Layers"
Cohesion: 0.28
Nodes (3): Layer, Two images are passed through the same backbone (shared weights).     The absolu, SiameseNetwork

### Community 7 - "Face Recognition (Legacy)"
Cohesion: 0.39
Nodes (7): load_images(), main(), make_pairs(), Reads every image whose filename matches  <letters><digits>.<ext>.     e.g.  a1., Builds every positive pair (same person, label=1) and     every negative pair (d, show_loaded_images(), show_match_results()

### Community 8 - "CNN Pooling Layer"
Cohesion: 0.36
Nodes (3): main(), make_data(), MaxPool

### Community 10 - "ResNet Backbone"
Cohesion: 0.40
Nodes (3): Residual backbone (96×96 → 512-d) fused with optional geometric face         fea, Two 3×3 convs with a residual skip. stride=2 halves spatial dimensions., _ResBlock

## Knowledge Gaps
- **1 isolated node(s):** `ndarray`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SiameseNetwork` connect `Scratch NN Layers` to `Scratch Activation Functions`, `LFW Dataset Training`, `Face Recognition (Legacy)`, `CNN Pooling Layer`, `Convolutional Layer`, `Flatten Layer & Scaffold`, `Scratch Network Core`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `FacePairDataset` connect `Siamese Training Pipeline` to `Gradio UI App Frontend`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `SiameseNet` connect `Siamese Training Pipeline` to `Gradio UI App Frontend`, `ResNet Backbone`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `SiameseNetwork` (e.g. with `Relu` and `Sigmoid`) actually correct?**
  _`SiameseNetwork` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Gradio UI for LFW face verification.  Tabs:   Train       — start/stop backgroun`, `label=1 → same person (push dist toward 0), label=0 → different.`, `Resize PIL image to a square by resizing the short edge then center-cropping.` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Gradio UI App Frontend` be split into smaller, more focused modules?**
  _Cohesion score 0.11692307692307692 - nodes in this community are weakly interconnected._
- **Should `Siamese Training Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._