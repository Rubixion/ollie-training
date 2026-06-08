# Graph Report - .  (2026-06-07)

## Corpus Check
- 76 files · ~50,344 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 551 nodes · 852 edges · 47 communities (36 shown, 11 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 54 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Next.js Config & Dependencies|Next.js Config & Dependencies]]
- [[_COMMUNITY_Neural Network Layers|Neural Network Layers]]
- [[_COMMUNITY_AI Explainer Page|AI Explainer Page]]
- [[_COMMUNITY_Auth UI & shadcn Components|Auth UI & shadcn Components]]
- [[_COMMUNITY_Model Weights & Project Docs|Model Weights & Project Docs]]
- [[_COMMUNITY_Celebrity Scraper Pipeline|Celebrity Scraper Pipeline]]
- [[_COMMUNITY_Gradio App & Dataset|Gradio App & Dataset]]
- [[_COMMUNITY_Face Feature Extraction|Face Feature Extraction]]
- [[_COMMUNITY_shadcn Aliases & Config|shadcn Aliases & Config]]
- [[_COMMUNITY_TypeScript Config|TypeScript Config]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_How-It-Works Page Layout|How-It-Works Page Layout]]
- [[_COMMUNITY_App Layout & Auth|App Layout & Auth]]
- [[_COMMUNITY_8-Bit UI & Changelog|8-Bit UI & Changelog]]
- [[_COMMUNITY_Blog Listing & Footer|Blog Listing & Footer]]
- [[_COMMUNITY_Homepage Content & Blog Posts|Homepage Content & Blog Posts]]
- [[_COMMUNITY_About Page|About Page]]
- [[_COMMUNITY_Homepage & Navigation|Homepage & Navigation]]
- [[_COMMUNITY_Dev Tooling Config|Dev Tooling Config]]
- [[_COMMUNITY_FAISS Search & Embeddings|FAISS Search & Embeddings]]
- [[_COMMUNITY_Info Page|Info Page]]
- [[_COMMUNITY_Core ML Concepts|Core ML Concepts]]
- [[_COMMUNITY_LFW Evaluation|LFW Evaluation]]
- [[_COMMUNITY_WebGL Visual Effects|WebGL Visual Effects]]
- [[_COMMUNITY_Design System|Design System]]
- [[_COMMUNITY_Face Recognition Baseline|Face Recognition Baseline]]
- [[_COMMUNITY_Siamese Network Architecture|Siamese Network Architecture]]
- [[_COMMUNITY_Blog Post Dynamic Route|Blog Post Dynamic Route]]
- [[_COMMUNITY_VGGFace2 Data Pipeline|VGGFace2 Data Pipeline]]
- [[_COMMUNITY_Blog Infrastructure|Blog Infrastructure]]
- [[_COMMUNITY_API Routes|API Routes]]
- [[_COMMUNITY_App Icon|App Icon]]
- [[_COMMUNITY_OpenGraph Image|OpenGraph Image]]
- [[_COMMUNITY_Sitemap|Sitemap]]
- [[_COMMUNITY_FAQ Component|FAQ Component]]
- [[_COMMUNITY_Next.js Docs|Next.js Docs]]
- [[_COMMUNITY_Testimonials|Testimonials]]
- [[_COMMUNITY_Domain & SEO Config|Domain & SEO Config]]
- [[_COMMUNITY_shadcn Config|shadcn Config]]
- [[_COMMUNITY_Test Transform|Test Transform]]
- [[_COMMUNITY_Train Transform|Train Transform]]
- [[_COMMUNITY_OG Image Root|OG Image Root]]

## God Nodes (most connected - your core abstractions)
1. `Nav()` - 19 edges
2. `Footer()` - 16 edges
3. `compilerOptions` - 16 edges
4. `extract_face_features()` - 15 edges
5. `_train_worker()` - 14 edges
6. `Ollie — celebrity face matching application` - 14 edges
7. `FacePairDataset` - 13 edges
8. `SiameseNet` - 13 edges
9. `README.md — project overview` - 12 edges
10. `Network` - 11 edges

## Surprising Connections (you probably didn't know these)
- `VGGFace2 kagglehub cache path` --semantically_similar_to--> `VGGFace2 dataset`  [INFERRED] [semantically similar]
  vggface2_path.txt → README.md
- `groq 1.4.0` --semantically_similar_to--> `Groq llama-4-scout AI image verification`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `mediapipe 0.10.9` --semantically_similar_to--> `MediaPipe FaceMesh (v0.10.9)`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `float` --uses--> `FacePairDataset`  [INFERRED]
  app.py → lfw_pytorch.py
- `float` --uses--> `SiameseNet`  [INFERRED]
  app.py → lfw_pytorch.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Training pipeline: dataset loading → feature cache → model training → checkpoint save** — app_train_worker, lfw_pytorch_facepair_dataset, face_features_build_feature_cache, lfw_pytorch_siamesenet, app_contrastive_loss, concept_vggface2 [EXTRACTED 0.95]
- **Search pipeline: upload → feature extract → embed → FAISS search → re-rank → results** — api_search_route, app_find_dataset_matches, face_features_extract_face_features, app_build_embed_index, app_build_faiss, concept_faiss_index [EXTRACTED 0.95]
- **Face feature extraction: InsightFace GPU or MediaPipe CPU → 32-dim vector** — face_features_extract_face_features, face_features_get_insight_app, face_features_feats_from_insight, face_features_get_mesh, face_features_feat_dim, concept_insightface_gpu, concept_mediapipe_fallback [EXTRACTED 0.95]
- **Next.js frontend pages (Ollie web app)** — root_layout, home_page, match_page, ai_neuralpage, feedback_page, about_aboutpage, blog_index_page, contact_page, info_page [EXTRACTED 0.95]
- **Authentication flow: AuthProvider + AuthModal + Supabase** — components_auth_provider, components_auth_modal, lib_supabase, concept_auth_context, concept_supabase_auth [INFERRED 0.95]
- **Canvas/WebGL visual effects components** — components_dotted_surface_root, effects_dotted_surface, effects_entropy, effects_face_feature_viz, effects_particle_text [INFERRED 0.85]
- **Legal/policy pages (privacy + terms)** — privacy_page, terms_page, components_footer [INFERRED 0.85]
- **SEO metadata config (robots + sitemap)** — robots_config, sitemap_config, lib_blog_posts [INFERRED 0.85]
- **Three.js WebGL Visualization Components** — components_neural_deep_viz_neuraldeepviz, components_neural_viz_threeviz, ui_dotted_surface_dottedsurface, concept_three_webgl_renderer [INFERRED 0.90]
- **Siamese Network Layer Pair** — components_neural_deep_viz_top_layers, components_neural_deep_viz_bottom_layers, components_neural_deep_viz_distance_layer, components_neural_deep_viz_siamese_architecture [EXTRACTED 1.00]
- **ShadCN UI Primitive Components** — ui_badge_badge, ui_button_button, ui_card_card, ui_input_input, ui_label_label [INFERRED 0.90]
- **8-bit Styled UI Wrappers** — ui_8bit_badge_badge, ui_8bit_card_card, ui_8bit_separator_separator [INFERRED 0.90]
- **Authentication UI Flow** — components_nav_nav, components_nav_profilemenu, components_sign_in_authform, concept_auth_provider [INFERRED 0.90]
- **Python ML dependency stack** — requirements_torch, requirements_mediapipe, requirements_gradio, requirements_faiss, requirements_groq, requirements_insightface [INFERRED 0.90]
- **Face verification core concepts** — readme_siamese_net, readme_contrastive_loss, readme_geometric_features, readme_feature_reranking, readme_mediapipe [EXTRACTED 0.95]
- **Training data sources** — readme_lfw_dataset, readme_vggface2, readme_celebrity_scraper [EXTRACTED 0.95]
- **Next.js configuration files** — next_app_postcss_config, next_app_tsconfig [INFERRED 0.90]
- **Vercel-inspired design system** — design_md_color_system, design_md_typography, design_md_spacing, design_md_components, design_md_mesh_gradient, design_md_elevation [EXTRACTED 0.95]

## Communities (47 total, 11 thin omitted)

### Community 0 - "Next.js Config & Dependencies"
Cohesion: 0.05
Nodes (39): Supabase, nextConfig, dependencies, class-variance-authority, clsx, framer-motion, lucide-react, motion (+31 more)

### Community 1 - "Neural Network Layers"
Cohesion: 0.11
Nodes (10): Relu, Sigmoid, Flatten, main(), make_data(), main(), Layer, Network (+2 more)

### Community 2 - "AI Explainer Page"
Cohesion: 0.08
Nodes (20): DottedSurfaceProps, BOTTOM_LAYERS, DISTANCE_LAYER, HoveredNode, LayerDef, Siamese Network Architecture, TOP_LAYERS, buildThreeScene() (+12 more)

### Community 3 - "Auth UI & shadcn Components"
Cohesion: 0.16
Nodes (19): AuthForm, Badge(), badgeVariants, BitButtonProps, Badge(), BadgeProps, badgeVariants, Button (+11 more)

### Community 4 - "Model Weights & Project Docs"
Cohesion: 0.11
Nodes (25): app_best.pt — best model weights, app_checkpoint.pt — training checkpoint, CLAUDE.md — project rules, Graphify query-first rule, Never-delete trained models rule, Celebrity scraper (~450 celebrities), Contrastive loss (margin=2.0), Feature re-ranking (skin tone, hair, face shape) (+17 more)

### Community 5 - "Celebrity Scraper Pipeline"
Cohesion: 0.12
Nodes (22): get_scrape_status(), _scrape_worker(), start_ai_verify(), ai_verify_all(), clean_non_faces(), count_images(), get_scraped_pairs(), _groq_client() (+14 more)

### Community 6 - "Gradio App & Dataset"
Cohesion: 0.13
Nodes (18): _celebrity_image_index(), _feedback_count(), _get_dataset(), Gradio UI (Face Verification app), label_diff(), label_same(), _lfw_root(), load_feedback_pair() (+10 more)

### Community 7 - "Face Feature Extraction"
Cohesion: 0.15
Nodes (21): build_feature_index(), _describe_features(), Generator. Runs parallel MediaPipe extraction on every index image that     curr, bool, InsightFace GPU feature extraction, MediaPipe FaceMesh CPU fallback, build_feature_cache(), extract_face_features() (+13 more)

### Community 8 - "shadcn Aliases & Config"
Cohesion: 0.09
Nodes (21): aliases, components, hooks, lib, ui, utils, iconLibrary, menuAccent (+13 more)

### Community 9 - "TypeScript Config"
Cohesion: 0.09
Nodes (21): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+13 more)

### Community 10 - "Training Pipeline"
Cohesion: 0.17
Nodes (16): _train_worker(), Dataset, FacePairDataset, find_file(), generate_pairs_from_filesystem, generate_pairs_from_flat_dir, get_names_from_csv(), lfw_img_path() (+8 more)

### Community 11 - "How-It-Works Page Layout"
Cohesion: 0.13
Nodes (11): NeuralPage (/ai route), BGMaskType, BGPattern(), BGPatternProps, BGVariantType, geBgImage(), maskClasses, metadata (+3 more)

### Community 12 - "App Layout & Auth"
Cohesion: 0.13
Nodes (11): fontMono, metadata, outfit, Tab, AuthContext, AuthContextValue, Match, AuthContext (React Context) (+3 more)

### Community 13 - "8-Bit UI & Changelog"
Cohesion: 0.14
Nodes (12): ChangelogEntry, defaultEntries, Team2Props, ENTRIES, BitCardProps, Card(), CardContent(), CardDescription() (+4 more)

### Community 14 - "Blog Listing & Footer"
Cohesion: 0.15
Nodes (9): FindCelebrityLookalikePage, company, Footer(), legal, product, metadata, jsonLd, metadata (+1 more)

### Community 15 - "Homepage Content & Blog Posts"
Cohesion: 0.13
Nodes (7): WORDS, Ollie — celebrity face matching application, allPosts (blog-posts lib), metadata, robots (MetadataRoute), sitemap (MetadataRoute), metadata

### Community 16 - "About Page"
Cohesion: 0.14
Nodes (7): changelogData, FAQ_ITEMS, TabId, TABS, AboutPage (server wrapper), AboutPageClient, metadata

### Community 17 - "Homepage & Navigation"
Cohesion: 0.23
Nodes (8): Page(), useAuth(), links, Nav(), ProfileMenu(), AuthProvider (useAuth hook), jsonLd, metadata

### Community 18 - "Dev Tooling Config"
Cohesion: 0.15
Nodes (11): devDependencies, eslint, prettier, prettier-plugin-tailwindcss, tailwindcss, @types/node, @types/react, @types/three (+3 more)

### Community 19 - "FAISS Search & Embeddings"
Cohesion: 0.22
Nodes (11): _build_embed_index(), _build_faiss(), find_dataset_matches(), find_online_matches(), _get_feat_cache(), _load_model(), ndarray, Return the in-memory feature cache, loading from disk if needed. (+3 more)

### Community 20 - "Info Page"
Cohesion: 0.20
Nodes (4): ENTRIES, FAQ_ITEMS, TabId, TABS

### Community 21 - "Core ML Concepts"
Cohesion: 0.36
Nodes (9): ArchitectureScroll, contrastive_loss(), label=1 → same person (push dist toward 0), label=0 → different., How AI Facial Recognition Works article, Siamese Neural Networks Explained article, 256-dim face embedding / fingerprint, FAISS nearest-neighbour index, Siamese Network architecture (+1 more)

### Community 22 - "LFW Evaluation"
Cohesion: 0.39
Nodes (8): download_dataset(), evaluate(), find_file(), load_lfw_image(), load_pairs(), main(), match CSV columns:    name, imagenum1, imagenum2     mismatch CSV columns: name,, train_epoch()

### Community 23 - "WebGL Visual Effects"
Cohesion: 0.29
Nodes (4): DottedSurface (root), Three.js particle wave animation, DottedSurfaceProps, EntropyProps

### Community 24 - "Design System"
Cohesion: 0.39
Nodes (8): DESIGN.md — Vercel design system analysis, Vercel color system, Vercel component library, Vercel stacked-shadow elevation system, Multi-stop mesh gradient decoration, Vercel design rationale — stark ink+gray+gradient, no extra accents, Vercel spacing system, Vercel typography system (Geist / Geist Mono)

### Community 25 - "Face Recognition Baseline"
Cohesion: 0.39
Nodes (7): load_images(), main(), make_pairs(), Reads every image whose filename matches  <letters><digits>.<ext>.     e.g.  a1., Builds every positive pair (same person, label=1) and     every negative pair (d, show_loaded_images(), show_match_results()

### Community 26 - "Siamese Network Architecture"
Cohesion: 0.43
Nodes (4): Residual backbone (96×96 → 512-d) fused with optional geometric face         fea, Two 3×3 convs with a residual skip. stride=2 halves spatial dimensions., _ResBlock, SiameseNet

### Community 28 - "VGGFace2 Data Pipeline"
Cohesion: 0.40
Nodes (5): download_vggface2(), Generator — downloads VGGFace2 from Kaggle and reports progress., Return the VGGFace2 train directory, or None if not downloaded yet., _vgg_root(), _vgg_status()

### Community 29 - "Blog Infrastructure"
Cohesion: 0.40
Nodes (3): BlogPage (index), metadata, BlogPostPage ([slug])

### Community 30 - "API Routes"
Cohesion: 0.67
Nodes (3): POST /api/feedback, POST /api/search, Gradio backend server (port 7860)

### Community 35 - "Next.js Docs"
Cohesion: 0.67
Nodes (3): next-app/AGENTS.md — Next.js agent rules, next-app/README.md — Next.js + shadcn/ui template, shadcn/ui component system

## Knowledge Gaps
- **169 isolated node(s):** `ndarray`, `bool`, `FAQ_ITEMS`, `changelogData`, `TABS` (+164 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Supabase` connect `Next.js Config & Dependencies` to `App Layout & Auth`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `Ollie — celebrity face matching application` connect `Homepage Content & Blog Posts` to `AI Explainer Page`, `Gradio App & Dataset`, `How-It-Works Page Layout`, `App Layout & Auth`, `Blog Listing & Footer`, `About Page`, `Info Page`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `Gradio UI (Face Verification app)` connect `Gradio App & Dataset` to `FAISS Search & Embeddings`, `Homepage Content & Blog Posts`, `Face Feature Extraction`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **What connects `Gradio UI for LFW face verification.  Tabs:   Train       — start/stop backgroun`, `label=1 → same person (push dist toward 0), label=0 → different.`, `Resize PIL image to a square by resizing the short edge then center-cropping.` to the rest of the system?**
  _206 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Next.js Config & Dependencies` be split into smaller, more focused modules?**
  _Cohesion score 0.04878048780487805 - nodes in this community are weakly interconnected._
- **Should `Neural Network Layers` be split into smaller, more focused modules?**
  _Cohesion score 0.11025641025641025 - nodes in this community are weakly interconnected._
- **Should `AI Explainer Page` be split into smaller, more focused modules?**
  _Cohesion score 0.0761904761904762 - nodes in this community are weakly interconnected._