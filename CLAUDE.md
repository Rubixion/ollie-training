## Trained models — NEVER DELETE

`app_checkpoint.pt` and `app_best.pt` are the user's trained model weights and took a long time to train.
**Never delete, overwrite, or modify these files under any circumstances.**
If code changes require a fresh start, tell the user and let them decide.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.
Last updated: 2026-06-07 — 551 nodes · 852 edges · 47 communities (full stack: Python ML backend + Next.js frontend).

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### God nodes (most connected — touch these carefully)
1. `Nav()` — 19 edges (used by every page)
2. `Footer()` — 16 edges
3. `extract_face_features()` — 15 edges (face_features.py, core of all search/train paths)
4. `_train_worker()` — 14 edges
5. `SiameseNet` — 13 edges
6. `FacePairDataset` — 13 edges

### Key communities
| Community | Files |
|---|---|
| Training Pipeline | lfw_pytorch.py — FacePairDataset, training loop |
| FAISS Search & Embeddings | app.py — build_faiss, build_embed_index, find_matches |
| Face Feature Extraction | face_features.py — InsightFace GPU / MediaPipe CPU fallback |
| Siamese Network Architecture | lfw_pytorch.py — ResBlock, SiameseNet, get_embedding |
| Celebrity Scraper Pipeline | celebrity_scraper.py — scraper, Groq AI verify |
| Gradio App & Dataset | app.py — all 4 tabs, feedback loop |
| AI Explainer Page | ollie-frontend/app/ai/page.tsx — NeuralDeepViz, sections 01–08 |
| Blog Post Dynamic Route | ollie-frontend/app/blog/[slug]/page.tsx — SEO, CTAs, JSON-LD |
| API Routes | ollie-frontend/app/api/search + feedback routes → Gradio port 7860 |
| WebGL Visual Effects | components/effects/ — Three.js particles, entropy, face-feature-viz |
| Auth UI & shadcn | components/sign-in, auth-modal, auth-provider, ui/ primitives |
| Design System | DESIGN.md — color tokens, Geist typography, mesh gradient |
