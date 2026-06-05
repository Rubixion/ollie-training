# Graph Report - C:\Users\barca\projects\neural network learning  (2026-06-04)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 15 nodes · 12 edges · 4 communities (2 shown, 2 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]

## God Nodes (most connected - your core abstractions)
1. `graphify-out/` - 4 edges
2. `graph.json` - 4 edges
3. `hooks` - 2 edges
4. `permissions` - 2 edges
5. `Graphify Project` - 2 edges
6. `PreToolUse` - 1 edges
7. `allow` - 1 edges
8. `graphify query` - 1 edges
9. `graphify path` - 1 edges
10. `graphify explain` - 1 edges

## Surprising Connections (you probably didn't know these)
- `graphify-out/` --references--> `graph.json`  [INFERRED]
  CLAUDE.md → CLAUDE.md  _Bridges community 0 → community 1_

## Import Cycles
- None detected.

## Communities (4 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.40
Nodes (5): GRAPH_REPORT.md, Graphify Project, graphify-out/, graphify update, wiki/index.md

### Community 1 - "Community 1"
Cohesion: 0.50
Nodes (4): graph.json, graphify explain, graphify path, graphify query

## Knowledge Gaps
- **8 isolated node(s):** `PreToolUse`, `allow`, `graphify query`, `graphify path`, `graphify explain` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `graphify-out/` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
- **Why does `graph.json` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.198) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `graphify-out/` (e.g. with `graph.json` and `GRAPH_REPORT.md`) actually correct?**
  _`graphify-out/` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PreToolUse`, `allow`, `graphify query` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._