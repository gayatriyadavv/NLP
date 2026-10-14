# SciFig-Controlled benchmark

A **synthetic, controlled** benchmark for figure element extraction, element→entity grounding,
3D-figure detection, figure↔geometry alignment and figure retrieval. It exists because the
repository's previous "SciCap" samples were four placeholder drawings (see `docs/AUDIT.md`)
and real SciCap / arXiv / HuggingFace hosts are not reachable from the build environment.

> Results on this benchmark measure behaviour under controlled conditions. They are **not**
> evidence of performance on real scientific papers.

## Generation

```bash
python -m src.data.benchmark_generator --out data/benchmark --seed 42
```

Deterministic for a given seed and matplotlib version (recorded in `metadata.json`).
Images and geometry are not committed (regenerate them); `metadata.json` is committed.

| Figure type | Count | 3D | Ground-truth geometry |
|---|---:|---|---|
| line | 50 | no | — |
| scatter | 50 | no | — |
| bar | 50 | no | — |
| heatmap | 50 | no | — |
| protein (Cα trace) | 80 (16 structures × 5 views) | yes | **real PDB coordinates** (`data/raw/pdb/SOURCES.md`) |
| surface | 60 | yes | analytic z = f(x, y) (5 families, random parameters) |
| scatter3d | 60 | yes | analytic shapes (sphere, torus, helix, cube, cylinder, cone) |
| **total** | **400** (282 train / 118 test) | 200 | |

Split: random 70/30 per figure, except proteins, which are split **by structure**
(test structures 1HVR, 1AKE, 4AKE, 2BEG, 5A7U never appear in training). Phase 4 model selection
uses a further structure-disjoint validation split of the training set (1A28, 2XHE).

## What each record contains

* `paragraph_text` — template text naming entities by canonical name, sometimes defining the
  figure abbreviation ("Graph Attention Network (GAT)"), plus **distractor entities** that are
  mentioned but not plotted.
* `entities` — gold entity list (METHOD / METRIC / VARIABLE / DATASET / MATERIAL).
* `elements` — every text artist in the figure (axis labels, legend entries, category tick
  labels, colour-bar labels, titles) with its exact pixel bbox from matplotlib, and its gold
  entity link (or none — e.g. "PC1", "x (A)", "Method" are deliberately unlinked).
* `gold_links` — element → entity with relation (REPRESENTS / CORRESPONDS_TO).
* `caption` — reference caption; used as the retrieval query and for caption overlap.
* `is_3d`, `geometry_path` (1024 × 3, centred, unit radius), `pdb_id` for proteins.

## Design choices that make it non-trivial

* Figures use **display aliases** ("GAT", "Top-1 Acc. (%)", "# params (M)") while paragraphs
  use canonical names, so exact substring matching fails for most links.
* Distractor METHOD / METRIC entities make type-based rules (LEGEND → every METHOD) produce
  false positives.
* Unlinked text elements (PC axes, Cartesian axes, "Method") must be left ungrounded.

## Known limitations

* Template text: entity mentions, captions and 3D wording are regular. In particular every 3D
  figure's paragraph/caption contains 3D vocabulary, so the **text-keyword 3D detector scores
  100% here by construction**; only the image classifier result says anything about visual 3D
  detection.
* Single rendering library and style (matplotlib defaults); an image classifier can exploit
  style cues that do not exist in real papers.
* Entity vocabulary is shared between train and test (figures differ, names recur), so the
  grounding model can partly memorise aliases.
* Small: 16 real structures; 118 test figures. Differences of a few points are within noise —
  learned models are reported as mean ± std over seeds.
* Entities are given (oracle NER) in the grounding experiments; the repository's regex NER
  does not cover this vocabulary.

## Metrics

| Task | Metric |
|---|---|
| Element extraction | text P/R/F1 (difflib similarity ≥ 0.75, one-to-one), type accuracy on matches, bbox IoU |
| Grounding | link P/R/F1 (micro, unextracted gold links count as misses), accuracy@1 and MRR per gold-linked element (ties broken in expectation) |
| 3D detection | is-3D accuracy / precision / recall, 7-way figure-type accuracy |
| Figure ↔ geometry | Recall@1/5, MRR over unique test geometries |
| Caption → figure | Recall@1/5, MRR over all 118 test figures |
| SCQ | 0.3·entity coverage + 0.3·technical correctness + 0.2·reconstruction fidelity + 0.2·retrieval accuracy, where coverage = gold-linked entities named in the generated caption, correctness = verifier SUPPORTED ratio, fidelity = token F1 vs reference caption, retrieval accuracy = reciprocal rank of the figure in caption → figure retrieval |
