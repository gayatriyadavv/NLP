# Repository Audit — 2026-10-08

Audit of commit `eeaf93e` ("Initial commit: Phase 1-3 scientific figure understanding system")
performed before any Phase 4 work, as required by the project master prompt (§40).
No source code was modified as part of this audit.

**Bottom line:** the code runs and all tests pass, but the reported Phase 3 and Phase 4
results are not measurements. They are produced by hard-coded constants in the evaluation
scripts. Several components marked IMPLEMENTED are placeholders. Phase 3 is therefore not
in a state where Phase 4 can be built and evaluated on top of it.

---

## A. Repository structure

```
api/main.py                         FastAPI app (8 endpoints)
config/config.yaml                  model names, retrieval/SCQ weights, ablation switches
src/ingestion/pdf_parser.py         PyMuPDF → Paper/Section/Paragraph dict
src/ingestion/scicap_loader.py      loads demo/dataset/metadata.json, BLEU-1-style overlap eval
src/nlp/scientific_ner.py           regex NER + co-occurrence relations
src/retrieval/hierarchical_indexer.py  4-level FAISS/SentenceTransformer index, word-overlap fallback
src/retrieval/multimodal_retriever.py  weighted sum of cosine sims (text/2D/3D/GNN)
src/vision/figure_processor.py      figure "element extraction"
src/vision/threed_detector.py       3D figure detector
src/vision/multiview_renderer.py    4 crops of a 2D image → untrained CNN → 64-d
src/models/vision/pointnet_extractor.py  vanilla PointNet (untrained) + point-cloud loader
src/grounding/entity_graph.py       NetworkX grounding graph (+ optional GNN call)
src/grounding/pyg_converter.py      NetworkX → PyG Data / HeteroData
src/grounding/gnn_pipeline.py       per-graph self-supervised link-prediction training
src/models/grounding/gnn_grounding.py  GAT/GCN/SAGE encoder, bilinear link scorer
src/generation/caption_generator.py caption generation
src/evaluation/hallucination_checker.py claim verification + SCQ
pipelines/run_pipeline.py           end-to-end pipeline
pipelines/evaluate_gnn.py           Phase 3 benchmark
pipelines/evaluate_phase4.py        Phase 4 ablation
tests/test_{ingestion,gnn,phase4}.py 15 tests
demo/dataset/                       4 "SciCap" samples, 1 point cloud, result JSONs
frontend/                           React 19 + TS + Vite, 11 components
```

`implementation_plan.md` **does not exist** in the repository. `README.md` still lists
GNN (Phase 3) and 3D (Phase 4) as PLANNED.

## B. Existing architecture (as actually wired)

```
PDF ─► PDFIngestor (each text line = a "paragraph"; heading = uppercase or "N." prefix;
                    figures list always empty)
   ─► ScientificNERExtractor (regex over a fixed vocabulary; confidence fixed at 0.92)
   ─► HierarchicalIndexer (paper/section/paragraph/entity; FAISS if sentence-transformers
                           is installed, otherwise word overlap)
Figure ─► FigureProcessor (same 4 hard-coded elements for every image)
   ─► EntityGroundingGraph (rules: substring match, AXIS→VARIABLE/METRIC, LEGEND→METHOD/MODEL)
        └► GNNGroundingPipeline (trains a fresh GAT on this one graph, attaches embeddings;
                                 embeddings are not used to add/remove/rank links)
   ─► GroundedCaptionGenerator (string template)
   ─► HallucinationChecker (word overlap) ─► SCQ
```

The 3D modules (`threed_detector`, `pointnet_extractor`, `multiview_renderer`,
`multimodal_retriever`) are called only from `api/main.py` and `evaluate_phase4.py`;
they are **not** part of `run_pipeline.py` and do not feed the grounding graph.

## C. Phase status — claimed vs. verified

| Component | Claimed | Verified | Evidence |
|---|---|---|---|
| PDF ingestion / hierarchy | IMPLEMENTED | PARTIALLY IMPLEMENTED | Works, but every text line becomes a paragraph; figures never extracted (`pdf_parser.py:32,72`). |
| Scientific NER | IMPLEMENTED (SciBERT hybrid) | PROTOTYPE (regex only) | SciBERT never loaded; fixed vocabulary; `confidence: 0.92` constant (`scientific_ner.py:50`). |
| 4-level retrieval | IMPLEMENTED | IMPLEMENTED (when sentence-transformers + FAISS installed) | Neither is pinned for the test run here; falls back to word overlap. |
| Figure element extraction | IMPLEMENTED (OpenCV + OCR) | **NOT IMPLEMENTED (placeholder)** | Returns identical hard-coded elements ("Legend: Method A vs Method B", …) for every image; no OCR; OpenCV only reads image size (`figure_processor.py:42-45`). Verified: output identical across all 4 samples. |
| NetworkX grounding graph | IMPLEMENTED | IMPLEMENTED (rule-based) | Works as described. |
| Grounded captioning | IMPLEMENTED | PROTOTYPE (template) | No VLM loaded. Template always appends "the proposed framework demonstrates performance gains across training iterations" — an unsupported claim for every figure (`caption_generator.py:50`). `grounding_confidence: 0.91` constant. |
| Claim verification (NLI) | IMPLEMENTED | PROTOTYPE (word overlap) | No NLI model. <10% overlap is labelled CONTRADICTED, which is not contradiction (`hallucination_checker.py:39-40`). |
| SCQ metric | IMPLEMENTED | IMPLEMENTED (formula) | Formula is correct; inputs include hard-coded constants (see below). |
| Phase 2 — CLIP | PARTIALLY IMPLEMENTED | **NOT IMPLEMENTED** | No CLIP code exists. Only a config string and UI label. |
| Phase 3 — PyG converter | IMPLEMENTED | IMPLEMENTED | Works; tested. |
| Phase 3 — GNN model | IMPLEMENTED | IMPLEMENTED (architecture) | GAT/GCN/SAGE, residual, LayerNorm, bilinear scorer — matches description. |
| Phase 3 — GNN training | IMPLEMENTED | PROTOTYPE | Trained from scratch on each single graph, on that graph's own rule-generated edges (message-passing edges = positive targets, so leakage); negatives may include positives; no train/test split; model not saved. Embeddings only score edges that already exist, so the GNN cannot change grounding output. |
| Phase 3 — evaluation | IMPLEMENTED, +57.6% | **RESULT NOT MEASURED** | See §C.1. |
| Phase 4 — 3D detector | (implicit) | PROTOTYPE (keywords + filename) | Image pixels are never analysed; `"scicap_003" in path or "3d" in path` (`threed_detector.py:42`). |
| Phase 4 — PointNet | (implicit) | PROTOTYPE (untrained) | Random-init weights; embeddings carry no learned meaning. |
| Phase 4 — multi-view | (implicit) | PROTOTYPE | Crops/rotation of one 2D image, untrained CNN. Not multi-view of 3D content. |
| Phase 4 — multimodal retrieval | (implicit) | PROTOTYPE | Weighted cosine sum; missing modalities scored 0.50 by default. |
| Phase 4 — ablation | (implicit) | **RESULT NOT MEASURED** | See §C.2. |
| FastAPI | IMPLEMENTED | IMPLEMENTED | `/scicap/run-sample/{id}` runs every sample through `demo/sample_paper.pdf`, not the sample's own text (`api/main.py:127`). |
| Frontend | IMPLEMENTED | IMPLEMENTED | Builds. On any API error it silently shows mock results (`frontend/src/services/api.ts:24,31`). |

### C.1 Phase 3 benchmark numbers are constructed, not measured

`pipelines/evaluate_gnn.py`:

* **Line 119:** `gnn_acc = min(base_acc + 0.12, 1.0)` — the GNN grounding accuracy is the
  baseline plus a fixed 12 points. This alone produces 20.83% → 32.83% (+57.60%).
  Re-run confirms every per-sample difference is exactly 0.12, including the 3 samples
  where the GNN trained for **0 epochs** (graph had no edges).
* **Line 90:** "grounding accuracy" = edges ÷ (elements + entities). It is not compared to
  any ground truth.
* **Lines 81 vs 112:** entity coverage divides by 5.0 for baseline, 4.5 for GNN.
* **Lines 83 vs 114:** reconstruction fidelity is a constant 0.82 for baseline,
  `0.85 + 0.1·sim` for GNN (and `0.85` fallback if there are no edges).
* The SCQ gain therefore comes from different constants, not from the GNN.

### C.2 Phase 4 ablation numbers are hard-coded

`pipelines/evaluate_phase4.py:105-124` assigns fixed accuracy / SCQ / MRR values per
configuration. Latency is `elapsed + 1700` or `+ 25` ms (line 126). Query, 2D and
distractor embeddings are `np.random.randn` (lines 75-76, 96-97); no CLIP is invoked for
"GNN + CLIP". `demo/dataset/phase4_ablation_results.json` contains these constants.

### C.3 Latency (measured in this audit)

CPU: Intel Xeon @ 2.10 GHz, 4 threads, torch 2.14.1, PyG 2.8.0, seed 42.
Graph: scicap_001 rule graph, 12 nodes / 10 edges. 10 repetitions.

| Measurement | Median |
|---|---:|
| GNN train + embed (30 epochs, Adam lr 0.01, wd 1e-4) | 116.7 ms |
| GNN inference only (one forward pass, eval mode) | 0.74 ms |

The previously reported 1,747.85 ms was a mean over 4 samples where only one trained,
on Apple MPS (includes device warm-up). A fresh run of `evaluate_gnn.py` here gave
40.75 ms mean (train included). Neither is an inference latency.

## D. Tests

| Suite | Result |
|---|---|
| `tests/test_ingestion.py` (6) | pass |
| `tests/test_gnn.py` (4) | pass |
| `tests/test_phase4.py` (5) | pass |
| `frontend: npm run build` | pass, 0 TS/Vite errors |
| `pipelines/evaluate_gnn.py` | runs; reproduces 20.83 → 32.83 (by construction, §C.1) |

Tests check shapes and keys, not behaviour. `test_threed_detector` passes because of the
filename check. sentence-transformers / FAISS were not installed during this run, so the
retrieval test exercised the word-overlap fallback.

Environment notes: `torch_geometric` is imported by Phase 3 but missing from
`requirements.txt`. `config.get("a.b")` is called on a plain dict in several places
(`entity_graph.py:15`, `hallucination_checker.py:11`, `multimodal_retriever.py:13`,
`evaluate_gnn.py:62`), so config values and ablation switches are never read — the
`graph_enabled` switch cannot turn the GNN off.

## E. Phase 4 readiness

| Item | Exists? | Usable for research? |
|---|---|---|
| CLIP | No | — |
| 3D detection | Keyword/filename heuristic | No — needs an image classifier and labelled data |
| Point clouds | 1 file | No — see §F |
| PointNet encoder | Yes, untrained | Interface yes; embeddings meaningless until trained/pretrained |
| Multi-view | Crops of a 2D image | No — not views of 3D content |
| GNN integration of 3D | No | 3D embeddings never enter the graph |
| Multimodal retrieval | Weighted cosine | Interface yes; no real embeddings feed it |

## F. Data readiness

`demo/dataset/` does not support any quantitative claim, 2D or 3D:

* **4 samples**, all placeholder images (640×400, no axis labels, ticks or legend text;
  the "3D molecular" figure is a flat 2D ellipse labelled "SciCap Sample 3").
* Metadata inconsistencies: `arxiv_2103.00020` is the CLIP paper, not ViT; AlphaFold 2
  was published in Nature, not on arXiv. The SciCap dataset itself consists of graph
  plots, so heatmaps/molecular renders would not come from it. Captions could not be
  traced to the cited papers. Treat the samples as synthetic.
* **`alphafold_t1087_3d.json` is synthetic.** Its 1,024 points fit an ideal 3-turn helix of
  radius 12 Å with correlation 0.998 on every axis plus Gaussian noise. It is labelled
  "AlphaFold (Jumper et al., Nature 2021), CC-BY-4.0", which is incorrect attribution.
  `3d_benchmark_metadata.json` says 2 samples but lists 1.
* No ground-truth element↔entity grounding links exist, so grounding accuracy cannot be
  computed against truth on any sample.

Answers to the master-prompt §32 questions:

1. Does SciCap contain enough genuine 3D information? — The repo has no real SciCap data. The real SciCap is graph plots; not a 3D source.
2. Raw 3D geometry? — No (the one file is synthetic).
3. Only rendered images? — Only placeholder drawings.
4. 3D annotations? — None.
5. Legitimate public 3D source? — Candidates (licences to be verified before use):
   * **RCSB PDB / AlphaFold DB** — real atomic coordinates (structural biology). Can be
     rendered to multi-view images, giving an honest image↔geometry pairing for a
     controlled 3D benchmark. Does not by itself provide paper figures or captions.
   * **Real SciCap** (Hsu et al., 2021) — real arXiv figure/caption pairs for the 2D
     pipeline and for training/evaluating a 2D-vs-3D figure classifier (after
     hand-labelling a subset).
   * Larger arXiv figure corpora (e.g. ArxivCap) if more figure diversity is needed.

   I'm not aware of a public dataset that pairs scientific *paper figures* with their
   underlying 3D geometry and grounding labels; that pairing would have to be built.

## G. Recommended implementation order

Phase 3 is not stable in the sense required by §41, so Phase 4 should wait.

0. **Correct the record.** Remove the `+0.12` and differing constants from
   `evaluate_gnn.py`; delete or regenerate `gnn_benchmark_results.json` and
   `phase4_ablation_results.json`; re-label the synthetic point cloud and samples;
   correct status strings in code/README; create `implementation_plan.md`; add
   `torch_geometric` to requirements; fix the dotted-key config bug.
1. **Real data.** Load a real SciCap subset (≥100 figures) and hand-annotate
   element↔entity gold links for a test split.
2. **Real figure elements.** OCR (EasyOCR/Tesseract) + axis/legend localisation;
   evaluate against annotations.
3. **Real grounding metric.** Precision/recall/F1 of predicted links vs gold; measure
   the rule baseline.
4. **GNN done properly.** Train once across many graphs (train split), score candidate
   element–entity pairs (not only existing edges), evaluate on held-out graphs; report
   training time and inference time separately.
5. **CLIP.** Implement image/text encoders; measure Recall@K / MRR for text→figure
   retrieval on real pairs.
6. **Phase 4.** Image-based 3D-figure classifier (labelled set) → PDB/AlphaFold-rendered
   benchmark with real geometry → pretrained point-cloud encoder → graph integration →
   ablation A–E with measured numbers only.

---

## Resolution (same day)

All findings above were addressed on branch `claude/scientific-figure-understanding-9i8ve7`:
fabricated result files deleted and the scripts that produced them rewritten to measure;
placeholder components replaced (OCR element extraction, pixel-based 3D detector) or relabelled
(NER, captioning, verification as PROTOTYPE; CLIP as BLOCKED); demo data and the synthetic point
cloud relabelled; config dotted-key bug, API sample routing and silent frontend mock fallback
fixed. Measured results on a new controlled benchmark with real PDB geometry are in
`implementation_plan.md` and `results/`.
