# Handover Guide — Continuing the Project

This guide is for whoever continues the project. It explains how to get running, how the
code fits together, what to work on next, and the rules that keep the results trustworthy.

Read these in order:

1. **This file**: setup, code map, how to continue.
2. [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) or [`docs/Project_Status_Report.pdf`](docs/Project_Status_Report.pdf): what is done and what is not.
3. [`implementation_plan.md`](implementation_plan.md): every measured result, per component.
4. [`docs/BENCHMARK.md`](docs/BENCHMARK.md): the synthetic benchmark. [`docs/AUDIT.md`](docs/AUDIT.md): history of the 2026-10-08 audit.

---

## 1. What you received

The handover comes as **two files** (split to stay under attachment size limits):

| File | Contents |
|---|---|
| `NLP-project-handover.zip` | The project folder: all code, docs, results, the SciCap subset, and the artefacts git does not track (trained models and OCR caches, listed below) |
| `NLP-git-history.bundle` | The complete git history (all commits and branches) in one file |

Artefacts included in the zip that git does not track:

| Artefact | Path | Recreate with |
|---|---|---|
| Trained models (3) | `models/checkpoints/*.pt` | `pipelines/evaluate_phase4.py`, `pipelines/evaluate_gnn.py` |
| OCR caches (save ~30 min) | `data/benchmark/cache/`, `data/scicap_subset/cache/` | filled automatically on first run |

**One step is needed after unzipping.** Regenerate the synthetic benchmark images (left out to save
space; about 1 minute). With the matplotlib version in `requirements-lock.txt` they are byte-identical
to the originals, so the OCR cache and all results still match:

```bash
python -m src.data.benchmark_generator --out data/benchmark
```

**To restore the git history** (recommended), run this inside the unzipped `NLP` folder:

```bash
git init -q
git fetch -q ../NLP-git-history.bundle "refs/heads/*:refs/heads/*"
git checkout -f claude/scientific-figure-understanding-9i8ve7
git remote add origin <your-repo-url>          # then: git push -u origin --all
```

`git status` should then show a clean working tree, and `git log` shows the full history.
Development happens on branch `claude/scientific-figure-understanding-9i8ve7`. The original
repository is https://github.com/gayatriyadavv/NLP.

## 2. Setup (about 10 minutes)

**Requirements:** Python 3.11+ (results were produced with 3.13), Node.js 20+ (22 used), and about 5 GB of
disk space. A GPU is not needed; everything runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # or requirements-lock.txt for the exact versions used
cd frontend && npm ci && cd ..
python -m pytest tests/                    # expect: all pass (39)
```

On first use, EasyOCR downloads its model weights (about 100 MB) from GitHub.

**Notes**
* `requirements-lock.txt` pins the exact versions behind `results/*.json`. Use it when you want
  to reproduce numbers exactly. On CPU-only machines, install torch from the CPU index; the lock
  file explains how.
* CPU results are not bit-for-bit identical across machines or thread counts. That is why the
  learned models are reported as mean ± std over 3 seeds.

## 3. Running things

| Goal | Command | Time (CPU) |
|---|---|---|
| End-to-end pipeline on one figure | `python pipelines/run_pipeline.py --paper demo/sample_paper.pdf --figure demo/sample_figure.png --query "What does this figure show?"` | ~10 s |
| Backend API | `uvicorn api.main:app --port 8000` | – |
| Web UI | `cd frontend && npm run dev`, then open http://localhost:5173 | – |
| Grounding evaluation (rules vs GNN) | `python pipelines/evaluate_gnn.py` | ~10 min |
| 3D models (train + evaluate) | `python pipelines/evaluate_phase4.py` | ~15 min |
| Ablation A–E with significance tests | `python pipelines/run_ablation.py` | ~20 min |
| Real-figure evaluation (SciCap) | `python pipelines/evaluate_scicap.py` | ~2 min cached / ~15 min fresh |
| 3D detector modes | `python pipelines/evaluate_3d_detector_modes.py` | ~1 min |
| Rebuild the PDF status report | `python scripts/make_status_report.py` | seconds |
| Fetch more SciCap figures (needs HuggingFace access) | `python scripts/fetch_scicap_subset.py --n 500` | ~10 min |

Each evaluation writes `results/<name>.json`, recording the dataset, seeds, hyperparameters, hardware
and timings. Run evaluations one at a time: they use all CPU cores, and running several together slows
every one of them down sharply.

## 4. How the code fits together

```
figure image + paper text
   │
   ├─ src/vision/ocr_engine.py          EasyOCR, two orientations, cached
   ├─ src/vision/figure_processor.py    OCR boxes → AXIS / LEGEND / LABEL / CHART_REGION
   ├─ src/nlp/scientific_ner.py         regex entities + co-occurrence relations
   ├─ src/retrieval/hierarchical_indexer.py   4-level context retrieval
   ├─ src/grounding/entity_graph.py     NetworkX graph; applies the trained GNN if a checkpoint exists
   │     ├─ grounding_features.py       pairwise element↔entity features, rule graph
   │     ├─ pyg_converter.py            NetworkX → PyG (node features, FIGURE_3D / FIGURE_IMAGE nodes)
   │     ├─ gnn_pipeline.py             SupervisedGroundingTrainer (train / predict / save / load)
   │     └─ ../models/grounding/gnn_grounding.py   GAT encoder + pair head
   ├─ src/vision/threed_detector.py     3D detection: image CNN, text cues, "agreement" mode
   │     └─ ../models/vision/figure_image_encoder.py
   ├─ src/models/vision/cross_modal_3d.py   figure ↔ 3D geometry contrastive training / evaluation
   │     ├─ pointnet_extractor.py       PointNet on point clouds
   │     └─ ../../vision/multiview_renderer.py   multi-view rendering of geometry + CNN
   ├─ src/retrieval/multimodal_retriever.py   fuse text / 2D / 3D / graph similarities
   ├─ src/generation/caption_generator.py     evidence-slot caption template
   └─ src/evaluation/hallucination_checker.py claim verification + SCQ metric

pipelines/run_pipeline.py   ties the stages together (also used by api/main.py)
src/data/benchmark_generator.py   builds the synthetic benchmark (real PDB structures in data/raw/pdb)
src/evaluation/bootstrap.py       paired bootstrap confidence intervals
src/evaluation/figure_extraction_eval.py   OCR element precision / recall
```

**Configuration** lives in `config/config.yaml`; load it with `src/utils/config.get_config()`, which
supports dotted keys such as `cfg.get("models.grounding.checkpoint")`.

**Data**
* `data/benchmark/`: synthetic benchmark (`metadata.json` is tracked; images are regenerated).
* `data/scicap_subset/`: 287 real arXiv figures with captions (CC BY-NC-SA 4.0: non-commercial, with attribution).
* `data/raw/pdb/`: 16 real protein structures used for the 3D figures.
* `demo/`: placeholder samples for the UI only; never report metrics on them.

## 5. What to work on next

In priority order. Each item names its starting point and how you will know it is done.

| # | Task | Start here | Done when |
|---|---|---|---|
| 1 | **Measure grounding on real figures**: hand-annotate element → entity links for ~100 SciCap figures | `data/scicap_subset/metadata.json`; reuse the gold-link format from `data/benchmark/metadata.json` (`elements`, `gold_links`) | `evaluate_gnn.py` (or a SciCap variant) reports rules vs GNN F1 on the annotated figures |
| 2 | **Enable pretrained models**: allow `huggingface.co` in your environment | `src/models/vision/clip_encoder.py` (ready), `hierarchical_indexer.py`, `hallucination_checker.py` (NLI loads automatically) | Ablation configuration C runs; retrieval uses Sentence-Transformers |
| 3 | **Better captions**: replace the template with a vision-language model | `src/generation/caption_generator.py`; keep the output dict format | BLEU-4 / ROUGE-L on SciCap improve over 0.051 / 0.084 |
| 4 | **Better entity extraction** (45% of real figures currently get none) | `src/nlp/scientific_ner.py`; keep the `extract_entities` output format | Fewer figures with zero entities; grounding links per figure go up |
| 5 | **Wire the Cross-Modal Search tab and file upload** in the UI | `frontend/src/components/CrossModalSearch.tsx` (hard-coded results), `UploadWorkspace.tsx` (sends only file names); add API endpoints in `api/main.py` | Search returns real results; uploaded files are processed |
| 6 | **Real 3D figures**: collect a small labelled set before claiming anything about 3D on real papers | `pipelines/evaluate_3d_detector_modes.py` | Recall and false positives measured on real 3D figures |
| 7 | **Message passing**: it currently adds nothing; redesign it (e.g. graphs across several figures) or remove it | `src/grounding/gnn_pipeline.py`; compare against `diagnostic_no_message_passing` in `evaluate_gnn.py` | A significant gain over the diagnostic, or a documented removal |

## 6. Rules that keep the results trustworthy

The project was audited after earlier results turned out to be hard-coded (`docs/AUDIT.md`).
Keep it honest:

1. **Never type a number into a report.** Results come from `results/*.json` produced by a script.
   The PDF report reads them automatically; tests fail if hard-coded metric offsets appear in the
   evaluation scripts.
2. **Measure training and inference time separately.**
3. **Report learned models over several seeds (mean ± std), and compare configurations with the paired
   bootstrap** (`src/evaluation/bootstrap.py`). A difference whose 95% confidence interval contains 0 is
   "no detectable effect".
4. **Choose hyperparameters on the validation split, never on test.**
5. **Label component status honestly** (IMPLEMENTED / PARTIAL / PROTOTYPE / PLANNED / BLOCKED) in
   `implementation_plan.md`, and update `docs/PROJECT_STATUS.md` when something changes.
6. **Run `python -m pytest tests/` before every push.** CI runs the tests and the frontend build on GitHub.

## 7. Known pitfalls (learned the hard way)

* **BatchNorm on small data:** eval-mode accuracy swung between 71% and 100% across seeds until
  BatchNorm statistics were recalibrated after training (`recalibrate_batchnorm` in
  `cross_modal_3d.py`). Keep that call when you change those models.
* **Single training runs are unreliable on CPU:** 3D alignment R@1 ranged from 0.13 to 0.22 for the same
  seed under different thread counts. Always use several seeds.
* **Caption leakage in SciCap:** the PDF-extracted paragraphs often contain the caption itself.
  `strip_caption` in `evaluate_scicap.py` removes it; without it, retrieval looks better than it really is.
* **The synthetic benchmark is easy for some tasks:** the text-keyword 3D detector scores 100% there
  only because the templates mention "3D". Check every claim on real data before trusting it.
* **The image-only 3D classifier does not transfer to real figures** (92.7% false positives on SciCap).
  Use agreement mode, which is the default when text is available.
* **SciCap's image archive is a multi-part (spanned) zip:** Python's `zipfile` cannot read it;
  `scripts/fetch_scicap_subset.py` contains a small reader that can.
* **Without checkpoints**, the system still runs, falling back to rule-based grounding and text-only 3D
  detection, and the `status` fields in its output say so.

## 8. Licences and data sources

* SciCap subset: CC BY-NC-SA 4.0 (Hsu, Giles, Huang, EMNLP Findings 2021); non-commercial use with attribution.
* PDB structures: wwPDB data (CC0); obtained from the MDAnalysisTests and Biopython test suites (`data/raw/pdb/SOURCES.md`).
* EasyOCR weights are downloaded at runtime under their own licence.
