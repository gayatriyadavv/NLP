# Project Status — What Is Done and What Is Not

Last updated: 2026-10-14 · Branch `claude/scientific-figure-understanding-9i8ve7`

A plain-language summary of where the project stands. Every number comes from a file in
`results/` produced by a script in `pipelines/`; detailed tables are in
[`implementation_plan.md`](../implementation_plan.md). Numbers reported before the
2026-10-08 audit ([`AUDIT.md`](AUDIT.md)) were hard-coded and must not be cited.

**Legend:** ✅ done and measured · 🟡 works but limited / prototype · ❌ not done · ⛔ blocked by the environment

---

## 1. One-paragraph summary

The full pipeline runs end to end — PDF/figure in, OCR'd figure elements, scientific entities,
grounding graph, GNN, 3D branch, retrieval, caption, verification, SCQ — through a script, a
FastAPI backend and a React UI, with 39 passing tests and CI on GitHub. It has been evaluated on a
400-figure controlled benchmark (with real protein structures for the 3D figures) and on **287 real
arXiv figures from SciCap**. On real figures, text retrieval works well (MRR 0.90) and OCR text adds
a small but significant gain; the generated captions are poor (template-based), entity extraction
misses many entities, and grounding accuracy on real figures **cannot be measured yet** because no
labelled data exists. The 3D branch works only on synthetic data; its image classifier fails on
real figures, so 3D is now only declared when image and text agree.

---

## 2. What is done

| Area | Status | Evidence / key numbers |
|---|---|---|
| Repository audit, fabricated results removed | ✅ | `docs/AUDIT.md`; regression tests forbid hard-coded metric offsets |
| Controlled benchmark (SciFig-Controlled) | ✅ | 400 figures, exact gold labels; 3D molecules from 16 real PDB structures, structure-disjoint split — `docs/BENCHMARK.md` |
| Real-data subset (SciCap) | ✅ | 287 real arXiv graph plots with author captions and paper paragraphs (`data/scicap_subset`, CC BY-NC-SA 4.0); fetched with `scripts/fetch_scicap_subset.py` |
| OCR figure-element extraction | ✅ | Synthetic: text F1 0.81, type accuracy 0.86. Real: 9.1 text elements / figure, 19 of 287 figures with no text found; ~2.8 s / figure on CPU |
| Entity grounding (rules vs GNN) | ✅ on synthetic | End-to-end F1 0.38 (rules) → 0.91 (GNN), significant (+0.525, 95% CI [+0.50, +0.55]); unseen entity names F1 0.83 |
| Message-passing diagnostic | ✅ | Same model without message passing scores the same → the gain comes from the learned pairwise features, not the graph |
| 3D detection | ✅ with caveat | Synthetic test: 100%. Real graph plots: image-only CNN wrongly calls 92.7% of them 3D; default "agreement" mode (image + text) lowers that to 2.4% — `results/threed_detector_modes.json` |
| 3D geometry encoders (PointNet, multi-view) | ✅ on synthetic | Figure ↔ structure retrieval R@1 0.14–0.16 (3 seeds) vs 0.02 chance, on unseen protein structures |
| Multimodal retrieval | ✅ | Real SciCap caption → figure: MRR 0.897 (text), 0.919 with OCR (+0.021, CI [+0.003, +0.041]); caption-leak in SciCap paragraphs detected and removed |
| Ablation A–E with significance tests | ✅ | 3D / full multimodal add nothing detectable over the GNN (all CIs contain 0) |
| Statistics | ✅ | Multi-seed means ± std; paired bootstrap CIs (`src/evaluation/bootstrap.py`) |
| Backend API | ✅ | Each sample runs on its own text; path-restricted figure-image endpoint; honest status fields |
| Frontend analysis view | ✅ | Shows the real document, figure image, OCR boxes and grounding links; banner when mock data is shown |
| CI | ✅ | GitHub Actions runs backend tests and frontend build on every push (all green) |
| Tests | ✅ | 39 passing (1 skipped without trained checkpoints) |

## 3. What works but is limited

| Area | Status | Limitation |
|---|---|---|
| PDF ingestion | 🟡 | Every text line becomes a "paragraph"; figures are not extracted from PDFs |
| Scientific NER | 🟡 | Regex over a fixed vocabulary; **found no entity in 45% (130/287) of real SciCap figures** |
| Caption generation | 🟡 | Fill-in-the-blanks template, no vision-language model. Real figures: BLEU-4 0.05, ROUGE-L 0.08 against author captions; often "Insufficient evidence" |
| Claim verification | 🟡 | Word-overlap + numeric-conflict check, no NLI model; real-figure technical correctness 0.26 |
| SCQ metric | 🟡 | Computed correctly, but on real data it is 0.30 — mostly reflects the weak captions |
| OCR quality | 🟡 | Real text is often garbled ("Accuracy Score" → "Accuangy Scine") |
| 3D alignment | 🟡 | Clearly above chance but weak (R@1 ≈ 0.15); trained on 137 synthetic figures |
| Overlay alignment fix in UI | 🟡 | Builds; not visually re-checked after the fix |

## 4. What is not done

| Item | Status | Why / what is needed |
|---|---|---|
| Grounding accuracy on **real** figures | ❌ | SciCap has no element → entity labels. Needs ~100 hand-annotated figures |
| Any evaluation on **real 3D** scientific figures | ❌ | SciCap contains graph plots only; no labelled real 3D figure set is available |
| Cross-Modal Search tab | ❌ | Still shows hard-coded example results; not wired to the retrieval code |
| Figure extraction from PDFs | ❌ | Figures must be supplied as separate images |
| Useful message passing in the GNN | ❌ | Currently contributes nothing measurable; needs a redesign (e.g. cross-figure graphs) or removal |
| Uploading your own PDF/figure in the UI | ❌ | UI runs the bundled demo samples; upload widgets do not send files to the backend |

## 5. What is blocked by the environment

| Item | Status | Unblock by |
|---|---|---|
| CLIP (ablation configuration C) | ⛔ | Allow `huggingface.co` in the cloud environment's network settings (code is ready: `src/models/vision/clip_encoder.py`) |
| Sentence-Transformers retrieval, SciBERT NER, NLI verification | ⛔ | Same — models are configured but cannot be downloaded |
| Downloading data directly in the cloud session | ⛔ | Same; SciCap was fetched on a local machine instead |

## 6. How to run

```bash
pip install -r requirements.txt
python -m src.data.benchmark_generator --out data/benchmark   # synthetic benchmark
python pipelines/evaluate_phase4.py                           # train 3D models (~15 min CPU)
python pipelines/evaluate_gnn.py                              # train/evaluate grounding (~10 min)
python pipelines/run_ablation.py                              # ablation A–E (~20 min)
python pipelines/evaluate_scicap.py                           # real SciCap figures (~15 min)
python pipelines/evaluate_3d_detector_modes.py                # 3D detector modes (~1 min)
python -m pytest tests/                                       # tests
uvicorn api.main:app --port 8000                              # backend
cd frontend && npm install && npm run dev                     # UI at http://localhost:5173
```

## 7. Recommended next steps (in order)

1. **Annotate ~100 SciCap figures** with element → entity links so grounding can be measured on real data.
2. **Allow HuggingFace** in the environment to enable CLIP, Sentence-Transformers, SciBERT and NLI, then re-run the evaluations.
3. **Replace the caption template** with a vision-language model (needs step 2).
4. **Improve NER** (SciBERT or a trained model) — 45% of real figures currently get no entities.
5. Wire the **Cross-Modal Search** tab to the real retrieval code.
6. Find or build a small **labelled set of real 3D figures** before making any claim about 3D on real papers.
