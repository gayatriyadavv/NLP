# Multi-Modal Scientific Visualization Understanding with Hierarchical Context Grounding and Cross-Modal Retrieval

Research prototype for scientific figure understanding: document hierarchy extraction,
scientific NER, 4-level hierarchical retrieval, OCR-based figure element extraction,
graph-based entity grounding (NetworkX + PyTorch Geometric GNN), 3D-figure detection and
figure ↔ 3D-geometry alignment, grounded caption generation, claim verification and the SCQ
composite metric.

**Continuing the project? Start with [`HANDOVER.md`](HANDOVER.md).**
**What is done / not done (plain-language summary): [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).**
**Per-component status, measured results and limitations: [`implementation_plan.md`](implementation_plan.md).**
The 2026-10-08 audit that preceded the current results is in [`docs/AUDIT.md`](docs/AUDIT.md);
results reported before that audit were not measurements and have been removed.

## System Architecture

```
Paper PDF / Image / Query
         │
         ├── Document Ingestion (PyMuPDF) -> Paper -> Section -> Paragraph
         ├── Scientific NER (regex prototype) -> METHOD, METRIC, DATASET, VARIABLE, ...
         ├── 4-Level Hierarchical Retrieval (Sentence-Transformers + FAISS, word-overlap fallback)
         ├── Figure Processing (EasyOCR + layout heuristics) -> AXIS / LEGEND / LABEL / CHART_REGION
         ├── 3D Branch: CNN 3D detector, PointNet + multi-view geometry encoders aligned to figure images
         ├── Entity Grounding Graph (NetworkX) + supervised PyG GAT grounding
         ├── Multimodal Retrieval (text / 2D / 3D / graph fusion, missing modalities renormalised)
         ├── Grounded Caption Generator (evidence-slot template, no VLM)
         ├── Claim Verification (lexical + numeric conflict; NLI when a model is available)
         └── SCQ Composite Evaluation
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Benchmark (synthetic figures; molecular 3D figures from real PDB structures)
python -m src.data.benchmark_generator --out data/benchmark --seed 42

# 2. Train + evaluate (writes models/checkpoints/* and results/*.json)
python pipelines/evaluate_phase4.py      # 3D detector + figure<->geometry alignment
python pipelines/evaluate_gnn.py         # grounding: rules vs GNN vs no-message-passing diagnostic
python pipelines/run_ablation.py         # configurations A-E

# 3. Tests
python -m pytest tests/

# 4. Pipeline / API / UI
python pipelines/run_pipeline.py --paper demo/sample_paper.pdf --figure demo/sample_figure.png --query "What does this figure show?"
uvicorn api.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

The demo samples in `demo/dataset/` are synthetic placeholders for the UI; do not report
metrics on them.
