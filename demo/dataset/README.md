# Demo samples (synthetic placeholders)

These four samples follow the SciCap schema but are **not SciCap data**. The images are
placeholder drawings without axis labels or legend text, and the captions/paragraphs were
written for the demo. They exist so the API (`/scicap/*`) and the frontend have something to
show. Do not report metrics computed on them.

`3d_samples/alphafold_t1087_3d.json` is a **synthetic helix** point cloud, not AlphaFold output.

Quantitative experiments use the controlled benchmark in `data/benchmark/`
(see `docs/BENCHMARK.md`), which uses real PDB geometry for its molecular 3D figures.

| ID | Placeholder for | Figure type |
|---|---|---|
| scicap_001 | ViT vs ResNet accuracy curves | 2D line plot |
| scicap_002 | Transformer attention heatmap | 2D heatmap |
| scicap_003 | Protein structure render | 3D (placeholder drawing is a flat ellipse) |
| scicap_004 | BatchNorm loss vs learning rate | scatter plot |
