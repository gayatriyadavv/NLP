| Config | Grounding F1 | Grounding Acc@1 | Grounding MRR | Caption->Fig R@1 | R@5 | MRR | Geometry->Fig R@1 | SCQ | Train time (s) | Inference ms/fig |
|---|---|---|---|---|---|---|---|---|---|---|
| A NetworkX rules | 0.3810 | 0.3200 | 0.4284 | 0.8051 | 0.9322 | 0.8598 | N/A | 0.5446 | 0.0 | 0.125 |
| B NetworkX + PyG GNN | 0.9054 ± 0.0034 | 0.8958 ± 0.0035 | 0.8981 ± 0.0019 | 0.8305 | 0.9407 | 0.8761 | N/A | 0.5689 | 4.459 | 4.47 |
| C GNN + CLIP | BLOCKED: CLIP weights 'openai/clip-vit-base-patch32' could n | | | | | | | | | |
| D GNN + 3D branch | 0.9086 ± 0.0012 | 0.8966 ± 0.0012 | 0.8987 ± 0.0006 | 0.8305 | 0.9407 | 0.8768 | 0.093 | 0.5706 | 4.523 | 4.106 |
| E Full multimodal (text + 2D + 3D + GNN) | 0.9052 ± 0.0025 | 0.8975 ± 0.0012 | 0.8991 ± 0.0006 | 0.8305 | 0.9407 | 0.8806 | 0.093 | 0.5715 | 4.66 | 5.443 |
