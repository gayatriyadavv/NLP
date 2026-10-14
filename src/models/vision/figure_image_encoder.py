"""
Figure image encoder (2D raster branch).

A small CNN that maps a rendered figure image to a 64-d embedding and, through a linear head,
to a figure-type class. It is trained from scratch (no pretrained weights are reachable in the
build environment) and is used for:
  * 3D-figure detection / visualization-type classification (src/vision/threed_detector.py)
  * the image side of figure <-> 3D-geometry contrastive alignment (cross_modal_3d.py)
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

FIGURE_CLASSES: List[str] = ["line", "scatter", "bar", "heatmap", "protein", "surface", "scatter3d"]
THREED_CLASSES = {"protein", "surface", "scatter3d"}
CLASS_TO_VIS_TYPE = {
    "line": "2D_PLOT", "scatter": "2D_PLOT", "bar": "2D_PLOT", "heatmap": "2D_PLOT",
    "protein": "3D_MOLECULAR_STRUCTURE", "surface": "3D_SURFACE_PLOT", "scatter3d": "3D_SCATTER_PLOT",
}
IMAGE_SIZE = 128


def load_figure_tensor(path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    """RGB image -> float tensor [3, size, size] in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class FigureImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 64, num_classes: int = len(FIGURE_CLASSES)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.embed = nn.Linear(64, out_dim)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        """Returns (normalised embedding [B, out_dim], class logits [B, num_classes])."""
        h = self.features(x)
        return F.normalize(self.embed(h), dim=-1), self.classifier(h)
