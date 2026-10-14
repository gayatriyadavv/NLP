"""
Multi-View 3D Representation Module.

Two input regimes, kept explicitly separate:

1. Geometry available (point cloud) -> ``render_views_from_points``: the points are rotated to
   ``num_views`` viewpoints and orthographically projected into 3-channel images
   (occupancy, nearest depth, farthest depth). This is a genuine multi-view representation of
   3D geometry (MVCNN-style), encoded by a shared CNN and max-pooled across views.

2. Only a 2D figure image available -> ``generate_views``: crops / rotation of that single
   image. These are NOT views of 3D content; a 2D screenshot does not give access to the
   underlying geometry. Kept for backward compatibility with the API.

The CNN weights are trained by ``src/models/vision/cross_modal_3d.py``; an untrained instance
produces embeddings with no learned meaning.
"""

import os
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

VIEW_SIZE = 64


def _rot(elev_deg: float, azim_deg: float) -> np.ndarray:
    e, a = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return (rx @ rz).astype(np.float32)


def render_views_from_points(points: np.ndarray, num_views: int = 6, size: int = VIEW_SIZE,
                             elevation: float = 30.0) -> List[np.ndarray]:
    """
    Orthographic multi-view rendering of a point cloud normalised to the unit ball.
    Returns ``num_views`` arrays of shape [size, size, 3] in [0, 1]:
    channel 0 = occupancy, 1 = nearest depth, 2 = farthest depth.
    """
    pts = np.asarray(points, dtype=np.float32)
    pts = pts - pts.mean(0, keepdims=True)
    pts = pts / max(np.linalg.norm(pts, axis=1).max(), 1e-8)
    views = []
    for k in range(num_views):
        p = pts @ _rot(elevation, 360.0 * k / num_views).T
        u = np.clip(((p[:, 0] + 1) / 2 * (size - 1)).round().astype(int), 0, size - 1)
        v = np.clip(((1 - (p[:, 1] + 1) / 2) * (size - 1)).round().astype(int), 0, size - 1)
        d = (p[:, 2] + 1) / 2
        img = np.zeros((size, size, 3), dtype=np.float32)
        near = np.ones((size, size), dtype=np.float32)
        far = np.zeros((size, size), dtype=np.float32)
        np.add.at(img[:, :, 0], (v, u), 1.0)
        np.minimum.at(near, (v, u), d)
        np.maximum.at(far, (v, u), d)
        occ = img[:, :, 0] > 0
        img[:, :, 0] = np.clip(img[:, :, 0] / 4.0, 0, 1)
        img[:, :, 1] = np.where(occ, 1 - near, 0)
        img[:, :, 2] = np.where(occ, far, 0)
        views.append(img)
    return views


class MultiViewRenderer(nn.Module):
    def __init__(self, num_views: int = 4, out_dim: int = 64):
        super(MultiViewRenderer, self).__init__()
        self.num_views = num_views
        self.out_dim = out_dim

        # Lightweight conv encoder shared across views
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # Fused multi-view linear head: (64 * 16) -> out_dim (64)
        self.fc_head = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    # ---- regime 1: genuine multi-view of geometry ------------------------
    def forward_views(self, views: torch.Tensor) -> torch.Tensor:
        """views: [B, V, 3, H, W] -> normalised embedding [B, out_dim] (max-pool over views)."""
        B, V = views.shape[:2]
        h = self.slice_encoder(views.flatten(0, 1)).view(B, V, -1)
        return F.normalize(self.fc_head(h.max(dim=1)[0]), p=2, dim=-1)

    @staticmethod
    def views_tensor_from_points(points: np.ndarray, num_views: int = 6) -> torch.Tensor:
        views = render_views_from_points(points, num_views=num_views)
        return torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2)  # [V, 3, H, W]

    def extract_embedding_from_points(self, points: np.ndarray, num_views: int = 6) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            v = self.views_tensor_from_points(points, num_views).unsqueeze(0)
            return self.forward_views(v).cpu().numpy()[0]

    # ---- regime 2: crops of a single 2D image (not 3D views) -------------
    def generate_views(self, figure_path: str) -> List[np.ndarray]:
        """
        Generates ``num_views`` crops/rotations of a 2D figure image. These are not views of
        3D geometry. Raises FileNotFoundError if the image does not exist.
        """
        if not os.path.exists(figure_path):
            raise FileNotFoundError(f"Figure image not found: {figure_path}")
        img = Image.open(figure_path).convert("RGB")
        w, h = img.size
        v1 = img.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))).resize((VIEW_SIZE, VIEW_SIZE))
        v2 = img.crop((0, 0, int(w*0.6), int(h*0.6))).resize((VIEW_SIZE, VIEW_SIZE))
        v3 = img.crop((int(w*0.4), int(h*0.4), w, h)).resize((VIEW_SIZE, VIEW_SIZE))
        v4 = img.rotate(90).crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))).resize((VIEW_SIZE, VIEW_SIZE))
        return [np.array(v, dtype=np.float32) / 255.0 for v in [v1, v2, v3, v4]][: self.num_views]

    def extract_multiview_embedding(self, figure_path: str) -> np.ndarray:
        """Embedding of image crops (regime 2). See module docstring for limitations."""
        views = self.generate_views(figure_path)
        v = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            return self.forward_views(v).cpu().numpy()[0]
