"""
PointNet Point-Cloud Feature Extraction Module.
Processes 3D point cloud spatial coordinates (N x 3) and extracts a symmetric global 3D feature embedding (64-D).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

class PointNetExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super(PointNetExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Shared MLP 1: 3 -> 64 -> 128
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Shared MLP 2: 128 -> 256 -> 512
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Output Feature Projection Head: 512 -> 256 -> out_dim (64)
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D PointNet feature extraction.
        Input:
            pts: Tensor of shape [B, N, 3] or [N, 3]
        Returns:
            z_3d: Normalized 3D point-cloud embedding [B, out_dim]
        """
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)

        B, N, C = pts.shape
        x = pts.transpose(1, 2)

        x = self.mlp1(x)
        x = self.mlp2(x)

        global_feat = torch.max(x, 2)[0]

        out_emb = self.fc_head(global_feat)
        z_3d = F.normalize(out_emb, p=2, dim=-1)
        return z_3d

    def extract_point_cloud_embedding(self, point_array: np.ndarray) -> np.ndarray:
        """
        Convenience function taking numpy array [N, 3] and returning numpy vector [64].
        """
        self.eval()
        with torch.no_grad():
            pts_tensor = torch.tensor(point_array, dtype=torch.float32)
            z_3d = self.forward(pts_tensor)
            return z_3d.cpu().numpy()[0]

def load_point_cloud_fixture(filepath: str, num_points: int = None) -> np.ndarray:
    """
    Loads 3D point coordinates [N, 3] from a ``.json`` file ({"points": [[x, y, z], ...]}) or a
    ``.npy`` array. Raises FileNotFoundError / ValueError instead of fabricating geometry.
    If ``num_points`` is given, the cloud is resampled to exactly that many points (deterministic).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")
    if filepath.endswith(".npy"):
        pts = np.load(filepath).astype(np.float32)
    elif filepath.endswith(".json"):
        import json
        with open(filepath, "r") as f:
            pts = np.array(json.load(f).get("points", []), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {filepath}")
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"Expected an [N, 3] point array in {filepath}, got shape {pts.shape}")
    if num_points is not None and len(pts) != num_points:
        idx = np.linspace(0, len(pts) - 1, num_points).round().astype(int)
        pts = pts[idx]
    return pts


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3D rotation (QR decomposition of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)
