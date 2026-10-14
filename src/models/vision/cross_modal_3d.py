"""
Training and evaluation of the Phase 4 3D branch on SciFig-Controlled.

1. ``train_figure_classifier``: FigureImageEncoder trained with cross-entropy on 7 figure classes
   (4 x 2D, 3 x 3D). Used by ThreeDVisualizationDetector.
2. ``train_alignment``: contrastive (InfoNCE) alignment between the figure image encoder and two
   geometry encoders - PointNet (point cloud) and MultiViewRenderer (rendered views of the point
   cloud). Figures that depict the same geometry (several views of one PDB structure) are treated
   as positives of each other, not as negatives.
3. ``evaluate_alignment``: figure-image -> geometry retrieval (Recall@1/5, MRR) over the unique
   test geometries.

Everything is trained from scratch on CPU; no pretrained weights are used.
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.vision.figure_image_encoder import (
    FIGURE_CLASSES, FigureImageEncoder, load_figure_tensor,
)
from src.models.vision.pointnet_extractor import PointNetExtractor, random_rotation_matrix
from src.vision.multiview_renderer import MultiViewRenderer, render_views_from_points

NUM_VIEWS = 6


def geometry_id(fig: Dict[str, Any]) -> str:
    """Figures rendered from the same PDB structure share one geometry id."""
    return fig.get("pdb_id") or fig["figure_id"]


@torch.no_grad()
def recalibrate_batchnorm(model: torch.nn.Module, forward_fn, batches) -> None:
    """
    "Precise BN": recompute BatchNorm running statistics as an exact average over the training
    data after optimisation. Without this, momentum-averaged running stats from few, unevenly sized
    batches made eval-mode accuracy swing between seeds (e.g. 100% vs 71% on validation with
    identical weights in train mode).
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    model.train()
    for b in batches:
        forward_fn(b)
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


def load_images(root: str, figs: List[Dict[str, Any]]) -> torch.Tensor:
    return torch.stack([load_figure_tensor(os.path.join(root, f["image_path"])) for f in figs])


# ---------------------------------------------------------------------------
# 1. Figure-type classifier (3D detection)
# ---------------------------------------------------------------------------
def train_figure_classifier(root: str, train_figs: List[Dict[str, Any]], epochs: int = 25,
                            batch_size: int = 32, lr: float = 1e-3, seed: int = 42
                            ) -> Tuple[FigureImageEncoder, Dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = load_images(root, train_figs)
    y = torch.tensor([FIGURE_CLASSES.index(f["figure_type"]) for f in train_figs])
    model = FigureImageEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    losses = []
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        ep = []
        for i in range(0, len(X), batch_size):
            idx = torch.from_numpy(perm[i:i + batch_size])
            _, logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep.append(loss.item())
        losses.append(float(np.mean(ep)))
    recalibrate_batchnorm(model, lambda b: model(b), [X[i:i + batch_size] for i in range(0, len(X), batch_size)])
    return model, {"train_time_s": round(time.perf_counter() - t0, 2), "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "optimizer": "Adam", "weight_decay": 1e-4,
                   "precise_bn": True,
                   "initial_loss": round(losses[0], 4), "final_loss": round(losses[-1], 4),
                   "num_train": len(train_figs)}


@torch.no_grad()
def evaluate_figure_classifier(model: FigureImageEncoder, root: str, figs: List[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    X = load_images(root, figs)
    t0 = time.perf_counter()
    _, logits = model(X)
    per_fig_ms = (time.perf_counter() - t0) * 1000 / len(figs)
    pred = logits.argmax(1).numpy()
    gold = np.array([FIGURE_CLASSES.index(f["figure_type"]) for f in figs])
    is3d_pred = np.isin(pred, [FIGURE_CLASSES.index(c) for c in ("protein", "surface", "scatter3d")])
    is3d_gold = np.array([f["is_3d"] for f in figs])
    tp = int((is3d_pred & is3d_gold).sum())
    fp = int((is3d_pred & ~is3d_gold).sum())
    fn = int((~is3d_pred & is3d_gold).sum())
    return {
        "figure_type_accuracy": round(float((pred == gold).mean()), 4),
        "is_3d_accuracy": round(float((is3d_pred == is3d_gold).mean()), 4),
        "is_3d_precision": round(tp / max(tp + fp, 1), 4),
        "is_3d_recall": round(tp / max(tp + fn, 1), 4),
        "num_test": len(figs),
        "inference_ms_per_figure_batched": round(per_fig_ms, 3),
    }


# ---------------------------------------------------------------------------
# 2. Figure image <-> 3D geometry contrastive alignment
# ---------------------------------------------------------------------------
class CrossModal3DAligner(torch.nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.image_encoder = FigureImageEncoder(out_dim=dim)
        self.point_encoder = PointNetExtractor(out_dim=dim)
        self.view_encoder = MultiViewRenderer(num_views=NUM_VIEWS, out_dim=dim)
        self.log_temp = torch.nn.Parameter(torch.tensor(np.log(1 / 0.1), dtype=torch.float32))

    def encode_image(self, x):
        return self.image_encoder(x)[0]


def _multi_positive_nce(a: torch.Tensor, b: torch.Tensor, gids: List[str], scale: torch.Tensor) -> torch.Tensor:
    logits = scale * a @ b.t()
    same = torch.tensor([[gi == gj for gj in gids] for gi in gids], dtype=torch.float32)
    target = same / same.sum(1, keepdim=True)
    return 0.5 * (-(target * logits.log_softmax(1)).sum(1).mean()
                  - (target.t() * logits.t().log_softmax(1)).sum(1).mean())


def _points_batch(geoms: List[np.ndarray], rng, augment: bool):
    pts, views = [], []
    for g in geoms:
        p = g @ random_rotation_matrix(rng).T if augment else g
        pts.append(torch.from_numpy(p.astype(np.float32)))
        views.append(MultiViewRenderer.views_tensor_from_points(p, NUM_VIEWS))
    return torch.stack(pts), torch.stack(views)


def train_alignment(root: str, train_figs: List[Dict[str, Any]], epochs: int = 40, batch_size: int = 32,
                    lr: float = 1e-3, seed: int = 42, augment_rotation: bool = True
                    ) -> Tuple[CrossModal3DAligner, Dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    figs = [f for f in train_figs if f["is_3d"] and f.get("geometry_path")]
    X = load_images(root, figs)
    geoms = [np.load(os.path.join(root, f["geometry_path"])) for f in figs]
    gids = [geometry_id(f) for f in figs]
    model = CrossModal3DAligner()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    losses = []
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(figs))
        ep = []
        for i in range(0, len(figs), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            img = model.encode_image(X[torch.from_numpy(idx)])
            pts, views = _points_batch([geoms[j] for j in idx], rng, augment_rotation)
            zp = model.point_encoder(pts)
            zv = model.view_encoder.forward_views(views)
            scale = model.log_temp.exp().clamp(max=100)
            g = [gids[j] for j in idx]
            loss = _multi_positive_nce(img, zp, g, scale) + _multi_positive_nce(img, zv, g, scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep.append(loss.item())
        losses.append(float(np.mean(ep)))
    order = list(range(len(figs)))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size) if len(order[i:i + batch_size]) > 1]

    def _fwd(idx):
        model.encode_image(X[torch.tensor(idx)])
        pts, views = _points_batch([geoms[j] for j in idx], None, augment=False)
        model.point_encoder(pts)
        model.view_encoder.forward_views(views)

    recalibrate_batchnorm(model, _fwd, chunks)
    return model, {"train_time_s": round(time.perf_counter() - t0, 2), "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "optimizer": "Adam", "weight_decay": 1e-4,
                   "precise_bn": True, "rotation_augmentation": augment_rotation, "num_views": NUM_VIEWS,
                   "initial_loss": round(losses[0], 4), "final_loss": round(losses[-1], 4),
                   "num_train_3d_figures": len(figs), "num_train_geometries": len(set(gids))}


def _retrieval_metrics(sim: np.ndarray, gold_cols: List[int]) -> Dict[str, float]:
    ranks = []
    for i, g in enumerate(gold_cols):
        order = np.argsort(-sim[i])
        ranks.append(int(np.where(order == g)[0][0]) + 1)
    ranks = np.array(ranks)
    return {"recall@1": round(float((ranks <= 1).mean()), 4), "recall@5": round(float((ranks <= 5).mean()), 4),
            "mrr": round(float((1.0 / ranks).mean()), 4)}


@torch.no_grad()
def evaluate_alignment(model: CrossModal3DAligner, root: str, test_figs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Figure image -> geometry retrieval over the unique test geometries (no rotation at test time)."""
    model.eval()
    figs = [f for f in test_figs if f["is_3d"] and f.get("geometry_path")]
    gal_ids, gal_geoms = [], []
    for f in figs:
        gid = geometry_id(f)
        if gid not in gal_ids:
            gal_ids.append(gid)
            gal_geoms.append(np.load(os.path.join(root, f["geometry_path"])))
    X = load_images(root, figs)
    t0 = time.perf_counter()
    zi = model.encode_image(X).numpy()
    t_img = (time.perf_counter() - t0) * 1000 / len(figs)
    pts, views = _points_batch(gal_geoms, None, augment=False)
    t0 = time.perf_counter()
    zp = model.point_encoder(pts).numpy()
    t_pts = (time.perf_counter() - t0) * 1000 / len(gal_geoms)
    t0 = time.perf_counter()
    zv = model.view_encoder.forward_views(views).numpy()
    t_mv = (time.perf_counter() - t0) * 1000 / len(gal_geoms)
    gold = [gal_ids.index(geometry_id(f)) for f in figs]
    by_type = {}
    for name, z in (("pointnet", zp), ("multiview", zv), ("pointnet+multiview", zp + zv)):
        sim = zi @ z.T
        res = _retrieval_metrics(sim, gold)
        for t in ("protein", "surface", "scatter3d"):
            rows = [i for i, f in enumerate(figs) if f["figure_type"] == t]
            if rows:
                res[f"{t}_recall@1"] = _retrieval_metrics(sim[rows], [gold[i] for i in rows])["recall@1"]
        by_type[name] = res
    return {
        "num_queries": len(figs), "gallery_size": len(gal_ids),
        "chance_recall@1": round(1 / len(gal_ids), 4),
        "results": by_type,
        "inference_ms": {"image_encoder_per_figure": round(t_img, 3),
                         "pointnet_per_geometry": round(t_pts, 3),
                         "multiview_per_geometry_excl_render": round(t_mv, 3)},
    }


def save_checkpoint(obj: torch.nn.Module, stats: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": obj.state_dict(), "stats": stats}, path)
