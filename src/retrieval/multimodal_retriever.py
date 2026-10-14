"""
Tri-Modal & Multimodal Retriever Module.
Fuses per-modality similarities (text, 2D visual, 3D, GNN/graph) for figure retrieval.

Fusion rule: score = sum_m w_m * s_m / sum_m w_m over the modalities m that are available for
BOTH the query and the candidate. Missing modalities are dropped and the remaining weights are
renormalised; they are never filled with a constant. 3D similarity only contributes when the
candidate is a 3D figure (``is_3d``), so 3D features never influence ordinary 2D charts.
"""

from typing import Dict, Any, List, Optional
import numpy as np

MODALITIES = ("text", "2d", "3d", "gnn")


class MultimodalRetriever:
    def __init__(self, config: Dict[str, Any] = None, weights: Optional[Dict[str, float]] = None):
        self.config = config or {}
        self.weights = weights or self.config.get("retrieval.multimodal_weights", None) or {
            "w_text": 0.40,
            "w_2d": 0.25,
            "w_3d": 0.20,
            "w_gnn": 0.15
        }

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """
        Cosine similarity clipped to [0, 1]; None if either vector is missing or zero.
        """
        if vec_a is None or vec_b is None:
            return None
        vec_a = np.asarray(vec_a, dtype=np.float32).ravel()
        vec_b = np.asarray(vec_b, dtype=np.float32).ravel()
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return None
        dot = float(np.dot(vec_a / norm_a, vec_b / norm_b))
        return float(round(max(dot, 0.0), 4))

    def retrieve_multimodal(
        self,
        query_text_emb: np.ndarray,
        figure_2d_emb: Optional[np.ndarray] = None,
        figure_3d_emb: Optional[np.ndarray] = None,
        gnn_graph_emb: Optional[np.ndarray] = None,
        candidate_figures: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query-side vectors per modality are compared with the candidate's ``text_embedding``,
        ``2d_embedding``, ``3d_embedding`` and ``gnn_embedding``.
        """
        query = {"text": query_text_emb, "2d": figure_2d_emb, "3d": figure_3d_emb, "gnn": gnn_graph_emb}
        results = []
        for cand in candidate_figures or []:
            is_cand_3d = bool(cand.get("is_3d", False))
            sims: Dict[str, Optional[float]] = {
                "text": self.compute_similarity(query["text"], cand.get("text_embedding")),
                "2d": self.compute_similarity(query["2d"], cand.get("2d_embedding")),
                "3d": self.compute_similarity(query["3d"], cand.get("3d_embedding")) if is_cand_3d else None,
                "gnn": self.compute_similarity(query["gnn"], cand.get("gnn_embedding")),
            }
            num = den = 0.0
            for m in MODALITIES:
                w = float(self.weights.get(f"w_{m}", 0.0))
                if sims[m] is not None and w > 0:
                    num += w * sims[m]
                    den += w
            total_score = num / den if den > 0 else 0.0
            results.append({
                "figure_id": cand.get("figure_id", "fig_cand"),
                "paper_title": cand.get("paper_title", ""),
                "is_3d": is_cand_3d,
                "multimodal_score": float(round(total_score, 4)),
                "_raw_score": float(total_score),
                "modality_breakdown": {
                    "text_similarity": sims["text"],
                    "visual_2d_similarity": sims["2d"],
                    "pointcloud_3d_similarity": sims["3d"],
                    "gnn_graph_similarity": sims["gnn"]
                }
            })
        results.sort(key=lambda x: x["_raw_score"], reverse=True)
        for r in results:
            del r["_raw_score"]
        return results
