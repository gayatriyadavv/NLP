"""
GNN Pipeline & Self-Supervised Training Pipeline Module.
Converts NetworkX graphs to PyG, runs self-supervised GNN link prediction training,
and attaches learned node/graph embeddings back to the grounding graph.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.optim as optim
import networkx as nx
import numpy as np

from src.grounding.pyg_converter import PyGGraphConverter
from src.models.grounding.gnn_grounding import GNNGroundingModel

class GNNGroundingPipeline:
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        gnn_type: str = "GAT",
        lr: float = 0.01,
        epochs: int = 30,
        device: str = None
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.lr = lr
        self.epochs = epochs
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.converter = PyGGraphConverter(feature_dim=feature_dim)
        self.model = GNNGroundingModel(
            in_channels=feature_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            gnn_type=gnn_type
        ).to(self.device)

    def train_and_embed(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Runs conversion, self-supervised GNN training, and attaches learned node embeddings to NetworkX graph.
        Returns:
            - graph: NetworkX graph with node['gnn_embedding'] attached
            - metrics: Dict containing loss history and GNN summary
        """
        pyg_data, node2idx, idx2node = self.converter.convert_to_pyg_data(graph)
        
        if pyg_data.num_nodes == 0 or pyg_data.edge_index.numel() == 0:
            # Fallback for empty/single node graphs
            metrics = {
                "epochs_trained": 0,
                "final_loss": 0.0,
                "gnn_type": self.gnn_type,
                "status": "PARTIAL (Insufficient edges for GNN message passing)"
            }
            return graph, metrics

        pyg_data = pyg_data.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Self-Supervised Link Prediction Training Loop
        self.model.train()
        loss_history = []
        
        pos_edge_index = pyg_data.edge_index
        num_pos = pos_edge_index.size(1)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            z = self.model(pyg_data.x, pos_edge_index)

            # Positive edge predictions
            src_pos, tgt_pos = pos_edge_index[0], pos_edge_index[1]
            pos_scores = self.model.predict_link(z[src_pos], z[tgt_pos])
            pos_loss = -torch.log(pos_scores + 1e-8).mean()

            # Sample negative edges for contrastive link prediction
            neg_src = torch.randint(0, pyg_data.num_nodes, (num_pos,), device=self.device)
            neg_tgt = torch.randint(0, pyg_data.num_nodes, (num_pos,), device=self.device)
            neg_scores = self.model.predict_link(z[neg_src], z[neg_tgt])
            neg_loss = -torch.log(1.0 - neg_scores + 1e-8).mean()

            total_loss = pos_loss + neg_loss
            total_loss.backward()
            optimizer.step()

            loss_history.append(float(total_loss.item()))

        # Inference: Extract learned embeddings
        self.model.eval()
        with torch.no_grad():
            z_final = self.model(pyg_data.x, pos_edge_index)
            graph_emb = self.model.get_graph_embedding(z_final).cpu().numpy()[0]

        # Attach GNN embeddings back into NetworkX graph nodes
        z_np = z_final.cpu().numpy()
        for idx, node_id in idx2node.items():
            if graph.has_node(node_id):
                graph.nodes[node_id]["gnn_embedding"] = z_np[idx].tolist()

        # Compute GNN-enhanced element-to-entity grounding similarity scores
        gnn_grounding_pairs = []
        for u, v, d in graph.edges(data=True):
            if u in node2idx and v in node2idx:
                u_idx, v_idx = node2idx[u], node2idx[v]
                sim = float(np.dot(z_np[u_idx], z_np[v_idx]))
                d["gnn_grounding_score"] = float(round(sim, 4))
                gnn_grounding_pairs.append({
                    "source": u,
                    "target": v,
                    "relation": d.get("relation", ""),
                    "gnn_similarity": float(round(sim, 4))
                })

        metrics = {
            "epochs_trained": self.epochs,
            "initial_loss": float(round(loss_history[0], 4)) if loss_history else 0.0,
            "final_loss": float(round(loss_history[-1], 4)) if loss_history else 0.0,
            "gnn_type": self.gnn_type,
            "gnn_grounding_pairs": gnn_grounding_pairs,
            "graph_embedding": graph_emb.tolist(),
            "status": "IMPLEMENTED (PyTorch Geometric GNN Message Passing)"
        }

        return graph, metrics


# ---------------------------------------------------------------------------
# Supervised, inductive grounding (trained once across many figures)
# ---------------------------------------------------------------------------
import time
from torch_geometric.data import Data, Batch

from src.grounding.grounding_features import (
    PAIR_FEATURE_NAMES, TEXT_ELEMENT_TYPES, build_grounding_graph, pair_features,
)
from src.models.grounding.gnn_grounding import PairGroundingModel


class SupervisedGroundingTrainer:
    """
    Trains ``PairGroundingModel`` on labelled element -> entity links from many figures and
    predicts links for unseen figures.

    Unlike ``GNNGroundingPipeline.train_and_embed`` (self-supervised, re-trained per graph on that
    graph's own rule edges), this model is trained once on a training split, its decision
    threshold is chosen on a validation split, and it is evaluated on held-out figures.
    Training time and per-figure inference time are recorded separately.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 64, gnn_type: str = "GAT",
                 use_message_passing: bool = True, lr: float = 0.005, weight_decay: float = 1e-4,
                 epochs: int = 150, seed: int = 42, device: str = "cpu"):
        self.cfg = dict(feature_dim=feature_dim, hidden_dim=hidden_dim, gnn_type=gnn_type,
                        use_message_passing=use_message_passing, lr=lr, weight_decay=weight_decay,
                        epochs=epochs, seed=seed)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.converter = PyGGraphConverter(feature_dim=feature_dim, text_encoding="trigram")
        self.model = PairGroundingModel(in_channels=feature_dim, hidden_channels=hidden_dim,
                                        pair_feature_dim=len(PAIR_FEATURE_NAMES), gnn_type=gnn_type,
                                        use_message_passing=use_message_passing).to(self.device)
        self.threshold = 0.5
        self.train_stats: Dict[str, Any] = {}

    # ---- example construction ------------------------------------------
    def make_example(self, elements: List[Dict[str, Any]], entities: List[Dict[str, Any]],
                     paragraph: str = "", relations: List[Dict[str, Any]] = None,
                     gold: Optional[set] = None, extra_nodes=None) -> Tuple[Data, List[Tuple[str, str]]]:
        """
        gold: set of (element_id, entity_text) positive pairs, or None at inference time.
        Returns (pyg Data with cand_index / cand_feat / y, list of (element_id, entity_text)).
        """
        g = build_grounding_graph(elements, entities, relations, extra_nodes)
        data, node2idx, _ = self.converter.convert_to_pyg_data(g)
        ei = data.edge_index
        if ei.numel():
            data.edge_index = torch.cat([ei, ei.flip(0)], dim=1)
            data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
        cands, idx, feats, ys = [], [], [], []
        for el in elements:
            if el.get("type") not in TEXT_ELEMENT_TYPES:
                continue
            u = node2idx.get(f"elem:{el['id']}")
            for en in entities:
                v = node2idx.get(f"ent:{en['text']}")
                if u is None or v is None:
                    continue
                rule = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                cands.append((el["id"], en["text"]))
                idx.append([u, v])
                feats.append(pair_features(el, en, paragraph, rule))
                ys.append(float((el["id"], en["text"]) in gold) if gold is not None else 0.0)
        data.cand_index = torch.tensor(idx, dtype=torch.long).t().contiguous() if idx else torch.empty((2, 0), dtype=torch.long)
        data.cand_feat = torch.tensor(np.array(feats), dtype=torch.float32) if feats else torch.empty((0, len(PAIR_FEATURE_NAMES)))
        data.y = torch.tensor(ys, dtype=torch.float32)
        return data, cands

    # ---- training ---------------------------------------------------------
    def _scores(self, batch: Batch) -> torch.Tensor:
        return torch.sigmoid(self.model(batch.x, batch.edge_index, batch.cand_index, batch.cand_feat))

    def fit(self, train: List[Tuple[Data, List]], val: List[Tuple[Data, List]]) -> Dict[str, Any]:
        torch.manual_seed(self.cfg["seed"])
        batch = Batch.from_data_list([d for d, _ in train]).to(self.device)
        pos = float(batch.y.sum())
        pos_weight = torch.tensor((len(batch.y) - pos) / max(pos, 1.0), device=self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        history = []
        t0 = time.perf_counter()
        self.model.train()
        for _ in range(self.cfg["epochs"]):
            opt.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.cand_index, batch.cand_feat)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
        train_time = time.perf_counter() - t0
        self.threshold = self._tune_threshold(val)
        self.train_stats = {
            "train_time_s": round(train_time, 3), "epochs": self.cfg["epochs"],
            "initial_loss": round(history[0], 4), "final_loss": round(history[-1], 4),
            "num_train_graphs": len(train), "num_train_candidates": int(len(batch.y)),
            "num_train_positives": int(pos), "threshold": self.threshold,
        }
        return self.train_stats

    def _tune_threshold(self, val: List[Tuple[Data, List]]) -> float:
        best_t, best_f1 = 0.5, -1.0
        preds = [self.predict_scores(d) for d, _ in val]
        for t in np.linspace(0.05, 0.95, 19):
            tp = fp = fn = 0
            for (d, cands), s in zip(val, preds):
                chosen = set(self.decode(cands, s, float(t)))
                gold = {c for c, y in zip(cands, d.y.tolist()) if y > 0.5}
                tp += len(chosen & gold)
                fp += len(chosen - gold)
                fn += len(gold - chosen)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_t, best_f1 = float(round(t, 2)), f1
        return best_t

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def predict_scores(self, data: Data) -> np.ndarray:
        self.model.eval()
        if data.cand_index.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        b = Batch.from_data_list([data]).to(self.device)
        return self._scores(b).cpu().numpy()

    @staticmethod
    def decode(cands: List[Tuple[str, str]], scores: np.ndarray, threshold: float) -> List[Tuple[str, str]]:
        """At most one entity per element: the arg-max, if its score clears the threshold."""
        best: Dict[str, Tuple[float, str]] = {}
        for (eid, ent), s in zip(cands, scores):
            if s >= threshold and (eid not in best or s > best[eid][0]):
                best[eid] = (float(s), ent)
        return [(eid, ent) for eid, (_, ent) in best.items()]

    # ---- persistence ------------------------------------------------------
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"cfg": self.cfg, "state_dict": self.model.state_dict(), "threshold": self.threshold,
                    "train_stats": self.train_stats}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SupervisedGroundingTrainer":
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["cfg"]
        t = cls(feature_dim=c["feature_dim"], hidden_dim=c["hidden_dim"], gnn_type=c["gnn_type"],
                use_message_passing=c["use_message_passing"], lr=c["lr"], weight_decay=c["weight_decay"],
                epochs=c["epochs"], seed=c["seed"], device=device)
        t.model.load_state_dict(ck["state_dict"])
        t.threshold = ck["threshold"]
        t.train_stats = ck.get("train_stats", {})
        return t
