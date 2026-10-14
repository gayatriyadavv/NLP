"""
Entity Grounding Graph Module
Constructs NetworkX graph connecting Figure Elements -> Scientific Entities -> Scientific Claims,
and integrates PyTorch Geometric GNN message passing for learned representations.

GNN modes (``gnn_metrics["mode"]``):
* "supervised": a trained grounding checkpoint (default models/checkpoints/grounding_gnn.pt,
  produced by pipelines/evaluate_gnn.py) predicts element -> entity links. These replace the
  rule-based element -> entity edges (rule edges are counted in gnn_metrics). This is the model
  evaluated in results/gnn_grounding_results.json.
* "self_supervised_embedding_only": no checkpoint available; the legacy per-graph link-prediction
  pipeline attaches embeddings but does not change any link.
"""

import os
import time
from typing import Dict, Any, List, Optional
import networkx as nx
from src.grounding.gnn_pipeline import GNNGroundingPipeline, SupervisedGroundingTrainer

DEFAULT_GROUNDING_CHECKPOINT = "models/checkpoints/grounding_gnn.pt"
_TRAINER_CACHE: Dict[str, SupervisedGroundingTrainer] = {}


def _relation_for(element_type: str, entity_type: str) -> str:
    if element_type in ("LEGEND", "LABEL") and entity_type in ("METHOD", "MODEL"):
        return "CORRESPONDS_TO"
    return "REPRESENTS"


class EntityGroundingGraph:
    def __init__(self, config: Dict[str, Any] = None, checkpoint_path: Optional[str] = None):
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.enable_gnn = self.config.get("ablation_switches.graph_enabled", True)
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else self.config.get(
            "models.grounding.checkpoint", DEFAULT_GROUNDING_CHECKPOINT)
        self.gnn_pipeline = None
        self.supervised: Optional[SupervisedGroundingTrainer] = None
        if self.enable_gnn:
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                if self.checkpoint_path not in _TRAINER_CACHE:
                    _TRAINER_CACHE[self.checkpoint_path] = SupervisedGroundingTrainer.load(self.checkpoint_path)
                self.supervised = _TRAINER_CACHE[self.checkpoint_path]
            else:
                self.gnn_pipeline = GNNGroundingPipeline()
        self.gnn_metrics: Dict[str, Any] = {}

    def build_graph(
        self, 
        figure_elements: List[Dict[str, Any]], 
        entities: List[Dict[str, Any]], 
        relations: List[Dict[str, Any]] = None,
        run_gnn: bool = True,
        paragraph: str = ""
    ) -> nx.DiGraph:
        """
        Builds a grounding graph linking figure visual elements to scientific entities and paper claims.
        Optionally enhances the graph using PyTorch Geometric GNN message-passing.
        """
        self.graph.clear()
        
        # Add figure element nodes
        for elem in figure_elements:
            node_id = f"elem:{elem['id']}"
            self.graph.add_node(
                node_id, 
                label=elem['text'], 
                node_type="FIGURE_ELEMENT", 
                element_type=elem['type']
            )

        # Add entity nodes
        for ent in entities:
            node_id = f"ent:{ent['text']}"
            self.graph.add_node(
                node_id, 
                label=ent['text'], 
                node_type="SCIENTIFIC_ENTITY", 
                entity_type=ent['type']
            )

        # Create grounding edges (FIGURE_ELEMENT -> REPRESENTS / CORRESPONDS_TO -> SCIENTIFIC_ENTITY)
        for elem in figure_elements:
            elem_node = f"elem:{elem['id']}"
            elem_text_lower = elem['text'].lower()
            
            for ent in entities:
                ent_node = f"ent:{ent['text']}"
                ent_text_lower = ent['text'].lower()
                
                if ent_text_lower in elem_text_lower or (elem['type'] == "AXIS" and ent['type'] in ["VARIABLE", "METRIC"]):
                    self.graph.add_edge(elem_node, ent_node, relation="REPRESENTS")
                elif elem['type'] == "LEGEND" and ent['type'] in ["METHOD", "MODEL"]:
                    self.graph.add_edge(elem_node, ent_node, relation="CORRESPONDS_TO")

        # Add entity-entity relations
        if relations:
            for rel in relations:
                src_node = f"ent:{rel['source']}"
                tgt_node = f"ent:{rel['target']}"
                if self.graph.has_node(src_node) and self.graph.has_node(tgt_node):
                    self.graph.add_edge(src_node, tgt_node, relation=rel['relation'])

        # Enhance graph with PyTorch Geometric GNN if enabled
        if run_gnn and self.enable_gnn and self.supervised is not None and figure_elements and entities:
            self._apply_supervised(figure_elements, entities, relations or [], paragraph)
        elif run_gnn and self.enable_gnn and self.gnn_pipeline and len(self.graph.nodes) > 1:
            try:
                self.graph, self.gnn_metrics = self.gnn_pipeline.train_and_embed(self.graph)
                self.gnn_metrics["mode"] = "self_supervised_embedding_only"
            except Exception as e:
                print(f"[EntityGroundingGraph] PyTorch Geometric GNN fallback notice: {e}")
                self.gnn_metrics = {"status": "PARTIAL (Fallback NetworkX baseline topology)"}

        return self.graph

    def _apply_supervised(self, figure_elements, entities, relations, paragraph: str):
        t0 = time.perf_counter()
        data, cands = self.supervised.make_example(figure_elements, entities, paragraph, relations)
        scores = self.supervised.predict_scores(data)
        links = self.supervised.decode(cands, scores, self.supervised.threshold)
        score_of = {c: float(s) for c, s in zip(cands, scores)}
        rule_edges = [(u, v) for u, v in self.graph.edges if u.startswith("elem:") and v.startswith("ent:")]
        self.graph.remove_edges_from(rule_edges)
        etype = {e["id"]: e["type"] for e in figure_elements}
        ntype = {e["text"]: e["type"] for e in entities}
        for eid, ent in links:
            self.graph.add_edge(f"elem:{eid}", f"ent:{ent}", relation=_relation_for(etype.get(eid), ntype.get(ent)),
                                link_source="gnn", gnn_grounding_score=round(score_of[(eid, ent)], 4))
        self.gnn_metrics = {
            "mode": "supervised",
            "checkpoint": self.checkpoint_path,
            "threshold": self.supervised.threshold,
            "num_rule_links_replaced": len(rule_edges),
            "num_gnn_links": len(links),
            "inference_ms": round((time.perf_counter() - t0) * 1000, 3),
            "train_stats": self.supervised.train_stats,
            "gnn_grounding_pairs": [{"source": f"elem:{e}", "target": f"ent:{n}",
                                     "gnn_score": round(score_of[(e, n)], 4)} for e, n in links],
            "status": "IMPLEMENTED (supervised PyG GAT grounding)",
        }

    def get_grounding_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of nodes, edges, grounded paths, and GNN learned metrics.
        """
        nodes = []
        for n in self.graph.nodes:
            ndata = dict(self.graph.nodes[n])
            # Exclude large vector arrays from basic summary
            if "gnn_embedding" in ndata:
                ndata["has_gnn_embedding"] = True
                del ndata["gnn_embedding"]
            nodes.append({"id": n, **ndata})

        edges = [{"source": u, "target": v, **self.graph.edges[u, v]} for u, v in self.graph.edges]
        
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "gnn_enhanced": bool(self.gnn_metrics.get("mode") == "supervised"
                                 or self.gnn_metrics.get("epochs_trained", 0) > 0),
            "gnn_metrics": self.gnn_metrics,
            "status": ("IMPLEMENTED (NetworkX + supervised PyG GNN grounding)"
                       if self.gnn_metrics.get("mode") == "supervised"
                       else "IMPLEMENTED (NetworkX rules; GNN embeddings attached, links unchanged)"
                       if self.gnn_metrics.get("epochs_trained", 0) > 0
                       else "IMPLEMENTED (NetworkX rule-based grounding)")
        }
