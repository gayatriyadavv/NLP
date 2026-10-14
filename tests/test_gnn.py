"""
Unit and Integration Tests for Phase 3 — PyTorch Geometric GNN Module.
Verifies NetworkX-to-PyG conversion, GNN forward pass, self-supervised link prediction, and pipeline integration.
"""

import os
import sys
import torch
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grounding.pyg_converter import PyGGraphConverter
from src.models.grounding.gnn_grounding import GNNGroundingModel
from src.grounding.gnn_pipeline import GNNGroundingPipeline
from src.grounding.entity_graph import EntityGroundingGraph

def test_pyg_converter():
    g = nx.DiGraph()
    g.add_node("elem:axis_y", label="Top-1 Accuracy (%)", node_type="FIGURE_ELEMENT", element_type="AXIS")
    g.add_node("ent:ResNet", label="ResNet-50", node_type="SCIENTIFIC_ENTITY", entity_type="METHOD")
    g.add_edge("elem:axis_y", "ent:ResNet", relation="REPRESENTS")

    converter = PyGGraphConverter(feature_dim=64)
    pyg_data, node2idx, idx2node = converter.convert_to_pyg_data(g)

    assert pyg_data.num_nodes == 2
    assert pyg_data.x.shape == (2, 64)
    assert pyg_data.edge_index.shape == (2, 1)
    print("✓ test_pyg_converter passed")

def test_gnn_model_forward():
    x = torch.randn(4, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    model = GNNGroundingModel(in_channels=64, hidden_channels=64, out_channels=64, gnn_type="GAT")
    z = model(x, edge_index)

    assert z.shape == (4, 64)
    graph_emb = model.get_graph_embedding(z)
    assert graph_emb.shape == (1, 64)
    print("✓ test_gnn_model_forward passed")

def test_gnn_pipeline_training():
    g = nx.DiGraph()
    g.add_node("elem:axis_x", label="Epochs", node_type="FIGURE_ELEMENT", element_type="AXIS")
    g.add_node("elem:legend_1", label="ResNet", node_type="FIGURE_ELEMENT", element_type="LEGEND")
    g.add_node("ent:ResNet", label="ResNet-50", node_type="SCIENTIFIC_ENTITY", entity_type="METHOD")
    g.add_node("ent:Accuracy", label="Top-1 Accuracy", node_type="SCIENTIFIC_ENTITY", entity_type="METRIC")

    g.add_edge("elem:axis_x", "ent:Accuracy", relation="REPRESENTS")
    g.add_edge("elem:legend_1", "ent:ResNet", relation="CORRESPONDS_TO")
    g.add_edge("ent:ResNet", "ent:Accuracy", relation="ACHIEVES")

    pipeline = GNNGroundingPipeline(feature_dim=64, hidden_dim=64, epochs=10, device="cpu")
    g_out, metrics = pipeline.train_and_embed(g)

    assert metrics["epochs_trained"] == 10
    assert "gnn_embedding" in g_out.nodes["ent:ResNet"]
    assert len(g_out.nodes["ent:ResNet"]["gnn_embedding"]) == 64
    print("✓ test_gnn_pipeline_training passed")

def test_entity_grounding_graph_gnn_integration():
    elements = [
        {"id": "axis_x", "type": "AXIS", "text": "Training Epochs"},
        {"id": "axis_y", "type": "AXIS", "text": "Top-1 Accuracy (%)"},
        {"id": "legend_1", "type": "LEGEND", "text": "ResNet-50"}
    ]
    entities = [
        {"text": "ResNet-50", "type": "METHOD"},
        {"text": "Top-1 Accuracy", "type": "METRIC"}
    ]
    relations = [
        {"source": "ResNet-50", "relation": "ACHIEVES", "target": "Top-1 Accuracy"}
    ]

    graph_builder = EntityGroundingGraph()
    g = graph_builder.build_graph(elements, entities, relations, run_gnn=True)
    summary = graph_builder.get_grounding_summary()

    assert summary["num_nodes"] > 0
    assert summary["gnn_enhanced"] == True
    assert "gnn_metrics" in summary
    print("✓ test_entity_grounding_graph_gnn_integration passed")

if __name__ == "__main__":
    test_pyg_converter()
    test_gnn_model_forward()
    test_gnn_pipeline_training()
    test_entity_grounding_graph_gnn_integration()
    print("\nALL PYTORCH GEOMETRIC GNN TESTS PASSED!")


def test_grounding_summary_edges_keep_node_ids(tmp_path):
    # Regression: an edge attribute named "source" used to overwrite the edge's source node id
    # in get_grounding_summary().
    import numpy as np
    from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
    elements = [{"id": "axis_x", "type": "AXIS", "text": "Epoch"},
                {"id": "legend_1", "type": "LEGEND", "text": "GCN"}]
    entities = [{"text": "Epoch", "type": "VARIABLE"}, {"text": "Graph Convolutional Network", "type": "METHOD"}]
    tr = SupervisedGroundingTrainer(epochs=1, seed=0)
    tr.threshold = 0.0  # force links so edges exist
    path = str(tmp_path / "g.pt")
    tr.save(path)
    builder = EntityGroundingGraph(checkpoint_path=path)
    builder.build_graph(elements, entities, run_gnn=True)
    summary = builder.get_grounding_summary()
    assert summary["gnn_metrics"]["mode"] == "supervised"
    gnn_edges = [e for e in summary["edges"] if e.get("link_source") == "gnn"]
    assert gnn_edges and all(e["source"].startswith("elem:") for e in gnn_edges)
