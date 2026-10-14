"""
Unit and Integration Tests for Phase 4 — 3D Scientific Visualization Branch Expansion.
Verifies 3D content detection, PointNet 3D feature extraction, multi-view rendering,
multimodal retrieval (Text + 2D + 3D + PyG GNN) and graph integration.
"""

import os
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.threed_detector import ThreeDVisualizationDetector, DEFAULT_CHECKPOINT
from src.models.vision.pointnet_extractor import PointNetExtractor, load_point_cloud_fixture
from src.vision.multiview_renderer import MultiViewRenderer, render_views_from_points
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.grounding.entity_graph import EntityGroundingGraph


def test_threed_detector_text_cues():
    detector = ThreeDVisualizationDetector(checkpoint_path="")  # text cues only
    res_3d = detector.detect_3d_content(
        "demo/dataset/samples/scicap_003.png",
        caption_text="3D point cloud molecular rendering of protein T1087 with pLDDT score"
    )
    assert res_3d["is_3d"] is True
    assert res_3d["method"] == "text_cues_only"

    res_2d = detector.detect_3d_content(
        "demo/dataset/samples/scicap_001.png",
        caption_text="Top-1 accuracy comparison of ViT-Base vs ResNet-50 across epochs"
    )
    assert res_2d["is_3d"] is False
    print("✓ test_threed_detector_text_cues passed")


def test_threed_detector_ignores_filename(tmp_path):
    # Regression: the detector used to return is_3d=True whenever the path contained "3d".
    dst = tmp_path / "my_3d_figure.png"
    shutil.copy("demo/dataset/samples/scicap_001.png", dst)
    detector = ThreeDVisualizationDetector(checkpoint_path="")
    assert detector.detect_3d_content(str(dst))["is_3d"] is False
    print("✓ test_threed_detector_ignores_filename passed")


@pytest.mark.skipif(not os.path.exists(DEFAULT_CHECKPOINT), reason="figure classifier checkpoint not trained")
def test_threed_detector_image_classifier(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    x, y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    ax.plot_surface(x, y, np.exp(-(x ** 2 + y ** 2)), cmap="viridis")
    p3 = tmp_path / "a.png"
    fig.savefig(p3)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.plot(np.arange(10), np.arange(10) ** 0.5, label="GCN")
    ax.legend()
    p2 = tmp_path / "b.png"
    fig.savefig(p2)
    plt.close(fig)

    det = ThreeDVisualizationDetector()
    r3, r2 = det.detect_3d_content(str(p3)), det.detect_3d_content(str(p2))
    assert r3["method"] == "image_classifier"
    assert r3["is_3d"] is True
    assert r2["is_3d"] is False
    print("✓ test_threed_detector_image_classifier passed")


def test_pointnet_extractor():
    extractor = PointNetExtractor(in_channels=3, out_dim=64)
    pts = load_point_cloud_fixture("demo/dataset/3d_samples/alphafold_t1087_3d.json", num_points=1024)

    assert pts.shape == (1024, 3)
    z_3d = extractor.extract_point_cloud_embedding(pts)

    assert z_3d.shape == (64,)
    assert np.isclose(np.linalg.norm(z_3d), 1.0, atol=1e-3)
    print("✓ test_pointnet_extractor passed")


def test_point_cloud_loader_does_not_fabricate():
    with pytest.raises(FileNotFoundError):
        load_point_cloud_fixture("does/not/exist.json")


def test_multiview_renderer():
    renderer = MultiViewRenderer(num_views=4, out_dim=64)
    views = renderer.generate_views("demo/dataset/samples/scicap_003.png")
    assert len(views) == 4

    mv_emb = renderer.extract_multiview_embedding("demo/dataset/samples/scicap_003.png")
    assert mv_emb.shape == (64,)
    assert np.isclose(np.linalg.norm(mv_emb), 1.0, atol=1e-3)
    print("✓ test_multiview_renderer passed")


def test_multiview_from_points():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)).astype(np.float32)
    views = render_views_from_points(pts, num_views=6)
    assert len(views) == 6 and views[0].shape == (64, 64, 3)
    assert all(v[:, :, 0].sum() > 0 for v in views)
    # different viewpoints of an anisotropic cloud should differ
    pts[:, 0] *= 3
    views = render_views_from_points(pts, num_views=4)
    assert not np.allclose(views[0], views[1])
    emb = MultiViewRenderer().extract_embedding_from_points(pts)
    assert emb.shape == (64,)


def test_multimodal_retriever():
    retriever = MultimodalRetriever()
    rng = np.random.default_rng(0)
    q_emb, v2d_emb, v3d_emb, gnn_emb = (rng.normal(size=64) for _ in range(4))

    candidates = [
        {"figure_id": "fig_3d", "is_3d": True, "text_embedding": q_emb, "2d_embedding": v2d_emb,
         "3d_embedding": v3d_emb, "gnn_embedding": gnn_emb},
        {"figure_id": "fig_2d", "is_3d": False, "text_embedding": q_emb, "2d_embedding": v2d_emb,
         "3d_embedding": v3d_emb, "gnn_embedding": gnn_emb},
    ]

    results = retriever.retrieve_multimodal(q_emb, v2d_emb, v3d_emb, gnn_emb, candidates)
    assert len(results) == 2
    by_id = {r["figure_id"]: r for r in results}
    # 3D similarity never contributes for a 2D candidate
    assert by_id["fig_2d"]["modality_breakdown"]["pointcloud_3d_similarity"] is None
    assert by_id["fig_3d"]["modality_breakdown"]["pointcloud_3d_similarity"] == pytest.approx(1.0, abs=1e-3)
    print("✓ test_multimodal_retriever passed")


def test_multimodal_retriever_missing_modality_not_imputed():
    retriever = MultimodalRetriever(weights={"w_text": 0.5, "w_2d": 0.5, "w_3d": 0.0, "w_gnn": 0.0})
    q = np.ones(8)
    res = retriever.retrieve_multimodal(q, None, None, None, [{"figure_id": "a", "text_embedding": q}])
    # only text is available -> score equals the text similarity, not averaged with a 0.5 constant
    assert res[0]["multimodal_score"] == pytest.approx(1.0)


def test_end_to_end_phase4_pipeline():
    detector = ThreeDVisualizationDetector(checkpoint_path="")
    det_res = detector.detect_3d_content("demo/dataset/samples/scicap_003.png", "3D molecular render of protein T1087")
    assert det_res["is_3d"] is True

    extractor = PointNetExtractor()
    pts = load_point_cloud_fixture("demo/dataset/3d_samples/alphafold_t1087_3d.json")
    z_3d = extractor.extract_point_cloud_embedding(pts)
    assert z_3d.shape == (64,)

    elements = [{"id": "3d_mesh", "type": "CHART_REGION", "text": "3D Point Cloud Mesh (AlphaFold T1087)"},
                {"id": "title", "type": "LABEL", "text": "AlphaFold"}]
    entities = [{"text": "AlphaFold", "type": "MODEL"}, {"text": "T1087", "type": "MATERIAL"}]

    graph_builder = EntityGroundingGraph()
    graph_builder.build_graph(elements, entities, run_gnn=True)
    summary = graph_builder.get_grounding_summary()

    assert summary["num_nodes"] > 0
    assert summary["gnn_enhanced"] is True
    print("✓ test_end_to_end_phase4_pipeline passed")


def test_figure_3d_node_feature_uses_embedding():
    import networkx as nx
    from src.grounding.pyg_converter import PyGGraphConverter
    g = nx.DiGraph()
    emb = np.random.default_rng(0).normal(size=64)
    g.add_node("fig3d", label="", node_type="FIGURE_3D", modal_embedding=emb.tolist())
    g.add_node("elem:a", label="x", node_type="FIGURE_ELEMENT", element_type="AXIS")
    g.add_edge("fig3d", "elem:a", relation="OTHER")
    data, n2i, _ = PyGGraphConverter().convert_to_pyg_data(g)
    row = data.x[n2i["fig3d"]].numpy()
    assert np.abs(row[-10:]).sum() > 0  # embedding occupies the text-feature slot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
