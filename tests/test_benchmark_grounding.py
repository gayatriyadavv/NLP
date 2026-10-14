"""
Tests for the controlled benchmark, supervised GNN grounding, evaluation utilities and
regression tests for issues found in the 2026-10-08 audit (docs/AUDIT.md).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.benchmark_generator import BenchmarkGenerator, load_benchmark
from src.evaluation.figure_extraction_eval import evaluate_extraction, match_elements
from src.evaluation.hallucination_checker import HallucinationChecker
from src.generation.caption_generator import GroundedCaptionGenerator
from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
from src.grounding.grounding_features import acronym_match, defined_in_paragraph, pair_features
from src.utils.config import ConfigDict, get_config
from src.vision.figure_processor import is_numeric_text


@pytest.fixture(scope="module")
def tiny_benchmark(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    BenchmarkGenerator(str(out), seed=0, n_per_2d_type=3, n_views_per_molecule=1,
                       n_surface=3, n_scatter3d=3).generate()
    return load_benchmark(str(out))


# ---------------------------------------------------------------- benchmark
def test_benchmark_structure(tiny_benchmark):
    m = tiny_benchmark
    assert m["num_figures"] == 4 * 3 + 16 + 3 + 3
    for f in m["figures"]:
        assert os.path.exists(os.path.join(m["root"], f["image_path"]))
        ids = {e["id"] for e in f["elements"]}
        ents = {e["text"] for e in f["entities"]}
        for link in f["gold_links"]:
            assert link["element_id"] in ids and link["entity"] in ents
        for e in f["elements"]:
            x0, y0, x1, y1 = e["bbox"]
            assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480
        if f["is_3d"]:
            g = np.load(os.path.join(m["root"], f["geometry_path"]))
            assert g.shape == (1024, 3)
            assert np.linalg.norm(g, axis=1).max() == pytest.approx(1.0, abs=1e-4)


def test_benchmark_protein_split_is_structure_disjoint(tiny_benchmark):
    split_of = {}
    for f in tiny_benchmark["figures"]:
        if f["figure_type"] == "protein":
            split_of.setdefault(f["pdb_id"], set()).add(f["split"])
    assert all(len(s) == 1 for s in split_of.values())


def test_benchmark_paragraph_contains_distractors(tiny_benchmark):
    # at least some figures mention entities that have no gold link (otherwise type rules are trivially right)
    n = sum(1 for f in tiny_benchmark["figures"]
            if len({e["text"] for e in f["entities"]} - {l["entity"] for l in f["gold_links"]}) > 0)
    assert n > len(tiny_benchmark["figures"]) // 2


# ---------------------------------------------------------------- features
def test_pair_features():
    assert acronym_match("GCN", "Graph Convolutional Network") == 1.0
    assert acronym_match("RF", "Graph Convolutional Network") == 0.0
    assert defined_in_paragraph("MAE (eV)", "Mean Absolute Error", "the Mean Absolute Error (MAE) is") == 1.0
    f = pair_features({"type": "AXIS", "text": "Top-1 Acc. (%)"}, {"type": "METRIC", "text": "Top-1 Accuracy"}, "", False)
    assert f.shape == (10,) and f[6] == 1.0


# ---------------------------------------------------------------- supervised GNN
def _instances(figs):
    out = []
    for f in figs:
        gold = {(l["element_id"], l["entity"]) for l in f["gold_links"]}
        out.append((f, gold))
    return out


def test_supervised_gnn_trains_and_predicts(tiny_benchmark, tmp_path):
    figs = tiny_benchmark["figures"]
    tr = SupervisedGroundingTrainer(epochs=30, seed=0)
    ex = [tr.make_example(f["elements"], f["entities"], f["paragraph_text"], gold=g) for f, g in _instances(figs)]
    stats = tr.fit(ex[:-5], ex[-5:])
    assert stats["final_loss"] < stats["initial_loss"]
    assert 0.0 < tr.threshold < 1.0
    f = figs[0]
    data, cands = tr.make_example(f["elements"], f["entities"], f["paragraph_text"])
    scores = tr.predict_scores(data)
    links = tr.decode(cands, scores, tr.threshold)
    assert len({e for e, _ in links}) == len(links)  # at most one entity per element

    path = str(tmp_path / "g.pt")
    tr.save(path)
    tr2 = SupervisedGroundingTrainer.load(path)
    assert np.allclose(tr2.predict_scores(data), scores, atol=1e-6)


def test_message_passing_switch_changes_nothing_but_edges(tiny_benchmark):
    f = tiny_benchmark["figures"][0]
    a = SupervisedGroundingTrainer(epochs=1, seed=0, use_message_passing=True)
    b = SupervisedGroundingTrainer(epochs=1, seed=0, use_message_passing=False)
    b.model.load_state_dict(a.model.state_dict())
    d, _ = a.make_example(f["elements"], f["entities"], f["paragraph_text"])
    a.model.eval(), b.model.eval()
    with torch.no_grad():
        sa = a.model(d.x, d.edge_index, d.cand_index, d.cand_feat)
        sb = b.model(d.x, d.edge_index, d.cand_index, d.cand_feat)
        sc = a.model(d.x, d.edge_index[:, :0], d.cand_index, d.cand_feat)
    assert torch.allclose(sb, sc)
    assert not torch.allclose(sa, sb)


# ---------------------------------------------------------------- evaluation utils
def test_extraction_eval_perfect_and_empty():
    gold = [{"type": "AXIS", "text": "Epoch", "bbox": [0, 0, 10, 10]},
            {"type": "LEGEND", "text": "GCN", "bbox": [20, 20, 30, 30]}]
    r = evaluate_extraction([(gold, gold)])
    assert r["text_f1"] == 1.0 and r["type_accuracy_on_matched"] == 1.0
    assert evaluate_extraction([([], gold)])["text_recall"] == 0.0
    assert len(match_elements([{"text": "Epochs"}], [{"text": "Epoch"}])) == 1


def test_numeric_tick_filter():
    assert is_numeric_text("0.25") and is_numeric_text("-1.0") and is_numeric_text("100%")
    assert not is_numeric_text("Top-1 Acc. (%)") and not is_numeric_text("ResNet-50")


# ---------------------------------------------------------------- audit regressions
def test_config_dotted_keys_resolve():
    cfg = get_config()
    assert isinstance(cfg, ConfigDict)
    assert cfg.get("ablation_switches.graph_enabled") is True
    assert cfg.get("hierarchical_retrieval.weights")["w_paragraph"] == 0.40
    assert cfg.get("no.such.key", "d") == "d"


def test_caption_generator_makes_no_unsupported_trend_claim():
    out = GroundedCaptionGenerator().generate_caption(
        {"elements": [{"type": "AXIS", "text": "Epoch"}]},
        {"level_results": {"paragraph": [{"text": "We compare GCN."}]}},
        [{"text": "GCN", "type": "METHOD"}],
        {"edges": [], "num_edges": 0})
    assert "performance gains" not in out["generated_caption"]
    assert out["grounding_confidence"] == 0.0  # nothing grounded -> no confidence


def test_verifier_contradiction_is_numeric_conflict_only():
    v = HallucinationChecker(use_nli=False)
    ev = [{"text": "ResNet achieved 85.3% accuracy on ImageNet."}]
    assert v.verify_caption("ResNet achieved 91.2% accuracy.", ev)["claims_verification"][0]["status"] == "CONTRADICTED"
    # unrelated wording is UNKNOWN, not CONTRADICTED
    assert v.verify_caption("Bananas are yellow.", ev)["claims_verification"][0]["status"] == "UNKNOWN"
    assert v.verify_caption("ResNet achieved 85.3% accuracy.", ev)["claims_verification"][0]["status"] == "SUPPORTED"


def test_no_hardcoded_metric_offsets_in_evaluation_scripts():
    for p in ("pipelines/evaluate_gnn.py", "pipelines/evaluate_phase4.py", "pipelines/run_ablation.py"):
        src = open(p).read()
        assert "+ 0.12" not in src and "1700.0" not in src and "np.random.randn" not in src, p


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------- statistics
def test_paired_bootstrap():
    from src.evaluation.bootstrap import compare_means, compare_micro_f1
    rng = np.random.default_rng(0)
    a = rng.random(200)
    same = compare_means(a, a.copy(), n_resamples=2000)
    assert same["difference"] == 0.0 and not same["significant_at_0.05"]
    better = compare_means(a, a + 0.1, n_resamples=2000)
    assert better["significant_at_0.05"] and better["ci95_low"] > 0
    noisy = compare_means(a, a + rng.normal(0, 0.3, 200), n_resamples=2000)
    assert noisy["ci95_low"] < 0 < noisy["ci95_high"]
    ca = np.array([[1, 1, 1]] * 50)
    cb = np.array([[2, 0, 0]] * 50)
    r = compare_micro_f1(ca, cb, n_resamples=500)
    assert r["difference"] == pytest.approx(1.0 - 0.5) and r["significant_at_0.05"]


def test_caption_metrics():
    from pipelines.evaluate_scicap import bleu4, rouge_l
    ref = "Top-1 accuracy of ResNet and ViT on ImageNet across epochs"
    assert bleu4(ref, ref) == pytest.approx(1.0)
    assert rouge_l(ref, ref) == pytest.approx(1.0)
    assert rouge_l("completely unrelated words", ref) == 0.0
    assert 0 < bleu4("accuracy of ResNet on ImageNet", ref) < 1


def test_strip_caption_removes_leakage():
    from pipelines.evaluate_scicap import strip_caption
    cap = "Accuracy of GCN on Cora across training epochs."
    para = ("We train GCN on Cora. Figure 3: Accuracy of GCN on Cora across training epochs. "
            "The model converges after 50 epochs.")
    out = strip_caption(para, cap)
    assert "Figure 3" not in out and "converges" in out and "We train GCN" in out
