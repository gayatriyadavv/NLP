"""
Phase 3 evaluation: element -> entity grounding on SciFig-Controlled (held-out test split).

Compared methods
  A. Rules       - the NetworkX rule graph of EntityGroundingGraph (substring match,
                   AXIS -> METRIC/VARIABLE, LEGEND -> METHOD/MODEL). No training.
  B. GNN         - PairGroundingModel: GAT message passing over the grounding graph + pair MLP,
                   trained on the train split (inductive), threshold tuned on a validation split.
  Diagnostic     - identical model and features with message passing disabled, to measure what
                   the graph structure itself contributes.

Element settings
  oracle - gold element texts (isolates grounding from OCR errors)
  ocr    - elements extracted by FigureProcessor (EasyOCR + layout heuristics); gold links whose
           element was not extracted count as misses.

Entities are the gold entity list of each figure (oracle NER) in both settings; see
docs/BENCHMARK.md for why.

Metrics: link precision / recall / F1 (micro), grounding accuracy@1 and MRR over gold-linked
elements (ties broken in expectation), training time, per-figure inference latency.
All numbers are measured; learned models are run with several seeds and reported as mean ± std.

Usage:
    python pipelines/evaluate_gnn.py [--seeds 42 43 44] [--settings oracle ocr]
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.benchmark_generator import load_benchmark
from src.evaluation.figure_extraction_eval import evaluate_extraction, match_elements
from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
from src.grounding.grounding_features import TEXT_ELEMENT_TYPES, build_grounding_graph
from src.nlp.scientific_ner import ScientificNERExtractor
from src.vision.figure_processor import FigureProcessor
from src.vision.ocr_engine import OCREngine

_NER = ScientificNERExtractor()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def figure_instance(fig: Dict[str, Any], setting: str, processor: FigureProcessor, root: str) -> Dict[str, Any]:
    """Elements, entities, relations, gold pairs (over the instance's element ids) and total gold links."""
    entities = fig["entities"]
    relations = _NER.extract_relations(entities, fig["paragraph_text"])
    gold_by_elem = {l["element_id"]: l["entity"] for l in fig["gold_links"]}
    if setting == "oracle":
        elements = [{"id": e["id"], "type": e["type"], "text": e["text"]} for e in fig["elements"]]
        gold = {(eid, ent) for eid, ent in gold_by_elem.items()}
    else:
        out = processor.process_figure(os.path.join(root, fig["image_path"]))
        elements = [e for e in out["elements"] if e["type"] in TEXT_ELEMENT_TYPES]
        gold = set()
        for i, j, _ in match_elements(elements, fig["elements"]):
            ent = gold_by_elem.get(fig["elements"][j]["id"])
            if ent:
                gold.add((elements[i]["id"], ent))
    return {"figure": fig, "elements": elements, "entities": entities, "relations": relations,
            "gold": gold, "num_gold_links": len(fig["gold_links"])}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _expected_rank(scores: Dict[str, float], gold_ent: str) -> float:
    g = scores.get(gold_ent, 0.0)
    greater = sum(1 for s in scores.values() if s > g)
    ties = sum(1 for s in scores.values() if s == g)  # includes gold
    return greater + (ties + 1) / 2.0


def score_instances(instances, predicted_links, entity_scores) -> Dict[str, float]:
    """
    predicted_links: list (per instance) of set((element_id, entity))
    entity_scores:   list (per instance) of {element_id: {entity: score}}
    """
    tp = fp = 0
    n_gold_total = 0
    acc1, rr = [], []
    for inst, pred, sc in zip(instances, predicted_links, entity_scores):
        tp += len(pred & inst["gold"])
        fp += len(pred - inst["gold"])
        n_gold_total += inst["num_gold_links"]
        # accuracy@1 / MRR over gold-linked elements of the instance (oracle: all gold elements;
        # ocr: extracted & matched ones; unextracted ones count as rank = inf -> 0)
        for eid, ent in inst["gold"]:
            r = _expected_rank(sc.get(eid, {}), ent)
            ties = sum(1 for s in sc.get(eid, {}).values() if s == sc.get(eid, {}).get(ent, 0.0))
            greater = sum(1 for s in sc.get(eid, {}).values() if s > sc.get(eid, {}).get(ent, 0.0))
            acc1.append((1.0 / ties) if greater == 0 and ties else 0.0)
            rr.append(1.0 / r)
        missing = inst["num_gold_links"] - len(inst["gold"])
        acc1.extend([0.0] * missing)
        rr.extend([0.0] * missing)
    fn = n_gold_total - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gold_total, 1)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "grounding_accuracy@1": round(float(np.mean(acc1)), 4) if acc1 else 0.0,
            "mrr": round(float(np.mean(rr)), 4) if rr else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def run_rules(instances) -> Tuple[Dict[str, float], float]:
    preds, scores, times = [], [], []
    for inst in instances:
        t0 = time.perf_counter()
        g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
        links = {(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")}
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(links)
        sc = {el["id"]: {en["text"]: float((el["id"], en["text"]) in links) for en in inst["entities"]}
              for el in inst["elements"]}
        scores.append(sc)
    return score_instances(instances, preds, scores), float(np.median(times))


def run_learned(train_i, val_i, test_i, use_mp: bool, seed: int, epochs: int):
    tr = SupervisedGroundingTrainer(use_message_passing=use_mp, seed=seed, epochs=epochs)
    mk = lambda inst: tr.make_example(inst["elements"], inst["entities"], inst["figure"]["paragraph_text"],
                                      inst["relations"], gold=inst["gold"], extra_nodes=inst.get("extra_nodes"))
    train_ex = [mk(i) for i in train_i]
    val_ex = [mk(i) for i in val_i]
    stats = tr.fit(train_ex, val_ex)
    preds, scores, times = [], [], []
    for inst in test_i:
        t0 = time.perf_counter()
        data, cands = tr.make_example(inst["elements"], inst["entities"], inst["figure"]["paragraph_text"],
                                      inst["relations"], extra_nodes=inst.get("extra_nodes"))
        s = tr.predict_scores(data)
        links = set(tr.decode(cands, s, tr.threshold))
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(links)
        sc: Dict[str, Dict[str, float]] = {}
        for (eid, ent), v in zip(cands, s):
            sc.setdefault(eid, {})[ent] = float(v)
        scores.append(sc)
    return score_instances(test_i, preds, scores), stats, float(np.median(times)), tr, preds


def heldout_vocab_experiment(fit_figs, val_figs, test_figs, processor, root, seeds, epochs, k_per_type=4):
    """
    Generalisation to unseen entity names: hold out k entities per type (METHOD / METRIC /
    VARIABLE), drop every training/validation figure that links to one of them, and score only the
    test elements whose gold entity is held out. Oracle elements.
    """
    from src.data.benchmark_generator import METHODS, METRICS, VARIABLES
    rng = np.random.default_rng(7)
    held = set()
    for table in (METHODS, METRICS, VARIABLES):
        names = sorted(table)
        held |= set(rng.choice(names, size=k_per_type, replace=False).tolist())
    linked = lambda f: {l["entity"] for l in f["gold_links"]}
    keep = lambda figs: [f for f in figs if not (linked(f) & held)]
    fit_i = [figure_instance(f, "oracle", processor, root) for f in keep(fit_figs)]
    val_i = [figure_instance(f, "oracle", processor, root) for f in keep(val_figs)]
    test_i = []
    for f in test_figs:
        inst = figure_instance(f, "oracle", processor, root)
        inst["gold"] = {(e, n) for e, n in inst["gold"] if n in held}
        inst["num_gold_links"] = len(inst["gold"])
        if inst["gold"]:
            test_i.append(inst)

    def restrict(preds):  # only score predictions on the held-out-entity elements
        return [{(e, n) for e, n in p if e in {g[0] for g in inst["gold"]}} for p, inst in zip(preds, test_i)]

    out = {"held_out_entities": sorted(held), "num_fit": len(fit_i), "num_val": len(val_i),
           "num_test_figures": len(test_i), "num_test_links": sum(len(i["gold"]) for i in test_i)}
    preds, scores = [], []
    for inst in test_i:
        g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
        links = {(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")}
        preds.append(links)
        scores.append({el["id"]: {en["text"]: float((el["id"], en["text"]) in links) for en in inst["entities"]}
                       for el in inst["elements"]})
    out["A_rules"] = score_instances(test_i, restrict(preds), scores)
    for name, mp in (("B_gnn", True), ("diagnostic_no_message_passing", False)):
        runs = []
        for seed in seeds:
            _, _, _, tr, _ = run_learned(fit_i, val_i, test_i, mp, seed, epochs)
            pr, sc = [], []
            for inst in test_i:
                data, cands = tr.make_example(inst["elements"], inst["entities"], inst["figure"]["paragraph_text"], inst["relations"])
                sv = tr.predict_scores(data)
                pr.append(set(tr.decode(cands, sv, tr.threshold)))
                d: Dict[str, Dict[str, float]] = {}
                for (e, n), v in zip(cands, sv):
                    d.setdefault(e, {})[n] = float(v)
                sc.append(d)
            runs.append(score_instances(test_i, restrict(pr), sc))
        out[name] = mean_std(runs)
    return out


def mean_std(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--settings", nargs="+", default=["oracle", "ocr"])
    ap.add_argument("--out", default="results/gnn_grounding_results.json")
    a = ap.parse_args()

    meta = load_benchmark(a.benchmark)
    root = meta["root"]
    ocr = OCREngine(cache_path=os.path.join(root, "cache", "ocr.json"))
    processor = FigureProcessor(ocr_engine=ocr)
    rng = np.random.default_rng(0)
    train_figs = [f for f in meta["figures"] if f["split"] == "train"]
    test_figs = [f for f in meta["figures"] if f["split"] == "test"]
    val_mask = rng.random(len(train_figs)) < 0.2
    fit_figs = [f for f, v in zip(train_figs, val_mask) if not v]
    val_figs = [f for f, v in zip(train_figs, val_mask) if v]

    report: Dict[str, Any] = {
        "benchmark": {"name": meta["dataset_name"], "version": meta["version"], "path": root,
                      "generator_seed": meta["seed"], "num_fit": len(fit_figs), "num_val": len(val_figs),
                      "num_test": len(test_figs)},
        "model": {"encoder": "GATConv x2 (64-d, 2 heads, residual, LayerNorm, dropout 0.1)",
                  "node_features": "one-hot node/element/entity type + hashed character trigrams (64-d total)",
                  "pair_head": "MLP([z_u, z_v, z_u*z_v, 10 pairwise features]) -> logit",
                  "loss": "BCE with positive-class weighting", "optimizer": "Adam", "lr": 0.005,
                  "weight_decay": 1e-4, "epochs": a.epochs, "batch": "full batch of all training graphs",
                  "decision": "per element arg-max entity if score >= threshold (threshold tuned on val F1)"},
        "seeds": a.seeds,
        "hardware": {"platform": platform.platform(), "torch": torch.__version__,
                     "threads": torch.get_num_threads(), "device": "cpu"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {},
    }

    for setting in a.settings:
        print(f"=== setting: {setting}")
        t0 = time.perf_counter()
        mk = lambda figs: [figure_instance(f, setting, processor, root) for f in figs]
        fit_i, val_i, test_i = mk(fit_figs), mk(val_figs), mk(test_figs)
        prep_s = time.perf_counter() - t0
        res: Dict[str, Any] = {"data_preparation_s": round(prep_s, 2)}
        if setting == "ocr":
            res["element_extraction_test"] = evaluate_extraction(
                [(processor.process_figure(os.path.join(root, f["image_path"]))["elements"], f["elements"])
                 for f in test_figs])
            res["test_gold_links_reachable_after_ocr"] = round(
                sum(len(i["gold"]) for i in test_i) / max(sum(i["num_gold_links"] for i in test_i), 1), 4)

        rules, rules_ms = run_rules(test_i)
        res["A_rules"] = {"test": rules, "train_time_s": 0.0, "inference_ms_per_figure_median": round(rules_ms, 3)}
        print(f"  A rules        F1={rules['f1']} acc@1={rules['grounding_accuracy@1']} MRR={rules['mrr']}")

        for name, use_mp in (("B_gnn", True), ("diagnostic_no_message_passing", False)):
            runs, stats_l, lat = [], [], []
            for seed in a.seeds:
                m, stats, ms, tr, _ = run_learned(fit_i, val_i, test_i, use_mp, seed, a.epochs)
                runs.append(m)
                stats_l.append(stats)
                lat.append(ms)
                if setting == "oracle" and use_mp and seed == a.seeds[0]:
                    tr.save("models/checkpoints/grounding_gnn.pt")
            res[name] = {"test": mean_std(runs), "per_seed": runs,
                         "train_time_s": {"mean": round(float(np.mean([s["train_time_s"] for s in stats_l])), 3),
                                          "per_seed": [s["train_time_s"] for s in stats_l]},
                         "thresholds": [s["threshold"] for s in stats_l],
                         "inference_ms_per_figure_median": round(float(np.median(lat)), 3)}
            print(f"  {name:<14} F1={res[name]['test']['f1']['mean']}±{res[name]['test']['f1']['std']} "
                  f"acc@1={res[name]['test']['grounding_accuracy@1']['mean']} MRR={res[name]['test']['mrr']['mean']}")
        report["settings"][setting] = res

    if "oracle" in a.settings:
        print("=== held-out entity vocabulary (oracle elements)")
        hv = heldout_vocab_experiment(fit_figs, val_figs, test_figs, processor, root, a.seeds, a.epochs)
        report["heldout_vocabulary"] = hv
        print(f"  A rules F1={hv['A_rules']['f1']}  B gnn F1={hv['B_gnn']['f1']['mean']}  "
              f"no-MP F1={hv['diagnostic_no_message_passing']['f1']['mean']}  (links={hv['num_test_links']})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results written to {a.out}")


if __name__ == "__main__":
    main()
