"""
Phase 4 ablation study (configurations A-E) on SciFig-Controlled, test split, end-to-end OCR elements.

  A  NetworkX rules                       (no learning)
  B  NetworkX + PyG GNN                   (supervised GAT grounding)
  C  GNN + CLIP                           (BLOCKED unless CLIP weights can be loaded)
  D  GNN + 3D branch                      (FIGURE_3D node = image->3D-aligned embedding, only for
                                           figures the detector classifies as 3D; 3D-type channel
                                           in retrieval)
  E  Full multimodal (text + 2D + 3D + GNN) (D + FIGURE_IMAGE node for every figure + OCR-text
                                           2D retrieval channel)

Tasks / metrics (all measured in this run):
  T1 grounding           link P/R/F1, accuracy@1, MRR       (mean ± std over seeds)
  T2 caption -> figure   Recall@1, Recall@5, MRR over the test gallery
  T3 geometry -> figure  Recall@1, MRR (only configurations with a 3D branch)
  SCQ                    per-figure composite, every component computed (see docs/BENCHMARK.md)
  Latency                training time (once) and per-figure inference time, reported separately;
                         OCR time reported separately because it is shared by all configurations.

Prerequisites: data/benchmark (generator), OCR cache (optional, speeds up), checkpoints from
pipelines/evaluate_phase4.py (figure_classifier.pt, cross_modal_3d.pt).

Usage:
    python pipelines/run_ablation.py [--seeds 42 43 44]
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.evaluate_gnn import figure_instance, mean_std, run_learned, run_rules
from src.data.benchmark_generator import load_benchmark
from src.evaluation.bootstrap import compare_means, compare_micro_f1
from src.evaluation.hallucination_checker import HallucinationChecker
from src.generation.caption_generator import GroundedCaptionGenerator
from src.grounding.grounding_features import build_grounding_graph
from src.models.vision.clip_encoder import CLIPEncoder, ModalityUnavailableError
from src.models.vision.cross_modal_3d import CrossModal3DAligner, geometry_id, _points_batch
from src.models.vision.figure_image_encoder import FigureImageEncoder, load_figure_tensor
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.vision.figure_processor import FigureProcessor
from src.vision.ocr_engine import OCREngine
from src.vision.threed_detector import ThreeDVisualizationDetector

VIS_TYPES = ["2D_PLOT", "3D_MOLECULAR_STRUCTURE", "3D_SURFACE_PLOT", "3D_SCATTER_PLOT"]
RETRIEVAL_WEIGHTS = {"w_text": 0.4, "w_gnn": 0.3, "w_2d": 0.2, "w_3d": 0.1}  # fixed a priori


def query_3d_intent(query: str) -> Optional[np.ndarray]:
    """Map 3D wording in a text query to a visualization type; None if the query has no 3D intent."""
    q = query.lower()
    vec = np.zeros(len(VIS_TYPES), dtype=np.float32)
    if re.search(r"c-alpha|pdb|protein|molecul", q):
        vec[1] = 1
    if re.search(r"surface|landscape", q):
        vec[2] = 1
    if re.search(r"3d (pca|scatter|embedding)|point cloud", q):
        vec[3] = 1
    return vec if vec.any() else None


def _tokens(t: str) -> set:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r)


def ranks_from_scores(results: List[Dict[str, Any]], gold_id: str) -> int:
    for k, r in enumerate(results):
        if r["figure_id"] == gold_id:
            return k + 1
    return len(results) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="results/ablation_results.json")
    a = ap.parse_args()

    meta = load_benchmark(a.benchmark)
    root = meta["root"]
    ocr = OCREngine(cache_path=os.path.join(root, "cache", "ocr.json"))
    processor = FigureProcessor(ocr_engine=ocr)
    detector = ThreeDVisualizationDetector()
    if detector.model is None:
        sys.exit("Missing models/checkpoints/figure_classifier.pt - run pipelines/evaluate_phase4.py first.")
    aligner = CrossModal3DAligner()
    aligner.load_state_dict(torch.load("models/checkpoints/cross_modal_3d.pt", weights_only=False)["state_dict"])
    aligner.eval()
    fig_enc = FigureImageEncoder()
    fig_enc.load_state_dict(torch.load("models/checkpoints/figure_classifier.pt", weights_only=False)["state_dict"])
    fig_enc.eval()

    rng = np.random.default_rng(0)
    train_figs = [f for f in meta["figures"] if f["split"] == "train"]
    test_figs = [f for f in meta["figures"] if f["split"] == "test"]
    val_mask = rng.random(len(train_figs)) < 0.2
    fit_figs = [f for f, v in zip(train_figs, val_mask) if not v]
    val_figs = [f for f, v in zip(train_figs, val_mask) if v]

    # ---- OCR latency (uncached, shared by all configs) -------------------
    fresh = OCREngine()
    ocr_ms = []
    for f in test_figs[:10]:
        t0 = time.perf_counter()
        fresh.read(os.path.join(root, f["image_path"]))
        ocr_ms.append((time.perf_counter() - t0) * 1000)

    # ---- per-figure modality features ------------------------------------
    print("Preparing instances ...")
    feats: Dict[str, Dict[str, Any]] = {}
    det_ms, emb_ms = [], []
    for f in meta["figures"]:
        path = os.path.join(root, f["image_path"])
        t0 = time.perf_counter()
        det = detector.detect_3d_content(path)  # pixels only: no caption / paragraph cues
        det_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        with torch.no_grad():
            x = load_figure_tensor(path).unsqueeze(0)
            z2d = fig_enc(x)[0][0].numpy()
            z3d = aligner.encode_image(x)[0].numpy() if det["is_3d"] else None
        emb_ms.append((time.perf_counter() - t0) * 1000)
        vt = np.zeros(len(VIS_TYPES), dtype=np.float32)
        vt[VIS_TYPES.index(det["visualization_type"]) if det["visualization_type"] in VIS_TYPES else 0] = 1
        feats[f["figure_id"]] = {"is_3d_pred": det["is_3d"], "vis_type_vec": vt, "z2d": z2d, "z3d": z3d}

    def instances(figs, cfg):
        out = []
        for f in figs:
            inst = figure_instance(f, "ocr", processor, root)
            elem_nodes = [f"elem:{e['id']}" for e in inst["elements"]]
            extra = []
            ft = feats[f["figure_id"]]
            if cfg in ("D", "E") and ft["z3d"] is not None:
                extra.append(("fig3d", {"label": "", "node_type": "FIGURE_3D",
                                        "modal_embedding": ft["z3d"].tolist()}, elem_nodes))
            if cfg == "E":
                extra.append(("figimg", {"label": "", "node_type": "FIGURE_IMAGE",
                                         "modal_embedding": ft["z2d"].tolist()}, elem_nodes))
            inst["extra_nodes"] = extra
            out.append(inst)
        return out

    report: Dict[str, Any] = {
        "benchmark": {"name": meta["dataset_name"], "version": meta["version"], "path": root,
                      "generator_seed": meta["seed"], "num_fit": len(fit_figs), "num_val": len(val_figs),
                      "num_test": len(test_figs), "num_test_3d": sum(f["is_3d"] for f in test_figs)},
        "element_setting": "end-to-end OCR elements; gold entity lists (oracle NER)",
        "seeds": a.seeds, "gnn_epochs": a.epochs,
        "retrieval_weights": RETRIEVAL_WEIGHTS,
        "hardware": {"platform": platform.platform(), "torch": torch.__version__,
                     "threads": torch.get_num_threads(), "device": "cpu"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shared_latency_ms_per_figure": {
            "ocr_uncached_median": round(float(np.median(ocr_ms)), 1),
            "3d_detector_median": round(float(np.median(det_ms)), 3),
            "image_embeddings_median": round(float(np.median(emb_ms)), 3)},
        "configurations": {},
    }

    # ---- TF-IDF spaces fitted on the train split only --------------------
    ocr_text = {f["figure_id"]: " ".join(e["text"] for e in processor.process_figure(
        os.path.join(root, f["image_path"]))["elements"] if e["text"]) for f in meta["figures"]}
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(
        [f["paragraph_text"] + " " + f["caption"] + " " + ocr_text[f["figure_id"]] for f in train_figs])
    vec = lambda t: tfidf.transform([t]).toarray()[0]

    # ---- C: CLIP ------------------------------------------------------------
    try:
        CLIPEncoder.load()
        clip_status = "AVAILABLE (not wired into this ablation run)"
    except ModalityUnavailableError as e:
        clip_status = f"BLOCKED: {e}"
    report["configurations"]["C"] = {"name": "GNN + CLIP", "status": clip_status,
                                     "note": "Not experimentally validated: pretrained CLIP weights unavailable."}
    print(f"C: {clip_status[:120]}")

    # ---- T3: geometry -> figure (shared by D and E) ----------------------
    test3d = [f for f in test_figs if f["is_3d"] and f.get("geometry_path")]
    gal_ids, gal_geoms = [], []
    for f in test3d:
        if geometry_id(f) not in gal_ids:
            gal_ids.append(geometry_id(f))
            gal_geoms.append(np.load(os.path.join(root, f["geometry_path"])))
    with torch.no_grad():
        pts, views = _points_batch(gal_geoms, None, augment=False)
        zq = (aligner.point_encoder(pts) + aligner.view_encoder.forward_views(views)).numpy()
    gallery = [f for f in test_figs if feats[f["figure_id"]]["z3d"] is not None]
    rr = []
    for gi, gid in enumerate(gal_ids):
        sims = sorted(((float(zq[gi] @ feats[f["figure_id"]]["z3d"]), geometry_id(f)) for f in gallery), reverse=True)
        hit = next((k + 1 for k, (_, g) in enumerate(sims) if g == gid), len(test_figs) + 1)
        rr.append(hit)
    rr = np.array(rr)
    t3 = {"recall@1": round(float((rr <= 1).mean()), 4), "recall@5": round(float((rr <= 5).mean()), 4),
          "mrr": round(float((1 / rr).mean()), 4), "num_queries": len(gal_ids),
          "gallery": "test figures classified as 3D by the detector",
          "chance_recall@1_approx": round(1 / max(len(gal_ids), 1), 4)}

    per_item: Dict[str, Dict[str, Any]] = {}  # cfg -> per-figure arrays for paired bootstrap
    retriever = MultimodalRetriever(weights=RETRIEVAL_WEIGHTS)
    captioner = GroundedCaptionGenerator()
    verifier = HallucinationChecker(use_nli=False)

    for cfg, name in (("A", "NetworkX rules"), ("B", "NetworkX + PyG GNN"),
                      ("D", "GNN + 3D branch"), ("E", "Full multimodal (text + 2D + 3D + GNN)")):
        print(f"=== {cfg}: {name}")
        fit_i, val_i, test_i = instances(fit_figs, cfg), instances(val_figs, cfg), instances(test_figs, cfg)
        res: Dict[str, Any] = {"name": name, "status": "MEASURED"}
        if cfg == "A":
            m, ms = run_rules(test_i)
            res["grounding"] = {"test": m, "train_time_s": 0.0, "inference_ms_per_figure_median": round(ms, 3)}
            links_per_fig = []
            for inst in test_i:
                g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
                links_per_fig.append({(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")})
        else:
            runs, tts, lats, links_per_fig = [], [], [], None
            for seed in a.seeds:
                m, stats, ms, _, preds = run_learned(fit_i, val_i, test_i, True, seed, a.epochs)
                runs.append(m)
                tts.append(stats["train_time_s"])
                lats.append(ms)
                if links_per_fig is None:
                    links_per_fig = preds  # first seed drives retrieval / SCQ
            res["grounding"] = {"test": mean_std(runs), "per_seed": runs,
                                "train_time_s": {"mean": round(float(np.mean(tts)), 3), "per_seed": tts},
                                "inference_ms_per_figure_median": round(float(np.median(lats)), 3)}
        g = res["grounding"]["test"]
        f1 = g["f1"] if isinstance(g["f1"], float) else g["f1"]["mean"]
        print(f"  grounding F1={f1}")

        # ---- T2 retrieval -------------------------------------------------
        cands = []
        for f, inst, links in zip(test_figs, test_i, links_per_fig):
            ft = feats[f["figure_id"]]
            grounded = " ".join(ent for _, ent in links)
            cands.append({"figure_id": f["figure_id"], "is_3d": bool(ft["is_3d_pred"]),
                          "text_embedding": vec(f["paragraph_text"]),
                          "gnn_embedding": vec(grounded) if grounded else None,
                          "2d_embedding": vec(ocr_text[f["figure_id"]]) if cfg == "E" else None,
                          "3d_embedding": ft["vis_type_vec"] if cfg in ("D", "E") else None})
        ranks, t_ret = [], []
        for f in test_figs:
            q = vec(f["caption"])
            t0 = time.perf_counter()
            out = retriever.retrieve_multimodal(q, q if cfg == "E" else None,
                                                query_3d_intent(f["caption"]) if cfg in ("D", "E") else None,
                                                q, cands)
            t_ret.append((time.perf_counter() - t0) * 1000)
            ranks.append(ranks_from_scores(out, f["figure_id"]))
        ranks = np.array(ranks)
        res["retrieval_caption_to_figure"] = {
            "recall@1": round(float((ranks <= 1).mean()), 4), "recall@5": round(float((ranks <= 5).mean()), 4),
            "mrr": round(float((1 / ranks).mean()), 4), "gallery_size": len(test_figs),
            "latency_ms_per_query_median": round(float(np.median(t_ret)), 3)}
        res["retrieval_geometry_to_figure"] = t3 if cfg in ("D", "E") else "N/A (no 3D branch)"

        # ---- SCQ ------------------------------------------------------------
        scq, comps = [], {"entity_coverage": [], "technical_correctness": [],
                          "reconstruction_fidelity": [], "retrieval_accuracy": []}
        for f, inst, links, rk in zip(test_figs, test_i, links_per_fig, ranks):
            ent_types = {e["text"]: e["type"] for e in f["entities"]}
            grounded_ents = [{"text": ent, "type": ent_types.get(ent, "METHOD")} for _, ent in links]
            elems = {"elements": inst["elements"]}
            ctx = {"level_results": {"paragraph": [{"text": f["paragraph_text"]}], "section": [{"text": ""}]}}
            gsum = {"edges": [{"source": f"elem:{e}", "target": f"ent:{n}"} for e, n in links], "num_edges": len(links)}
            cap = captioner.generate_caption(elems, ctx, grounded_ents, gsum)["generated_caption"]
            ver = verifier.verify_caption(cap, [{"text": f["paragraph_text"]}])
            gold_ents = {l["entity"] for l in f["gold_links"]}
            cov = sum(1 for e in gold_ents if e.lower() in cap.lower()) / max(len(gold_ents), 1)
            vals = {"entity_coverage": cov, "technical_correctness": ver["technical_correctness_ratio"],
                    "reconstruction_fidelity": token_f1(cap, f["caption"]), "retrieval_accuracy": 1.0 / rk}
            for k, v in vals.items():
                comps[k].append(v)
            scq.append(verifier.compute_scq_metric(**vals)["scq_composite_score"])
        res["scq"] = {"mean": round(float(np.mean(scq)), 4), "std": round(float(np.std(scq)), 4),
                      "components_mean": {k: round(float(np.mean(v)), 4) for k, v in comps.items()}}
        # 3D-figure subset
        idx3 = [i for i, f in enumerate(test_figs) if f["is_3d"]]
        res["subset_3d_figures"] = {"n": len(idx3), "scq_mean": round(float(np.mean([scq[i] for i in idx3])), 4),
                                    "retrieval_mrr": round(float(np.mean([1 / ranks[i] for i in idx3])), 4)}
        print(f"  T2 MRR={res['retrieval_caption_to_figure']['mrr']}  SCQ={res['scq']['mean']}")
        report["configurations"][cfg] = res
        per_item[cfg] = {
            "rr": (1.0 / ranks).tolist(), "scq": scq,
            "link_counts": [[len(l & i["gold"]), len(l - i["gold"]), i["num_gold_links"] - len(l & i["gold"])]
                            for l, i in zip(links_per_fig, test_i)]}

    # ---- paired bootstrap comparisons (resampling test figures) ------------
    comparisons = {}
    for a_cfg, b_cfg in (("A", "B"), ("B", "D"), ("B", "E"), ("D", "E")):
        A, B = per_item[a_cfg], per_item[b_cfg]
        comparisons[f"{b_cfg}_minus_{a_cfg}"] = {
            "grounding_f1_seed42": compare_micro_f1(np.array(A["link_counts"]), np.array(B["link_counts"])),
            "retrieval_mrr": compare_means(A["rr"], B["rr"]),
            "scq": compare_means(A["scq"], B["scq"]),
        }
        c = comparisons[f"{b_cfg}_minus_{a_cfg}"]
        print(f"  {b_cfg}-{a_cfg}: F1 {c['grounding_f1_seed42']['difference']:+.4f} "
              f"[{c['grounding_f1_seed42']['ci95_low']:+.4f},{c['grounding_f1_seed42']['ci95_high']:+.4f}]  "
              f"MRR {c['retrieval_mrr']['difference']:+.4f} [{c['retrieval_mrr']['ci95_low']:+.4f},"
              f"{c['retrieval_mrr']['ci95_high']:+.4f}]  SCQ {c['scq']['difference']:+.4f} "
              f"[{c['scq']['ci95_low']:+.4f},{c['scq']['ci95_high']:+.4f}]")
    report["paired_bootstrap"] = {"method": "paired bootstrap over test figures, 10000 resamples, 95% percentile CI",
                                  "comparisons": comparisons}
    report["per_figure"] = {"figure_ids": [f["figure_id"] for f in test_figs], **per_item}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    write_table(report, a.out.replace(".json", ".md"))
    print(f"Results written to {a.out}")


def _fmt(g, k):
    v = g[k]
    return f"{v['mean']:.4f} ± {v['std']:.4f}" if isinstance(v, dict) else f"{v:.4f}"


def write_table(report: Dict[str, Any], path: str):
    lines = ["| Config | Grounding F1 | Grounding Acc@1 | Grounding MRR | Caption->Fig R@1 | R@5 | MRR | "
             "Geometry->Fig R@1 | SCQ | Train time (s) | Inference ms/fig |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in ("A", "B", "C", "D", "E"):
        r = report["configurations"].get(cfg)
        if r is None:
            continue
        if r["status"] != "MEASURED":
            lines.append(f"| {cfg} {r['name']} | {r['status'][:60]} | | | | | | | | | |")
            continue
        g = r["grounding"]["test"]
        rt = r["retrieval_caption_to_figure"]
        t3 = r["retrieval_geometry_to_figure"]
        tt = r["grounding"]["train_time_s"]
        lines.append(
            f"| {cfg} {r['name']} | {_fmt(g, 'f1')} | {_fmt(g, 'grounding_accuracy@1')} | {_fmt(g, 'mrr')} | "
            f"{rt['recall@1']:.4f} | {rt['recall@5']:.4f} | {rt['mrr']:.4f} | "
            f"{t3['recall@1'] if isinstance(t3, dict) else 'N/A'} | {r['scq']['mean']:.4f} | "
            f"{tt['mean'] if isinstance(tt, dict) else tt} | {r['grounding']['inference_ms_per_figure_median']} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
