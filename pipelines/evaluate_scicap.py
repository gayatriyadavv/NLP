"""
Evaluation on REAL arXiv figures: the SciCap subset fetched by scripts/fetch_scicap_subset.py
(data/scicap_subset/metadata.json + images/).

SciCap provides author-written captions and (in the val metadata) referring paragraphs, but no
element -> entity grounding labels and no 3D figures. Measured here:

  1. OCR element extraction: number of text elements per figure (no gold boxes in SciCap, so no
     precision/recall).
  2. 3D detector on real graph plots: SciCap figures are graph plots, so every "is_3d" prediction
     is a false positive -> out-of-domain false-positive rate of the CNN trained on SciFig-Controlled.
  3. Caption -> figure retrieval over the subset (query = author caption), with channels:
       text      paragraph / mention text of the figure (TF-IDF)
       ocr       OCR text of the figure image
       grounded  entities grounded to figure elements by the GNN (or rules if no checkpoint)
     reported as R@1, R@5, MRR per channel combination, with paired bootstrap CIs vs text-only.
  4. Generated caption vs author caption: BLEU-4 and ROUGE-L (corpus mean).
  5. SCQ with all components computed.

Entities come from the regex NER (no gold entities in SciCap) - a known weak point.

Usage:
    python pipelines/evaluate_scicap.py [--subset data/scicap_subset] [--limit N]
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.bootstrap import compare_means
from src.evaluation.hallucination_checker import HallucinationChecker
from src.generation.caption_generator import GroundedCaptionGenerator
from src.grounding.entity_graph import EntityGroundingGraph
from src.nlp.scientific_ner import ScientificNERExtractor
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.utils.config import get_config
from src.vision.figure_processor import FigureProcessor
from src.vision.ocr_engine import OCREngine
from src.vision.threed_detector import ThreeDVisualizationDetector


# ---------------------------------------------------------------- caption metrics
def _toks(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def bleu4(hyp: str, ref: str) -> float:
    """Sentence BLEU-4 with add-one smoothing (Lin & Och 2004) and brevity penalty."""
    h, r = _toks(hyp), _toks(ref)
    if not h or not r:
        return 0.0
    logp = 0.0
    for n in range(1, 5):
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        match = sum(min(c, rc[g]) for g, c in hc.items())
        total = max(sum(hc.values()), 0)
        logp += math.log((match + 1) / (total + 1))
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(logp / 4)


def rouge_l(hyp: str, ref: str) -> float:
    h, r = _toks(hyp), _toks(ref)
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def strip_caption(paragraph: str, caption: str) -> str:
    """
    Remove caption leakage from SciCap's PDF-extracted paragraph text: the verbatim caption,
    "Figure N:" caption labels, and any sentence sharing >= 60% of the caption's tokens.
    """
    text = paragraph.replace(caption, " ")
    # caption labels such as "Figure 3:" / "Fig. 3:" (in-text references like "in Figure 3," stay)
    text = re.sub(r"\b(Figure|Fig\.?)\s*\d+[a-z]?\s*:\s*", " ", text)
    cap = set(_toks(caption))
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        st = set(_toks(sent))
        if cap and st and len(st & cap) / len(cap) >= 0.6:
            continue
        kept.append(sent)
    return " ".join(kept).strip()


def ranks_for(retriever, queries, cands) -> np.ndarray:
    ranks = []
    for gold_id, q in queries:
        out = retriever.retrieve_multimodal(q, q, None, q, cands)
        ranks.append(next(k + 1 for k, r in enumerate(out) if r["figure_id"] == gold_id))
    return np.array(ranks)


def summarize(ranks: np.ndarray) -> Dict[str, float]:
    return {"recall@1": round(float((ranks <= 1).mean()), 4), "recall@5": round(float((ranks <= 5).mean()), 4),
            "mrr": round(float((1 / ranks).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/scicap_subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/scicap_results.json")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.subset, "metadata.json")))
    from PIL import Image
    recs, unreadable = [], []
    for r in meta["records"]:
        if not r.get("caption"):
            continue
        try:
            Image.open(os.path.join(a.subset, r["image_path"])).verify()
            recs.append(r)
        except Exception:
            unreadable.append(r["figure_id"])
    if unreadable:
        print(f"skipping {len(unreadable)} unreadable image(s): {unreadable}")
    if a.limit:
        recs = recs[: a.limit]
    if len(recs) < 10:
        sys.exit(f"Only {len(recs)} records with captions in {a.subset} - nothing meaningful to evaluate.")
    print(f"{len(recs)} SciCap figures with captions")

    config = get_config()
    ocr = OCREngine(cache_path=os.path.join(a.subset, "cache", "ocr.json"))
    processor = FigureProcessor(ocr_engine=ocr)
    detector = ThreeDVisualizationDetector()
    ner = ScientificNERExtractor()
    grapher = EntityGroundingGraph(config)
    captioner = GroundedCaptionGenerator()
    verifier = HallucinationChecker(config, use_nli=False)

    rows: List[Dict[str, Any]] = []
    t_ocr, t_ground = [], []
    for i, r in enumerate(recs):
        path = os.path.join(a.subset, r["image_path"])
        t0 = time.perf_counter()
        fig = processor.process_figure(path)
        t_ocr.append((time.perf_counter() - t0) * 1000)
        text_elems = [e for e in fig["elements"] if e["type"] != "CHART_REGION"]
        para_raw = r.get("paragraph_text") or ""
        para = strip_caption(para_raw, r["caption"])
        ents = ner.extract_entities(para)
        rels = ner.extract_relations(ents, para)
        t0 = time.perf_counter()
        grapher.build_graph(fig["elements"], ents, rels, paragraph=para)
        summary = grapher.get_grounding_summary()
        t_ground.append((time.perf_counter() - t0) * 1000)
        grounded = [e["target"][4:] for e in summary["edges"]
                    if e["source"].startswith("elem:") and e["target"].startswith("ent:")]
        det = detector.detect_3d_content(path, mode="image")
        ctx = {"level_results": {"paragraph": [{"text": para}], "section": [{"text": ""}]}}
        cap = captioner.generate_caption(fig, ctx, ents, summary)["generated_caption"]
        ver = verifier.verify_caption(cap, [{"text": para + " " + r.get("ocr_text", "")}])
        rows.append({"figure_id": r["figure_id"], "paragraph": para, "paragraph_raw": para_raw,
                     "ocr": " ".join(e["text"] for e in text_elems), "grounded": " ".join(grounded),
                     "n_text_elements": len(text_elems), "n_entities": len(ents), "n_grounded": len(grounded),
                     "is_3d_pred": det["is_3d"], "generated_caption": cap, "caption": r["caption"],
                     "bleu4": bleu4(cap, r["caption"]), "rouge_l": rouge_l(cap, r["caption"]),
                     "technical_correctness": ver["technical_correctness_ratio"]})
        if i % 25 == 0:
            ocr.save_cache()
            print(f"  {i}/{len(recs)}")
    ocr.save_cache()

    # OCR latency on figures NOT in the cache (the loop above mostly hits the cache)
    fresh = OCREngine()
    t_fresh = []
    for x in recs[:10]:
        t0 = time.perf_counter()
        fresh.read(os.path.join(a.subset, x["image_path"]))
        t_fresh.append((time.perf_counter() - t0) * 1000)

    # ---- retrieval: caption -> figure --------------------------------------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(
        [x["paragraph"] + " " + x["ocr"] + " " + x["caption"] for x in rows])
    vec = lambda t: tfidf.transform([t]).toarray()[0] if t.strip() else None
    queries = [(x["figure_id"], vec(x["caption"])) for x in rows]
    configs = {
        "text_only": {"w_text": 1.0},
        "text+ocr": {"w_text": 0.6, "w_2d": 0.4},
        "text+grounded": {"w_text": 0.7, "w_gnn": 0.3},
        "text+ocr+grounded": {"w_text": 0.5, "w_2d": 0.3, "w_gnn": 0.2},
        "ocr_only": {"w_2d": 1.0},
    }
    cands = [{"figure_id": x["figure_id"], "is_3d": False, "text_embedding": vec(x["paragraph"]),
              "2d_embedding": vec(x["ocr"]), "gnn_embedding": vec(x["grounded"])} for x in rows]
    retrieval, rr = {}, {}
    for name, w in configs.items():
        ranks = ranks_for(MultimodalRetriever(weights=w), queries, cands)
        retrieval[name] = summarize(ranks)
        rr[name] = (1 / ranks).tolist()
        print(f"  retrieval {name:<20} {retrieval[name]}")
    # Same text-only retrieval on the RAW paragraph (caption leakage not removed), for reference
    cands_raw = [dict(c, text_embedding=vec(x["paragraph_raw"])) for c, x in zip(cands, rows)]
    leak = summarize(ranks_for(MultimodalRetriever(weights={"w_text": 1.0}), queries, cands_raw))
    print(f"  retrieval text_only on RAW paragraph (leaky) {leak}")
    sig = {f"{n}_vs_text_only": compare_means(rr["text_only"], rr[n]) for n in configs if n != "text_only"}

    scq = [verifier.compute_scq_metric(
        entity_coverage=(sum(1 for e in x["grounded"].split() if e.lower() in x["generated_caption"].lower())
                         / max(len(x["grounded"].split()), 1)) if x["grounded"] else 0.0,
        technical_correctness=x["technical_correctness"],
        reconstruction_fidelity=x["rouge_l"],
        retrieval_accuracy=v)["scq_composite_score"] for x, v in zip(rows, rr["text+ocr+grounded"])]

    n = len(rows)
    report = {
        "dataset": {"name": meta.get("dataset"), "license": meta.get("license"), "num_figures": n,
                    "source": "SciCap val split subset (real arXiv graph plots, author captions)",
                    "skipped_unreadable_images": unreadable},
        "hardware": {"platform": platform.platform(), "torch": torch.__version__, "device": "cpu"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ocr": {"mean_text_elements_per_figure": round(float(np.mean([x["n_text_elements"] for x in rows])), 2),
                "figures_with_no_text_detected": int(sum(x["n_text_elements"] == 0 for x in rows)),
                "latency_ms_median_uncached": round(float(np.median(t_fresh)), 1),
                "note": "3D detection for this dataset is reported per mode in results/threed_detector_modes.json"},
        "ner_and_grounding": {"mean_regex_entities_per_figure": round(float(np.mean([x["n_entities"] for x in rows])), 2),
                              "figures_with_no_entities": int(sum(x["n_entities"] == 0 for x in rows)),
                              "mean_grounded_links_per_figure": round(float(np.mean([x["n_grounded"] for x in rows])), 2),
                              "grounding_mode": grapher.gnn_metrics.get("mode", "rules"),
                              "latency_ms_median": round(float(np.median(t_ground)), 2),
                              "note": "No grounding labels in SciCap - link accuracy cannot be measured."},
        "threed_detector_out_of_domain": {
            "mode": "image (pixels only)",
            "false_positive_rate_on_graph_plots": round(float(np.mean([x["is_3d_pred"] for x in rows])), 4)},
        "leakage_check": {
            "paragraphs_containing_caption_verbatim": int(sum(x["caption"][:60].lower() in x["paragraph_raw"].lower() for x in rows)),
            "mitigation": "caption text, 'Figure N:' labels and sentences sharing >=60% of caption tokens removed from the paragraph before retrieval and verification",
            "text_only_retrieval_on_raw_paragraph_LEAKY": leak},
        "retrieval_caption_to_figure": {"gallery_size": n, "configs": retrieval,
                                        "paired_bootstrap_vs_text_only": sig},
        "caption_quality_vs_author_caption": {
            "bleu4_mean": round(float(np.mean([x["bleu4"] for x in rows])), 4),
            "rouge_l_mean": round(float(np.mean([x["rouge_l"] for x in rows])), 4),
            "technical_correctness_mean": round(float(np.mean([x["technical_correctness"] for x in rows])), 4)},
        "scq": {"mean": round(float(np.mean(scq)), 4), "std": round(float(np.std(scq)), 4)},
        "examples": [{k: x[k] for k in ("figure_id", "caption", "generated_caption", "ocr")} for x in rows[:5]],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("ocr", "threed_detector_out_of_domain",
                                             "caption_quality_vs_author_caption", "scq")}, indent=1))
    print(f"Results written to {a.out}")


if __name__ == "__main__":
    main()
