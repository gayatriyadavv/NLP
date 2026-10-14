"""
Compare the 3D detector's decision modes on synthetic and real figures (no training).

  image      - CNN on pixels only
  text       - 3D keywords in caption / paragraph only
  agreement  - 3D only if both say 3D (default when text is available)

Datasets:
  SciFig-Controlled test split (data/benchmark, 118 figures, 63 3D) - accuracy / precision / recall
  SciCap subset (data/scicap_subset, real arXiv graph plots, all 2D) - false-positive rate

Usage:
    python pipelines/evaluate_3d_detector_modes.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.threed_detector import ThreeDVisualizationDetector

MODES = ("image", "text", "agreement")


def scores(pred, gold):
    pred, gold = np.array(pred, bool), np.array(gold, bool)
    tp, fp, fn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
    return {"accuracy": round(float((pred == gold).mean()), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "false_positive_rate": round(fp / max(int((~gold).sum()), 1), 4),
            "n": int(len(gold))}


def main():
    det = ThreeDVisualizationDetector()
    if det.model is None:
        sys.exit("Missing models/checkpoints/figure_classifier.pt - run pipelines/evaluate_phase4.py first.")
    report = {"modes": list(MODES)}

    bench = json.load(open("data/benchmark/metadata.json"))
    test = [f for f in bench["figures"] if f["split"] == "test"]
    report["scifig_controlled_test"] = {
        m: scores([det.detect_3d_content(os.path.join("data/benchmark", f["image_path"]), f["caption"],
                                         f["paragraph_text"], mode=m)["is_3d"] for f in test],
                  [f["is_3d"] for f in test]) for m in MODES}

    if os.path.exists("data/scicap_subset/metadata.json"):
        recs = []
        for r in json.load(open("data/scicap_subset/metadata.json"))["records"]:
            path = os.path.join("data/scicap_subset", r["image_path"])
            try:
                Image.open(path).verify()
                recs.append((path, r))
            except Exception:
                pass
        report["scicap_real_graph_plots"] = {
            m: scores([det.detect_3d_content(p, r["caption"], r["paragraph_text"], mode=m)["is_3d"] for p, r in recs],
                      [False] * len(recs)) for m in MODES}

    os.makedirs("results", exist_ok=True)
    with open("results/threed_detector_modes.json", "w") as f:
        json.dump(report, f, indent=2)
    for ds in ("scifig_controlled_test", "scicap_real_graph_plots"):
        for m, v in report.get(ds, {}).items():
            print(f"{ds:<26} {m:<10} acc={v['accuracy']} P={v['precision']} R={v['recall']} FPR={v['false_positive_rate']}")


if __name__ == "__main__":
    main()
