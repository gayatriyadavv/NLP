"""
Evaluation of figure element extraction against gold elements.

Matching: a predicted text element matches a gold element when their normalised texts have a
difflib similarity >= ``threshold`` (greedy one-to-one, best pairs first).
Reported: text precision / recall / F1, type accuracy on matched pairs, mean bbox IoU.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9+#/]", "", t.lower())


def text_sim(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def bbox_iou(a: List[int], b: List[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_elements(pred: List[Dict[str, Any]], gold: List[Dict[str, Any]],
                   threshold: float = 0.75) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching. Returns (pred_idx, gold_idx, similarity)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s = text_sim(p.get("text", ""), g.get("text", ""))
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, s))
    return out


def evaluate_extraction(per_figure: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
                        threshold: float = 0.75) -> Dict[str, float]:
    """per_figure: list of (predicted_text_elements, gold_elements)."""
    n_pred = n_gold = n_match = n_type = 0
    ious = []
    for pred, gold in per_figure:
        pred = [p for p in pred if p.get("type") != "CHART_REGION"]
        m = match_elements(pred, gold, threshold)
        n_pred += len(pred)
        n_gold += len(gold)
        n_match += len(m)
        for i, j, _ in m:
            n_type += int(pred[i]["type"] == gold[j]["type"])
            if pred[i].get("bbox") and gold[j].get("bbox"):
                ious.append(bbox_iou(pred[i]["bbox"], gold[j]["bbox"]))
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_gold if n_gold else 0.0
    return {
        "text_precision": round(p, 4),
        "text_recall": round(r, 4),
        "text_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "type_accuracy_on_matched": round(n_type / n_match, 4) if n_match else 0.0,
        "mean_bbox_iou_on_matched": round(float(sum(ious) / len(ious)), 4) if ious else 0.0,
        "num_pred": n_pred, "num_gold": n_gold, "num_matched": n_match,
        "match_threshold": threshold,
    }
