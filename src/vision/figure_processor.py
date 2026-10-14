"""
Figure Processing & Element Detection Module
Extracts text-bearing visual elements (axes, legends, labels) and the chart region from
scientific figures using real OCR plus layout heuristics.

Pipeline:
1. OCR (EasyOCR, two orientations) -> text boxes            [src/vision/ocr_engine.py]
2. Drop numeric tick labels.
3. Classify remaining boxes by position/orientation:
   - vertical text, left half                 -> AXIS   (y-axis label)
   - vertical text, right half                -> LABEL  (colour-bar label)
   - horizontal text, top band                -> LABEL  (title)
   - >=2 left-aligned stacked lines inside     -> LEGEND (legend entries)
   - bottom band, lowest row                  -> AXIS   (x-axis label)
   - bottom band, rows above the lowest       -> LABEL  (categorical tick labels)
   - anything else                            -> LABEL
4. CHART_REGION = bounding box of non-background pixels outside text boxes.

Accuracy of these heuristics is measured against gold boxes by
``src/evaluation/figure_extraction_eval.py``; see docs/BENCHMARK.md for numbers.
No element is ever returned that was not detected in the image.
"""

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from src.vision.ocr_engine import OCREngine, OCRUnavailableError

_NUMERIC_CHARS = set("0123456789.,-−+eE%xX ()[]")


def is_numeric_text(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    digits = sum(c.isdigit() for c in t)
    numeric_like = sum(c in _NUMERIC_CHARS for c in t)
    return digits > 0 and numeric_like / len(t) >= 0.8


class FigureProcessor:
    def __init__(self, config: Dict[str, Any] = None, ocr_engine: Optional[OCREngine] = None,
                 bottom_band: float = 0.80, top_band: float = 0.13, row_tol: int = 10):
        self.config = config or {}
        self.ocr = ocr_engine or OCREngine()
        self.bottom_band = bottom_band
        self.top_band = top_band
        self.row_tol = row_tol

    # ------------------------------------------------------------------
    def _find_legend(self, boxes: List[Dict[str, Any]], H: int) -> set:
        """Indices of horizontal boxes forming a left-aligned vertical stack (legend)."""
        cand = [i for i, b in enumerate(boxes)
                if b["orientation"] == "horizontal"
                and b["bbox"][1] > self.top_band * H * 0.5
                and b["bbox"][3] < self.bottom_band * H]
        legend = set()
        for i in cand:
            bi = boxes[i]["bbox"]
            hi = bi[3] - bi[1]
            stack = [j for j in cand if abs(boxes[j]["bbox"][0] - bi[0]) <= 12
                     and abs((boxes[j]["bbox"][3] - boxes[j]["bbox"][1]) - hi) <= 8]
            if len(stack) >= 2:
                ys = sorted(boxes[j]["bbox"][1] for j in stack)
                gaps = np.diff(ys)
                if len(gaps) and gaps.max() <= 3.5 * hi:
                    legend.update(stack)
        return legend

    def _classify(self, boxes: List[Dict[str, Any]], W: int, H: int) -> List[str]:
        types = ["LABEL"] * len(boxes)
        legend = self._find_legend(boxes, H)
        bottom = []
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = b["bbox"]
            cx = (x0 + x1) / 2
            if b["orientation"] == "vertical":
                types[i] = "AXIS" if cx < 0.5 * W else "LABEL"
            elif i in legend:
                types[i] = "LEGEND"
            elif y1 < self.top_band * H:
                types[i] = "LABEL"
            elif y0 > self.bottom_band * H:
                bottom.append(i)
        if bottom:
            centers = {i: (boxes[i]["bbox"][1] + boxes[i]["bbox"][3]) / 2 for i in bottom}
            lowest = max(centers.values())
            for i in bottom:
                types[i] = "AXIS" if lowest - centers[i] <= self.row_tol else "LABEL"
        return types

    @staticmethod
    def _chart_region(img: np.ndarray, text_boxes: List[List[int]]) -> Optional[List[int]]:
        gray = img.mean(axis=2)
        mask = gray < 245
        for x0, y0, x1, y1 in text_boxes:
            mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = False
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return [int(np.percentile(xs, 1)), int(np.percentile(ys, 1)),
                int(np.percentile(xs, 99)), int(np.percentile(ys, 99))]

    # ------------------------------------------------------------------
    def process_figure(self, figure_path: str) -> Dict[str, Any]:
        """
        Processes a scientific figure and returns detected elements:
            {"figure_id", "dimensions", "status", "elements": [{"id","type","text","bbox","confidence"}]}
        """
        figure_id = os.path.basename(figure_path).replace(".", "_")
        if not os.path.exists(figure_path):
            return {"figure_id": figure_id, "dimensions": None, "elements": [],
                    "status": "NOT AVAILABLE (figure file not found)"}

        from PIL import Image
        img = np.array(Image.open(figure_path).convert("RGB"))
        H, W = img.shape[:2]

        try:
            dets = self.ocr.read(figure_path)
        except OCRUnavailableError as e:
            return {"figure_id": figure_id, "dimensions": {"width": W, "height": H}, "elements": [],
                    "status": f"UNAVAILABLE (OCR engine not available: {e})"}

        text_boxes = [d for d in dets if not is_numeric_text(d["text"])]
        types = self._classify(text_boxes, W, H)

        elements: List[Dict[str, Any]] = []
        for d, t in zip(text_boxes, types):
            elements.append({"id": f"{t.lower()}_{len(elements)}", "type": t,
                             "text": re.sub(r"\s+", " ", d["text"]).strip(),
                             "bbox": d["bbox"], "confidence": round(d["confidence"], 4)})

        region = self._chart_region(img, [d["bbox"] for d in dets])
        if region:
            elements.append({"id": f"chart_region_{len(elements)}", "type": "CHART_REGION",
                             "text": "", "bbox": region, "confidence": 1.0})

        return {
            "figure_id": figure_id,
            "dimensions": {"width": W, "height": H},
            "status": "IMPLEMENTED (EasyOCR + layout heuristics)",
            "elements": elements,
        }
