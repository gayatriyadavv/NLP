"""
OCR engine wrapper for scientific figures.

Uses EasyOCR (CRAFT detector + CRNN recogniser). Two passes are run per image:
the original orientation and a 90-degree clockwise rotation, so that vertical axis labels
(y-axis, colour-bar labels) are read as horizontal text. Boxes from the rotated pass are mapped
back into original image coordinates.

If EasyOCR or its weights are unavailable, ``OCREngine.available`` is False and ``read`` raises
``OCRUnavailableError``; callers must report that instead of inventing text.
"""

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np


class OCRUnavailableError(RuntimeError):
    pass


_READER = None
_READER_LOCK = threading.Lock()
_READER_ERROR: Optional[str] = None


def _get_reader():
    global _READER, _READER_ERROR
    with _READER_LOCK:
        if _READER is None and _READER_ERROR is None:
            try:
                import easyocr
                _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as e:  # pragma: no cover - depends on environment
                _READER_ERROR = f"{type(e).__name__}: {e}"
    return _READER


def _iou(a: List[int], b: List[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class OCREngine:
    """Thin, cached wrapper around EasyOCR."""

    def __init__(self, cache_path: Optional[str] = None, min_confidence: float = 0.2):
        self.cache_path = cache_path
        self.min_confidence = min_confidence
        self._cache: Dict[str, Any] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    @property
    def available(self) -> bool:
        return _get_reader() is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        _get_reader()
        return _READER_ERROR

    @staticmethod
    def _key(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()

    def save_cache(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)

    def read(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of text detections:
            {"text", "bbox": [x0, y0, x1, y1], "confidence", "orientation": "horizontal"|"vertical"}
        """
        key = self._key(image_path)
        if key in self._cache:
            return self._cache[key]
        reader = _get_reader()
        if reader is None:
            raise OCRUnavailableError(f"EasyOCR unavailable: {_READER_ERROR}")

        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
        H, W = img.shape[:2]
        dets: List[Dict[str, Any]] = []

        for quad, text, conf in reader.readtext(img):
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            dets.append({"text": str(text), "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                         "confidence": float(conf), "orientation": "horizontal"})

        # 90° clockwise rotation: original (x, y) -> rotated (H-1-y, x)
        rot = np.ascontiguousarray(np.rot90(img, k=-1))
        for quad, text, conf in reader.readtext(rot):
            rxs, rys = [p[0] for p in quad], [p[1] for p in quad]
            # inverse: rotated (u, v) -> original (x=v, y=H-1-u)
            ox = [int(v) for v in rys]
            oy = [int(H - 1 - u) for u in rxs]
            box = [min(ox), min(oy), max(ox), max(oy)]
            # keep only genuinely vertical text (taller than wide in original frame)
            if (box[3] - box[1]) > 1.5 * (box[2] - box[0]):
                dets.append({"text": str(text), "bbox": box, "confidence": float(conf), "orientation": "vertical"})

        dets = [d for d in dets if d["confidence"] >= self.min_confidence and d["text"].strip()]
        # de-duplicate overlapping detections, keep higher confidence
        dets.sort(key=lambda d: -d["confidence"])
        kept: List[Dict[str, Any]] = []
        for d in dets:
            if all(_iou(d["bbox"], k["bbox"]) < 0.5 for k in kept):
                kept.append(d)
        self._cache[key] = kept
        return kept
