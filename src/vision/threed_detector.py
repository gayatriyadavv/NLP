"""
3D Scientific Visualization Detector Module.
Identifies whether a scientific figure contains 3D content (molecular structures, 3D surface
plots, 3D scatter / point-cloud plots).

Two signals, reported separately:
* image classifier (primary): FigureImageEncoder trained on SciFig-Controlled
  (checkpoint ``models/checkpoints/figure_classifier.pt``, produced by
  ``pipelines/evaluate_phase4.py``). Uses only the pixels.
* text cues (secondary): 3D-related keywords in the caption / paper context.

Decision modes:
* "agreement" (default whenever caption / paper text is supplied): 3D only if the image classifier
  AND the text cues both say 3D. Measured on 288 real SciCap graph plots (results/scicap_results.json)
  the image classifier alone flags ~93% of them as 3D (it was trained only on synthetic matplotlib
  renders and does not transfer), the text cues alone ~2%; requiring agreement keeps the
  synthetic-benchmark accuracy and the low real-figure false-positive rate.
* "image": classifier only (used when no text is available; unreliable out of domain).
* "text": text cues only (also used when no classifier checkpoint exists).

The file name is never used as a signal.
"""

import os
import re
from typing import Dict, Any, Optional

import torch

from src.models.vision.figure_image_encoder import (
    CLASS_TO_VIS_TYPE, FIGURE_CLASSES, THREED_CLASSES, FigureImageEncoder, load_figure_tensor,
)

THREED_KEYWORDS = [
    "3d", "point cloud", "pointcloud", "molecular", "surface render",
    "volumetric", "protein structure", "plddt", "mesh", "atomic coordinates",
    "alphafold", "pdb", "3d scatter", "voxel", "stereographic", "crystal lattice",
    "c-alpha trace", "surface plot"
]
DEFAULT_CHECKPOINT = "models/checkpoints/figure_classifier.pt"


class ThreeDVisualizationDetector:
    def __init__(self, config: Dict[str, Any] = None, checkpoint_path: Optional[str] = None):
        self.config = config or {}
        # checkpoint_path="" explicitly disables the image classifier (text cues only)
        if checkpoint_path is None:
            checkpoint_path = self.config.get("models.vision.figure_classifier_checkpoint", DEFAULT_CHECKPOINT)
        self.checkpoint_path = checkpoint_path
        self.model: Optional[FigureImageEncoder] = None
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.model = FigureImageEncoder()
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()

    @staticmethod
    def text_cues(caption_text: str = "", paper_context: str = "") -> list:
        corpus = f"{caption_text} {paper_context}".lower()
        return [kw for kw in THREED_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", corpus)]

    @torch.no_grad()
    def classify_image(self, figure_path: str) -> Optional[Dict[str, Any]]:
        if self.model is None or not os.path.exists(figure_path):
            return None
        _, logits = self.model(load_figure_tensor(figure_path).unsqueeze(0))
        probs = torch.softmax(logits, -1)[0]
        cls = FIGURE_CLASSES[int(probs.argmax())]
        p3d = float(sum(probs[FIGURE_CLASSES.index(c)] for c in THREED_CLASSES))
        return {"figure_class": cls, "p_3d": p3d, "class_probability": float(probs.max())}

    def detect_3d_content(self, figure_path: str, caption_text: str = "", paper_context: str = "",
                          mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns is_3d, visualization_type, confidence, detected_cues, method and status.
        mode: "agreement" | "image" | "text"; default = "agreement" if any text is given, else "image".
        """
        cues = self.text_cues(caption_text, paper_context)
        img = self.classify_image(figure_path)
        has_text = bool((caption_text or "").strip() or (paper_context or "").strip())
        mode = mode or ("agreement" if has_text else "image")

        if img is not None and mode == "agreement":
            img_3d = img["p_3d"] >= 0.5
            is_3d = img_3d and len(cues) > 0
            vis_type = (CLASS_TO_VIS_TYPE[img["figure_class"]] if img["figure_class"] in THREED_CLASSES
                        else "3D_SCIENTIFIC_RENDER") if is_3d else "2D_PLOT"
            confidence = img["p_3d"] if is_3d else max(1 - img["p_3d"], 0.5 if img_3d else 0.0)
            method = "image_and_text_agreement"
            status = ("IMPLEMENTED (3D only when the image classifier and text cues agree; image classifier "
                      "alone has a high false-positive rate on real figures)")
        elif img is not None and mode == "image":
            is_3d = img["p_3d"] >= 0.5
            vis_type = CLASS_TO_VIS_TYPE[img["figure_class"]] if is_3d == (img["figure_class"] in THREED_CLASSES) \
                else ("3D_SCIENTIFIC_RENDER" if is_3d else "2D_PLOT")
            confidence = img["p_3d"] if is_3d else 1 - img["p_3d"]
            method = "image_classifier"
            status = ("PARTIAL (image classifier only; trained on synthetic renders - flags ~93% of real "
                      "SciCap graph plots as 3D, provide caption/paper text for agreement mode)")
        else:
            is_3d = len(cues) > 0
            vis_type = "3D_SCIENTIFIC_RENDER" if is_3d else "2D_PLOT"
            confidence = min(0.5 + 0.1 * len(cues), 0.9) if is_3d else 0.5
            method = "text_cues_only"
            reason = ("requested" if mode == "text" else "figure file not found" if not os.path.exists(figure_path)
                      else "no classifier checkpoint")
            status = f"PARTIAL (text keyword cues only: {reason})"

        return {
            "is_3d": bool(is_3d),
            "visualization_type": vis_type,
            "confidence": float(round(confidence, 4)),
            "detected_cues": cues,
            "image_classifier": img,
            "method": method,
            "status": status,
        }
