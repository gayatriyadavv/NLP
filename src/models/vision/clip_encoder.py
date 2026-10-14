"""
CLIP cross-modal encoder (Phase 2).

Loads a pretrained CLIP model (config ``models.retrieval.multimodal_embedding_model``,
default ``openai/clip-vit-base-patch32``) through HuggingFace ``transformers`` and exposes
normalised image and text embeddings in a shared space.

Status: BLOCKED in the current build environment - huggingface.co is not reachable, so the
weights cannot be downloaded. ``CLIPEncoder.load`` raises ``ModalityUnavailableError`` with the
underlying reason; callers (retrieval, ablation) must report the configuration as not run rather
than substituting other numbers. When the weights are present in the local HF cache (or the
network allows the download) this module works without code changes.

Note: CLIP is a general-domain vision-language model. Its zero-shot alignment for scientific
plots is limited and it does not process raw 3D geometry.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class ModalityUnavailableError(RuntimeError):
    pass


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @classmethod
    def load(cls, config: Optional[Dict[str, Any]] = None, device: str = "cpu") -> "CLIPEncoder":
        name = (config or {}).get("models.retrieval.multimodal_embedding_model", None) or "openai/clip-vit-base-patch32"
        enc = cls(name, device)
        try:
            from transformers import CLIPModel, CLIPProcessor
            enc.model = CLIPModel.from_pretrained(name).to(device).eval()
            enc.processor = CLIPProcessor.from_pretrained(name)
        except Exception as e:
            raise ModalityUnavailableError(f"CLIP weights '{name}' could not be loaded: {type(e).__name__}: {e}")
        return enc

    def encode_images(self, paths: List[str]) -> np.ndarray:
        import torch
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad():
            inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
            z = self.model.get_image_features(**inp)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            inp = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            z = self.model.get_text_features(**inp)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy()
