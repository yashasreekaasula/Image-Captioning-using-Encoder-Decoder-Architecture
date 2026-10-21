"""
Vision Transformer (ViT) encoder via HuggingFace transformers.
Uses the CLS token embedding as the global image representation.
"""

import torch
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor
from loguru import logger

from .base import BaseEncoder
from backend.config import settings


class ViTEncoder(BaseEncoder):
    """
    HuggingFace ViT-Base-16 encoder.
    CLS token → 768-d feature vector.
    """

    def __init__(self):
        logger.info(f"Loading ViT encoder: {settings.vit_model_id} …")
        self._processor = ViTFeatureExtractor.from_pretrained(settings.vit_model_id)
        self._model = ViTModel.from_pretrained(settings.vit_model_id)
        self._model.eval()
        self._model.to(settings.device)
        logger.info("ViT encoder ready.")

    @property
    def feature_dim(self) -> int:
        return 768

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Returns (1, 768) tensor (CLS token)."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(settings.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        # last_hidden_state[:, 0, :] → CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        return cls_token.cpu()