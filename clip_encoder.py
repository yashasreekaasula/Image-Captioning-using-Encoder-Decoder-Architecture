"""
CLIP Vision Encoder via HuggingFace transformers.
Produces a 512-dimensional image embedding aligned with text.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from loguru import logger

from .base import BaseEncoder
from backend.config import settings


class CLIPEncoder(BaseEncoder):
    """
    OpenAI CLIP-ViT-B/32 vision encoder.
    image_embeds → 512-d feature vector.
    """

    def __init__(self):
        logger.info(f"Loading CLIP encoder: {settings.clip_model_id} …")
        self._processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
        self._model = CLIPModel.from_pretrained(settings.clip_model_id)
        self._model.eval()
        self._model.to(settings.device)
        logger.info("CLIP encoder ready.")

    @property
    def feature_dim(self) -> int:
        return 512

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Returns (1, 512) tensor."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(settings.device) for k, v in inputs.items()}
        image_features = self._model.get_image_features(**inputs)  # (1, 512)
        # L2-normalise (CLIP convention)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu()