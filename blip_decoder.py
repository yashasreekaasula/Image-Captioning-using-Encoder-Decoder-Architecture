"""
BLIP pretrained image captioning decoder.
Uses Salesforce/blip-image-captioning-base end-to-end; no separate encoder needed.
Confidence is approximated as the mean of token probabilities from a scoring pass.
"""

from __future__ import annotations

import math
import torch
from PIL import Image
from typing import Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration
from loguru import logger

from .base import BaseDecoder
from backend.config import settings


class BLIPDecoder(BaseDecoder):
    """
    BLIP end-to-end caption generator.
    NOTE: BLIP contains its own vision encoder, so the external encoder
          features are NOT used (pass `features=None`). The raw PIL image
          must be supplied via `decode_image()`.
    """

    def __init__(self):
        logger.info(f"Loading BLIP model: {settings.blip_model_id} …")
        self._processor = BlipProcessor.from_pretrained(settings.blip_model_id)
        self._model = BlipForConditionalGeneration.from_pretrained(
            settings.blip_model_id,
            torch_dtype=torch.float32,
        )
        self._model.eval()
        self._model.to(settings.device)
        logger.info("BLIP decoder ready.")

    # ------------------------------------------------------------------
    # BaseDecoder contract (features unused – BLIP is self-contained)
    # ------------------------------------------------------------------

    def decode(self, features: torch.Tensor) -> Tuple[str, float]:
        """Fallback: not useful without the raw image. Use decode_image()."""
        return "use decode_image() for BLIP", 0.0

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Generate a caption directly from a PIL image.

        Returns:
            (caption_str, confidence_float)
        """
        inputs = self._processor(images=image, return_tensors="pt").to(settings.device)

        # --- Generation ---
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=1.3,
        )
        caption = self._processor.decode(output_ids[0], skip_special_tokens=True)

        # --- Approximate confidence via teacher-forced log-probs ---
        labels = output_ids.clone()
        outputs = self._model(
            pixel_values=inputs["pixel_values"],
            input_ids=output_ids,
            labels=labels,
        )
        # Cross-entropy loss → perplexity → pseudo-confidence
        nll = outputs.loss.item()
        confidence = math.exp(-nll)
        confidence = round(min(max(confidence, 0.0), 1.0), 4)

        return caption.strip(), confidence