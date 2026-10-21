"""
Feature Fusion Module.

When YOLO is enabled we extract features from each detected bounding-box
crop, mean-pool them, and concatenate with the global encoder feature.
A linear projection maps back to the original `feature_dim`.

This is a simple but effective strategy used in region-feature captioners
(e.g., Bottom-Up Top-Down by Anderson et al., 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List
from PIL import Image
from loguru import logger

from backend.schemas import BoundingBox
from backend.models.encoders.base import BaseEncoder


class FeatureFusion(nn.Module):
    """
    Fuses global image features with YOLO region features.

    Output dim == input `feature_dim` (identity when YOLO disabled).
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # 2× feature_dim → feature_dim after concat
        self.proj = nn.Linear(feature_dim * 2, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def fuse(
        self,
        global_feat: torch.Tensor,
        image: Image.Image,
        boxes: List[BoundingBox],
        encoder: BaseEncoder,
    ) -> torch.Tensor:
        """
        Fuse global features with region features from detected boxes.

        Args:
            global_feat: (1, feature_dim) from encoder
            image:       original PIL image
            boxes:       YOLO bounding boxes
            encoder:     the same encoder used for the global feature

        Returns:
            fused: (1, feature_dim)
        """
        if not boxes:
            logger.debug("No YOLO boxes – skipping fusion.")
            return global_feat

        W, H = image.size
        region_feats: List[torch.Tensor] = []

        for box in boxes:
            # Clamp coordinates to image bounds
            x1 = max(int(box.x1), 0)
            y1 = max(int(box.y1), 0)
            x2 = min(int(box.x2), W)
            y2 = min(int(box.y2), H)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            feat = encoder.encode(crop)  # (1, feature_dim)
            region_feats.append(feat)

        if not region_feats:
            return global_feat

        region_mean = torch.stack(region_feats, dim=0).mean(0)  # (1, feature_dim)
        combined = torch.cat([global_feat, region_mean], dim=1)  # (1, 2 * feature_dim)
        fused = self.proj(combined)                               # (1, feature_dim)
        logger.debug(f"Fused {len(region_feats)} region features with global feature.")
        return fused