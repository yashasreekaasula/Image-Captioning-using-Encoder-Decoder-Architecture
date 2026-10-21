"""
ResNet50 encoder.
Uses global average pooling of the final conv block to produce
a 2048-dimensional feature vector.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from loguru import logger

from .base import BaseEncoder
from backend.config import settings


class ResNetEncoder(BaseEncoder):
    """
    Pretrained ResNet50 feature extractor.
    Removes the final classification head; outputs 2048-d features.
    """

    _TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def __init__(self):
        logger.info("Loading ResNet50 encoder …")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Strip the final FC layer; keep avgpool → (B, 2048, 1, 1)
        self._backbone = nn.Sequential(*list(base.children())[:-1])
        self._backbone.eval()
        self._backbone.to(settings.device)
        logger.info("ResNet50 encoder ready.")

    @property
    def feature_dim(self) -> int:
        return 2048

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Returns (1, 2048) tensor."""
        x = self._TRANSFORM(image).unsqueeze(0).to(settings.device)  # (1,3,224,224)
        feat = self._backbone(x)                                       # (1,2048,1,1)
        feat = feat.view(1, -1)                                        # (1,2048)
        return feat.cpu()