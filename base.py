"""
Abstract base class for all caption decoders.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseDecoder(ABC):
    """
    Contract for caption decoders.

    decode(features) → (caption_str, confidence_float)
    """

    @abstractmethod
    def decode(self, features: torch.Tensor) -> Tuple[str, float]:
        """
        Generate a caption from encoder feature tensor.

        Args:
            features: (1, feature_dim) encoder output

        Returns:
            (caption, confidence)  where confidence ∈ [0, 1]
        """