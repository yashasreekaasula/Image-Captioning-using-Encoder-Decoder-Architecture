"""
Abstract base class for all image encoders.
Every encoder must implement `encode` and expose `feature_dim`.
"""

from abc import ABC, abstractmethod
import torch
from PIL import Image


class BaseEncoder(ABC):
    """
    Contract for image encoders.

    encode(image) → torch.Tensor of shape (1, feature_dim)
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""

    @abstractmethod
    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a single PIL image.

        Args:
            image: RGB PIL Image

        Returns:
            Tensor of shape (1, feature_dim) on CPU
        """