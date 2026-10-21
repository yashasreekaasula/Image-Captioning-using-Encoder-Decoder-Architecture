"""
Image validation and pre-processing utilities.
"""

import io
import base64
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from backend.config import settings


class ImageValidationError(ValueError):
    """Raised when the uploaded image fails validation."""


def validate_image(filename: str, content_type: str, data: bytes) -> Image.Image:
    """
    Validate file extension, MIME type, file size, and PIL readability.
    Returns a PIL Image on success; raises ImageValidationError otherwise.
    """
    # 1. Extension check
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_extensions:
        raise ImageValidationError(
            f"Extension '.{ext}' is not allowed. "
            f"Use: {settings.allowed_extensions}"
        )

    # 2. MIME type check
    if content_type not in settings.allowed_mime_types:
        raise ImageValidationError(
            f"MIME type '{content_type}' is not allowed. "
            f"Use: {settings.allowed_mime_types}"
        )

    # 3. Size check
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File size {len(data)} bytes exceeds limit "
            f"of {settings.max_upload_bytes} bytes."
        )

    # 4. PIL readability check
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.verify()  # checks for truncated files
    except Exception as exc:
        raise ImageValidationError(f"Cannot read image data: {exc}")

    # Re-open after verify (verify exhausts the stream)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    logger.info(f"Image validated: {filename} | size={image.size} | mode={image.mode}")
    return image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(array: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_model(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize image to the standard model input size."""
    return image.resize(size, Image.LANCZOS)