"""
Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel
from typing import Optional, List


class BoundingBox(BaseModel):
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class CaptionResponse(BaseModel):
    caption: str
    confidence: float
    inference_time_ms: float
    encoder_used: str
    decoder_used: str
    yolo_enabled: bool
    detected_objects: Optional[List[BoundingBox]] = None
    annotated_image_b64: Optional[str] = None  # base64 PNG if YOLO enabled


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None