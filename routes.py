"""
FastAPI route definitions for the Image Captioning Studio.
"""

from __future__ import annotations

import time
import io
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from loguru import logger
from PIL import Image

from backend.schemas import CaptionResponse, ErrorResponse
from backend.utils.image_utils import validate_image, image_to_base64, ImageValidationError
from backend.models.object_detection import yolo_detector
from backend.models.encoders import ResNetEncoder, ViTEncoder, CLIPEncoder
from backend.models.decoders import LSTMAttentionDecoder, TransformerCaptionDecoder, BLIPDecoder
from backend.models.fusion import FeatureFusion

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy model registry (models loaded on first request, cached thereafter)
# ---------------------------------------------------------------------------

_encoder_cache: dict = {}
_decoder_cache: dict = {}
_fusion_cache: dict = {}


def _get_encoder(name: str):
    if name not in _encoder_cache:
        if name == "resnet":
            _encoder_cache[name] = ResNetEncoder()
        elif name == "vit":
            _encoder_cache[name] = ViTEncoder()
        elif name == "clip":
            _encoder_cache[name] = CLIPEncoder()
        else:
            raise ValueError(f"Unknown encoder: {name}")
    return _encoder_cache[name]


def _get_decoder(name: str, encoder_dim: int):
    if name not in _decoder_cache:
        if name == "lstm":
            _decoder_cache[name] = LSTMAttentionDecoder(encoder_dim=encoder_dim)
        elif name == "transformer":
            _decoder_cache[name] = TransformerCaptionDecoder(encoder_dim=encoder_dim)
        elif name == "blip":
            _decoder_cache[name] = BLIPDecoder()
        else:
            raise ValueError(f"Unknown decoder: {name}")
    return _decoder_cache[name]


def _get_fusion(feature_dim: int):
    key = f"fusion_{feature_dim}"
    if key not in _fusion_cache:
        _fusion_cache[key] = FeatureFusion(feature_dim=feature_dim)
    return _fusion_cache[key]


# ---------------------------------------------------------------------------
# Caption endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/caption",
    response_model=CaptionResponse,
    summary="Generate caption for an uploaded image",
)
async def generate_caption(
    file: UploadFile = File(..., description="Image file (jpg/jpeg/png)"),
    encoder: str = Form("resnet", description="resnet | vit | clip"),
    decoder: str = Form("blip", description="lstm | transformer | blip"),
    use_yolo: bool = Form(False, description="Enable YOLO object detection"),
):
    """
    Full pipeline:
    1. Validate image
    2. Optionally run YOLO
    3. Encode with selected encoder (skip for BLIP-only path)
    4. Fuse features (if YOLO enabled)
    5. Decode caption
    6. Return structured response
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Read & validate                                                   #
    # ------------------------------------------------------------------ #
    data = await file.read()
    try:
        image: Image.Image = validate_image(
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            data=data,
        )
    except ImageValidationError as exc:
        logger.warning(f"Image validation failed: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))

    detected_objects = None
    annotated_b64: Optional[str] = None

    # ------------------------------------------------------------------ #
    # 2. Optional YOLO detection                                           #
    # ------------------------------------------------------------------ #
    if use_yolo:
        try:
            detected_objects, annotated_image = yolo_detector.detect(image)
            annotated_b64 = image_to_base64(annotated_image)
        except Exception as exc:
            logger.error(f"YOLO detection failed: {exc}")
            raise HTTPException(status_code=500, detail=f"YOLO error: {exc}")

    # ------------------------------------------------------------------ #
    # 3. BLIP fast-path (no external encoder needed)                       #
    # ------------------------------------------------------------------ #
    if decoder == "blip":
        try:
            blip: BLIPDecoder = _get_decoder("blip", encoder_dim=0)
            caption, confidence = blip.decode_image(image)
        except Exception as exc:
            logger.error(f"BLIP decoding failed: {exc}")
            raise HTTPException(status_code=500, detail=f"BLIP error: {exc}")

        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
        return CaptionResponse(
            caption=caption,
            confidence=confidence,
            inference_time_ms=inference_ms,
            encoder_used="BLIP-internal",
            decoder_used="blip",
            yolo_enabled=use_yolo,
            detected_objects=detected_objects,
            annotated_image_b64=annotated_b64,
        )

    # ------------------------------------------------------------------ #
    # 4. Encode                                                            #
    # ------------------------------------------------------------------ #
    try:
        enc_module = _get_encoder(encoder)
        features = enc_module.encode(image)  # (1, D)
    except Exception as exc:
        logger.error(f"Encoding failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Encoder error: {exc}")

    # ------------------------------------------------------------------ #
    # 5. Feature fusion (YOLO regions + global)                           #
    # ------------------------------------------------------------------ #
    if use_yolo and detected_objects:
        try:
            fusion = _get_fusion(enc_module.feature_dim)
            features = fusion.fuse(
                global_feat=features,
                image=image,
                boxes=detected_objects,
                encoder=enc_module,
            )
        except Exception as exc:
            logger.warning(f"Fusion failed (using global only): {exc}")

    # ------------------------------------------------------------------ #
    # 6. Decode                                                            #
    # ------------------------------------------------------------------ #
    try:
        dec_module = _get_decoder(decoder, encoder_dim=enc_module.feature_dim)
        caption, confidence = dec_module.decode(features)
    except Exception as exc:
        logger.error(f"Decoding failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Decoder error: {exc}")

    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    return CaptionResponse(
        caption=caption,
        confidence=confidence,
        inference_time_ms=inference_ms,
        encoder_used=encoder,
        decoder_used=decoder,
        yolo_enabled=use_yolo,
        detected_objects=detected_objects,
        annotated_image_b64=annotated_b64,
    )


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}