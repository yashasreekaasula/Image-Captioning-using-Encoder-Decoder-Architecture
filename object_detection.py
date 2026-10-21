"""
YOLOv8 object detection wrapper.
Detects objects, draws bounding boxes, and returns structured results.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
from loguru import logger

from backend.config import settings
from backend.schemas import BoundingBox
from backend.utils.image_utils import pil_to_cv2, cv2_to_pil


class YOLODetector:
    """
    Wraps YOLOv8 for object detection.
    Lazy-loads the model on first use to keep startup fast.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info(f"Loading YOLO model: {settings.yolo_model_id}")
            from ultralytics import YOLO  # deferred import
            self._model = YOLO(settings.yolo_model_id)
            logger.info("YOLO model loaded.")

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.35,
    ) -> Tuple[List[BoundingBox], Image.Image]:
        """
        Run inference on a PIL image.

        Returns:
            boxes    – list of BoundingBox pydantic objects
            annotated – PIL image with drawn bounding boxes
        """
        self._load()
        bgr = pil_to_cv2(image)

        results = self._model(bgr, conf=conf_threshold, verbose=False)
        result = results[0]

        boxes: List[BoundingBox] = []
        annotated_bgr = bgr.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            boxes.append(
                BoundingBox(
                    label=label,
                    confidence=round(conf, 4),
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    x2=round(x2, 2),
                    y2=round(y2, 2),
                )
            )

            # Draw rectangle + label
            cv2.rectangle(
                annotated_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_bgr,
                f"{label} {conf:.2f}",
                (int(x1), max(int(y1) - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        annotated_pil = cv2_to_pil(annotated_bgr)
        logger.info(f"YOLO detected {len(boxes)} objects.")
        return boxes, annotated_pil


# Module-level singleton
yolo_detector = YOLODetector()