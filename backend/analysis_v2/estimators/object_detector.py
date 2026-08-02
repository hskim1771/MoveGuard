from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..domain import EstimatorStatus, ObjectDetection
from ..errors import EstimatorUnavailable
from .base import ObjectDetector


class NoopObjectDetector(ObjectDetector):
    """Explicitly unconfigured detector; it never fabricates detections."""

    def detect(self, image_rgb: Any) -> list[ObjectDetection]:
        return []

    def status(self) -> EstimatorStatus:
        return EstimatorStatus(
            name="NoopObjectDetector",
            configured=False,
            loaded=False,
            detail="OBJECT_DETECTOR_MODEL_PATH is not configured",
        )


class UltralyticsObjectDetector(ObjectDetector):
    """Optional adapter for a custom YOLO model.

    The dependency and model are lazy-loaded. A missing runtime is reported as
    unavailable; it is never replaced by arbitrary demo boxes.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.35,
        allowed_labels: set[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence = float(confidence)
        self.allowed_labels = {item.lower() for item in (allowed_labels or set())}
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise EstimatorUnavailable(
                f"Object detector model not found: {self.model_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise EstimatorUnavailable(
                "Install backend/requirements-object-detection.txt to use the YOLO adapter."
            ) from error
        self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image_rgb: Any) -> list[ObjectDetection]:
        model = self._get_model()
        height, width = image_rgb.shape[:2]
        results = model.predict(image_rgb, conf=self.confidence, verbose=False)
        if not results:
            return []

        first = results[0]
        names = first.names
        detections: list[ObjectDetection] = []
        for index, box in enumerate(first.boxes):
            class_index = int(box.cls[0].item())
            label = str(names[class_index])
            if self.allowed_labels and label.lower() not in self.allowed_labels:
                continue
            x_min, y_min, x_max, y_max = [
                float(value) for value in box.xyxy[0].tolist()
            ]
            detections.append(
                ObjectDetection(
                    detection_id=f"object-{index + 1}",
                    label=label,
                    confidence=float(box.conf[0].item()),
                    bbox=(
                        x_min / width,
                        y_min / height,
                        x_max / width,
                        y_max / height,
                    ),
                )
            )
        return detections

    def status(self) -> EstimatorStatus:
        configured = self.model_path.is_file()
        return EstimatorStatus(
            name="UltralyticsObjectDetector",
            configured=configured,
            loaded=self._model is not None,
            detail=None if configured else f"model not found: {self.model_path}",
        )


def default_object_detector() -> ObjectDetector:
    model_path = os.getenv("OBJECT_DETECTOR_MODEL_PATH", "").strip()
    if not model_path:
        return NoopObjectDetector()
    labels = {
        item.strip().lower()
        for item in os.getenv(
            "MOVEGUARD_OBJECT_CLASSES",
            "box,tool,cart,workbench,load",
        ).split(",")
        if item.strip()
    }
    return UltralyticsObjectDetector(
        model_path,
        confidence=float(os.getenv("MOVEGUARD_OBJECT_MIN_CONFIDENCE", "0.35")),
        allowed_labels=labels,
    )
