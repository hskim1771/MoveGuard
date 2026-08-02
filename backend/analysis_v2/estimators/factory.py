from __future__ import annotations

import os

from ..errors import EstimatorUnavailable
from .base import HandEstimator, ObjectDetector, PoseEstimator
from .mediapipe_hand import default_hand_estimator
from .mediapipe_pose import default_pose_estimator
from .object_detector import NoopObjectDetector, default_object_detector


def build_pose_estimator() -> PoseEstimator:
    backend = os.getenv("MOVEGUARD_POSE_ESTIMATOR", "mediapipe").strip().lower()
    if backend == "mediapipe":
        return default_pose_estimator()
    raise EstimatorUnavailable(
        f"Unsupported MOVEGUARD_POSE_ESTIMATOR={backend!r}. "
        "Add an adapter implementing PoseEstimator before selecting it."
    )


def build_hand_estimator() -> HandEstimator:
    backend = os.getenv("MOVEGUARD_HAND_ESTIMATOR", "mediapipe").strip().lower()
    if backend == "mediapipe":
        return default_hand_estimator()
    raise EstimatorUnavailable(
        f"Unsupported MOVEGUARD_HAND_ESTIMATOR={backend!r}. "
        "Add an adapter implementing HandEstimator before selecting it."
    )


def build_object_detector() -> ObjectDetector:
    backend = os.getenv("MOVEGUARD_OBJECT_DETECTOR", "auto").strip().lower()
    if backend in {"auto", "none", "ultralytics"}:
        if backend == "none":
            return NoopObjectDetector()
        if backend == "ultralytics" and not os.getenv(
            "OBJECT_DETECTOR_MODEL_PATH", ""
        ).strip():
            raise EstimatorUnavailable(
                "OBJECT_DETECTOR_MODEL_PATH is required for the ultralytics adapter."
            )
        return default_object_detector()
    raise EstimatorUnavailable(
        f"Unsupported MOVEGUARD_OBJECT_DETECTOR={backend!r}. "
        "Add an adapter implementing ObjectDetector before selecting it."
    )
