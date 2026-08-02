from __future__ import annotations

import os

from ..errors import EstimatorUnavailable
from .base import HandEstimator, ObjectDetector, PoseEstimator
from .mediapipe_hand import default_hand_estimator
from .mediapipe_pose import default_pose_estimator
from .object_detector import NoopObjectDetector, default_object_detector
from .rtmw_pose import default_rtmw_estimators


MODEL_ALIASES = {
    "baseline": "baseline",
    "mediapipe": "baseline",
    "v1": "baseline",
    "rtmw": "rtmw",
    "rtmw-l": "rtmw",
    "rtmw_l": "rtmw",
    "v2": "rtmw",
}


def normalize_pose_model(value: str | None = None) -> str:
    selected = (
        value
        or os.getenv("POSE_MODEL")
        or os.getenv("MOVEGUARD_POSE_ESTIMATOR")
        or "baseline"
    ).strip().lower()
    try:
        return MODEL_ALIASES[selected]
    except KeyError as error:
        raise EstimatorUnavailable(
            f"Unsupported pose model {selected!r}; use baseline or rtmw."
        ) from error


def build_model_estimators(model: str | None = None) -> tuple[str, PoseEstimator, HandEstimator]:
    selected = normalize_pose_model(model)
    if selected == "baseline":
        return selected, default_pose_estimator(), default_hand_estimator()
    pose, hand = default_rtmw_estimators()
    return selected, pose, hand


def build_pose_estimator() -> PoseEstimator:
    return build_model_estimators()[1]


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
