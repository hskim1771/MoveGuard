from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ..domain import EstimatorStatus, HandDetection, Landmark, PoseDetection
from ..errors import EstimatorUnavailable
from .base import HandEstimator

_SIDE_INDEXES = {
    "left": (13, 15),
    "right": (14, 16),
}


def _roi_for_wrist(
    pose: PoseDetection,
    side: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    elbow_index, wrist_index = _SIDE_INDEXES[side]
    if len(pose.landmarks) <= wrist_index:
        return None
    elbow = pose.landmarks[elbow_index]
    wrist = pose.landmarks[wrist_index]
    if min(elbow.confidence, wrist.confidence) < 0.35:
        return None

    elbow_x, elbow_y = elbow.x * width, elbow.y * height
    wrist_x, wrist_y = wrist.x * width, wrist.y * height
    forearm = math.hypot(wrist_x - elbow_x, wrist_y - elbow_y)
    if forearm < 4:
        return None

    center_x = wrist_x + (wrist_x - elbow_x) * 0.28
    center_y = wrist_y + (wrist_y - elbow_y) * 0.28
    half = max(56.0, min(max(width, height) * 0.28, forearm * 1.35))
    x_min = max(0, int(center_x - half))
    y_min = max(0, int(center_y - half))
    x_max = min(width, int(center_x + half))
    y_max = min(height, int(center_y + half))
    if x_max - x_min < 32 or y_max - y_min < 32:
        return None
    return x_min, y_min, x_max, y_max


def _category_score(result: Any, index: int) -> float:
    try:
        return float(result.handedness[index][0].score)
    except (AttributeError, IndexError, TypeError):
        return 0.0


class MediaPipeHandEstimator(HandEstimator):
    """MediaPipe Hand Landmarker adapter using per-wrist regions of interest."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.min_confidence = float(min_confidence)
        self._landmarker: Any | None = None

    def _get_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker
        if not self.model_path.is_file():
            raise EstimatorUnavailable(
                f"MediaPipe hand model not found: {self.model_path}"
            )
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as error:
            raise EstimatorUnavailable(
                "The mediapipe package is required for hand inference."
            ) from error

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        return self._landmarker

    def estimate(
        self,
        image_rgb: Any,
        pose: PoseDetection,
    ) -> dict[str, HandDetection]:
        if not self.model_path.is_file():
            return {}
        try:
            import mediapipe as mp
        except ImportError as error:
            raise EstimatorUnavailable(
                "The mediapipe package is required for hand inference."
            ) from error

        height, width = image_rgb.shape[:2]
        landmarker = self._get_landmarker()
        hands: dict[str, HandDetection] = {}
        for side in ("left", "right"):
            roi = _roi_for_wrist(pose, side, width, height)
            if roi is None:
                continue
            x_min, y_min, x_max, y_max = roi
            crop = image_rgb[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            result = landmarker.detect(mp_image)
            if not result.hand_landmarks:
                continue

            score = _category_score(result, 0)
            if score < self.min_confidence:
                continue
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            landmarks = tuple(
                Landmark(
                    x=(x_min + float(point.x) * crop_width) / width,
                    y=(y_min + float(point.y) * crop_height) / height,
                    z=float(point.z) * crop_width / max(width, 1),
                    confidence=score,
                )
                for point in result.hand_landmarks[0]
            )
            world_landmarks = None
            if result.hand_world_landmarks:
                world_landmarks = tuple(
                    Landmark(
                        x=float(point.x),
                        y=float(point.y),
                        z=float(point.z),
                        confidence=score,
                    )
                    for point in result.hand_world_landmarks[0]
                )
            hands[side] = HandDetection(
                side=side,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                confidence=score,
                roi=(
                    x_min / width,
                    y_min / height,
                    x_max / width,
                    y_max / height,
                ),
            )
        return hands

    def status(self) -> EstimatorStatus:
        configured = self.model_path.is_file()
        return EstimatorStatus(
            name="MediaPipeHandEstimator",
            configured=configured,
            loaded=self._landmarker is not None,
            detail=None if configured else f"model not found: {self.model_path}",
        )


def default_hand_estimator() -> MediaPipeHandEstimator:
    backend_root = Path(__file__).resolve().parents[2]
    return MediaPipeHandEstimator(
        os.getenv(
            "HAND_MODEL_PATH",
            str(backend_root / "hand_landmarker.task"),
        ),
        min_confidence=float(os.getenv("MOVEGUARD_HAND_MIN_CONFIDENCE", "0.5")),
    )
