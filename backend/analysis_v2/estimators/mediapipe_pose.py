from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..domain import EstimatorStatus, Landmark, PoseDetection
from ..errors import EstimatorUnavailable
from .base import PoseEstimator

_CONFIDENCE_INDEXES = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def _landmark_confidence(point: Any) -> float:
    visibility = getattr(point, "visibility", None)
    presence = getattr(point, "presence", None)
    values = [float(value) for value in (visibility, presence) if value is not None]
    return min(values) if values else 1.0


def _convert_landmarks(points: Any) -> tuple[Landmark, ...]:
    return tuple(
        Landmark(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            confidence=_landmark_confidence(point),
        )
        for point in points
    )


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe Tasks pose adapter.

    RTMW can replace this class later by implementing the same ``PoseEstimator``
    interface. Downstream angle and RULA code does not import MediaPipe.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_poses: int = 1,
        min_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.max_poses = max(1, min(int(max_poses), 4))
        self.min_confidence = float(min_confidence)
        self._landmarker: Any | None = None

    def _get_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker
        if not self.model_path.is_file():
            raise EstimatorUnavailable(
                f"MediaPipe pose model not found: {self.model_path}"
            )

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as error:
            raise EstimatorUnavailable(
                "The mediapipe package is required for pose inference."
            ) from error

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=self.max_poses,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def estimate(self, image_rgb: Any) -> list[PoseDetection]:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise EstimatorUnavailable(
                "The mediapipe package is required for pose inference."
            ) from error

        landmarker = self._get_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
        world_sets = result.pose_world_landmarks or []
        detections: list[PoseDetection] = []

        for index, points in enumerate(result.pose_landmarks or []):
            landmarks = _convert_landmarks(points)
            world_landmarks = (
                _convert_landmarks(world_sets[index])
                if index < len(world_sets)
                else None
            )
            available = [
                landmarks[item].confidence
                for item in _CONFIDENCE_INDEXES
                if item < len(landmarks)
            ]
            confidence = sum(available) / len(available) if available else 0.0
            detections.append(
                PoseDetection(
                    person_id=f"person-{index + 1}",
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    confidence=confidence,
                )
            )
        return detections

    def status(self) -> EstimatorStatus:
        configured = self.model_path.is_file()
        return EstimatorStatus(
            name="MediaPipePoseEstimator",
            configured=configured,
            loaded=self._landmarker is not None,
            detail=None if configured else f"model not found: {self.model_path}",
        )


def default_pose_estimator() -> MediaPipePoseEstimator:
    backend_root = Path(__file__).resolve().parents[2]
    return MediaPipePoseEstimator(
        os.getenv(
            "POSE_MODEL_PATH",
            str(backend_root / "pose_landmarker_lite.task"),
        ),
        max_poses=int(os.getenv("MOVEGUARD_MAX_POSES", "1")),
        min_confidence=float(os.getenv("MOVEGUARD_POSE_MIN_CONFIDENCE", "0.5")),
    )
