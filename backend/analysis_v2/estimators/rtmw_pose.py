from __future__ import annotations

import importlib.util
import os
from typing import Any

from ..domain import EstimatorStatus, HandDetection, Landmark, PoseDetection
from ..errors import EstimatorUnavailable
from .base import HandEstimator, PoseEstimator

# Official OpenMMLab ONNX packages used by rtmlib's RTMW "performance" mode.
# Keep these explicit so an upstream default change cannot silently alter an A/B run.
OFFICIAL_RTM_W_L_384_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
OFFICIAL_YOLOX_M_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)

# COCO-WholeBody 133 -> MediaPipe-compatible 33 slots consumed by the shared
# MoveGuard angle engine. Unmapped slots remain confidence=0 and are never
# treated as detected.
WHOLEBODY_TO_MEDIAPIPE = {
    0: 0,   # nose
    7: 3,   # left ear
    8: 4,   # right ear
    11: 5,  # left shoulder
    12: 6,  # right shoulder
    13: 7,  # left elbow
    14: 8,  # right elbow
    15: 9,  # left wrist
    16: 10, # right wrist
    23: 11, # left hip
    24: 12, # right hip
    25: 13, # left knee
    26: 14, # right knee
    27: 15, # left ankle
    28: 16, # right ankle
    29: 19, # left heel
    30: 22, # right heel
    31: 17, # left big toe / foot index
    32: 20, # right big toe / foot index
}
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


def _landmark(x: float, y: float, score: float, width: int, height: int) -> Landmark:
    return Landmark(
        x=float(x) / max(width, 1),
        y=float(y) / max(height, 1),
        z=0.0,
        confidence=max(0.0, min(1.0, float(score))),
    )


def map_rtmw_wholebody_person(
    keypoints: Any,
    scores: Any,
    *,
    width: int,
    height: int,
    person_id: str,
    min_confidence: float = 0.5,
) -> tuple[PoseDetection, dict[str, HandDetection]]:
    """Map one RTMW COCO-WholeBody result to MoveGuard pose and hand domains."""

    if len(keypoints) < 133 or len(scores) < 133:
        raise ValueError("RTMW output must contain 133 COCO-WholeBody keypoints.")

    pose_points = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    for target_index, source_index in WHOLEBODY_TO_MEDIAPIPE.items():
        pose_points[target_index] = _landmark(
            keypoints[source_index][0],
            keypoints[source_index][1],
            scores[source_index],
            width,
            height,
        )

    body_confidences = [
        pose_points[index].confidence
        for index in (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    ]
    pose = PoseDetection(
        person_id=person_id,
        landmarks=tuple(pose_points),
        world_landmarks=None,
        confidence=sum(body_confidences) / len(body_confidences),
    )

    hands: dict[str, HandDetection] = {}
    for side, keypoint_slice in (
        ("left", LEFT_HAND_SLICE),
        ("right", RIGHT_HAND_SLICE),
    ):
        side_points = keypoints[keypoint_slice]
        side_scores = scores[keypoint_slice]
        hand_landmarks = tuple(
            _landmark(point[0], point[1], score, width, height)
            for point, score in zip(side_points, side_scores)
        )
        # A hand is exposed only when the exact landmarks required by the
        # shared wrist-angle formula (wrist and middle MCP) pass threshold.
        if (
            len(hand_landmarks) == 21
            and hand_landmarks[0].confidence >= min_confidence
            and hand_landmarks[9].confidence >= min_confidence
        ):
            xs = [point.x for point in hand_landmarks if point.confidence >= min_confidence]
            ys = [point.y for point in hand_landmarks if point.confidence >= min_confidence]
            hands[side] = HandDetection(
                side=side,  # type: ignore[arg-type]
                landmarks=hand_landmarks,
                world_landmarks=None,
                confidence=sum(point.confidence for point in hand_landmarks) / 21,
                roi=(min(xs), min(ys), max(xs), max(ys)) if xs and ys else None,
            )
    return pose, hands


class RTMWPoseEstimator(PoseEstimator):
    """Lazy RTMW-l 384x288 whole-body estimator using official ONNX weights."""

    def __init__(
        self,
        *,
        device: str | None = None,
        backend: str = "onnxruntime",
        min_confidence: float | None = None,
    ) -> None:
        self.device = (device or os.getenv("MOVEGUARD_RTMW_DEVICE", "cpu")).strip().lower()
        self.backend = backend
        self.min_confidence = float(
            min_confidence
            if min_confidence is not None
            else os.getenv("MOVEGUARD_RTMW_MIN_CONFIDENCE", "0.5")
        )
        self._model: Any | None = None
        self._hands_by_person: dict[str, dict[str, HandDetection]] = {}

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from rtmlib import Wholebody
        except ImportError as error:
            raise EstimatorUnavailable(
                "RTMW requires rtmlib==0.0.15 and onnxruntime. "
                "Install backend/requirements-rtmw.txt."
            ) from error
        try:
            self._model = Wholebody(
                det=OFFICIAL_YOLOX_M_URL,
                det_input_size=(640, 640),
                pose=OFFICIAL_RTM_W_L_384_URL,
                pose_input_size=(288, 384),
                to_openpose=False,
                backend=self.backend,
                device=self.device,
            )
        except Exception as error:
            raise EstimatorUnavailable(f"RTMW-l model initialization failed: {error}") from error
        return self._model

    def estimate(self, image_rgb: Any) -> list[PoseDetection]:
        try:
            import numpy as np
        except ImportError as error:
            raise EstimatorUnavailable("numpy is required for RTMW inference.") from error
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("RTMW expects an RGB image with shape HxWx3.")
        height, width = image.shape[:2]
        image_bgr = image[:, :, :3][:, :, ::-1].copy()
        try:
            model = self._load()
            bboxes = model.det_model(image_bgr)
            if np.asarray(bboxes).size == 0:
                self._hands_by_person = {}
                return []
            keypoints, scores = model.pose_model(image_bgr, bboxes=bboxes)
        except EstimatorUnavailable:
            raise
        except Exception as error:
            raise EstimatorUnavailable(f"RTMW-l inference failed: {error}") from error

        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.size == 0 or scores.size == 0:
            self._hands_by_person = {}
            return []
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]

        poses: list[PoseDetection] = []
        hands_by_person: dict[str, dict[str, HandDetection]] = {}
        for index, (person_keypoints, person_scores) in enumerate(
            zip(keypoints, scores), start=1
        ):
            person_id = f"person-{index}"
            pose, hands = map_rtmw_wholebody_person(
                person_keypoints,
                person_scores,
                width=width,
                height=height,
                person_id=person_id,
                min_confidence=self.min_confidence,
            )
            poses.append(pose)
            hands_by_person[person_id] = hands
        self._hands_by_person = hands_by_person
        return poses

    def hands_for(self, person_id: str) -> dict[str, HandDetection]:
        return dict(self._hands_by_person.get(person_id, {}))

    def status(self) -> EstimatorStatus:
        dependency_available = importlib.util.find_spec("rtmlib") is not None
        return EstimatorStatus(
            name="RTMWPoseEstimator[RTMW-l 384x288]",
            configured=dependency_available,
            loaded=self._model is not None,
            detail=(
                f"official OpenMMLab ONNX; backend={self.backend}; device={self.device}; "
                "wholebody=133 keypoints"
                if dependency_available
                else "rtmlib is not installed; install requirements-rtmw.txt"
            ),
        )


class RTMWCachedHandEstimator(HandEstimator):
    """Expose hand keypoints produced by the same RTMW whole-body inference."""

    def __init__(self, pose_estimator: RTMWPoseEstimator) -> None:
        self.pose_estimator = pose_estimator

    def estimate(
        self, image_rgb: Any, pose: PoseDetection
    ) -> dict[str, HandDetection]:
        return self.pose_estimator.hands_for(pose.person_id)

    def status(self) -> EstimatorStatus:
        pose_status = self.pose_estimator.status()
        return EstimatorStatus(
            name="RTMWWholeBodyHands[not Hand5]",
            configured=pose_status.configured,
            loaded=pose_status.loaded,
            detail="COCO-WholeBody hand keypoints from the same RTMW-l inference.",
        )


def default_rtmw_estimators() -> tuple[RTMWPoseEstimator, RTMWCachedHandEstimator]:
    pose = RTMWPoseEstimator()
    return pose, RTMWCachedHandEstimator(pose)
