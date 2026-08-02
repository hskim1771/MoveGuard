from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analysis_v2.domain import EstimatorStatus, Landmark, PoseDetection
from analysis_v2.estimators.base import PoseEstimator

CANONICAL_INDEX_TO_NAME = {
    0: "head",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass(frozen=True)
class JointFusionProfile:
    name: str
    source_priority_by_index: dict[int, tuple[str, ...]]
    default_priority: tuple[str, ...]
    confidence_threshold: float = 0.5


@dataclass(frozen=True)
class FusedPose:
    pose: PoseDetection
    source_by_index: dict[int, str]

    @property
    def canonical_source_overrides(self) -> dict[str, str]:
        return {
            name: self.source_by_index[index]
            for index, name in CANONICAL_INDEX_TO_NAME.items()
            if index in self.source_by_index
        }


def moveguard_specialist_profile() -> JointFusionProfile:
    """Planned priority profile; unavailable models must fall back explicitly."""

    return JointFusionProfile(
        name="moveguard-rtmw-vitpose-specialists",
        source_priority_by_index={
            0: ("rtmwx", "mediapipe_pose"),
            7: ("rtmwx", "mediapipe_pose"),
            8: ("rtmwx", "mediapipe_pose"),
            11: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            12: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            13: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            14: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            15: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            16: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            23: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            24: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            25: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            26: ("vitpose_h", "rtmwx", "mediapipe_pose"),
            27: ("rtmwx", "vitpose_h", "mediapipe_pose"),
            28: ("rtmwx", "vitpose_h", "mediapipe_pose"),
        },
        default_priority=("rtmwx", "vitpose_h", "mediapipe_pose"),
    )


def _point(pose: PoseDetection, index: int, *, world: bool = False) -> Landmark | None:
    points = pose.world_landmarks if world else pose.landmarks
    if points is None or index >= len(points):
        return None
    return points[index]


def fuse_pose_detections(
    detections: dict[str, PoseDetection],
    profile: JointFusionProfile,
) -> FusedPose | None:
    if not detections:
        return None
    max_length = max(len(pose.landmarks) for pose in detections.values())
    if max_length < 33:
        max_length = 33
    fused_image = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(max_length)]
    fused_world = [Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(max_length)]
    has_world = False
    sources: dict[int, str] = {}

    for index in range(max_length):
        priority = profile.source_priority_by_index.get(index, profile.default_priority)
        selected_source: str | None = None
        selected_point: Landmark | None = None
        for source in priority:
            pose = detections.get(source)
            point = _point(pose, index) if pose else None
            if point is not None and point.confidence >= profile.confidence_threshold:
                selected_source = source
                selected_point = point
                break
        if selected_point is None:
            candidates = [
                (source, point)
                for source, pose in detections.items()
                if (point := _point(pose, index)) is not None
                and point.confidence >= profile.confidence_threshold
            ]
            if candidates:
                selected_source, selected_point = max(
                    candidates, key=lambda item: item[1].confidence
                )
        if selected_source is None or selected_point is None:
            continue
        fused_image[index] = selected_point
        sources[index] = selected_source
        world_point = _point(detections[selected_source], index, world=True)
        if world_point is not None:
            fused_world[index] = world_point
            has_world = True

    confidence_points = [
        fused_image[index].confidence
        for index in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        if index < len(fused_image)
    ]
    confidence = (
        sum(confidence_points) / len(confidence_points) if confidence_points else 0.0
    )
    return FusedPose(
        pose=PoseDetection(
            person_id="person-1",
            landmarks=tuple(fused_image),
            world_landmarks=tuple(fused_world) if has_world else None,
            confidence=confidence,
        ),
        source_by_index=sources,
    )


class FusionPoseEstimator(PoseEstimator):
    """Run normalized pose adapters and fuse their joint coordinates by priority.

    Each adapter must already convert its native keypoints to the shared 33-slot
    MoveGuard pose layout. The class records the selected source for every joint.
    """

    def __init__(
        self,
        estimators: dict[str, PoseEstimator],
        profile: JointFusionProfile | None = None,
    ) -> None:
        self.estimators = estimators
        self.profile = profile or moveguard_specialist_profile()
        self._last_overrides: dict[str, str] = {}

    def estimate(self, image_rgb: Any) -> list[PoseDetection]:
        primary_by_source: dict[str, PoseDetection] = {}
        for source, estimator in self.estimators.items():
            poses = estimator.estimate(image_rgb)
            if poses:
                primary_by_source[source] = max(poses, key=lambda pose: pose.confidence)
        fused = fuse_pose_detections(primary_by_source, self.profile)
        self._last_overrides = fused.canonical_source_overrides if fused else {}
        return [fused.pose] if fused else []

    def joint_source_overrides(self) -> dict[str, str]:
        return dict(self._last_overrides)

    def status(self) -> EstimatorStatus:
        states = {name: estimator.status() for name, estimator in self.estimators.items()}
        configured = bool(states) and any(state.configured for state in states.values())
        loaded = bool(states) and any(state.loaded for state in states.values())
        detail = "; ".join(
            f"{name}={'ready' if state.configured else 'unavailable'}"
            for name, state in states.items()
        )
        return EstimatorStatus(
            name=f"FusionPoseEstimator[{self.profile.name}]",
            configured=configured,
            loaded=loaded,
            detail=detail,
        )
