from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from analysis_v2.domain import HandDetection, Landmark, PoseDetection

JOINT_NAMES = (
    "head",
    "neck",
    "trunk",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "neck": ("neck",),
    "trunk": ("trunk",),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "elbow": ("left_elbow", "right_elbow"),
    "wrist": ("left_wrist", "right_wrist"),
    "hip": ("left_hip", "right_hip"),
    "knee": ("left_knee", "right_knee"),
    "ankle": ("left_ankle", "right_ankle"),
}

_POSE_INDEX = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass(frozen=True)
class CanonicalJoint:
    joint: str
    x: float | None
    y: float | None
    z: float | None
    confidence: float
    source_model: str
    valid: bool
    world_x: float | None = None
    world_y: float | None = None
    world_z: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalFrame:
    frame_id: str
    person_id: str | None
    joints: dict[str, CanonicalJoint]
    wrist_angle_ready: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "person_id": self.person_id,
            "joints": {name: joint.as_dict() for name, joint in self.joints.items()},
            "wrist_angle_ready": self.wrist_angle_ready,
        }


def _missing(name: str, source: str = "not_detected") -> CanonicalJoint:
    return CanonicalJoint(name, None, None, None, 0.0, source, False)


def _joint(
    name: str,
    image: Landmark | None,
    world: Landmark | None,
    source: str,
    threshold: float,
) -> CanonicalJoint:
    if image is None:
        return _missing(name, source)
    confidence = float(image.confidence)
    return CanonicalJoint(
        joint=name,
        x=float(image.x),
        y=float(image.y),
        z=float(image.z),
        confidence=confidence,
        source_model=source,
        valid=confidence >= threshold,
        world_x=None if world is None else float(world.x),
        world_y=None if world is None else float(world.y),
        world_z=None if world is None else float(world.z),
    )


def _mean_landmarks(points: list[Landmark]) -> Landmark | None:
    if not points:
        return None
    count = len(points)
    return Landmark(
        x=sum(point.x for point in points) / count,
        y=sum(point.y for point in points) / count,
        z=sum(point.z for point in points) / count,
        confidence=min(point.confidence for point in points),
    )


def _pose_point(pose: PoseDetection, index: int, *, world: bool = False) -> Landmark | None:
    points = pose.world_landmarks if world else pose.landmarks
    if points is None or index >= len(points):
        return None
    return points[index]


def canonicalize(
    frame_id: str,
    pose: PoseDetection | None,
    hands: dict[str, HandDetection],
    *,
    pose_source: str,
    hand_source: str | None,
    confidence_threshold: float = 0.5,
    joint_source_overrides: dict[str, str] | None = None,
) -> CanonicalFrame:
    """Convert model-specific output into the MoveGuard common joint schema.

    Hand landmark 0 takes priority for the canonical wrist when it is available.
    The pose wrist remains available to the shared angle engine as the forearm end.
    """

    if pose is None:
        return CanonicalFrame(
            frame_id=frame_id,
            person_id=None,
            joints={name: _missing(name) for name in JOINT_NAMES},
            wrist_angle_ready={"left": False, "right": False},
        )

    overrides = joint_source_overrides or {}
    joints: dict[str, CanonicalJoint] = {}
    for name, index in _POSE_INDEX.items():
        joints[name] = _joint(
            name,
            _pose_point(pose, index),
            _pose_point(pose, index, world=True),
            overrides.get(name, pose_source),
            confidence_threshold,
        )

    shoulder_image = [
        point
        for point in (_pose_point(pose, 11), _pose_point(pose, 12))
        if point is not None
    ]
    hip_image = [
        point
        for point in (_pose_point(pose, 23), _pose_point(pose, 24))
        if point is not None
    ]
    shoulder_world = [
        point
        for point in (_pose_point(pose, 11, world=True), _pose_point(pose, 12, world=True))
        if point is not None
    ]
    hip_world = [
        point
        for point in (_pose_point(pose, 23, world=True), _pose_point(pose, 24, world=True))
        if point is not None
    ]
    neck_image = _mean_landmarks(shoulder_image)
    neck_world = _mean_landmarks(shoulder_world)
    hip_mid_image = _mean_landmarks(hip_image)
    hip_mid_world = _mean_landmarks(hip_world)
    trunk_image = _mean_landmarks(
        [point for point in (neck_image, hip_mid_image) if point is not None]
    )
    trunk_world = _mean_landmarks(
        [point for point in (neck_world, hip_mid_world) if point is not None]
    )
    derived_source = f"{pose_source}:derived"
    joints["neck"] = _joint(
        "neck", neck_image, neck_world, overrides.get("neck", derived_source), confidence_threshold
    )
    joints["trunk"] = _joint(
        "trunk", trunk_image, trunk_world, overrides.get("trunk", derived_source), confidence_threshold
    )

    wrist_ready: dict[str, bool] = {}
    for side in ("left", "right"):
        hand = hands.get(side)
        wrist_name = f"{side}_wrist"
        if hand and len(hand.landmarks) >= 10:
            world_wrist = (
                hand.world_landmarks[0]
                if hand.world_landmarks and len(hand.world_landmarks) > 0
                else None
            )
            # The hand landmark replaces the pose wrist here, so its provenance
            # must remain the hand estimator even when pose-fusion overrides exist.
            source = hand_source or "hand_model"
            joints[wrist_name] = _joint(
                wrist_name,
                hand.landmarks[0],
                world_wrist,
                source,
                confidence_threshold,
            )
            wrist_ready[side] = (
                hand.confidence >= confidence_threshold
                and hand.landmarks[0].confidence >= confidence_threshold
                and hand.landmarks[9].confidence >= confidence_threshold
            )
        else:
            wrist_ready[side] = False

    return CanonicalFrame(
        frame_id=frame_id,
        person_id=pose.person_id,
        joints={name: joints.get(name, _missing(name)) for name in JOINT_NAMES},
        wrist_angle_ready=wrist_ready,
    )
