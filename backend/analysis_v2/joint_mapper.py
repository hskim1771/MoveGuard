from __future__ import annotations

from typing import Any

from .angle_calculator import midpoint
from .domain import HandDetection, Landmark, PoseDetection

COMMON_JOINT_INDEX = {
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


def _item(
    name: str,
    point: Landmark,
    *,
    source_model: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "joint": name,
        "x": round(float(point.x), 6),
        "y": round(float(point.y), 6),
        "z": round(float(point.z), 6) if point.z is not None else None,
        "confidence": round(float(point.confidence), 4),
        "source_model": source_model,
        "valid": point.confidence >= threshold,
    }


def common_joint_schema(
    pose: PoseDetection,
    hands: dict[str, HandDetection],
    *,
    pose_source: str,
    hand_source: str | None,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Return named joints shared by MediaPipe and RTMW API responses."""

    joints = {
        name: _item(
            name,
            pose.landmarks[index],
            source_model=pose_source,
            threshold=threshold,
        )
        for name, index in COMMON_JOINT_INDEX.items()
        if index < len(pose.landmarks)
    }
    if len(pose.landmarks) > 12:
        neck = midpoint(pose.landmarks[11], pose.landmarks[12])
        joints["neck"] = _item(
            "neck",
            neck,
            source_model=f"{pose_source}:derived",
            threshold=threshold,
        )
    for side in ("left", "right"):
        hand = hands.get(side)
        name = f"{side}_wrist"
        if hand and len(hand.landmarks) >= 10 and hand.landmarks[0].confidence >= threshold:
            joints[name] = _item(
                name,
                hand.landmarks[0],
                source_model=hand_source or pose_source,
                threshold=threshold,
            )
    return joints
