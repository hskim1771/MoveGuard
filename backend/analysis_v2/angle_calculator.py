from __future__ import annotations

import math
from collections.abc import Iterable

from .domain import HandDetection, JointMeasurement, Landmark, PoseDetection

MIN_LANDMARK_CONFIDENCE = 0.5


def _vector(first: Landmark, second: Landmark) -> tuple[float, float, float]:
    return second.x - first.x, second.y - first.y, second.z - first.z


def _dot(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _magnitude(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    magnitude = _magnitude(vector)
    if magnitude < 1e-9:
        return None
    return tuple(value / magnitude for value in vector)  # type: ignore[return-value]


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def angle_between(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float | None:
    magnitude = _magnitude(first) * _magnitude(second)
    if magnitude < 1e-9:
        return None
    cosine = max(-1.0, min(1.0, _dot(first, second) / magnitude))
    return math.degrees(math.acos(cosine))


def angle_at(first: Landmark, pivot: Landmark, third: Landmark) -> float | None:
    return angle_between(_vector(pivot, first), _vector(pivot, third))


def midpoint(first: Landmark, second: Landmark) -> Landmark:
    return Landmark(
        x=(first.x + second.x) / 2,
        y=(first.y + second.y) / 2,
        z=(first.z + second.z) / 2,
        confidence=min(first.confidence, second.confidence),
    )


def _measurement(
    name: str,
    points: Iterable[Landmark],
    calculate,
    *,
    side: str = "unknown",
    min_confidence: float = MIN_LANDMARK_CONFIDENCE,
    auxiliary: dict | None = None,
) -> JointMeasurement:
    points = tuple(points)
    confidence = min((point.confidence for point in points), default=0.0)
    if confidence < min_confidence:
        return JointMeasurement(
            name=name,
            angle=None,
            confidence=confidence,
            status="low_confidence",
            reason="required_landmark_confidence_below_threshold",
            side=side,  # type: ignore[arg-type]
            auxiliary=auxiliary or {},
        )
    angle = calculate()
    if angle is None or not math.isfinite(angle):
        return JointMeasurement(
            name=name,
            angle=None,
            confidence=confidence,
            status="not_measurable",
            reason="degenerate_landmark_geometry",
            side=side,  # type: ignore[arg-type]
            auxiliary=auxiliary or {},
        )
    return JointMeasurement(
        name=name,
        angle=float(angle),
        confidence=confidence,
        status="valid",
        side=side,  # type: ignore[arg-type]
        auxiliary=auxiliary or {},
    )


def _unavailable(name: str, side: str, reason: str) -> JointMeasurement:
    return JointMeasurement(
        name=name,
        angle=None,
        confidence=0.0,
        status="not_measurable",
        reason=reason,
        side=side,  # type: ignore[arg-type]
    )


def _wrist_measurements(
    pose: PoseDetection,
    hand: HandDetection | None,
    side: str,
) -> tuple[JointMeasurement, JointMeasurement]:
    prefix = f"{side}_wrist"
    if hand is None:
        return (
            _unavailable(prefix, side, "hand_not_detected"),
            _unavailable(f"{prefix}_twist", side, "single_camera_twist_not_reliable"),
        )
    if len(hand.landmarks) < 21:
        return (
            _unavailable(prefix, side, "hand_landmarks_incomplete"),
            _unavailable(f"{prefix}_twist", side, "single_camera_twist_not_reliable"),
        )

    elbow_index, wrist_index = ((13, 15) if side == "left" else (14, 16))
    pose_points = pose.world_landmarks or pose.landmarks
    if len(pose_points) <= wrist_index:
        return (
            _unavailable(prefix, side, "pose_landmarks_incomplete"),
            _unavailable(f"{prefix}_twist", side, "single_camera_twist_not_reliable"),
        )

    elbow = pose_points[elbow_index]
    wrist = pose_points[wrist_index]
    hand_points = hand.world_landmarks
    method = "world_3d"
    deviation: float | None = None

    if hand_points and len(hand_points) >= 18 and pose.world_landmarks:
        hand_wrist = hand_points[0]
        index_mcp = hand_points[5]
        middle_mcp = hand_points[9]
        pinky_mcp = hand_points[17]
        forearm = _normalize(_vector(elbow, wrist))
        hand_axis = _normalize(_vector(hand_wrist, middle_mcp))
        lateral = _normalize(_vector(pinky_mcp, index_mcp))
        if forearm and hand_axis and lateral:
            normal = _normalize(_cross(lateral, hand_axis))
            if normal:
                longitudinal = _dot(forearm, hand_axis)
                flexion = abs(
                    math.degrees(math.atan2(_dot(forearm, normal), longitudinal))
                )
                deviation = abs(
                    math.degrees(math.atan2(_dot(forearm, lateral), longitudinal))
                )
            else:
                flexion = angle_between(forearm, hand_axis)
        else:
            flexion = None
        points_for_confidence = (elbow, wrist, hand_wrist, index_mcp, middle_mcp, pinky_mcp)
    else:
        method = "image_2d_fallback"
        pose_elbow = pose.landmarks[elbow_index]
        pose_wrist = pose.landmarks[wrist_index]
        hand_wrist = hand.landmarks[0]
        middle_mcp = hand.landmarks[9]
        flexion = angle_between(
            _vector(pose_elbow, pose_wrist)[:2],
            _vector(hand_wrist, middle_mcp)[:2],
        )
        points_for_confidence = (pose_elbow, pose_wrist, hand_wrist, middle_mcp)

    wrist_measurement = _measurement(
        prefix,
        points_for_confidence,
        lambda: flexion,
        side=side,
        auxiliary={
            "deviation_angle": None if deviation is None else round(deviation, 2),
            "method": method,
        },
    )
    twist_measurement = JointMeasurement(
        name=f"{prefix}_twist",
        angle=None,
        confidence=hand.confidence,
        status="not_measurable",
        reason="single_camera_twist_not_reliable",
        side=side,  # type: ignore[arg-type]
    )
    return wrist_measurement, twist_measurement


def calculate_joint_angles(
    pose: PoseDetection,
    hands: dict[str, HandDetection],
) -> dict[str, JointMeasurement]:
    if len(pose.landmarks) < 33:
        return {
            name: _unavailable(name, "unknown", "pose_landmarks_incomplete")
            for name in ("neck", "trunk", "legs")
        }

    landmarks = pose.landmarks
    shoulder_mid = midpoint(landmarks[11], landmarks[12])
    hip_mid = midpoint(landmarks[23], landmarks[24])
    ear_mid = midpoint(landmarks[7], landmarks[8])
    body_up = _vector(hip_mid, shoulder_mid)

    measurements: dict[str, JointMeasurement] = {
        "neck": _measurement(
            "neck",
            (landmarks[7], landmarks[8], landmarks[11], landmarks[12], landmarks[23], landmarks[24]),
            lambda: angle_between(_vector(shoulder_mid, ear_mid), body_up),
        ),
        "trunk": _measurement(
            "trunk",
            (landmarks[11], landmarks[12], landmarks[23], landmarks[24]),
            lambda: angle_between(body_up[:2], (0.0, -1.0)),
        ),
    }

    for side, shoulder_index, elbow_index, wrist_index, hip_index in (
        ("left", 11, 13, 15, 23),
        ("right", 12, 14, 16, 24),
    ):
        shoulder = landmarks[shoulder_index]
        elbow = landmarks[elbow_index]
        wrist = landmarks[wrist_index]
        hip = landmarks[hip_index]
        measurements[f"{side}_upper_arm"] = _measurement(
            f"{side}_upper_arm",
            (shoulder, elbow, hip),
            # Both vectors point down from the shoulder. Neutral arms therefore
            # produce approximately 0 degrees instead of the legacy 180-degree bug.
            lambda shoulder=shoulder, elbow=elbow, hip=hip: angle_between(
                _vector(shoulder, elbow)[:2],
                _vector(shoulder, hip)[:2],
            ),
            side=side,
        )
        measurements[f"{side}_lower_arm"] = _measurement(
            f"{side}_lower_arm",
            (shoulder, elbow, wrist),
            lambda shoulder=shoulder, elbow=elbow, wrist=wrist: angle_at(
                shoulder, elbow, wrist
            ),
            side=side,
        )
        wrist_measurement, twist_measurement = _wrist_measurements(
            pose, hands.get(side), side
        )
        measurements[wrist_measurement.name] = wrist_measurement
        measurements[twist_measurement.name] = twist_measurement

    left_knee_points = (landmarks[23], landmarks[25], landmarks[27])
    right_knee_points = (landmarks[24], landmarks[26], landmarks[28])
    leg_points = left_knee_points + right_knee_points
    left_knee_angle = angle_at(*left_knee_points)
    right_knee_angle = angle_at(*right_knee_points)
    supported = None
    bend = None
    if left_knee_angle is not None and right_knee_angle is not None:
        bend = max(abs(180.0 - left_knee_angle), abs(180.0 - right_knee_angle))
        supported = bend <= 25 and abs(left_knee_angle - right_knee_angle) <= 15
    measurements["legs"] = _measurement(
        "legs",
        leg_points,
        lambda: bend,
        auxiliary={
            "left_knee_angle": None if left_knee_angle is None else round(left_knee_angle, 2),
            "right_knee_angle": None if right_knee_angle is None else round(right_knee_angle, 2),
            "support_inferred": supported,
        },
    )
    return measurements
