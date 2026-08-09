"""2D projected wrist flexion/extension measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WristMeasurement:
    angle: float | None
    signed_angle: float | None
    posture: str
    rula_score: int | None
    confidence: float
    valid: bool
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def projected_wrist_angle(
    elbow: Sequence[float],
    wrist: Sequence[float],
    hand_points: Sequence[Sequence[float]],
    *,
    confidence: Sequence[float],
    orientation_sign: int | None = None,
    neutral_threshold_deg: float = 15.0,
    min_confidence: float = 0.45,
) -> WristMeasurement:
    points = np.asarray([elbow, wrist, *hand_points], dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if len(hand_points) == 0:
        return WristMeasurement(None, None, "unknown", None, 0.0, False, "hand_landmark_missing")
    if not np.isfinite(points).all() or not np.isfinite(conf).all():
        return WristMeasurement(None, None, "unknown", None, 0.0, False, "nonfinite_input")
    joint_confidence = float(conf.min())
    if joint_confidence < min_confidence:
        return WristMeasurement(None, None, "unknown", None, joint_confidence, False, "low_confidence")
    hand_reference = points[2:, :2].mean(axis=0)
    forearm = points[1, :2] - points[0, :2]
    hand = hand_reference - points[1, :2]
    if np.linalg.norm(forearm) < 1e-6 or np.linalg.norm(hand) < 1e-6:
        return WristMeasurement(None, None, "unknown", None, joint_confidence, False, "degenerate_vector")
    cross = forearm[0] * hand[1] - forearm[1] * hand[0]
    signed = float(np.degrees(np.arctan2(cross, np.dot(forearm, hand))))
    angle = abs(signed)
    if angle <= neutral_threshold_deg:
        posture = "neutral"
    elif orientation_sign in {-1, 1}:
        posture = "flexion" if signed * orientation_sign > 0 else "extension"
    else:
        posture = "unknown"
    # Existing MoveGuard/AI-Hub-derived rule: neutral=1, bending=2.
    rula_score = 1 if angle <= neutral_threshold_deg else 2
    return WristMeasurement(
        angle=angle,
        signed_angle=signed,
        posture=posture,
        rula_score=rula_score,
        confidence=joint_confidence,
        valid=True,
        reason=None if posture != "unknown" else "camera_orientation_unverified",
    )


def mediapipe_wrist_measurements(landmarks: Sequence[object]) -> dict[str, dict[str, object]]:
    values = np.asarray([[p.x, p.y] for p in landmarks], dtype=np.float64)
    confidence = np.asarray([p.visibility for p in landmarks], dtype=np.float64)
    mapping = {
        "left": (13, 15, (17, 19, 21)),
        "right": (14, 16, (18, 20, 22)),
    }
    output = {}
    for side, (elbow, wrist, hand) in mapping.items():
        indices = (elbow, wrist, *hand)
        output[side] = projected_wrist_angle(
            values[elbow], values[wrist], values[list(hand)],
            confidence=confidence[list(indices)],
        ).to_dict()
    return output
