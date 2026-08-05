"""Confidence- and anatomy-aware pose outlier stabilization.

This module replaces only isolated, supported outliers. It deliberately does
not apply a moving average, so sustained fast motion is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import warnings

import numpy as np


MEDIAPIPE_SEGMENTS = (
    (11, 12), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
)


@dataclass(frozen=True)
class StabilizerConfig:
    min_confidence: float = 0.45
    coordinate_margin: float = 0.20
    segment_ratio_min: float = 0.45
    segment_ratio_max: float = 2.20
    isolated_jump_scale: float = 0.35
    neighbor_support_scale: float = 0.18
    max_neighbor_gap_seconds: float = 0.50
    max_bad_segments: int = 2


@dataclass
class PoseFrame:
    keypoints: np.ndarray
    confidence: np.ndarray
    timestamp: float
    frame_id: str = ""


@dataclass
class StabilizedFrame:
    keypoints: np.ndarray
    confidence: np.ndarray
    timestamp: float
    frame_id: str
    valid: bool
    outlier: bool
    outlier_joints: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class TemporalPoseStabilizer:
    def __init__(
        self,
        segments: Sequence[tuple[int, int]] = MEDIAPIPE_SEGMENTS,
        config: StabilizerConfig | None = None,
    ) -> None:
        self.segments = tuple(segments)
        self.config = config or StabilizerConfig()

    def stabilize(self, frames: Sequence[PoseFrame]) -> list[StabilizedFrame]:
        if not frames:
            return []
        points = [np.asarray(frame.keypoints, dtype=np.float32) for frame in frames]
        confidence = [np.asarray(frame.confidence, dtype=np.float32) for frame in frames]
        segment_reference = self._segment_reference(points, confidence)
        output: list[StabilizedFrame] = []

        for index, frame in enumerate(frames):
            current = points[index].copy()
            conf = confidence[index].copy()
            reasons: list[str] = []
            outliers: set[int] = set()

            nonfinite = np.flatnonzero(~np.isfinite(current).all(axis=1))
            out_of_bounds = np.flatnonzero(
                (current[:, 0] < -self.config.coordinate_margin)
                | (current[:, 0] > 1 + self.config.coordinate_margin)
                | (current[:, 1] < -self.config.coordinate_margin)
                | (current[:, 1] > 1 + self.config.coordinate_margin)
            )
            outliers.update(nonfinite.tolist())
            outliers.update(out_of_bounds.tolist())
            if len(nonfinite):
                reasons.append("nonfinite_coordinate")
            if len(out_of_bounds):
                reasons.append("coordinate_out_of_bounds")

            bad_segments = self._bad_segments(current, segment_reference)
            if bad_segments:
                reasons.append("body_segment_inconsistent")

            if self._neighbors_are_temporally_close(frames, index):
                isolated = self._isolated_jumps(points, confidence, index)
                outliers.update(isolated)
                if isolated:
                    reasons.append("isolated_temporal_jump")

            for a, b in bad_segments:
                if conf[a] < conf[b]:
                    outliers.add(a)
                elif conf[b] < conf[a]:
                    outliers.add(b)
                elif index > 0 and index + 1 < len(points):
                    da = self._neighbor_deviation(points, index, a)
                    db = self._neighbor_deviation(points, index, b)
                    outliers.add(a if da >= db else b)

            replaced: list[int] = []
            for joint in sorted(outliers):
                replacement = self._replacement(points, confidence, frames, index, joint)
                if replacement is not None:
                    current[joint] = replacement
                    conf[joint] = min(confidence[index - 1][joint], confidence[index + 1][joint])
                    replaced.append(joint)

            unresolved = [joint for joint in outliers if joint not in replaced]
            valid = not len(nonfinite) and len(bad_segments) <= self.config.max_bad_segments
            if unresolved and len(unresolved) > max(2, len(current) // 5):
                valid = False
                reasons.append("too_many_unresolved_joints")

            output.append(
                StabilizedFrame(
                    keypoints=current,
                    confidence=conf,
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                    valid=valid,
                    outlier=bool(outliers),
                    outlier_joints=sorted(outliers),
                    reasons=sorted(set(reasons)),
                )
            )
        return output

    def _segment_reference(
        self, points: Sequence[np.ndarray], confidence: Sequence[np.ndarray]
    ) -> np.ndarray:
        values = []
        for frame, conf in zip(points, confidence):
            lengths = []
            for a, b in self.segments:
                valid = np.isfinite(frame[[a, b]]).all() and min(conf[a], conf[b]) >= self.config.min_confidence
                lengths.append(float(np.linalg.norm(frame[a, :2] - frame[b, :2])) if valid else np.nan)
            values.append(lengths)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reference = np.nanmedian(np.asarray(values, dtype=np.float32), axis=0)
        reference[~np.isfinite(reference) | (reference < 1e-5)] = np.nan
        return reference

    def _bad_segments(
        self, points: np.ndarray, reference: np.ndarray
    ) -> list[tuple[int, int]]:
        bad = []
        for segment_index, (a, b) in enumerate(self.segments):
            expected = reference[segment_index]
            if not np.isfinite(expected) or not np.isfinite(points[[a, b]]).all():
                continue
            ratio = np.linalg.norm(points[a, :2] - points[b, :2]) / expected
            if ratio < self.config.segment_ratio_min or ratio > self.config.segment_ratio_max:
                bad.append((a, b))
        return bad

    def _neighbors_are_temporally_close(self, frames: Sequence[PoseFrame], index: int) -> bool:
        if index == 0 or index + 1 >= len(frames):
            return False
        return (
            frames[index].timestamp - frames[index - 1].timestamp <= self.config.max_neighbor_gap_seconds
            and frames[index + 1].timestamp - frames[index].timestamp <= self.config.max_neighbor_gap_seconds
        )

    def _isolated_jumps(
        self,
        points: Sequence[np.ndarray],
        confidence: Sequence[np.ndarray],
        index: int,
    ) -> set[int]:
        previous, current, following = points[index - 1], points[index], points[index + 1]
        torso = np.linalg.norm(current[11, :2] - current[23, :2])
        if not np.isfinite(torso) or torso < 1e-4:
            torso = 0.25
        midpoint = (previous[:, :2] + following[:, :2]) / 2
        deviation = np.linalg.norm(current[:, :2] - midpoint, axis=1) / torso
        support = np.linalg.norm(previous[:, :2] - following[:, :2], axis=1) / torso
        candidates = np.flatnonzero(
            (deviation > self.config.isolated_jump_scale)
            & (support < self.config.neighbor_support_scale)
        )
        return {
            int(joint)
            for joint in candidates
            if confidence[index][joint] < 0.75 or self._neighbor_deviation(points, index, int(joint)) > 0.5
        }

    @staticmethod
    def _neighbor_deviation(points: Sequence[np.ndarray], index: int, joint: int) -> float:
        midpoint = (points[index - 1][joint, :2] + points[index + 1][joint, :2]) / 2
        return float(np.linalg.norm(points[index][joint, :2] - midpoint))

    def _can_interpolate(
        self,
        points: Sequence[np.ndarray],
        confidence: Sequence[np.ndarray],
        index: int,
        joint: int,
    ) -> bool:
        return (
            index > 0
            and index + 1 < len(points)
            and np.isfinite(points[index - 1][joint]).all()
            and np.isfinite(points[index + 1][joint]).all()
            and confidence[index - 1][joint] >= self.config.min_confidence
            and confidence[index + 1][joint] >= self.config.min_confidence
        )

    def _replacement(
        self,
        points: Sequence[np.ndarray],
        confidence: Sequence[np.ndarray],
        frames: Sequence[PoseFrame],
        index: int,
        joint: int,
    ) -> np.ndarray | None:
        if not self._can_interpolate(points, confidence, index, joint):
            return None
        before, after = frames[index - 1].timestamp, frames[index + 1].timestamp
        ratio = 0.5 if after <= before else (frames[index].timestamp - before) / (after - before)
        return points[index - 1][joint] * (1 - ratio) + points[index + 1][joint] * ratio
