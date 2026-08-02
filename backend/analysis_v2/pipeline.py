from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from .angle_calculator import calculate_joint_angles
from .domain import HandDetection, ObjectDetection, PoseDetection
from .errors import EstimatorUnavailable
from .estimators.base import HandEstimator, ObjectDetector, PoseEstimator
from .estimators.factory import (
    build_hand_estimator,
    build_object_detector,
    build_pose_estimator,
)
from .rula_engine import RulaOptions, calculate_rula
from .runtime import heavy_inference_lock


def _serialize_pose(pose: PoseDetection) -> dict[str, Any]:
    return {
        "person_id": pose.person_id,
        "landmarks": [point.as_dict() for point in pose.landmarks],
        "world_landmarks": (
            [point.as_dict() for point in pose.world_landmarks]
            if pose.world_landmarks
            else None
        ),
        "confidence": round(pose.confidence, 4),
    }


def _serialize_hands(hands: dict[str, HandDetection]) -> dict[str, Any]:
    response: dict[str, Any] = {"left": None, "right": None}
    for side, hand in hands.items():
        response[side] = {
            "landmarks": [point.as_dict() for point in hand.landmarks],
            "world_landmarks": (
                [point.as_dict() for point in hand.world_landmarks]
                if hand.world_landmarks
                else None
            ),
            "confidence": round(hand.confidence, 4),
            "roi": hand.roi,
        }
    return response


def _object_relationships(
    pose: PoseDetection,
    objects: list[ObjectDetection],
) -> list[dict[str, Any]]:
    joints = {
        "left_elbow": pose.landmarks[13],
        "right_elbow": pose.landmarks[14],
        "left_wrist": pose.landmarks[15],
        "right_wrist": pose.landmarks[16],
    }
    relationships: list[dict[str, Any]] = []
    for detection in objects:
        x_min, y_min, x_max, y_max = detection.bbox
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        nearest_name, nearest_point = min(
            joints.items(),
            key=lambda item: math.hypot(item[1].x - center[0], item[1].y - center[1]),
        )
        distance = math.hypot(nearest_point.x - center[0], nearest_point.y - center[1])
        box_diagonal = math.hypot(x_max - x_min, y_max - y_min)
        relationships.append(
            {
                "person_id": pose.person_id,
                "object_id": detection.detection_id,
                "closest_joint": nearest_name,
                "normalized_distance": round(distance, 4),
                "possibly_held": nearest_name.endswith("wrist")
                and distance <= max(0.08, box_diagonal * 0.45),
            }
        )
    return relationships


class PostureAnalysisPipeline:
    def __init__(
        self,
        pose_estimator: PoseEstimator,
        hand_estimator: HandEstimator,
        object_detector: ObjectDetector,
    ) -> None:
        self.pose_estimator = pose_estimator
        self.hand_estimator = hand_estimator
        self.object_detector = object_detector
        self._inference_lock = heavy_inference_lock

    def status(self) -> dict[str, Any]:
        return {
            "pose": self.pose_estimator.status().as_dict(),
            "hand": self.hand_estimator.status().as_dict(),
            "object": self.object_detector.status().as_dict(),
        }

    def analyze_image(
        self,
        image_rgb: Any,
        *,
        worker_id: str | None = None,
        measurement_id: str | None = None,
        frame_id: str | None = None,
        timestamp_ms: float | None = None,
        rula_options: RulaOptions | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        object_issue: str | None = None
        with self._inference_lock:
            poses = self.pose_estimator.estimate(image_rgb)
            try:
                objects = self.object_detector.detect(image_rgb)
            except EstimatorUnavailable as error:
                objects = []
                object_issue = str(error)

            hands_by_person: dict[str, dict[str, HandDetection]] = {}
            hand_issues: dict[str, str] = {}
            for pose in poses:
                try:
                    hands_by_person[pose.person_id] = self.hand_estimator.estimate(
                        image_rgb, pose
                    )
                except EstimatorUnavailable as error:
                    hands_by_person[pose.person_id] = {}
                    hand_issues[pose.person_id] = str(error)

        return self._build_response(
            poses,
            hands_by_person,
            objects,
            worker_id=worker_id,
            measurement_id=measurement_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rula_options=rula_options,
            elapsed_ms=(perf_counter() - started) * 1000,
            object_issue=object_issue,
            hand_issues=hand_issues,
        )

    def analyze_landmarks(
        self,
        pose: PoseDetection,
        hands: dict[str, HandDetection],
        *,
        worker_id: str | None = None,
        measurement_id: str | None = None,
        frame_id: str | None = None,
        timestamp_ms: float | None = None,
        rula_options: RulaOptions | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        return self._build_response(
            [pose],
            {pose.person_id: hands},
            [],
            worker_id=worker_id,
            measurement_id=measurement_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rula_options=rula_options,
            elapsed_ms=(perf_counter() - started) * 1000,
            object_issue=None,
            hand_issues={},
        )

    def _build_response(
        self,
        poses: list[PoseDetection],
        hands_by_person: dict[str, dict[str, HandDetection]],
        objects: list[ObjectDetection],
        *,
        worker_id: str | None,
        measurement_id: str | None,
        frame_id: str | None,
        timestamp_ms: float | None,
        rula_options: RulaOptions | None,
        elapsed_ms: float,
        object_issue: str | None,
        hand_issues: dict[str, str],
    ) -> dict[str, Any]:
        resolved_frame_id = frame_id or f"frame-{uuid4().hex[:12]}"
        analyzed_at = datetime.now(timezone.utc).isoformat()
        object_status = self.object_detector.status().as_dict()
        if not poses:
            return {
                "schema_version": "2.0",
                "processor": "moveguard-pose-analysis-v2",
                "frame_id": resolved_frame_id,
                "timestamp_ms": timestamp_ms,
                "worker_id": worker_id,
                "measurement_id": measurement_id,
                "analyzed_at": analyzed_at,
                "pose": {"landmarks": [], "world_landmarks": None, "confidence": 0.0},
                "hands": {"left": None, "right": None},
                "objects": [item.as_dict() for item in objects],
                "object_relationships": [],
                "joint_angles": {},
                "measurements": {},
                "rula": {
                    "final_score": None,
                    "risk_level": "not_measurable",
                    "valid": False,
                    "issues": ["person_not_detected"],
                    "assumptions": [],
                },
                "quality": {
                    "status": "not_measurable",
                    "pose_valid": False,
                    "hand_valid": False,
                    "occlusion_detected": False,
                    "issues": ["person_not_detected"],
                    "object_detector": object_status,
                },
                "people": [],
                "elapsed_ms": round(elapsed_ms, 2),
                "video_stored": False,
            }

        people: list[dict[str, Any]] = []
        for pose in poses:
            hands = hands_by_person.get(pose.person_id, {})
            measurements = calculate_joint_angles(pose, hands)
            rula = calculate_rula(measurements, rula_options)
            measurement_data = {
                name: measurement.as_dict()
                for name, measurement in measurements.items()
            }
            joint_angles = {
                name: (None if measurement.angle is None else round(measurement.angle, 2))
                for name, measurement in measurements.items()
            }
            pose_valid = pose.confidence >= 0.5 and all(
                measurements.get(name) is not None and measurements[name].valid
                for name in ("neck", "trunk", "legs")
            )
            selected_side = rula.get("scored_side")
            hand_valid = bool(
                selected_side
                and measurements.get(f"{selected_side}_wrist")
                and measurements[f"{selected_side}_wrist"].valid
            )
            low_confidence = any(
                measurement.status == "low_confidence"
                for measurement in measurements.values()
            )
            issues = list(rula.get("issues", []))
            if pose.person_id in hand_issues:
                issues.append(f"hand_estimator:{hand_issues[pose.person_id]}")
            if object_issue:
                issues.append(f"object_detector:{object_issue}")
            quality_status = (
                "valid"
                if rula.get("valid")
                else ("low_confidence" if low_confidence else "not_measurable")
            )
            people.append(
                {
                    "person_id": pose.person_id,
                    "pose": _serialize_pose(pose),
                    "hands": _serialize_hands(hands),
                    "joint_angles": joint_angles,
                    "measurements": measurement_data,
                    "rula": rula,
                    "quality": {
                        "status": quality_status,
                        "pose_valid": pose_valid,
                        "hand_valid": hand_valid,
                        "occlusion_detected": low_confidence,
                        "issues": issues,
                    },
                    "object_relationships": _object_relationships(pose, objects),
                }
            )

        primary = max(
            people,
            key=lambda person: (
                person["rula"].get("final_score") is not None,
                person["rula"].get("final_score") or 0,
                person["pose"]["confidence"],
            ),
        )
        quality = {
            **primary["quality"],
            "object_detector": object_status,
        }
        return {
            "schema_version": "2.0",
            "processor": "moveguard-pose-analysis-v2",
            "frame_id": resolved_frame_id,
            "timestamp_ms": timestamp_ms,
            "worker_id": worker_id,
            "measurement_id": measurement_id,
            "analyzed_at": analyzed_at,
            "pose": primary["pose"],
            "hands": primary["hands"],
            "objects": [item.as_dict() for item in objects],
            "object_relationships": primary["object_relationships"],
            "joint_angles": primary["joint_angles"],
            "measurements": primary["measurements"],
            "rula": primary["rula"],
            "quality": quality,
            "people": people,
            "elapsed_ms": round(elapsed_ms, 2),
            "video_stored": False,
        }


_pipeline: PostureAnalysisPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> PostureAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = PostureAnalysisPipeline(
                    pose_estimator=build_pose_estimator(),
                    hand_estimator=build_hand_estimator(),
                    object_detector=build_object_detector(),
                )
    return _pipeline
