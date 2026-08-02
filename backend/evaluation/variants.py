from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, process_time
from typing import Any

from analysis_v2.angle_calculator import calculate_joint_angles
from analysis_v2.domain import EstimatorStatus, HandDetection, PoseDetection
from analysis_v2.errors import EstimatorUnavailable
from analysis_v2.estimators.base import HandEstimator, PoseEstimator
from analysis_v2.estimators.mediapipe_hand import default_hand_estimator
from analysis_v2.estimators.mediapipe_pose import default_pose_estimator
from analysis_v2.estimators.rtmw_pose import default_rtmw_estimators
from analysis_v2.rula_engine import RulaOptions, calculate_rula

from .canonical import CanonicalFrame, canonicalize


class NoHandEstimator(HandEstimator):
    def estimate(
        self, image_rgb: Any, pose: PoseDetection
    ) -> dict[str, HandDetection]:
        return {}

    def status(self) -> EstimatorStatus:
        return EstimatorStatus(
            name="NoHandEstimator",
            configured=True,
            loaded=True,
            detail="Baseline intentionally uses pose landmarks only.",
        )


@dataclass(frozen=True)
class VariantResult:
    variant_id: str
    variant_label: str
    frame_id: str
    frame_index: int
    timestamp_ms: float
    person_detected: bool
    pose_count: int
    canonical: CanonicalFrame
    measurements: dict[str, dict[str, Any]]
    rula: dict[str, Any]
    latency_ms: float
    process_cpu_ms: float
    process_rss_mb: float | None
    gpu_utilization_percent: float | None
    gpu_memory_mb: float | None
    issues: tuple[str, ...]
    posture_labels: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_label": self.variant_label,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "person_detected": self.person_detected,
            "pose_count": self.pose_count,
            "canonical": self.canonical.as_dict(),
            "measurements": self.measurements,
            "rula": self.rula,
            "latency_ms": round(self.latency_ms, 4),
            "process_cpu_ms": round(self.process_cpu_ms, 4),
            "process_rss_mb": (
                None if self.process_rss_mb is None else round(self.process_rss_mb, 4)
            ),
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "issues": list(self.issues),
            "posture_labels": list(self.posture_labels) if self.posture_labels else None,
        }


class LiveVariant:
    def __init__(
        self,
        *,
        variant_id: str,
        label: str,
        pose_estimator: PoseEstimator,
        hand_estimator: HandEstimator,
        pose_source: str,
        hand_source: str | None,
        rula_options: RulaOptions | None = None,
    ) -> None:
        self.variant_id = variant_id
        self.label = label
        self.pose_estimator = pose_estimator
        self.hand_estimator = hand_estimator
        self.pose_source = pose_source
        self.hand_source = hand_source
        self.rula_options = rula_options or RulaOptions()

    def status(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "available": self.pose_estimator.status().configured
            and self.hand_estimator.status().configured,
            "pose": self.pose_estimator.status().as_dict(),
            "hand": self.hand_estimator.status().as_dict(),
            "rula_engine": "analysis_v2.rula_engine.calculate_rula",
            "angle_engine": "analysis_v2.angle_calculator.calculate_joint_angles",
        }

    def analyze(
        self,
        image_rgb: Any,
        *,
        frame_id: str,
        frame_index: int,
        timestamp_ms: float,
    ) -> VariantResult:
        wall_started = perf_counter()
        cpu_started = process_time()
        issues: list[str] = []
        poses: list[PoseDetection] = []
        hands: dict[str, HandDetection] = {}
        try:
            poses = self.pose_estimator.estimate(image_rgb)
        except EstimatorUnavailable as error:
            issues.append(f"pose_estimator:{error}")

        primary = max(poses, key=lambda item: item.confidence) if poses else None
        if primary is not None:
            try:
                hands = self.hand_estimator.estimate(image_rgb, primary)
            except EstimatorUnavailable as error:
                issues.append(f"hand_estimator:{error}")

        if primary is None:
            measurements: dict[str, dict[str, Any]] = {}
            rula = {
                "final_score": None,
                "risk_level": "not_measurable",
                "valid": False,
                "issues": ["person_not_detected"],
                "assumptions": [],
            }
        else:
            raw_measurements = calculate_joint_angles(primary, hands)
            measurements = {
                name: measurement.as_dict()
                for name, measurement in raw_measurements.items()
            }
            rula = calculate_rula(raw_measurements, self.rula_options)

        canonical = canonicalize(
            frame_id,
            primary,
            hands,
            pose_source=self.pose_source,
            hand_source=self.hand_source,
            joint_source_overrides=(
                self.pose_estimator.joint_source_overrides()
                if hasattr(self.pose_estimator, "joint_source_overrides")
                else None
            ),
        )
        latency_ms = (perf_counter() - wall_started) * 1000
        process_cpu_ms = (process_time() - cpu_started) * 1000
        process_rss_mb: float | None = None
        try:
            import psutil

            process_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except (ImportError, OSError):
            pass
        gpu_utilization, gpu_memory = _gpu_snapshot()
        return VariantResult(
            variant_id=self.variant_id,
            variant_label=self.label,
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            person_detected=primary is not None,
            pose_count=len(poses),
            canonical=canonical,
            measurements=measurements,
            rula=rula,
            latency_ms=latency_ms,
            process_cpu_ms=process_cpu_ms,
            process_rss_mb=process_rss_mb,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_mb=gpu_memory,
            issues=tuple(issues),
        )


def _gpu_snapshot() -> tuple[float | None, float | None]:
    """Read an actual NVML sample; return missing rather than inventing zero."""

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        memory = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024 * 1024)
        return utilization, memory
    except Exception:
        return None, None


def build_rtmw_ab_variants() -> tuple[LiveVariant, LiveVariant]:
    """Build existing MediaPipe Baseline and official RTMW-l 384x288 candidate."""

    baseline = LiveVariant(
        variant_id="baseline_mediapipe",
        label="Existing MediaPipe Pose + MediaPipe Hand + shared RULA",
        pose_estimator=default_pose_estimator(),
        hand_estimator=default_hand_estimator(),
        pose_source="mediapipe_pose",
        hand_source="mediapipe_hand",
    )
    rtmw_pose, rtmw_hands = default_rtmw_estimators()
    candidate = LiveVariant(
        variant_id="rtmw_l_384x288",
        label="RTMW-l 384x288 WholeBody + shared RULA",
        pose_estimator=rtmw_pose,
        hand_estimator=rtmw_hands,
        pose_source="rtmw_l_384x288",
        hand_source="rtmw_l_384x288",
    )
    return baseline, candidate


def rtmw_ab_variants() -> list[dict[str, Any]]:
    return [
        {
            "id": "A",
            "configuration": "Existing MediaPipe Pose + MediaPipe Hand + shared RULA",
            "available": True,
            "evaluated_as": "baseline_mediapipe",
        },
        {
            "id": "B",
            "configuration": "RTMW-l 384x288 WholeBody + shared RULA",
            "available": True,
            "evaluated_as": "rtmw_l_384x288",
        },
    ]


def build_default_variants() -> tuple[LiveVariant, LiveVariant]:
    """Build the two variants that are executable in the current repository.

    The available Hybrid is deliberately named Hybrid-lite. RTMW-x, ViTPose-H,
    and RTMPose Hand5 must not be claimed until their adapters and weights are
    installed and evaluated on the same frames.
    """

    baseline = LiveVariant(
        variant_id="baseline_v1",
        label="MediaPipe Pose + shared RULA",
        pose_estimator=default_pose_estimator(),
        hand_estimator=NoHandEstimator(),
        pose_source="mediapipe_pose",
        hand_source=None,
    )
    hybrid = LiveVariant(
        variant_id="hybrid_v2_available",
        label="MediaPipe Pose + MediaPipe Hand + shared RULA",
        pose_estimator=default_pose_estimator(),
        hand_estimator=default_hand_estimator(),
        pose_source="mediapipe_pose",
        hand_source="mediapipe_hand",
    )
    return baseline, hybrid


def planned_ablation_variants() -> list[dict[str, Any]]:
    return [
        {
            "id": "A",
            "configuration": "MediaPipe Pose + shared RULA",
            "available": True,
            "evaluated_as": "baseline_v1",
        },
        {
            "id": "B",
            "configuration": "MediaPipe Pose + MediaPipe Hand + shared RULA",
            "available": True,
            "evaluated_as": "hybrid_v2_available",
        },
        {
            "id": "C",
            "configuration": "RTMW-x + RTMPose Hand5 + shared RULA",
            "available": False,
            "reason": "RTMW-x/RTMPose Hand5 runtime, configs, and weights are not installed.",
        },
        {
            "id": "D",
            "configuration": "RTMW-x + ViTPose-H + RTMPose Hand5 + shared RULA",
            "available": False,
            "reason": "ViTPose-H runtime, fusion adapter, and weights are not installed.",
        },
    ]
