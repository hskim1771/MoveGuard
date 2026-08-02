from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Side = Literal["left", "right", "unknown"]
MeasurementStatus = Literal["valid", "low_confidence", "not_measurable"]


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "z": round(float(self.z), 6),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class PoseDetection:
    person_id: str
    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] | None
    confidence: float


@dataclass(frozen=True)
class HandDetection:
    side: Side
    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] | None
    confidence: float
    roi: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ObjectDetection:
    detection_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x_min": round(self.bbox[0], 6),
                "y_min": round(self.bbox[1], 6),
                "x_max": round(self.bbox[2], 6),
                "y_max": round(self.bbox[3], 6),
            },
        }


@dataclass(frozen=True)
class JointMeasurement:
    name: str
    angle: float | None
    confidence: float
    status: MeasurementStatus
    reason: str | None = None
    side: Side = "unknown"
    auxiliary: dict[str, float | str | bool | None] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.angle is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "angle": None if self.angle is None else round(self.angle, 2),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "valid": self.valid,
            "reason": self.reason,
            **self.auxiliary,
        }


@dataclass(frozen=True)
class EstimatorStatus:
    name: str
    configured: bool
    loaded: bool
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
