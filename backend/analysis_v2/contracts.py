from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .domain import HandDetection, Landmark, PoseDetection
from .rula_engine import RulaOptions


class LandmarkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Annotated[float, Field(ge=-2, le=3)]
    y: Annotated[float, Field(ge=-2, le=3)]
    z: Annotated[float, Field(ge=-10, le=10)] = 0.0
    visibility: Annotated[float, Field(ge=0, le=1)] | None = None
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None

    def to_domain(self) -> Landmark:
        return Landmark(
            x=self.x,
            y=self.y,
            z=self.z,
            confidence=(
                self.confidence
                if self.confidence is not None
                else (self.visibility if self.visibility is not None else 0.0)
            ),
        )


class PoseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: Annotated[str, Field(min_length=1, max_length=40)] = "person-1"
    landmarks: Annotated[list[LandmarkInput], Field(min_length=33, max_length=33)]
    world_landmarks: Annotated[
        list[LandmarkInput], Field(min_length=33, max_length=33)
    ] | None = None
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None

    def to_domain(self) -> PoseDetection:
        landmarks = tuple(point.to_domain() for point in self.landmarks)
        confidence = self.confidence
        if confidence is None:
            confidence = sum(point.confidence for point in landmarks) / len(landmarks)
        return PoseDetection(
            person_id=self.person_id,
            landmarks=landmarks,
            world_landmarks=(
                tuple(point.to_domain() for point in self.world_landmarks)
                if self.world_landmarks
                else None
            ),
            confidence=confidence,
        )


class HandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    landmarks: Annotated[list[LandmarkInput], Field(min_length=21, max_length=21)]
    world_landmarks: Annotated[
        list[LandmarkInput], Field(min_length=21, max_length=21)
    ] | None = None
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None

    def to_domain(self, side: str) -> HandDetection:
        landmarks = tuple(point.to_domain() for point in self.landmarks)
        confidence = self.confidence
        if confidence is None:
            confidence = sum(point.confidence for point in landmarks) / len(landmarks)
        return HandDetection(
            side=side,  # type: ignore[arg-type]
            landmarks=landmarks,
            world_landmarks=(
                tuple(point.to_domain() for point in self.world_landmarks)
                if self.world_landmarks
                else None
            ),
            confidence=confidence,
        )


class RulaOptionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_kg: Annotated[float, Field(ge=0, le=200)] = 0.0
    static_force: bool = False
    muscle_static: bool = False
    legs_unsupported: bool | None = None
    wrist_twist_at_end_range: bool | None = None

    def to_domain(self) -> RulaOptions:
        return RulaOptions(**self.model_dump())


class LandmarkPostureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    timestamp_ms: Annotated[float, Field(ge=0)] | None = None
    worker_id: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    measurement_id: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    pose: PoseInput
    hands: dict[str, HandInput] = Field(default_factory=dict)
    rula_options: RulaOptionsInput = Field(default_factory=RulaOptionsInput)

    def domain_hands(self) -> dict[str, HandDetection]:
        return {
            side: hand.to_domain(side)
            for side, hand in self.hands.items()
            if side in {"left", "right"}
        }
