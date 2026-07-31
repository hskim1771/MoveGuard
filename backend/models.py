from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Risk = Literal["normal", "caution", "danger"]


class Landmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Annotated[float, Field(ge=-2, le=3)]
    y: Annotated[float, Field(ge=-2, le=3)]
    z: Annotated[float, Field(ge=-10, le=10)]
    visibility: Annotated[float, Field(ge=0, le=1)] = 0


LandmarkSet = Annotated[list[Landmark], Field(min_length=33, max_length=33)]


class PersonPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: Annotated[str, Field(min_length=1, max_length=40)]
    landmarks: LandmarkSet
    world_landmarks: LandmarkSet | None = None


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_ms: Annotated[float, Field(ge=0)]
    people: Annotated[list[PersonPose], Field(max_length=4)]


class MetricResult(BaseModel):
    angle: float | None
    risk: Risk
    experimental: bool = False


class PersonAnalysis(BaseModel):
    person_id: str
    metrics: dict[str, MetricResult]
    confidence: int
    full_body: bool
    overall_risk: Risk


class AnalyzeResponse(BaseModel):
    schema_version: str = "1.0"
    processor: str = "moveguard-landmark-rules-v1"
    people: list[PersonAnalysis]
    scene_risk: Risk
    elapsed_ms: float
    video_stored: bool = False
