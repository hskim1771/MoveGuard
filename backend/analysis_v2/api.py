from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from .contracts import LandmarkPostureRequest
from .errors import EstimatorUnavailable, InvalidAnalysisInput
from .pipeline import get_pipeline
from .rula_engine import RulaOptions

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter(tags=["posture-analysis-v2"])


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _decode_image(image_bytes: bytes):
    if not image_bytes:
        raise InvalidAnalysisInput("Image body is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidAnalysisInput("Image exceeds the 15 MB request limit.")
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise EstimatorUnavailable(
            "opencv-python and numpy are required for image decoding."
        ) from error

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidAnalysisInput("Image is not a readable JPEG, PNG, or WebP file.")
    height, width = bgr.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise InvalidAnalysisInput("Image dimensions exceed the 40-megapixel limit.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None or value == "" else _as_bool(value)


@router.post("/api/analysis/posture")
async def analyze_posture(
    request: Request,
    model: str = "baseline",
    worker_id: str | None = None,
    measurement_id: str | None = None,
    frame_id: str | None = None,
    timestamp_ms: float | None = None,
    weight_kg: float = 0.0,
    static_force: bool = False,
    muscle_static: bool = False,
    legs_unsupported: bool | None = None,
    wrist_twist_at_end_range: bool | None = None,
) -> dict[str, Any]:
    """Analyze a raw image, multipart frame, or precomputed landmark JSON.

    A video client should sample frames and send each frame as an image. Original
    image bytes are processed in memory and are never written to storage.
    """

    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    try:
        if content_type == "application/json":
            if model.strip().lower() not in {"baseline", "mediapipe", "v1"}:
                raise InvalidAnalysisInput(
                    "Model selection applies to image inference. Landmark JSON is already precomputed."
                )
            payload = LandmarkPostureRequest.model_validate(await request.json())
            pipeline = get_pipeline("baseline")
            return pipeline.analyze_landmarks(
                payload.pose.to_domain(),
                payload.domain_hands(),
                worker_id=payload.worker_id or worker_id,
                measurement_id=payload.measurement_id or measurement_id,
                frame_id=payload.frame_id or frame_id,
                timestamp_ms=payload.timestamp_ms if payload.timestamp_ms is not None else timestamp_ms,
                rula_options=payload.rula_options.to_domain(),
            )

        image_bytes: bytes
        if content_type == "multipart/form-data":
            form = await request.form()
            upload = form.get("image") or form.get("frame")
            if upload is None or not hasattr(upload, "read"):
                raise InvalidAnalysisInput(
                    "Multipart request must include an 'image' or 'frame' file."
                )
            upload_type = str(getattr(upload, "content_type", "")).lower()
            if upload_type not in SUPPORTED_IMAGE_TYPES:
                raise InvalidAnalysisInput("Only JPEG, PNG, and WebP images are supported.")
            image_bytes = await upload.read()
            model = str(form.get("model") or model)
            worker_id = str(form.get("worker_id") or worker_id or "") or None
            measurement_id = str(form.get("measurement_id") or measurement_id or "") or None
            frame_id = str(form.get("frame_id") or frame_id or "") or None
            weight_kg = float(form.get("weight_kg") or weight_kg)
            static_force = _as_bool(form.get("static_force"), static_force)
            muscle_static = _as_bool(form.get("muscle_static"), muscle_static)
            legs_unsupported = _optional_bool(form.get("legs_unsupported"))
            wrist_twist_at_end_range = _optional_bool(
                form.get("wrist_twist_at_end_range")
            )
        elif content_type in SUPPORTED_IMAGE_TYPES:
            image_bytes = await request.body()
        else:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Use application/json, multipart/form-data, or a raw JPEG/PNG/WebP body. "
                    "Video clients must send sampled image frames."
                ),
            )

        image_rgb = _decode_image(image_bytes)
        pipeline = get_pipeline(model)
        return pipeline.analyze_image(
            image_rgb,
            worker_id=worker_id,
            measurement_id=measurement_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rula_options=RulaOptions(
                weight_kg=weight_kg,
                static_force=static_force,
                muscle_static=muscle_static,
                legs_unsupported=legs_unsupported,
                wrist_twist_at_end_range=wrist_twist_at_end_range,
            ),
        )
    except HTTPException:
        raise
    except (InvalidAnalysisInput, ValueError, ValidationError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EstimatorUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


async def _comparison_image(request: Request) -> Any:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "multipart/form-data":
        form = await request.form()
        upload = form.get("image") or form.get("frame")
        if upload is None or not hasattr(upload, "read"):
            raise InvalidAnalysisInput(
                "Multipart request must include an 'image' or 'frame' file."
            )
        upload_type = str(getattr(upload, "content_type", "")).lower()
        if upload_type not in SUPPORTED_IMAGE_TYPES:
            raise InvalidAnalysisInput("Only JPEG, PNG, and WebP images are supported.")
        return _decode_image(await upload.read())
    if content_type in SUPPORTED_IMAGE_TYPES:
        return _decode_image(await request.body())
    raise HTTPException(
        status_code=415,
        detail="Use multipart/form-data or a raw JPEG/PNG/WebP body for A/B inference.",
    )


def _response_detection_summary(response: dict[str, Any]) -> dict[str, Any]:
    joints = response.get("joints", {})
    common_names = (
        "neck",
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
    detected = sum(bool(joints.get(name, {}).get("valid")) for name in common_names)
    measurements = response.get("measurements", {})
    wrist_ready = sum(
        measurements.get(f"{side}_wrist", {}).get("status") == "valid"
        for side in ("left", "right")
    )
    return {
        "common_joint_detected": detected,
        "common_joint_expected": len(common_names),
        "common_joint_detection_percent": round(100 * detected / len(common_names), 2),
        "wrist_angle_ready": wrist_ready,
        "wrist_angle_expected": 2,
        "rula_score": response.get("rula", {}).get("final_score"),
        "risk_level": response.get("rula", {}).get("risk_level", "not_measurable"),
        "elapsed_ms": response.get("elapsed_ms"),
    }


@router.post("/api/analysis/compare")
async def compare_posture(
    request: Request,
    frame_id: str | None = None,
    timestamp_ms: float | None = None,
    weight_kg: float = 0.0,
    static_force: bool = False,
    muscle_static: bool = False,
) -> dict[str, Any]:
    """Run Baseline and RTMW-l on the same decoded image without inventing accuracy."""

    try:
        image_rgb = await _comparison_image(request)
        options = RulaOptions(
            weight_kg=weight_kg,
            static_force=static_force,
            muscle_static=muscle_static,
        )
        outputs: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        for model in ("baseline", "rtmw"):
            try:
                output = get_pipeline(model).analyze_image(
                    image_rgb,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    rula_options=options,
                )
                outputs[model] = output
                summaries[model] = _response_detection_summary(output)
            except EstimatorUnavailable as error:
                outputs[model] = {
                    "model": model,
                    "error": str(error),
                    "quality": {"status": "server_error", "issues": [str(error)]},
                }
                summaries[model] = {"status": "unavailable", "reason": str(error)}
        return {
            "schema_version": "1.0",
            "frame_id": frame_id,
            "baseline": outputs["baseline"],
            "rtmw": outputs["rtmw"],
            "summary": summaries,
            "accuracy": {
                "status": "not_measurable",
                "reason": "A/B API input has no expert Ground Truth.",
            },
        }
    except HTTPException:
        raise
    except (InvalidAnalysisInput, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
