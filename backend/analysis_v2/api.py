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
    pipeline = get_pipeline()
    try:
        if content_type == "application/json":
            payload = LandmarkPostureRequest.model_validate(await request.json())
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
