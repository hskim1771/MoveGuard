import os
from time import perf_counter

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .aihub_classifier import (
    InvalidImageError,
    ModelUnavailableError,
    classifier_status,
    classify_image,
)
from .analyzer import analyze_request, highest_risk
from .models import AnalyzeRequest, AnalyzeResponse


SERVICE_VERSION = "0.2.0"

app = FastAPI(
    title="MoveGuard Analysis API",
    description=(
        "MediaPipe 관절 좌표를 받아 사람별 자세 각도와 시험용 위험도를 "
        "반환합니다. 영상 원본은 받거나 저장하지 않습니다."
    ),
    version=SERVICE_VERSION,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("MOVEGUARD_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "MoveGuard Analysis API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "moveguard-analysis-api",
        "version": SERVICE_VERSION,
        "videoStored": False,
        "aihub": classifier_status(),
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    started_at = perf_counter()
    people = analyze_request(payload)
    return AnalyzeResponse(
        people=people,
        scene_risk=highest_risk([person.overall_risk for person in people]),
        elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
    )


@app.post("/v1/aihub/classify")
def classify_aihub_frame(
    request: Request,
    image: bytes = Body(..., media_type="image/jpeg"),
) -> dict[str, object]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG, PNG, or WebP image as the raw request body.",
        )

    try:
        return classify_image(image)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
