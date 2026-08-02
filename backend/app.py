from __future__ import annotations

import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.rula import calculate

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv("MEDIAPIPE_MODEL_PATH", ROOT / "mediapipe/models/pose_landmarker_lite.task"))
app = FastAPI(title="MoveGuard MediaPipe Analysis API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_methods=["*"], allow_headers=["*"])


def angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    u, v = points[a] - points[b], points[c] - points[b]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1)))) if denom else 0.0


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "pose_model": "mediapipe", "landmark_count": 33, "classifier_available": (ROOT / "artifacts/aihub/models/final.joblib").exists()}


@app.post("/rula")
def rula(scores: dict[str, int]) -> dict[str, int | str]:
    try:
        return calculate(scores)
    except KeyError as exc:
        raise HTTPException(422, f"missing RULA component: {exc.args[0]}") from exc


@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...)) -> dict[str, object]:
    return analyze_bytes(await file.read())


@app.post("/api/analysis/posture")
async def analyze_posture(request: Request, model: str = "baseline") -> dict[str, object]:
    if model not in {"baseline", "mediapipe"}:
        raise HTTPException(422, "This server uses MediaPipe Pose as the operational model")
    result = analyze_bytes(await request.body())
    if not result.get("landmarks"):
        return {"model": "baseline", "pose": {"confidence": 0}, "joints": {}, "measurements": {}, "rula": {"final_score": None, "risk_level": "not_measurable"}, **result}
    landmarks = result["landmarks"]
    names = {"neck":0,"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,"left_wrist":15,"right_wrist":16,"left_hip":23,"right_hip":24,"left_knee":25,"right_knee":26,"left_ankle":27,"right_ankle":28}
    confidence = result["quality"]["joint_detection_confidence"]
    return {"model": "baseline", "pose": {"confidence": confidence}, "joints": {name:{**landmarks[idx],"valid":landmarks[idx]["visibility"]>=.5} for name,idx in names.items()}, "measurements": {}, "rula": {"final_score": None, "risk_level": "not_measurable"}, **result}


def analyze_bytes(raw: bytes) -> dict[str, object]:
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "invalid image")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    opts = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return {"model": "mediapipe", "landmarks": [], "quality": {"status": "not_measurable", "reason": "no_person"}, "rula_validation": "평가 불가"}
    lm = result.pose_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)
    confidence = float(np.mean([p.visibility for p in lm]))
    angles = {
        "left_elbow": angle(pts, 11, 13, 15), "right_elbow": angle(pts, 12, 14, 16),
        "left_shoulder": angle(pts, 23, 11, 13), "right_shoulder": angle(pts, 24, 12, 14),
        "left_knee": angle(pts, 23, 25, 27), "right_knee": angle(pts, 24, 26, 28),
    }
    return {
        "model": "mediapipe", "landmark_count": 33,
        "landmarks": [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility, "presence": p.presence} for p in lm],
        "angles": angles,
        "quality": {"status": "valid" if confidence >= .5 else "low_confidence", "joint_detection_confidence": confidence, "reference_only": True},
        "posture_classification": None,
        "rula_validation": "평가 불가",
    }
