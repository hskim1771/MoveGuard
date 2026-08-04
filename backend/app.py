from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.rula import calculate
from backend.multimodal_rula import RulaPosturePredictor
from backend.pose_refiner import PoseRefiner

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv("MEDIAPIPE_MODEL_PATH", ROOT / "mediapipe/models/pose_landmarker_lite.task"))
RULA_MODEL = Path(os.getenv("MULTIMODAL_RULA_MODEL_PATH", ROOT / "backend/models/multimodal_rula_v1.pt"))
POSE_REFINER_MODEL = Path(os.getenv("POSE_REFINER_MODEL_PATH", ROOT / "checkpoints/wrist_refiner_joint_domain_v1/best.pt"))
POSE_REFINER_ENABLED = os.getenv("POSE_REFINER_ENABLED", "true").lower() not in {"0", "false", "no"}
POSE_REFINER_APPLY_TO_RULA = os.getenv("POSE_REFINER_APPLY_TO_RULA", "false").lower() in {"1", "true", "yes"}
app = FastAPI(title="MoveGuard MediaPipe Analysis API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_methods=["*"], allow_headers=["*"])


def angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    u, v = points[a] - points[b], points[c] - points[b]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1)))) if denom else 0.0


@app.get("/health")
def health() -> dict[str, object]:
    refiner_available = POSE_REFINER_ENABLED and POSE_REFINER_MODEL.exists()
    return {"ok": True, "pose_model": "mediapipe", "landmark_count": 33, "pose_refiner_available": refiner_available, "pose_refiner": "joint_domain_v1" if refiner_available else None, "pose_refiner_applied_to_rula": refiner_available and POSE_REFINER_APPLY_TO_RULA, "classifier_available": RULA_MODEL.exists(), "rula_model": "multimodal_rula_multihead_v1" if RULA_MODEL.exists() else None, "posture_analysis_v2": {"default_model": "baseline", "models": {"baseline": {"pose": {"configured": MODEL.exists(), "backend": "mediapipe", "refiner": "joint_domain_v1" if refiner_available else None}}, "rtmw": {"pose": {"configured": False, "backend": "rtmw"}}}}}


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
async def analyze_posture(
    request: Request, model: str = "baseline", weight_kg: float = 0,
    static_force: bool = False, muscle_static: bool = False,
    legs_unsupported: bool = False, timestamp_ms: float | None = None,
) -> dict[str, object]:
    if model not in {"baseline", "mediapipe"}:
        raise HTTPException(422, "This server uses MediaPipe Pose as the operational model")
    result = analyze_bytes(await request.body(), weight_kg=weight_kg, static_force=static_force, muscle_static=muscle_static, legs_unsupported=legs_unsupported)
    if not result.get("landmarks"):
        quality = {"status": "not_measurable", "pose_valid": False, "hand_valid": False, "issues": [result.get("quality", {}).get("reason", "no_person")]}
        rula_result = {"final_score": None, "risk_level": "not_measurable", "valid": False, "issues": quality["issues"]}
        return {**result, "schema_version": "1.1", "processor": "mediapipe+multimodal-rula", "model": "baseline", "people": [], "pose": {"confidence": 0}, "joints": {}, "joint_angles": {}, "measurements": {}, "rula": rula_result, "quality": quality, "elapsed_ms": result.get("elapsed_ms", 0), "video_stored": False}
    landmarks = result["landmarks"]
    names = {"neck":0,"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,"left_wrist":15,"right_wrist":16,"left_hip":23,"right_hip":24,"left_knee":25,"right_knee":26,"left_ankle":27,"right_ankle":28}
    confidence = result["quality"]["joint_detection_confidence"]
    source_model = "mediapipe+joint-domain-refiner" if result.get("pose_refiner", {}).get("active") else "mediapipe"
    joints = {name:{**landmarks[idx], "joint": name, "confidence": landmarks[idx]["visibility"], "source_model": source_model, "valid": landmarks[idx]["visibility"]>=.5} for name,idx in names.items()}
    quality = {**result["quality"], "pose_valid": confidence >= .5, "hand_valid": joints["left_wrist"]["valid"] and joints["right_wrist"]["valid"], "issues": [] if confidence >= .5 else ["low_joint_detection_confidence"]}
    rula_result = result.get("rula", {"final_score": None, "risk_level": "not_measurable", "valid": False, "issues": ["posture_model_unavailable"]})
    person = {"person_id": "person-1", "model": "baseline", "pose": {"confidence": confidence}, "joints": joints, "joint_angles": result["angles"], "measurements": {}, "rula": rula_result, "quality": quality}
    return {"schema_version": "1.1", "processor": "mediapipe+joint-domain-refiner+multimodal-rula" if result.get("pose_refiner", {}).get("active") else "mediapipe+multimodal-rula", "model": "baseline", "people": [person], "pose": person["pose"], "joints": joints, "joint_angles": result["angles"], "measurements": {}, "rula": rula_result, "quality": quality, "elapsed_ms": result["elapsed_ms"], "video_stored": False, "landmarks": landmarks, "pose_refiner": result.get("pose_refiner"), "posture_classification": result.get("posture_classification")}


@lru_cache(maxsize=1)
def posture_predictor() -> RulaPosturePredictor:
    return RulaPosturePredictor(RULA_MODEL)


@lru_cache(maxsize=1)
def pose_refiner() -> PoseRefiner:
    return PoseRefiner(POSE_REFINER_MODEL)


def analyze_bytes(raw: bytes, *, weight_kg: float = 0, static_force: bool = False, muscle_static: bool = False, legs_unsupported: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "invalid image")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    opts = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return {"model": "mediapipe", "landmarks": [], "quality": {"status": "not_measurable", "reason": "no_person"}, "rula_validation": "평가 불가", "elapsed_ms": (time.perf_counter()-started)*1000}
    raw_lm = result.pose_landmarks[0]
    refiner_active = POSE_REFINER_ENABLED and POSE_REFINER_MODEL.exists()
    lm = pose_refiner().predict(raw_lm) if refiner_active else raw_lm
    scoring_lm = lm if refiner_active and POSE_REFINER_APPLY_TO_RULA else raw_lm
    pts = np.array([[p.x, p.y, p.z] for p in scoring_lm], dtype=float)
    confidence = float(np.mean([p.visibility for p in lm]))
    angles = {
        "left_elbow": angle(pts, 11, 13, 15), "right_elbow": angle(pts, 12, 14, 16),
        "left_shoulder": angle(pts, 23, 11, 13), "right_shoulder": angle(pts, 24, 12, 14),
        "left_knee": angle(pts, 23, 25, 27), "right_knee": angle(pts, 24, 26, 28),
    }
    posture = None
    rula_result = {"final_score": None, "risk_level": "not_measurable", "valid": False, "issues": ["posture_model_unavailable"]}
    if RULA_MODEL.exists():
        muscle = 1 if muscle_static else 0
        force = 2 if weight_kg > 10 else 1 if weight_kg >= 2 or static_force else 0
        prediction = posture_predictor().predict(rgb, scoring_lm, muscle=muscle, force=force, legs_unsupported=legs_unsupported)
        risk = "low" if prediction["final_score"] <= 2 else "medium" if prediction["final_score"] <= 4 else "high" if prediction["final_score"] <= 6 else "very_high"
        components = prediction["components"]
        rula_result = {**prediction, "valid": True, "risk_level": risk, "issues": [], "scored_side": "model", "neck_score": components["neck"], "trunk_score": components["trunk"], "upper_arm_score": components["upper_arm"], "lower_arm_score": components["lower_arm"], "wrist_score": components["wrist"], "wrist_twist_score": components["wrist_twist"], "leg_score": components["legs"], "force_score": force, "muscle_score": muscle}
        posture = {"model": "multimodal_rula_multihead_v1", "labels": prediction["labels"], "confidence": prediction["label_confidence"], "confidence_notice": "모델 출력 확률이며 실제 정확도가 아닌 참고값"}
    return {
        "model": "mediapipe", "landmark_count": 33,
        "pose_refiner": {"active": refiner_active, "model": "joint_domain_v1" if refiner_active else None, "purpose": "2D joint coordinate correction", "industrial_test_pck_0_10": 0.8848025916823593, "applied_to_rula": refiner_active and POSE_REFINER_APPLY_TO_RULA, "rula_accuracy_claimed": False},
        "landmarks": [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility, "presence": p.presence} for p in lm],
        "angles": angles,
        "quality": {"status": "valid" if confidence >= .5 else "low_confidence", "joint_detection_confidence": confidence, "reference_only": True},
        "posture_classification": posture, "rula": rula_result,
        "rula_validation": "AI-Hub 인체검수 라벨 파생 자세점수와의 일치도; 전문가 최종 RULA 정확도 아님",
        "elapsed_ms": (time.perf_counter()-started)*1000,
    }
