"""
MoveGuard 심화버전 백엔드 — 영상 업로드 → MediaPipe 분석 → RULA 채점 → (선택) Firebase 저장

실행: uvicorn main:app --reload --port 8000
필요 환경변수: FIREBASE_CREDENTIALS_PATH (서비스 계정 키 JSON 경로) — 없으면 Firebase 저장은 건너뛰고 분석 결과만 응답.
"""
import base64
import os
import tempfile

import cv2
import mediapipe as mp
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import rula

MODEL_PATH = os.environ.get("POSE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task"))
MIN_CONFIDENCE = 0.5   # 이 아래면 "인식 신뢰도 낮음" 프레임으로 취급, 전부 낮으면 재촬영 안내
SAMPLE_FRAMES = 30

app = FastAPI(title="MoveGuard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 실제 프론트 도메인으로 좁힐 것
    allow_methods=["*"],
    allow_headers=["*"],
)

_landmarker = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def try_init_firebase():
    """서비스 계정 키가 있으면 Firebase Admin 초기화, 없으면 None 반환 (기능만 꺼짐, 서버는 계속 동작)."""
    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    if not cred_path or not os.path.exists(cred_path):
        print("⚠️  FIREBASE_CREDENTIALS_PATH 없음 — Firebase 저장 없이 분석만 동작합니다.")
        return None
    try:
        import firebase_admin
        from firebase_admin import credentials
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", "moveguard-c6e06.firebasestorage.app"),
        })
        return firebase_admin
    except Exception as e:
        print(f"⚠️  Firebase 초기화 실패, 분석만 동작합니다: {e}")
        return None


_firebase = try_init_firebase()


@app.get("/health")
def health():
    return {"status": "ok", "firebase_connected": _firebase is not None}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    weight_kg: float = Form(0),
    static_force: bool = Form(False),
    muscle_static: bool = Form(False),
    legs_unsupported: bool = Form(False),
    job_name: str = Form(""),
):
    tmp_path = None
    try:
        # 1. 영상을 임시 파일로 저장 (분석 끝나면 즉시 삭제 — 개인정보 정책)
        suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "영상을 열 수 없습니다. 파일 형식을 확인해주세요.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // SAMPLE_FRAMES, 1)

        landmarker = get_landmarker()
        best = None          # 신뢰도 기준을 넘긴 프레임 중 가장 위험한(점수 높은) 것
        low_conf_count = 0
        checked_count = 0
        best_frame_image = None

        idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            checked_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                lm = [(p.x, p.y, p.visibility) for p in result.pose_landmarks[0]]
                angles = rula.compute_frame_angles(lm)

                if angles["confidence"] < MIN_CONFIDENCE:
                    low_conf_count += 1
                else:
                    total_score = (
                        rula.neck_score(angles["neckAngle"]) + rula.trunk_score(angles["trunkAngle"])
                        + rula.upper_arm_score(angles["upperArmAngle"]) + rula.lower_arm_score(angles["elbowAngle"])
                    )
                    if best is None or total_score > best["total_score"]:
                        best = {"angles": angles, "total_score": total_score}
                        best_frame_image = frame
            idx += step
            if idx >= total_frames:
                break

        cap.release()

        # 2. 저신뢰 프레임 필터링 — 쓸만한 프레임이 하나도 없으면 재촬영 안내 (기획서 약속 기능)
        if best is None:
            raise HTTPException(
                422,
                "자세를 신뢰할 수 있게 인식한 프레임이 없습니다. "
                "조명이 밝은 곳에서, 몸 전체가 보이도록 측면에서 다시 촬영해주세요."
                + (f" (분석 시도 {checked_count}프레임 중 {low_conf_count}개가 저신뢰였습니다)" if checked_count else ""),
            )

        # 3. RULA 채점
        rula_result = rula.compute_rula(best["angles"], weight_kg, static_force, muscle_static, legs_unsupported)

        # 4. 대표 프레임 사진 캡처 (base64) — 원본 영상은 이 시점 이후 바로 삭제
        _, buf = cv2.imencode(".jpg", best_frame_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_b64 = base64.b64encode(buf).decode("utf-8")

        # 5. Firebase 저장 (있으면), 실패해도 분석 결과 응답은 그대로 나가도록 감싸기
        firebase_saved = False
        if _firebase is not None:
            try:
                firebase_saved = save_to_firebase(rula_result, photo_b64, job_name)
            except Exception as e:
                print(f"⚠️  Firebase 저장 실패 (분석 결과는 정상 반환): {e}")

        return {
            "rula": rula_result,
            "photoBase64": photo_b64,
            "firebaseSaved": firebase_saved,
            "framesChecked": checked_count,
            "lowConfidenceFrames": low_conf_count,
        }

    finally:
        # 원본 영상 즉시 삭제 (성공하든 실패하든)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_to_firebase(rula_result, photo_b64, job_name):
    from datetime import datetime, timezone

    from firebase_admin import firestore, storage

    # 사진을 Storage에 업로드
    bucket = storage.bucket()
    photo_bytes = base64.b64decode(photo_b64)
    blob_name = f"photos/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jpg"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(photo_bytes, content_type="image/jpeg")

    # 결과를 Firestore에 기록
    db = firestore.client()
    db.collection("rula_results").add({
        "jobName": job_name,
        "grandScore": rula_result["grandScore"],
        "actionLevel": rula_result["actionLevel"]["label"],
        "scoreA": rula_result["scoreA"],
        "scoreB": rula_result["scoreB"],
        "photoPath": blob_name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    })
    return True
