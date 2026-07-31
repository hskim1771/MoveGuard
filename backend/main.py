"""
MoveGuard 심화버전 백엔드 — 영상 업로드 → MediaPipe 분석 → RULA 채점 → (선택) Firebase 저장

실행: uvicorn main:app --reload --port 8000
필요 환경변수: FIREBASE_CREDENTIALS_PATH (서비스 계정 키 JSON 경로) — 없으면 Firebase 저장은 건너뛰고 분석 결과만 응답.
"""
import base64
import os
import tempfile
import threading

import cv2
import mediapipe as mp
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import rula
from analyzer import analyze_request, highest_risk
from models import AnalyzeRequest, AnalyzeResponse
from aihub_classifier import (
    InvalidImageError,
    ModelUnavailableError,
    classifier_status,
    classify_image,
)

MODEL_PATH = os.environ.get("POSE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task"))
MIN_CONFIDENCE = 0.5   # 이 아래면 "인식 신뢰도 낮음" 프레임으로 취급, 전부 낮으면 재촬영 안내
SAMPLE_FRAMES = 30

# 배포된 프론트엔드 도메인만 쉼표로 구분해서 넣는다 (예: https://moveguard.onrender.com).
# 안 넣으면(로컬 개발 단계) 아래 정규식만 적용돼서 localhost/사설망 IP에서는 계속 열려있다.
_EXTRA_ORIGINS = [o.strip() for o in os.environ.get("MOVEGUARD_ALLOWED_ORIGINS", "").split(",") if o.strip()]
# 로컬 테스트(팀원 각자 python http.server, 같은 와이파이 IP 접속)는 항상 허용
_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?$"

app = FastAPI(title="MoveGuard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_EXTRA_ORIGINS,
    allow_origin_regex=_LOCAL_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드가 공개되면 이 서버 주소도 페이지 소스에 그대로 보인다.
# 완전한 보안은 아니지만(키도 프론트 코드에 있어 결국 보임), 무작위 봇/스캐너의
# 자동 남용은 이걸로 충분히 막는다. MOVEGUARD_API_KEY 안 정해두면(로컬 개발) 검사 자체를 건너뛴다.
_API_KEY = os.environ.get("MOVEGUARD_API_KEY")


def require_api_key(x_moveguard_key: str | None = Header(default=None)):
    if _API_KEY and x_moveguard_key != _API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")

# 영상 분석(mediapipe)과 AI Hub 분류(onnx)를 무료 서버 메모리 한도 안에서
# 동시에 안 돌게 하려고 무거운 작업 전체를 이 락 하나로 순차 처리한다.
_heavy_op_lock = threading.Lock()

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
    """
    서비스 계정 키가 있으면 Firebase Admin 초기화, 없으면 None 반환 (기능만 꺼짐, 서버는 계속 동작).
    두 가지 방식 지원:
      1) FIREBASE_CREDENTIALS_JSON — JSON 내용 전체를 환경변수 값으로 (권장, Render Secret File보다 안정적)
      2) FIREBASE_CREDENTIALS_PATH — JSON 파일 경로
    """
    import json

    import firebase_admin
    from firebase_admin import credentials

    cred = None
    json_str = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if json_str:
        try:
            cred = credentials.Certificate(json.loads(json_str))
        except Exception as e:
            print(f"⚠️  FIREBASE_CREDENTIALS_JSON 파싱 실패: {e}")

    if cred is None:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        if cred_path and os.path.exists(cred_path) and os.path.getsize(cred_path) > 0:
            try:
                cred = credentials.Certificate(cred_path)
            except Exception as e:
                print(f"⚠️  FIREBASE_CREDENTIALS_PATH 읽기 실패: {e}")

    if cred is None:
        print("⚠️  Firebase 인증정보 없음 — Firebase 저장 없이 분석만 동작합니다.")
        return None

    try:
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
    return {"status": "ok", "firebase_connected": _firebase is not None, "aihub": classifier_status()}


@app.post("/v1/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
def analyze_landmarks(payload: AnalyzeRequest) -> AnalyzeResponse:
    from time import perf_counter

    started_at = perf_counter()
    people = analyze_request(payload)
    return AnalyzeResponse(
        people=people,
        scene_risk=highest_risk([person.overall_risk for person in people]),
        elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
    )


@app.post("/v1/aihub/classify", dependencies=[Depends(require_api_key)])
def classify_aihub_frame(
    request: Request,
    image: bytes = Body(..., media_type="image/jpeg"),
) -> dict:
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG, PNG, or WebP image as the raw request body.",
        )

    try:
        with _heavy_op_lock:
            return classify_image(image)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/analyze", dependencies=[Depends(require_api_key)])
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

        best = None          # 신뢰도 기준을 넘긴 프레임 중 가장 위험한(점수 높은) 것
        low_conf_count = 0
        checked_count = 0
        best_frame_image = None

        with _heavy_op_lock:
            landmarker = get_landmarker()
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
