# MoveGuard backend

기존 백엔드 API와 자세 분석 v2 API를 함께 제공하는 FastAPI 서버입니다.
원본 영상은 기존 `/analyze`의 처리 중 임시 파일을 제외하면 저장하지 않으며,
새 `/api/analysis/posture`는 이미지 프레임을 메모리에서만 처리합니다.

## 실행

Python 3.12를 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

환경변수 예시는 `.env.example`에 있습니다. `MOVEGUARD_API_KEY`를 지정하면
분석 API 요청에 `X-MoveGuard-Key` 헤더가 필요합니다.

## 기존 API 유지

- `GET /health`
- `POST /analyze`: 영상 업로드 MediaPipe + 기존 RULA
- `POST /v1/analyze`: 브라우저에서 검출한 33개 관절 분석
- `POST /v1/aihub/classify`: 기존 AI Hub ONNX 비교 분석

## 자세 분석 v2

`POST /api/analysis/posture`

지원 입력:

1. 원본 JPEG/PNG/WebP 바이트
2. `multipart/form-data`의 `image` 또는 `frame` 파일
3. 이미 프런트에서 검출한 Pose/Hand landmark JSON

영상 파일 자체는 받지 않습니다. 프런트에서 프레임을 샘플링해 이미지로
전송하면 됩니다.

`model` 쿼리로 기존 방식과 RTMW를 선택합니다. 기본값은 기존 동작을 보존하는
`baseline`이며, `rtmw`를 사용하려면 선택 의존성을 설치해야 합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rtmw.txt
$env:MOVEGUARD_RTMW_DEVICE="cpu"
```

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/analysis/posture?model=rtmw&worker_id=W-001&measurement_id=M-001" `
  -ContentType "image/jpeg" `
  -InFile ".\frame.jpg"
```

응답에는 다음 감사·검증용 필드가 포함됩니다.

- `frame_id`, `timestamp_ms`, `worker_id`, `measurement_id`
- 선택된 `model`과 공통 관절명 기준의 `joints`, 각 관절의 `source_model`
- Pose/Hand landmark와 confidence
- 객체 탐지 결과와 사람-객체 거리 관계
- 관절 각도 원시값과 부위별 `valid`, `low_confidence`, `not_measurable`
- RULA 부분 점수, 최종 점수, 위험 수준, 가정 및 측정 실패 이유

사람이 없거나 필요한 관절이 가려졌거나 손을 검출하지 못한 경우 임의 좌표나
임의 최종 점수를 만들지 않습니다. 객체 전용 모델이 설정되지 않은 기본 상태도
`objects: []`와 `configured: false`로 명시합니다.

동일 이미지에서 두 모델을 함께 확인하려면 다음 A/B 엔드포인트를 사용합니다.

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/analysis/compare" `
  -ContentType "image/jpeg" `
  -InFile ".\frame.jpg"
```

이 응답은 관절 검출 수와 계산 가능 여부만 비교합니다. 전문가 Ground Truth가
없는 단일 입력에서 정확도 수치를 만들지 않고 `accuracy.status`를
`not_measurable`로 반환합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*_v2.py" -v
```

객체 탐지 모델을 연결할 때만 선택 의존성을 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-object-detection.txt
$env:MOVEGUARD_OBJECT_DETECTOR="ultralytics"
$env:OBJECT_DETECTOR_MODEL_PATH="C:\models\moveguard-objects.pt"
```

## Baseline vs Hybrid 동일조건 평가

`evaluation/`에는 동일 영상·동일 frame sampling·동일 각도/RULA 엔진으로
Baseline과 Hybrid를 비교하는 도구가 있습니다. Ground Truth가 없으면
confidence나 공식 AP를 accuracy로 바꾸지 않고 정확도 지표를
`not_measurable`로 출력합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rtmw.txt
.\.venv\Scripts\python.exe -m evaluation.compare `
  --video ..\test_clip.mp4 `
  --output evaluation\results\rtmw_test_clip_current `
  --candidate rtmw
```

상세 지표 정의, annotation 방법, 전문 모델 fusion 위치는
`evaluation/README.md`를 참고하세요.

자세 분석 v2 구조와 향후 모델 교체 위치는 `analysis_v2/README.md`를
참고하세요.
