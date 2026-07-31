# MoveGuard Analysis API

MediaPipe가 브라우저에서 만든 33개 관절 좌표를 받아 사람별 자세 각도와
시험용 위험도를 반환하는 FastAPI 서버입니다. 영상 원본은 받거나 저장하지
않습니다.

## Endpoints

- `GET /health`: 서버 상태 확인
- `POST /v1/analyze`: 최대 4명의 관절 좌표 분석
- `POST /v1/aihub/classify`: 위험 프레임 JPEG/PNG/WebP를 AI Hub 모델로 분류
- `GET /docs`: FastAPI 자동 API 문서

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

프론트 개발환경은 기본적으로 `http://127.0.0.1:8000`을 사용합니다.
배포된 서버를 연결할 때는 프론트의 `NEXT_PUBLIC_ANALYSIS_API_URL` 값을
서버 주소로 설정합니다.

## AI Hub 자세 분류

배포 서버는 `aihub_server/aihub_models`의 ONNX 모델을 사용합니다. 원본 PyTorch
모델과 같은 분류 결과를 유지하면서 무료 서버의 메모리 사용량을 줄인
형식입니다.

```powershell
Set-Location aihub_server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

다른 위치를 사용할 때는 `AIHUB_MODEL_DIR` 환경 변수로 경로를 지정합니다.
이미지 요청은 파일 업로드 폼이 아니라 원본 이미지 바이트를 본문에 넣고
`Content-Type: image/jpeg`로 전송합니다.

AI Hub 제공 자세 분류기는 손목 모델과 일부 RULA 각도 구간이 없으므로,
응답의 결과는 기존 MediaPipe RULA 결과를 대체하지 않는 비교용입니다.

모델 출처는 AI Hub 데이터셋 71439
`근골격계 질환 예측을 위한 작업자 자세 데이터`이며, 서비스와 외부 결과에는
AI Hub 데이터 활용 사실을 표시합니다. 원본 이미지 데이터는 저장소에
포함하지 않습니다.
