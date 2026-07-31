# AI Hub 분류 서버 인계

## 위치

- 저장소: `https://github.com/hskim1771/MoveGuard`
- 브랜치: `frontend-features`
- 소스 폴더: `aihub_server`
- 현재 배포 주소: `https://moveguard-analysis-api-4que.onrender.com`

## 주요 파일

- `aihub_server/app/main.py`: FastAPI 엔드포인트
- `aihub_server/app/aihub_classifier.py`: ONNX 자세 분류
- `aihub_server/app/analyzer.py`: MediaPipe 관절 좌표 보조 분석
- `aihub_server/aihub_models`: 부위별 ONNX 모델 5개
- `aihub_server/requirements.txt`: Python 의존성
- `aihub_server/render.yaml`: Render 배포 설정 참고
- `aihub_server/tests`: 분석 및 분류 테스트

## API

- `GET /health`: 서버와 AI Hub 모델 준비 상태 확인
- `POST /v1/analyze`: MediaPipe 관절 좌표 보조 분석
- `POST /v1/aihub/classify`: JPEG/PNG/WebP 이미지 바이트를 부위별 ONNX 모델로 분류
- `GET /docs`: FastAPI 자동 API 문서

`/v1/aihub/classify`는 업로드 폼이 아니라 이미지 원본 바이트를 요청 본문에 넣고 `Content-Type: image/jpeg` 같은 실제 이미지 형식을 지정합니다.

## 역할 구분

이 서버의 AI Hub 분류값은 참고용입니다. 손목 모델과 일부 세부 각도 정보가 없으므로 팀 FastAPI가 계산한 최종 RULA Score A/B와 Grand Score를 덮어쓰지 않습니다.

모델은 AI Hub 데이터셋 71439 `근골격계 질환 예측을 위한 작업자 자세 데이터`의 모델 패키지에서 변환했습니다. 원본 AI Hub 학습 이미지는 저장소에 포함하지 않았습니다.
