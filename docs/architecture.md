# MoveGuard 아키텍처

## 목적

MoveGuard는 제조 작업자의 영상을 분석해 관절, 작업 자세, RULA 위험도를 보여주고
개인 기록과 작업장 전체 통계를 분리해 저장하는 시스템이다.

## 처리 흐름

```text
카메라 또는 업로드 영상
  → 프론트엔드가 원본 영상 프레임 표시
  → FastAPI 분석 API
  → MediaPipe Pose가 33개 관절 추출
  → MoveGuard 멀티모달 모델이 부위별 자세 분류
  → 사용자 입력 하중·반복 조건 결합
  → 공식 RULA Table A/B/C 계산
  → 프론트엔드에 관절·점수·위험단계 반환
  → 개인 기록 및 작업장 통계 저장
```

## 구성요소

- 프론트엔드: Next.js/Vinext 화면과 `public/moveguard-mvp.html`
- 분석 API: `backend/app.py`
- 관절 추출: MediaPipe Pose Landmarker Lite
- 자세 모델: `backend/models/multimodal_rula_v1.pt`
- 모델 실행: `backend/multimodal_rula.py`
- RULA 표 계산: `backend/rula.py`
- 기록 API: `app/api/sessions/route.ts`
- 학습 도구: `training/`

## 환경변수

- 프론트엔드 분석 서버: `NEXT_PUBLIC_ANALYSIS_API_URL`
- MediaPipe 모델: `MEDIAPIPE_MODEL_PATH`
- 자세 모델: `MULTIMODAL_RULA_MODEL_PATH`
- 허용 프론트엔드 Origin: `CORS_ORIGINS`

## 책임 분리

- MediaPipe는 관절 추출기이며 MoveGuard가 학습한 모델이 아니다.
- MoveGuard 모델은 이미지와 MediaPipe 특징으로 AI-Hub 자세 항목을 예측한다.
- RULA 총점은 신경망이 직접 생성하지 않고 공식 표 코드가 계산한다.
- 하중·힘·반복·정적 근육 사용은 영상만으로 확정하지 않고 사용자 입력을 받는다.

## 현재 Git 기준

- 통합 브랜치: `feature/mediapipe-aihub-120k-integration`
- 모델 통합 커밋: `3369804`
- 관절 평가 커밋: `6d94d2b`
- `main` merge: 금지
