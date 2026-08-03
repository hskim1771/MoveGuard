# 백엔드와 모델

## 용어

- AI 모델 파일 자체는 백엔드가 아니다.
- FastAPI 서버, MediaPipe, 자세 모델, RULA 코드가 연결된 전체 서버 기능이 백엔드다.
- 정확한 설명: `새 자세 모델을 개발해 FastAPI 백엔드에 연결했다.`

## 운영 파이프라인

1. `POST /api/analysis/posture`가 JPEG 프레임과 작업 조건을 받는다.
2. MediaPipe가 33개 관절과 visibility를 반환한다.
3. 사람 crop 이미지와 정규화 관절 특징을 멀티모달 모델에 입력한다.
4. 모델이 목·몸통·위팔·아래팔·다리 및 14개 보정 항목을 예측한다.
5. 예측값을 RULA 구성 점수로 변환한다.
6. 하중·힘·근육 사용 조건을 추가해 Table A/B/C로 최종점수를 계산한다.

## 새로 학습한 것과 아닌 것

- 새로 학습: `multimodal_rula_v1.pt` 자세 분류 모델
- 새로 학습하지 않음: MediaPipe Pose
- 학습 모델 아님: `backend/rula.py` 공식 표 계산 코드

## 주요 파일

- `backend/app.py`: API와 파이프라인 조립
- `backend/multimodal_rula.py`: 모델 구조·전처리·추론
- `backend/models/multimodal_rula_v1.pt`: 약 989KB 배포 체크포인트
- `backend/rula.py`: RULA Table A/B/C
- `backend/requirements.txt`: 서버 Python 의존성

## API 표현 원칙

- `joint_detection_confidence`: 참고용 관절 검출 신뢰도
- `posture_classification.confidence`: 모델 출력 확률, 실제 정확도 아님
- `rula.final_score`: 공식 표로 계산한 점수
- `rula_validation`: 현재 전문가 최종점수 검증이 아님을 명시

## 알려진 한계

- 단일 RGB 영상은 손목 비틀림, 실제 하중, 반복 작업을 완전히 판별하지 못한다.
- 현장 배포 전 전문가 최종 RULA 채점과 비교 검증해야 한다.
- 가림·원거리·화면 밖 신체는 MediaPipe 성능을 낮춘다.
