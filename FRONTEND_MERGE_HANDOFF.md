# MoveGuard 프론트엔드 병합 인계

## 브랜치

- 저장소: `https://github.com/hskim1771/MoveGuard`
- 기준 브랜치: `main`
- 프론트엔드 브랜치: `frontend-features`
- 주요 변경 파일: `moveguard_mvp.html`

## 포함된 기능

- 5단계 조사 위저드
- 근로자 촬영·분석 동의
- 실시간 카메라 측정과 촬영 영상 업로드 선택
- 측정 중 추정 RULA 안내
- 최고 위험 자세 자동 선택
- 하중 3단계 및 반복·지지 조건 입력
- 동일한 RULA Table A/B/C 결과표
- 팀 FastAPI 백엔드 최종 영상 분석
- 서버 분석 결과와 Firebase 저장 여부 표시
- AI Hub 분류 결과 비교
- 법정서식 자동 생성과 대표 위험 프레임 첨부
- 서버 연결 실패 시 브라우저 RULA 결과 유지

실시간 카메라와 업로드 영상은 같은 각도 계산, 위험 프레임 선택, RULA 결과표와 법정서식을 사용합니다. 실시간 화면의 점수는 하중·반복 조건 입력 전의 안내값입니다. 사용자가 하중·반복 조건을 입력하고 결과 계산을 누르면 영상과 보충 입력을 팀 백엔드로 전송하고, 백엔드 RULA 결과를 최종 결과표와 법정서식에 반영합니다.

## 팀 백엔드 연결

- 배포 주소: `https://moveguard-backend.onrender.com`
- 상태 확인: `GET /health`
- 영상 분석: `POST /analyze` (`multipart/form-data`)
- 전송 항목: `video`, `weight_kg`, `static_force`, `muscle_static`, `legs_unsupported`, `job_name`
- 응답 반영: RULA 점수, 대표 프레임, 분석 프레임 수, 저신뢰 프레임 수, Firebase 저장 여부

업로드 모드는 선택한 원본 영상을 전송합니다. 실시간 모드는 측정 중 브라우저가 임시 녹화한 영상을 전송합니다. 브라우저가 녹화를 지원하지 않거나 서버 분석에 실패하면 기존 브라우저 RULA 결과를 유지합니다.

## AI Hub 분류 서버 소스

- 소스 폴더: `aihub_server`
- 배포 서버: `https://moveguard-analysis-api-4que.onrender.com`
- 핵심 분류 코드: `aihub_server/app/aihub_classifier.py`
- ONNX 모델: `aihub_server/aihub_models`
- 실행 및 API 설명: `aihub_server/README.md`
- 별도 인계 문서: `AIHUB_SERVER_HANDOFF.md`

AI Hub 분류 서버는 팀 FastAPI RULA 백엔드와 별도 서비스입니다. 프론트는 대표 위험 프레임 1장을 `/v1/aihub/classify`로 보내 참고용 자세 분류 결과를 받고, 이 결과는 최종 RULA 점수를 덮어쓰지 않습니다.

## 병합 순서

1. `frontend-features` 브랜치의 `moveguard_mvp.html`을 먼저 확인합니다.
2. 정상 동작을 확인하면 `main`에 병합합니다.
3. 이후 `offline-mediapipe` 변경을 합쳐 CDN과 서버 없이 동작하도록 조정합니다.
4. 오프라인 변경 후에도 실시간 측정과 업로드 분석이 같은 RULA 결과로 이어지는지 다시 확인합니다.

## 유지할 연결점

- `window.MOVEGUARD_CONFIG.teamBackendUrl`의 팀 배포 주소(필요하면 개발 환경에서 덮어쓰기 가능)
- `window.MOVEGUARD_CONFIG.aihubApiUrl`의 로컬·배포 주소 분기
- 팀 FastAPI 결과를 최종 RULA 값으로 사용하고, AI Hub 비교 결과는 점수를 덮어쓰지 않는 구조
- 원본 영상은 분석 중 임시 파일로만 사용하고 서버 분석 종료 후 삭제하는 구조
- 최고 위험 프레임 사진 1장과 RULA 결과·작업명은 Firebase에 저장될 수 있음을 동의 화면에 안내하는 구조
- AI Hub 비교가 켜진 환경에서는 대표 사진 1장이 별도 분류 서버로 전송됨을 동의 화면에 안내하는 구조
- 최고 위험 프레임 사진 1장만 법정서식에 사용하는 구조
- 서버가 없어도 브라우저 RULA 분석이 계속되는 예외 처리

## 확인 항목

- 실시간 카메라 시작·종료
- 영상 업로드 분석
- 유효 프레임 및 저신뢰 프레임 안내
- 하중 3단계 입력
- RULA 결과표
- 팀 FastAPI 최종 결과 및 Firebase 저장 여부
- 법정서식 생성 및 인쇄
- 서버 연결 상태에 따른 브라우저 분석 유지
