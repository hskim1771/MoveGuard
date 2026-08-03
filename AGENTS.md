# MoveGuard AI 작업 지침

이 파일은 Codex 등 AI 작업자가 저장소에서 가장 먼저 읽어야 하는 인수인계서다.
긴 대화의 기억보다 이 저장소의 코드와 아래 문서를 우선 근거로 사용한다.

## 시작 순서

1. 현재 브랜치와 `git status`를 확인한다.
2. [docs/architecture.md](docs/architecture.md)를 읽는다.
3. 요청 영역에 해당하는 문서만 추가로 읽는다.
4. 기존 구현과 실제 산출물을 확인한 뒤 변경한다.

## 절대 규칙

- `main`을 수정하거나 merge하지 않는다.
- 작업 브랜치는 `feature/mediapipe-aihub-120k-integration`이다.
- 프론트엔드를 새로 만들거나 다른 브랜치 파일로 덮어쓰지 않는다.
- API 연결에 필요한 최소 프론트엔드 변경만 허용한다.
- MediaPipe를 AI-Hub 데이터로 재학습했다고 표현하지 않는다.
- 모델 confidence를 정확도라고 표현하지 않는다.
- 전문가 최종 RULA 정답이 없으므로 현재 결과를 전문가 RULA 정확도로 표현하지 않는다.
- 원본 AI-Hub 데이터는 수정·이동·삭제하거나 GitHub에 올리지 않는다.
- 검증되지 않은 수치를 만들지 않는다.

## 문서 경로

- 전체 구조: [docs/architecture.md](docs/architecture.md)
- 프론트엔드 보호 범위: [docs/frontend-scope.md](docs/frontend-scope.md)
- 백엔드와 모델: [docs/backend-model.md](docs/backend-model.md)
- 데이터·학습 재실행: [docs/training-pipeline.md](docs/training-pipeline.md)
- 지표와 발표 기준: [docs/evaluation-guide.md](docs/evaluation-guide.md)
- 다음 작업 프롬프트: [docs/prompt-template.md](docs/prompt-template.md)
- 전체 실험 보고서: [reports/aihub_full_120k_training_report.md](reports/aihub_full_120k_training_report.md)

## 변경 원칙

- 요청 범위를 벗어난 리팩터링을 하지 않는다.
- 관련 없는 사용자 변경을 되돌리지 않는다.
- 생성 데이터와 원본 이미지는 `artifacts/` 또는 로컬 경로에 두고 Git에서 제외한다.
- 작은 모델 체크포인트만 `backend/models/`에 배포 자산으로 저장한다.
- 변경 후 관련 테스트, 실제 API 요청, `git diff --check`를 수행한다.
- 커밋과 push는 별도 통합 브랜치에만 수행한다.

## 문서 관리

- 각 문서는 가급적 200줄 이내로 유지한다.
- 길어지면 주제별 문서로 분리하고 상위 문서에 링크한다.
- 구조·수치·브랜치가 바뀌면 관련 문서도 같은 커밋에서 갱신한다.
