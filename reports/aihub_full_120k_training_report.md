# MoveGuard MediaPipe + AI-Hub 전체 학습 통합 보고서

> 상태: 데이터 감사 및 학습 진행 중. 확인되지 않은 수치는 기입하지 않는다.

## 기준 소스

- 프론트엔드 기준 브랜치: `feature/figma-ui-rtmw-integration`
- 프론트엔드 기준 커밋: `17eb855cc5274ce511fa66545630c616d734ba19`
- 통합 브랜치: `feature/mediapipe-aihub-120k-integration`
- 백엔드 커밋: 최종 커밋 후 기록
- `main` merge: 수행하지 않음

## AI-Hub 실제 데이터

- 원본: `/Users/admin/Downloads/253.제조업 노동자 근골격계 부담요인 예방을 위한 인공지능 데이터 구축/01-1.정식개방데이터`
- 원본 ZIP 크기: 약 164GB
- ZIP 목록 기준 이미지: 108,000장(Training 96,000 + Validation 12,000)
- ZIP 목록 기준 JSON: 108,000개(Training 96,000 + Validation 12,000)
- 요청 명칭의 “12만 장”과 실제 보유 파일 수가 다르므로 120,000장으로 과장하지 않는다.
- 라벨: 27개 AI-Hub 관절점, 작업자/공장/공정 식별자, 목·몸통·위팔·아래팔·다리 자세 및 세부 플래그.
- 전문가 RULA 총점: 샘플에는 없음. 전체 감사 결과에도 없으면 RULA 검증은 `평가 불가`.

## 데이터 분리

- seed: `20260803`
- 파일명 마지막 프레임 번호를 제거한 촬영 시퀀스를 그룹으로 사용한다.
- 그룹 단위 70%/15%/15% 분할이며 동일 영상·연속 프레임이 split 사이에 섞이지 않는다.
- 실제 분할: train 77,523 / validation 14,267 / test 16,210 (141개 그룹)
- manifest: `artifacts/aihub/audit/split_manifest.jsonl`(생성물은 Git 제외)
- manifest SHA-256: `399cb0e465ce1e2439989049e1a4214eaad2d7340d923b06c35a2151388c0c64`
- pairing 108,000쌍, 누락 0, 중복 0, invalid JSON 0.

## MediaPipe 전처리

- MediaPipe는 재학습 모델이 아니라 33개 관절 추출기다.
- AI-Hub 모델은 MediaPipe 관절 좌표와 파생 각도를 입력으로 받는 후단 자세 분류기다.
- 원본 ZIP을 풀지 않고 스트리밍하며 `features.jsonl`과 `checkpoint.json`으로 재실행한다.
- 원본 프레임 0.5 임계값: 사람 71.0%, 전신 49.6%, 9.51장/초, peak 1.16GB.
- 원본 프레임 0.3 임계값: 사람 81.0%, 전신 51.4%, 9.46장/초, peak 1.16GB.
- AI-Hub bbox + 15% 문맥, 0.3 임계값: 사람 96.9%, 10.40장/초, peak 1.03GB, 전체 예상 2.89시간.
- bbox 점검에서 전신 8관절 visibility ≥0.3은 전체의 66.3%였다.
- 전체 검출률 및 실패 사례: 실행 완료 후 실제 수치 기록

## 모델 비교

- 선형 기준 모델: 실행 완료 후 실제 지표 기록
- Gradient Boosting: 실행 완료 후 실제 지표 기록
- PyTorch MLP: 실행 완료 후 실제 지표 기록
- 테스트 세트는 validation 기반 모델 선택이 끝난 뒤 한 번만 평가한다.

## RULA 검증

- 공식 Table A/B/C 룩업은 `backend/rula.py`에 구현했다.
- 전문가 총점 정답이 없으면 정확 일치율, ±1, MAE, Weighted Kappa, 부위별 일치율은 모두 `평가 불가`로 표시한다.
- 모델 confidence를 RULA 정확도로 표시하지 않는다.

### AI-Hub 인간 검수 라벨 기반 자세 전용 RULA 일치도

- 검증 스크립트: `training/evaluate_rula_agreement.py`
- 평가 대상: 독립 test 그룹 중 MediaPipe 검출 성공 14,744프레임
- 기준값: AI-Hub 사람이 검수한 목·몸통·위팔·아래팔·다리 및 부가 자세 라벨을 공식 Table A/B/C에 입력한 파생 점수
- 양쪽 공통 고정 조건: 하중/힘 0, 근육·반복 사용 0. 다리 지지는 동일한 AI-Hub 인간 라벨 사용.
- MoveGuard 손목: 단일 RGB MediaPipe에서 확정하지 않고 중립값 사용.
- 최종점수 정확 일치율: 24.29%
- ±1점 이내 일치율: 72.26%
- 평균 절대 오차: 1.09점
- Quadratic Weighted Cohen's Kappa: 0.248
- 조치수준 일치율: 43.76%
- 조치수준 Quadratic Weighted Kappa: 0.142
- 부위별 정확 일치율: 목 32.10%, 몸통 35.66%, 위팔 25.70%, 아래팔 19.86%
- 이 결과는 전문가가 확정한 최종 RULA 총점과의 임상/현장 검증이 아니다. 원본 JSON에 전문가 최종점수, 하중, 반복·근육 사용 정답이 없으므로 `AI-Hub 인간 검수 라벨 기반 자세 전용 RULA 일치도`로만 보고한다.

## 프론트엔드 연결 테스트

- `NEXT_PUBLIC_ANALYSIS_API_URL` 사용: 확인
- MediaPipe 기본 선택: 확인
- 원본 프레임 위 스켈레톤 canvas 구조: 확인
- `/dashboard` 필수 문구 DOM 확인: 통과
- `/history` 개인/전체 통계 분리 문구 DOM 확인: 통과
- `/` 최신 통합 측정 iframe 및 환경변수 API URL 확인: 통과
- 브라우저 콘솔 오류: 없음(검증 시점)
- FastAPI `/health`: 통과
- AI-Hub 실제 이미지 → MediaPipe 33관절 응답: 통과
- 카메라·업로드·결과 저장 전체 클릭 테스트: 진행 중

## 재실행 및 resume

```bash
.venv/bin/python training/aihub_audit.py --data-root "/Users/admin/Downloads/253.제조업 노동자 근골격계 부담요인 예방을 위한 인공지능 데이터 구축/01-1.정식개방데이터" --output artifacts/aihub/audit --seed 20260803
.venv/bin/python training/preprocess_mediapipe.py --manifest artifacts/aihub/audit/split_manifest.jsonl --output artifacts/aihub/pilot --limit 1000
.venv/bin/python training/preprocess_mediapipe.py --manifest artifacts/aihub/audit/split_manifest.jsonl --output artifacts/aihub/full
.venv/bin/python training/train_models.py --features artifacts/aihub/full/features.jsonl --output artifacts/aihub/models --seed 20260803
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
NEXT_PUBLIC_ANALYSIS_API_URL=http://127.0.0.1:8000 pnpm dev
```

## 알려진 한계

- 실제 데이터는 120,000장이 아니라 현재 ZIP 기준 108,000장이다.
- 단일 RGB 카메라의 손목 비틀림·하중·반복 여부는 관절 추출만으로 확정할 수 없다.
- 전문가 RULA 정답이 없으면 RULA 정확도를 평가할 수 없다.
- 개인 기록은 작업자 이름 입력 이후 생성된 세션부터 분리된다.
