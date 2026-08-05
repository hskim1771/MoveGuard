# MoveGuard Current Pipeline Report

조사일: 2026-08-05  
브랜치: `feature/figma-mediapipe-joint-domain-integration`  
기준 커밋: `605a5fa468d7c575ff82a780a3e2106ab28a8088`

## 1. 현재 영상 분석 흐름

### 업로드 영상

1. 브라우저가 영상을 로드한다.
2. `step = max(duration / 30, 0.1)`로 영상 전체에서 최대 약 30개 시점을 균등 추출한다.
3. 각 시점에서 브라우저 MediaPipe Pose가 한 사람의 33개 landmark를 검출한다.
4. 상지 대표 측의 목·몸통·위팔·팔꿈치 각도를 계산한다.
5. 평균 상지 confidence, 전신 8관절 confidence, 화면 내부 여부를 검사한다.
6. 품질을 통과한 각 frame에 기존 RULA Table A/B/C 기반 provisional score를 계산한다.
7. `grand * 10 + component scores`가 가장 큰 단일 frame을 최종 위험 frame으로 선택한다.
8. 선택 frame 한 장을 FastAPI `/api/analysis/posture`로 전송한다.
9. 서버 MediaPipe → optional joint-domain 좌표 보정 → 멀티모달 자세분류 → 기존 `backend/rula.py` Table A/B/C 순으로 결과를 만든다.
10. 프론트가 서버 결과가 유효하면 서버 RULA를 사용하고, 실패하면 브라우저 provisional RULA를 사용한다.

### 실시간 카메라

- MediaPipe 호출 간격: 최소 160ms, 즉 최대 약 6.25회/초.
- 유효 frame마다 provisional RULA를 계산한다.
- 업로드 영상과 동일한 가중치로 최고 위험 frame을 보존한다.
- 서버 분석은 선택된 최고 위험 frame 한 장에만 수행한다.

### 별도 React 카메라 화면

- `app/CameraPrototype.tsx`는 MediaPipe를 최소 70ms 간격으로 실행한다.
- UI 통계 갱신은 최소 140ms 간격이다.
- MediaPipe 서버 분석은 1초, RTMW 선택 시 2.5초 간격이다.
- 세션 통계는 부위별 최대/평균과 위험 sample 수를 저장하며 temporal outlier 처리는 없다.

## 2. Frame sampling

- 고정 `SAMPLE_FRAMES` 상수는 없다.
- 업로드 영상은 전체 길이를 30구간으로 나누므로 긴 영상일수록 frame 간격이 크다.
- 30초 영상은 약 1초 간격, 3분 영상은 약 6초 간격이다.
- 따라서 업로드 분석은 연속 frame smoothing에 부적합한 sparse sampling이다.
- 안정화는 confidence, 해부학적 비율, 화면 범위, 이웃 frame 지지 기반 robust outlier 판정을 우선해야 한다.

## 3. 현재 품질 기준

- `MIN_FRAME_CONFIDENCE = 0.55`
- `MIN_USABLE_FRAMES = 5`
- 전신 검사 관절: 좌우 어깨·엉덩이·무릎·발목 8개
- 각 관절은 visibility 0.4 이상이고 정규화 좌표 0.015~0.985 범위여야 한다.
- 조건을 통과하지 못한 frame은 최종 후보에서 제외한다.

## 4. 현재 없는 기능

- keypoint smoothing 없음
- temporal filtering 없음
- NaN/Inf, 좌표 급점프, 관절 길이 변화 기반 outlier 판정 없음
- 좌우 swap 판정 없음
- 앞뒤 frame 지지 기반 transient/suspected outlier 분류 없음
- RULA 계산 직전 stabilized keypoint 적용 없음
- 위험 점수의 support frame/persistence 계산 없음

## 5. 현재 최종 frame 선택 방식

- 품질 기준을 통과한 frame만 후보가 된다.
- 최종값은 frame 전체 provisional RULA 점수가 가장 큰 단일 frame이다.
- joint별 최대각도를 조합해 가상의 frame을 만들지는 않는다.
- 동점 해소는 목·몸통·위팔·아래팔 component 합을 사용한다.
- 최고점이 한 frame만 나타났는지 여러 frame 지속됐는지는 구분하지 않는다.

## 6. 서버 Pose/RULA 흐름

- 기본 검출기: MediaPipe Pose 33 landmarks
- 공동학습 모델: `joint_domain_v1`, 2D 공통 관절 좌표 보정용
- 좌표 보정 API 출력: 기본 활성
- 보정 좌표의 RULA 적용: 기본 비활성 (`POSE_REFINER_APPLY_TO_RULA=false`)
- 자세분류: `multimodal_rula_multihead_v1`
- 공식 lookup: `backend/rula.py`의 기존 Table A/B/C
- 서버는 현재 단일 이미지 endpoint이므로 자체 시간축 정보가 없다.

## 7. 산업현장 데이터 실제 schema

- 전체 108,000 image/JSON pair, 기존 독립 test 16,210장은 변경하지 않는다.
- keypoint: 27개 `[x, y, visibility]`
- 손목 각도에 필요한 점:
  - Left elbow 6, wrist 7, index 8, pinky 9, thumb 10
  - Right elbow 12, wrist 13, index 14, pinky 15, thumb 16
- 손 방향은 유효한 index/pinky/thumb의 평균점으로 구성할 수 있다.
- 실제 label field: `wrist_bending`, `wrist_twisted`, `wrist_twisted_end`
- 전문가 최종 RULA field는 없다.

## 8. 프론트가 의존하는 기존 API field

- 최상위: `schema_version`, `processor`, `model`, `people`, `pose`, `joints`, `joint_angles`, `measurements`, `rula`, `quality`, `elapsed_ms`, `video_stored`
- RULA: `final_score`, `risk_level`, `valid`, `issues`, component score와 Table A/B/C 결과
- 품질: `status`, `pose_valid`, `hand_valid`, `issues`
- 공동 보정 정보: optional `pose_refiner`
- 기존 field는 삭제하거나 이름을 변경하지 않는다. 신규 temporal/wrist 결과는 optional field로만 추가한다.

## 9. Baseline 결론

현재 방식은 confidence와 전신 화면 포함 여부로 저품질 frame을 일부 제거한 뒤 신뢰 가능한 후보 중 최고 RULA를 선택한다. 그러나 confidence가 높게 나온 순간 좌표 spike나 비정상 관절 길이는 제거하지 못한다. 다음 개선은 RULA 공식과 최고위험 선택 원칙은 유지하면서, 최고값 선택 전에 robust outlier 검증을 추가하는 것이다.
