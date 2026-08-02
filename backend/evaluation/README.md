# MoveGuard Baseline vs Hybrid evaluation

이 디렉터리는 같은 영상·같은 프레임·같은 각도 계산기·같은 RULA 엔진으로
관절 추론 구성만 바꿔 비교한다. Ground Truth가 없으면 정확도 수치를 만들지
않으며, 모델 confidence나 공식 AP를 MoveGuard 정확도로 바꾸어 쓰지 않는다.

## 현재 비교 가능한 구성

| ID | 구성 | 현재 상태 |
|---|---|---|
| A / Baseline V1 | MediaPipe Pose → 공통 관절각 → 공통 RULA | 실행 가능 |
| B / Hybrid-lite V2 | MediaPipe Pose + MediaPipe Hand → 공통 관절각 → 공통 RULA | 실행 가능 |
| C | RTMW-x + RTMPose Hand5 → fusion → 공통 관절각 → 공통 RULA | 모델·런타임 미설치 |
| D | RTMW-x + ViTPose-H + RTMPose Hand5 → fusion → 공통 관절각 → 공통 RULA | 모델·런타임 미설치 |

현재 B는 최종 전문 모델 Hybrid가 아니라, 손 모델을 추가했을 때의 기여도를
확인하는 ablation 단계다. C/D가 준비되기 전에는 B의 결과를 RTMW-x 또는
ViTPose 성능이라고 부르면 안 된다.

OpenMMLab 공식 MMPose는 whole-body 및 hand inference 진입점을 제공한다.
실제 adapter를 설치할 때는 아래 공식 문서를 기준으로 config/checkpoint를
고정하고, 버전과 해시를 실험 manifest에 기록한다.

- https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md
- https://github.com/open-mmlab/mmpose

공식 benchmark는 후보 모델 선정에만 사용하고 MoveGuard 정확도 결론에는
사용하지 않는다.

## 실행

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.\.venv\Scripts\python.exe -m evaluation.compare `
  --video ..\test_clip.mp4 `
  --output evaluation\results\test_clip_current `
  --sample-every 1
```

전문가 라벨을 입력한 뒤에는 같은 명령에 다음 옵션만 추가한다.

```powershell
  --ground-truth evaluation\results\test_clip_current\ground_truth_annotation.csv
```

프레임 수를 제한하려면 `--max-frames`, 일정 간격으로 샘플링하려면
`--sample-every`를 사용한다. Baseline과 Hybrid는 항상 동일한 샘플을 쓰며,
프레임별 실행 순서를 번갈아 고정 순서 편향을 줄인다. 첫 warm-up 결과는
속도 통계에서 제외한다.

## 출력 파일

- `report.json`: 전체 지표, 상태, 사유, 실험 환경
- `per_frame.jsonl`: 동일 프레임의 Baseline/Hybrid/GT 결과
- `overall_metrics.csv`: 심사용 전체 성능표
- `joint_detection.csv`: 관절별 검출 성공률
- `joint_angle_accuracy.csv`: 각도 MAE 비교
- `performance.csv`: 지연시간, FPS, CPU, 메모리, GPU 상태
- `condition_analysis.csv`: 가림·촬영각·거리·조명 조건별 결과
- `ablation.csv`: A/B/C/D 실행 가능 여부와 결과 연결
- `ground_truth_annotation.csv` / `.json`: 입력 프레임에 맞춘 라벨 템플릿

모든 미측정 값은 빈 칸과 `not_measurable` 사유로 남는다.

## 지표 정의

### 관절 검출 성공률

각 샘플 프레임에서 confidence 0.5 이상인 공통 관절 수를 기대 관절 수로
나눈 값이다. 사람을 찾지 못한 프레임도 누락으로 포함한다.

- `wrist`: 손목 좌표 자체의 검출률
- `wrist_angle_ready`: 팔꿈치-손목-손 축이 모두 있어 손목 각도를 계산할 수
  있는 비율

두 값을 분리해야 pose wrist confidence를 실제 손목 굽힘 정확도로 오해하지
않는다.

### 위치 정확도

GT 좌표가 있을 때만 Pixel Error, torso/shoulder 기준 Normalized Joint Error,
PCK@0.2를 계산한다. `coordinate_space=world_m`이고 GT와 예측 모두 world 좌표가
있을 때만 MPJPE(mm)를 계산한다.

### 각도 정확도

전문가 각도가 있을 때만 관절별 MAE, RMSE, Median Absolute Error를 degree로
계산한다. 양쪽 관절은 같은 부위 안에서 함께 집계하되 원본 값은 per-frame에
남긴다.

### 자세 분류

GT 라벨과 실제 classifier 출력이 모두 있을 때만 Accuracy, macro Precision,
Recall, F1 및 Confusion Matrix를 계산한다. 현재 MoveGuard에는 독립 자세
classifier 출력이 없으므로 임의 규칙 라벨을 만들지 않는다.

### RULA

두 구성 모두 다음 함수를 그대로 공유한다.

- `analysis_v2.angle_calculator.calculate_joint_angles`
- `analysis_v2.rula_engine.calculate_rula`

전문가 RULA가 있을 때 Final Score exact, ±1, MAE, risk-level 분류 지표와
부위별 점수 일치율을 계산한다. 손이 검출되지 않으면 중립 손목 점수를
가정하지 않고 `not_measurable`로 남긴다.

## Ground Truth 라벨링 방법

1. 인접 프레임을 무작위로 섞지 말고 영상·작업자 단위로 train/validation/test를
   분리한다.
2. 관절 좌표는 원본 프레임 기준 `x/width`, `y/height`의 0~1 값으로 기록한다.
3. 가려진 관절은 `visible=0`으로 두고 좌표를 억지로 추정하지 않는다.
4. 각도는 동일한 해부학적 기준을 문서화하고 degree 숫자만 입력한다.
5. 자세 라벨은 허용 목록을 먼저 확정하고 다중 라벨은 `;`로 구분한다.
6. RULA는 교육받은 평가자가 원본 프레임과 작업 조건을 보고 입력한다.
7. 전체의 일부는 두 명이 독립 평가하고 불일치를 합의해 최종 GT를 만든다.
8. `occlusion`, `camera_angle`, `distance`, `lighting`은 빈 칸 없이 통제된
   범주로 기록한다.

### 권장 데이터 규모

이는 정확도 보장이 아니라 실험 설계를 위한 최소 권장안이다.

- 파일 형식 검증용: 30~50 frame
- pilot 비교: 서로 다른 영상 10개 이상, 작업자 5명 이상, 300~500 labeled frame
- 본 비교: 영상 20개 이상, 작업자·공정·촬영 조건을 나누어 1,000~2,000
  labeled frame
- 손목·가림·측면·등진 자세는 별도 strata로 충분히 포함

연속 프레임은 서로 매우 유사하므로 frame 수만 늘리지 말고 작업자·동작·촬영
조건의 다양성을 우선한다.

## 전문 모델 adapter와 fusion

`FusionPoseEstimator`는 여러 `PoseEstimator`가 반환한 정규화 33-slot 관절을
관절별 우선순위로 합친다. 선택된 source는 canonical joint마다 기록된다.

- 머리/목/몸통/발목: RTMW-x 우선
- 어깨/팔꿈치/손목/엉덩이/무릎: ViTPose-H 우선, RTMW-x fallback
- 손 세부 관절: RTMPose Hand5 우선, MediaPipe Hand fallback
- 어떤 모델도 confidence 기준을 통과하지 못하면 좌표를 만들지 않음

새 adapter는 native keypoint index를 MoveGuard 33-slot으로 변환한 뒤
`PoseEstimator` 또는 `HandEstimator`를 구현해야 한다. 이렇게 하면 아래
요소는 바뀌지 않는다.

- 영상 decoding과 RGB 전처리
- frame sampling
- 관절각 공식
- RULA 표와 threshold
- 결과 metric 계산

## 현재 test_clip 실측 결과

Ground Truth가 없는 저장소의 `test_clip.mp4` 40 frame 전체를 측정한 결과:

| Metric | Baseline | Hybrid-lite | 판단 |
|---|---:|---:|---|
| 공통 관절 검출률 | 80.71% | 80.71% | 차이 0.00%p |
| 손목 좌표 검출률 | 57.50% | 57.50% | 차이 0.00%p |
| 손목 각도 계산 가능률 | 0.00% | 0.00% | 손 모델 기여 확인 안 됨 |
| 평균 latency | 41.79 ms | 65.37 ms | Hybrid-lite가 23.58 ms 느림 |
| 처리속도 | 23.93 FPS | 15.30 FPS | Hybrid-lite가 8.63 FPS 낮음 |
| 증분 모델 load RSS | 122.27 MB | 127.93 MB | 동일 프로세스 순차 warm-up 기준 |

이 한 영상에서는 Hybrid-lite를 채택할 근거가 없다. 다만 GT가 없으므로
“정확도가 같다/낮다”는 결론도 낼 수 없다. 최종 모델 추천은 C/D까지 같은
조건으로 실행하고 전문가 GT 결과가 쌓인 뒤에만 가능하다.

