# MoveGuard Large Scale Wrist Refinement Experiment

## 핵심 결론

- Human Pose 기반 보정 모델 중 최고: **refiner_50k**
- MoveGuard 최종 추천 파이프라인: **mediapipe** (산업현장 PCK@0.10 우선 선택)
- Human Pose 최고: **Refiner 20k 94.31%**
- 산업현장 MediaPipe: **82.25%**
- 산업현장 추천 파이프라인: **82.25%** (+0.00%p)
- 데이터 누수 검사: **PASS** (action+actor 그룹 및 이미지 ID 교집합 0)

## Data

기존 5k 실험의 실제 학습분 3,828장 ⊂ 20,000장 ⊂ 50,000장이다. 검증 545장과 테스트 627장은 기존 목록 그대로 고정했다. seed는 20260804이다.

## Human Pose Wrist (PCK@0.10)

| 모델 | Left | Right | Mean |
|---|---:|---:|---:|
| MediaPipe | 80.33% | 77.24% | 78.78% |
| Refiner 5k | 93.33% | 93.17% | 93.25% |
| Refiner 20k | 94.31% | 94.31% | 94.31% |
| Refiner 50k | 94.80% | 93.66% | 94.23% |

## Industrial Transfer Wrist (PCK@0.10)

Human Pose에서 학습한 모델을 산업현장 테스트에서 재학습하지 않고 적용했다. Human Pose 수치를 산업현장 수치로 대신하지 않았다.

- mediapipe: 82.25%
- refiner_20k: 79.95%
- refiner_50k: 80.61%
- refiner_5k: 79.28%

## 작업자세 분류

기존에 인용된 71.7%는 전체 자세 정확도가 아니라 validation의 `neck_bending` binary F1(0.717103)이었다. 독립 test 14,744장에서 자세 head별 결과를 `posture_classification_metrics.csv`에 기록했다. Wrist Refiner는 현재 이 분류기 뒤에 있으므로 분류 결과 자체는 바뀌지 않는다.

## 관절 추론 지표 정의

PCK@0.10은 예측 관절과 정답 관절의 픽셀 거리를 사람 bbox 대각선으로 나눈 값이 0.10 이하인 비율이다. 기존 91.2%는 전체 관절 정확도로 재현되지 않았으며, 보고서에서 발견된 91.2846%는 `left_pinky PCK@0.20` 한 항목이었다. 산업 데이터의 재현 가능한 전체 값은 all-annotated PCK@0.10 85.2845%, visible-only PCK@0.10 89.9564%이다.

## 객체 탐지

저장소에 객체 탐지 checkpoint, 객체 GT, class schema, 평가 코드가 없어 기존 약 91%의 출처를 재현할 수 없다. 따라서 신규 mAP를 만들지 않고 **평가 불가**로 기록했다.

## RULA 관련 2D 자세

Shoulder/Elbow/Wrist/Neck proxy/Trunk의 MAE, median, p90 및 RULA-like 각도 구간 일치율은 `rula_angle_metrics.csv`에 있다. 2D 좌표만으로 wrist twist, pronation/supination을 해결했다고 주장하지 않는다. 기존 약 29cm 3D 결과는 본 2D 실험과 분리한다.

## 재현 및 Resume

```bash
.venv/bin/python training/run_wrist_refiner_large_scale.py
.venv/bin/python training/evaluate_wrist_refiner_industrial.py
.venv/bin/python training/export_posture_classification_metrics.py
.venv/bin/python training/finalize_wrist_refiner_experiment.py
```

MediaPipe 캐시는 250장 단위 NPZ로 저장되어 첫 명령을 다시 실행하면 완료 지점부터 이어진다. 기존 5k 및 3D checkpoint는 덮어쓰지 않았다.

## 한계와 결정

최종 제품 적용 가능 여부는 산업현장 전이 결과와 RULA 각도 지표를 우선해 판단한다. 산업현장 개선이 없으면 Human Pose 점수가 높더라도 큰 모델을 채택하지 않는다. 모델이 Human Pose의 두 actor와 제한된 촬영 환경에 과적합됐을 가능성, 2D 손목 회전의 근본적 한계, 실제 전문가 최종 RULA 정답 검증이 남아 있다.

- Best Wrist Refiner: **50k** (산업 전이 모델끼리만 비교할 때 80.61%)
- Recommended MoveGuard model: **MediaPipe-only** (82.25%)
- Industrial generalization: **Poor** (모든 refiner가 baseline보다 낮음)
- Ready for MoveGuard MVP: **YES, MediaPipe-only / wrist refiner 적용은 NO**
- Need 120k + Human Pose joint training: **YES, refiner를 제품에 쓰려면 산업 도메인 적응 실험 필요**
- Refiner checkpoint size: 약 **444 KiB**
- CPU batch-1 refiner inference: 약 **0.038 ms** (MediaPipe 시간 제외)
