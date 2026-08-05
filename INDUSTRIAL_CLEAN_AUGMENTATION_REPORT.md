# 제조업 정제 데이터 증강 모델 실험 보고서

작성일: 2026-08-05

## 목적

AI-Hub 제조업 노동자 근골격계 데이터의 기존 train split에서 학습하기 좋은 데이터만 선별하고, MediaPipe 기반 좌표 보정 모델을 산업현장 데이터만으로 학습했다. 기존 split, 원본 데이터, checkpoint는 수정하지 않았다.

## 데이터 선별

- 원본 train: 65,060장
- 품질 후보: 17,276장
- 연속 중복 제거 후 최종: 9,979장
- validation/test 포함: 0장
- 선별률: 15.34%
- 원본 wrist_bending 양성: 4,376/65,060(6.73%)
- 선별본 wrist_bending 양성: 721/9,979(7.23%)

선별 조건은 팔·손 MediaPipe confidence, GT crop 내부 포함, 관절/라벨 불일치 상한, 신체 segment 좌우 비율, 연속 프레임 중복이다. 원본 파일을 삭제하거나 이동하지 않았다.

## 증강

9,979장마다 다음 세 가지를 시도했다.

1. 원본 정제 이미지
2. 밝기 80~120% + 대비 85~115%
3. JPEG 품질 55~90 + 약한 Gaussian blur

요청 29,937건 중 MediaPipe가 검출한 29,655건을 관절 캐시로 저장했다. 이미지 복사본은 저장하지 않았고 캐시는 약 21MB다. 오류 파일은 0건이다.

## 학습

- 모델: MediaPipe 뒤 132차원 입력 → residual 24×2 Corrector
- 데이터: 제조업 clean/augmentation train만 사용
- 좌표 loss 가중치: wrist 4, hand center 3, elbow 2, 기타 1
- 추가 loss: elbow→wrist와 wrist→hand center의 signed 2D angle
- 모델 선택: 기존 industrial validation의 전체 PCK와 wrist PCK만 사용
- 독립 test: 모델 선택 이후 한 번 평가
- checkpoint: `checkpoints/industrial_clean_aug_v1/best.pt`

## 독립 테스트 결과

| 지표 | MediaPipe 원본 | 기존 공동학습 | 정제·증강 신규 모델 |
|---|---:|---:|---:|
| 공통 관절 PCK@0.10 | 86.87% | 88.48% | **88.77%** |
| 왼쪽 손목 PCK@0.10 | 82.70% | **83.90%** | 83.52% |
| 오른쪽 손목 PCK@0.10 | 81.80% | **84.69%** | 83.06% |
| 손목 각도 MAE | 16.28° | 33.40° | **14.63°** |
| 손목 각도 ±10° | 54.22% | 32.96% | **59.95%** |
| 손목 점수 일치율 | 60.46% | 51.05% | **64.35%** |

신규 모델은 개별 손목 위치 PCK만 보면 기존 공동학습보다 조금 낮지만, 이번 핵심 목표인 손목 각도와 손목 점수는 크게 개선했다.

## 촬영 변화 반복 신뢰도

독립 테스트 21개 그룹의 210장에 밝기, JPEG 압축, 가장자리 5% crop을 적용했다.

| 지표 | MediaPipe 원본 | 정제·증강 신규 모델 |
|---|---:|---:|
| 검출 유지율 | 95.41% | 95.41% |
| 손목 각도 반복 MAE | 11.38° | **6.54°** |
| 손목 점수 반복 일치율 | 75.57% | **85.88%** |
| 프레임 RULA 반복 일치율 | 92.82% | **95.22%** |

손목 점수 반복 일치율 80% 목표를 통과했다. 가장 어려운 5% crop에서도 80.43%로 기준을 넘었다.

## 해석과 제품 적용

- 반복 신뢰도 목표는 달성했다.
- GT 관절 파생 손목 점수 일치율은 64.35%이므로 전문가 정확도라고 표현하면 안 된다.
- 전문가 최종 RULA GT와 비교하지 않았으므로 Final RULA Accuracy는 평가 불가다.
- 현재 production 모델을 자동 교체하지 않는다.
- 다음 단계는 전문가 검토 표본에서 손목 점수와 최종 RULA를 확인한 뒤 feature flag로 비교 적용하는 것이다.

## 재실행

```bash
.venv/bin/python training/select_clean_industrial_training.py
.venv/bin/python -u training/cache_clean_industrial_augmentation.py
.venv/bin/python -u training/train_industrial_clean_aug_refiner.py
.venv/bin/python training/evaluate_clean_aug_independent.py
.venv/bin/python training/evaluate_pose_perturbation_reliability.py \
  --checkpoint checkpoints/industrial_clean_aug_v1/best.pt \
  --output experiments/pose_perturbation_reliability/industrial_clean_aug_v1_report.json
```
