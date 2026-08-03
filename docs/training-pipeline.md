# AI-Hub 학습 파이프라인

## 데이터 사실

- 요청 명칭은 12만 장이지만 현재 실제 파일은 이미지 108,000장과 JSON 108,000개다.
- pairing 누락 0, 중복 0, invalid JSON 0이다.
- 원본 데이터는 로컬 Downloads에 있으며 GitHub에 올리지 않는다.
- `artifacts/`는 생성물 경로이며 Git에서 제외된다.

## 누수 방지 분할

- seed: `20260803`
- 같은 작업자 촬영 시퀀스와 연속 프레임을 하나의 그룹으로 묶는다.
- 그룹 기준 train 70%, validation 15%, test 15%다.
- 전체 분할: 77,523 / 14,267 / 16,210장
- MediaPipe 성공 후: 66,250 / 13,782 / 14,744장
- test는 모델 선택과 하이퍼파라미터 조정에 사용하지 않는다.

## 실행 순서

```bash
.venv/bin/python training/aihub_audit.py \
  --data-root "<AI-Hub 원본 절대경로>" \
  --output artifacts/aihub/audit --seed 20260803

.venv/bin/python training/preprocess_mediapipe.py \
  --manifest artifacts/aihub/audit/split_manifest.jsonl \
  --output artifacts/aihub/full

.venv/bin/python training/cache_aihub_crops.py \
  --manifest artifacts/aihub/audit/split_manifest.jsonl \
  --features artifacts/aihub/full/features.jsonl \
  --output artifacts/aihub/image_cache_128 --image-size 128 --workers 4

.venv/bin/python training/train_multimodal_rula.py \
  --manifest artifacts/aihub/audit/split_manifest.jsonl \
  --features artifacts/aihub/full/features.jsonl \
  --image-cache artifacts/aihub/image_cache_128 \
  --output artifacts/aihub/multimodal_full \
  --epochs 8 --batch-size 64 --image-size 128 --seed 20260803
```

## 재개

- MediaPipe 전처리는 `features.jsonl`의 완료 ID를 읽고 남은 항목만 처리한다.
- crop 캐시는 checkpoint와 ID index를 사용한다.
- 원본 ZIP은 읽기 전용으로 사용한다.
- 모델 재학습 시 test 결과를 보고 학습 설정을 변경하지 않는다.

## 산출물

- 전체 전처리 특징: `artifacts/aihub/full/features.jsonl`
- crop 캐시: `artifacts/aihub/image_cache_128/`
- 학습 보고서: `artifacts/aihub/multimodal_full/training_report.json`
- 배포 모델: `backend/models/multimodal_rula_v1.pt`
