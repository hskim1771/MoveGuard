# MoveGuard RULA / Wrist Pipeline Validation Report

작성일: 2026-08-05  
브랜치: `feature/figma-mediapipe-joint-domain-integration`  
작업 시작 커밋: `605a5fa468d7c575ff82a780a3e2106ab28a8088`  
평가 데이터: 기존 AI-Hub 산업현장 데이터의 고정 test split 16,210장(21 groups)  
원본 데이터, split, checkpoint, 기존 결과는 수정하거나 덮어쓰지 않았다. GitHub push와 main merge도 수행하지 않았다.

## 결론 요약

- 현재 제품 기본값은 **MediaPipe raw 유지**가 타당하다.
- MediaPipe 검출률은 독립 테스트에서 **90.96%**(14,744/16,210)였다.
- 손목 위치 PCK@0.10은 안정화 및 공동학습 모델에서 올라갔지만, 손목 각도와 RULA 손목 점수는 좋아지지 않았다.
- 독립 테스트 최고 손목 각도 결과는 MediaPipe raw: **MAE 16.28°, ±10° 54.22%, RULA 손목 점수 일치율 60.46%**였다.
- 현재 stabilizer는 테스트에서 MediaPipe PCK를 87.78%→89.28%로 높였으나 각도 MAE를 16.28°→18.58°로 악화시켰다. 따라서 production RULA에 기본 적용하지 않는다.
- 공동학습 모델은 PCK는 더 높았지만 손목 각도 MAE가 33.40°로 MediaPipe보다 나빴다. production RULA 입력으로 자동 교체하지 않는다.
- 전문가가 매긴 최종 RULA 정답은 이 데이터에 없으므로 **Final RULA Accuracy는 평가 불가**다.
- 손목 기능은 optional API field와 기본값 OFF인 `ENABLE_WRIST_ANGLE` 플래그로만 연결했다. 기존 UI와 API 필드는 유지했다.

## A. 기존 pipeline

상세 내용은 `CURRENT_PIPELINE_REPORT.md`에 기록했다.

영상 → 최대 약 30개 균등 간격 frame sampling → 브라우저 MediaPipe → 33 landmark → angle → 기존 RULA Table A/B/C → 프레임 전체 점수가 가장 높은 한 프레임 선택 → 최고 위험 프레임을 API로 전송 → FastAPI MediaPipe/선택적 joint refiner/다중출력 RULA 모델 → 기존 frontend 결과 표시 순서다.

업로드 간격은 `max(video duration / 30, 0.1초)`, 실시간 검출은 약 160ms 간격이다. 기존 품질 기준은 평균 및 전신 confidence 0.55, 최소 유효 프레임 5장이다. 기존에는 smoothing, temporal filtering, RULA 전 이상치 제거가 없었다. MAX는 관절별 MAX가 아니라 **한 프레임 전체 RULA 후보 점수의 MAX**다.

## B. 발견된 문제

- 순간 landmark jump가 최고 위험 프레임으로 선택될 수 있다.
- 영상 전체에서 듬성듬성 뽑는 프레임에는 연속 프레임 보간을 그대로 쓰면 안 된다.
- 위치 PCK 개선이 작은 벡터 차이에 민감한 손목 각도 개선을 보장하지 않는다.
- 산업현장 `wrist_bending` 라벨은 degree GT가 아니며 GT keypoint 파생 각도와 충돌이 많다.
- 단일 2D 카메라에서 flexion/extension 방향과 twist를 안정적으로 분리하기 어렵다.

## C. Pose 이상치 유형

NaN/Inf, 허용 범위 밖 좌표, 신체 segment 길이 급변, 앞뒤 프레임이 지지하지 않는 단일 프레임 jump, 낮은 confidence, 너무 많은 미해결 joint를 처리한다. 좌우 swap은 자동 교정하지 않으며 review 대상으로 남긴다.

## D. Stabilizer 구현 방법

`backend/temporal_stabilizer.py`에 모델과 독립적인 모듈을 추가했다. 이동평균은 쓰지 않는다. confidence + segment consistency + neighbor support를 함께 사용한다. 양쪽 이웃이 0.5초 이내이고 신뢰 가능한 경우에만 고립된 outlier joint를 보간한다. 연속된 빠른 움직임과 sparse frame은 보존한다.

검증셋 14,267장으로 기준을 고정했다. 테스트셋을 기준 조정에 사용하지 않았다. 검증 결과에서 outlier 표시가 과다함을 확인해 production 활성화는 보류했다.

## E. RULA 적용 방식

평가 파이프라인에서는 raw keypoint와 stabilized keypoint 각각에 **같은 angle/RULA 규칙**을 적용했다. 공식과 threshold는 바꾸지 않았다. `valid frames → RULA → 전체 프레임 score MAX`를 raw와 비교했다. 제품 API는 단일 프레임 endpoint이므로 temporal stabilization을 적용했다고 거짓 표시하지 않고 `temporal_stabilization_applied: false`로 둔다.

## F. 기존 max 방식과 개선 방식 비교

MAX 자체는 위험 자세를 놓치지 않기 위해 유지할 가치가 있다. 다만 검증된 valid/outlier filter가 MAX 앞에 있어야 한다. 이번 stabilizer는 독립 테스트 그룹 21개에서 최종 파생 RULA를 바꾸지 않았지만, 각도 성능을 악화하고 너무 많은 frame에 outlier 표시를 했으므로 제품 MAX 앞에 기본 적용할 수준은 아니다.

## G. RULA 전/후 결과

### Pose / RULA

| Method | Valid Rate | Outlier 표시율 | Final RULA Change | False high-risk change |
|---|---:|---:|---:|---:|
| MediaPipe raw | 100.00% | 0.00% | baseline | baseline |
| MediaPipe stabilized | 95.00% | 50.58% | 0/21 (0.00%) | 0→0 |
| Joint raw | 100.00% | 0.00% | baseline | baseline |
| Joint stabilized | 86.88% | 58.46% | 0/21 (0.00%) | 0→0 |

여기서 outlier 표시율은 frame에 하나 이상의 의심 joint가 있었다는 뜻이며 frame 폐기율과 같지 않다. GT 파생 방식에서 high-risk group이 0개여서 실제 high-risk 보존율은 계산할 수 없다. 전문가 최종 RULA GT도 없어 final RULA accuracy는 평가 불가다.

## H. 사용 가능한 Wrist keypoint schema

산업현장 GT는 27개 `[x, y, visibility]` keypoint다. 왼쪽은 elbow 6, wrist 7, hand/index 8, pinky 9, thumb 10, 오른쪽은 elbow 12, wrist 13, hand/index 14, pinky 15, thumb 16이다. 따라서 forearm과 hand direction을 모두 만들 수 있다. 실제 label에는 `wrist_bending`, `wrist_twisted`, `wrist_twisted_end`가 있다. 전문가 최종 RULA 점수는 없다.

## I. Wrist angle 계산법

`forearm = wrist - elbow`, `hand = mean(valid hand landmarks) - wrist`의 signed 방향 차이를 구한다. 중립 자세가 약 0°가 되도록 정의한 **2D projected wrist flexion/extension angle**이다. 15° 이하는 기존 규칙의 wrist score 1, 초과는 score 2로 매핑한다. 카메라 방향을 확인할 수 없으면 굽힘/폄 방향은 `unknown`으로 둔다.

## J. GT Wrist angle 정의

GT keypoint에도 prediction과 완전히 같은 공식을 적용했다. 이는 motion capture 기반 절대 해부학 각도가 아니라 **2D keypoint 기반 reference angle**이다.

## K. MediaPipe Wrist 결과

독립 테스트 raw 결과: 왼쪽 MAE 18.15°(8,491), 오른쪽 14.10°(7,276), 전체 16.28°(15,767). 전체 Median AE 8.90°, RMSE 28.12°, ±5° 32.10%, ±10° 54.22%, ±15° 68.33%, PCK@0.10 87.78%다.

## L. Joint Pose Wrist 결과

독립 테스트 raw 결과: 왼쪽 MAE 39.03°(8,604), 오른쪽 26.84°(7,390), 전체 33.40°(15,994). 전체 Median AE 19.34°, RMSE 49.76°, ±5° 19.44%, ±10° 32.96%, ±15° 42.74%, PCK@0.10 88.56%다.

## M. Stabilization 전/후 Wrist 결과

| Model | Wrist PCK@0.10 | Angle MAE | ±10° | Valid Rate | RULA Wrist Accuracy |
|---|---:|---:|---:|---:|---:|
| MediaPipe raw | 87.78% | 16.28° | 54.22% | 100.00% | 60.46% |
| MediaPipe stabilized | 89.28% | 18.58° | 53.11% | 95.00% | 60.27% |
| Joint raw | 88.56% | 33.40° | 32.96% | 100.00% | 51.05% |
| Joint stabilized | 90.64% | 35.07° | 31.65% | 86.88% | 50.62% |

## N. RULA Wrist Score 결과

GT keypoint 파생 wrist score와의 일치율은 MediaPipe raw 60.46%, MediaPipe stabilized 60.27%, Joint raw 51.05%, Joint stabilized 50.62%다. 이는 **손목 구간/점수 일치율**이며 전체 최종 RULA 정확도가 아니다.

## O. dataset wrist label과의 비교

GT keypoint angle의 bending/neutral과 `wrist_bending` label 비교는 14,740건에서 accuracy 46.98%, precision 69.97%, recall 6.85%, F1 12.48%였다. label이 degree GT가 아니며 정의·시점·경계 차이가 크므로 어느 한쪽을 자동 정답으로 덮어쓰지 않는다.

## P. 불일치 라벨 수

| Category | Count |
|---|---:|
| Total comparable GT/label | 14,740 |
| Agreement | 6,925 |
| GT angle vs dataset label conflict | 7,815 |
| Joint stabilized vs GT angle conflict | 5,400 / 12,634 |
| Manual review required | 11,457 |

## Q. Manual Review 대상 수

11,457건을 자동 overwrite하지 않고 review queue로 생성했다. HIGH 8,071, MEDIUM 2,116, LOW 1,270이다. 대표 200건용 `expert_manual_review_template.csv`도 생성했다. 전문가가 작성하기 전에는 expert GT로 간주하지 않는다.

## R. frontend/API 영향

- 기존 endpoint와 response field를 삭제하거나 이름 변경하지 않았다.
- `ENABLE_WRIST_ANGLE=false`가 기본값이다.
- 켠 경우에만 `ergonomics.wrist.left/right`가 optional로 추가된다.
- 응답에는 angle, signed_angle, posture, rula_score, confidence, valid, reason이 포함된다.
- health에 flag 상태만 additive로 표시한다.
- 기존 frontend 파일은 수정하지 않았다.
- production build 성공, 렌더링 테스트 4/4 통과.
- 실제 브라우저에서 `/dashboard`, `/history`, `/`를 열어 필수 문구와 렌더링을 확인했고 console error/warning은 없었다.

## S. 한계

- 단일 2D 카메라라서 flexion/extension 방향은 camera orientation 검증 없이는 `unknown`이다.
- twist, pronation, supination은 이번 범위가 아니다.
- source JSON에 FPS가 없어 filename frame index / 30fps를 시간 proxy로 사용했다.
- test/validation은 이미지 sequence이며 실제 업로드 영상의 연속 프레임과 동일하지 않다.
- 전문가 최종 RULA GT가 없어 Final RULA Accuracy는 평가 불가다.
- 현재 stabilizer threshold는 PCK와 angle/RULA 사이의 trade-off가 좋지 않다.
- 공동학습 refiner crop/domain 조건이 제품 실시간 카메라와 완전히 같다고 보장할 수 없다.

## T. MVP 적용 여부

1. MediaPipe는 production 기본값으로 유지한다.
2. Joint refiner의 RULA 적용은 기본 OFF를 유지한다.
3. Wrist optional API는 실험용으로 보존하되 UI 기본 노출은 보류한다.
4. Temporal stabilizer는 모듈과 평가 도구만 보존하고 production RULA에는 기본 적용하지 않는다.
5. 다음 개선은 validation에서 outlier definition을 줄이고, 손목 각도 자체를 목적함수/검증 기준으로 삼아야 한다.
6. 전문가 100~300장 수기검증 후 손목 경계값 및 camera-view별 방향 판정을 재평가한다.

## 제품 적용 판단 14문항

1. production RULA에 Pose 안정화가 적용됐는가? **아니오. 평가 경로에는 적용했지만 성능 미달로 기본 적용하지 않았다.**
2. false spike가 감소했는가? **test 21 groups에서는 raw/stabilized 모두 false high-risk 0이라 입증 불가. validation MediaPipe에서 1→0이었으나 전체 근거로 부족하다.**
3. 실제 위험 자세 누락은 없는가? **GT high-risk group이 0이라 평가 불가.**
4. MAX 방식 유지 가치가 있는가? **있다. 실제 위험을 놓치지 않기 위해 유지한다.**
5. MAX 전 filter가 개선됐는가? **현재 filter는 PCK만 개선했고 angle/RULA는 악화해 제품 개선으로 판단하지 않는다.**
6. Joint가 MediaPipe보다 산업현장에서 안정적인가? **손목 위치 PCK는 높지만 손목 angle/RULA는 아니다.**
7. Wrist GT keypoint가 있는가? **있다. elbow, wrist, 3 hand points가 좌우에 있다.**
8. Wrist angle MAE는? **MediaPipe raw 16.28°가 최고.**
9. Joint가 wrist angle에서도 이기는가? **아니오. 33.40°로 더 나쁘다.**
10. RULA wrist score 정확도는? **MediaPipe raw 60.46%가 최고.**
11. `wrist_bending` label 신뢰도는? **GT 파생각과 accuracy 46.98%, F1 12.48%로 직접 degree GT로 쓰기 어렵다.**
12. 사람이 확인할 sample은? **11,457건, 우선 HIGH 8,071건 중 대표 200건.**
13. Wrist를 MVP에 노출할 수준인가? **아니오. optional 실험 API까지만 적절하다.**
14. 숨겨야 하는 기능은? **temporal stabilization, wrist flexion/extension UI, joint refiner의 RULA 강제 적용.**

## 산출물 및 재실행

- 독립 테스트 요약: `experiments/wrist_rula_pipeline_validation/validation_report.json`
- 그룹 비교: `experiments/wrist_rula_pipeline_validation/group_rula_comparison.csv`
- 로컬 frame cache/metrics/review queue는 `.gitignore` 처리했다.
- 재실행: `.venv/bin/python -u training/evaluate_wrist_temporal_pipeline.py --split test --workers 8 --output experiments/wrist_rula_pipeline_validation`
- resume: 같은 output 경로로 다시 실행하면 기존 `independent_test_pose_cache.jsonl`을 재사용한다.
