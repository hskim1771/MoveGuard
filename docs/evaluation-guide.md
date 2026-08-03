# 평가 수치와 발표 기준

## MoveGuard 측정 결과

### 새 자세 분류 모델

- 목 Accuracy: 55.66%
- 몸통 Accuracy: 88.30%
- 위팔 Accuracy: 70.57%
- 아래팔 Accuracy: 87.91%
- 다리 Accuracy: 56.16%
- 단순 평균: 약 71.7%

단순 평균은 부위별 난이도와 표본 수를 무시하므로 참고 요약으로만 사용한다.

### MediaPipe 관절 평가

- 독립 test 전체: 16,210장
- 사람·관절 검출 성공률: 90.96%
- 보이는 관절 PCK@0.10: 89.96%
- RULA 핵심 관절 PCK@0.10: 91.19%
- RULA 핵심 관절 PCK@0.20: 96.29%

PCK는 AI-Hub 작업자 bbox 대각선 대비 좌표 오차다. AI-Hub 정답 bbox로 crop한 조건이다.

### 자세 전용 RULA 일치도

- 정확 일치: 35.26%
- ±1점 이내: 79.62%
- 평균 절대 오차: 0.90점
- 조치 단계 일치: 65.84%

AI-Hub에는 전문가 최종 RULA 총점, 하중, 반복·근육 사용 정답이 없다. 따라서 위 수치는
AI-Hub 인간검수 자세 라벨로 파생한 점수와의 일치도이며 전문가 RULA 정확도가 아니다.

## 발표 문장

> AI-Hub 데이터로 학습한 MoveGuard 후단 자세 분류 모델은 부위별 평균 약 72%를
> 기록했습니다. 관절 추출기로 사용하는 MediaPipe는 독립 테스트에서 작업자·관절
> 검출 성공률 약 91%, RULA 핵심 관절 PCK@0.10 약 91%를 기록했습니다.

## 금지할 비교

- AI-Hub 공식 HRNet `AP@0.5 88%`와 MoveGuard `PCK@0.10 91.19%`를 직접 비교하지 않는다.
- AI-Hub EfficientNet 분류 정확도와 MoveGuard RULA 일치도를 직접 비교하지 않는다.
- 검출 성공률 90.96%를 관절 좌표 정확도라고 부르지 않는다.
- ±1점 79.62%를 전문가 허용 정확도라고 부르지 않는다.

## 상세 결과

- 전체 보고서: `reports/aihub_full_120k_training_report.md`
- 관절별 결과: `reports/aihub_keypoint_accuracy_report.json`
- 기존 규칙 비교: `training/evaluate_rula_agreement.py`
