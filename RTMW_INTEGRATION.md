# MoveGuard UI ↔ RTMW 분석 연결

이 브랜치는 `agent/figma-ui-mvp`의 화면 구조와 디자인을 유지하면서 자세 추론
백엔드를 실제 프레임 단위로 연결한다. 분석 백엔드 기준 브랜치는
`feature/rtmw-rula-abtest`다.

## 실행

`.env.local`에 분석 서버 주소를 지정한다.

```text
NEXT_PUBLIC_ANALYSIS_API_URL=http://127.0.0.1:8000
```

이 값은 `/video-analysis`가 여는 MVP iframe에 한 번만 전달된다. 정적 HTML 안의
개발·배포 기본값도 유지하되, 환경변수가 있으면 그 값을 우선한다.

## 실제 연결 범위

- 기존 5단계 MVP 흐름, 촬영·업로드 UI, RULA 결과표와 법정서식 디자인 유지
- 개발 검증용 모델 선택: `Baseline · MediaPipe Pose` / `RTMW-l 384×288`
- 최고 위험 대표 프레임 JPEG 1장을
  `POST /api/analysis/posture?model=baseline|rtmw`로 전송
- 서버 응답의 모델명, 공통 관절 검출 수, pose confidence, 손목 각도 계산 가능
  수, RULA, 위험 수준, 처리시간을 화면에 표시
- 하중·정적 힘·반복 작업·다리 지지 입력을 같은 API에 다시 보내 최종 RULA 계산
- 사람 미검출, 낮은 신뢰도, 측정 불가, 시간 초과, 서버 오류를 서로 다른 상태로
  표시하고 가짜 성공값을 만들지 않음
- RTMW 선택 상태에서 결과가 없으면 Baseline RULA로 자동 대체하지 않음
- 원본 영상은 서버에 보내지 않고 대표 프레임만 메모리 분석하며 저장하지 않음

브라우저 MediaPipe는 실시간 미리보기와 최고 위험 프레임 선택을 위해 그대로
유지한다. 현재 CPU RTMW는 실시간 매 프레임 처리용이 아니라 선택된 대표
프레임을 비동기로 정밀 재분석하는 용도다.

## 결과 해석

공통 관절 검출률과 RULA 계산 가능률은 정확도가 아니다. 전문가 Ground Truth가
없는 경우 화면에도 정확도 수치를 표시하지 않는다. 관절 위치·각도·RULA 정확도는
전문가 라벨을 추가한 별도 A/B 리포트로 판단해야 한다.
