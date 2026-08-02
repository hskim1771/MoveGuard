# MoveGuard Figma UI MVP

Figma의 `MoveGuard UI Design`을 참고해 기존 MoveGuard MVP 기능을 새 UI로 구성한 별도 브랜치입니다.

- 기준 브랜치: `main`
- 작업 브랜치: `agent/figma-ui-mvp`
- 배포 화면: <https://moveguard-figma-mvp.ohsin12.chatgpt.site/>
- 기존 정적 MVP 파일과 AI 분석 자료는 삭제하지 않고 유지합니다.

## 주요 화면

- 전체 작업장 대시보드
- 전체 작업자의 부위별 위험도 및 최근 위험 추이
- 개인별 분석 기록과 개인 부위별 위험도
- 실시간 카메라 및 영상 업로드 분석
- 분석 세션 저장 API

## 로컬 실행

Node.js 22.13.0 이상이 필요합니다.

```bash
npm install
npm run dev
```

기본 분석 서버 주소를 바꾸려면 `.env.example`을 참고해 `NEXT_PUBLIC_ANALYSIS_API_URL`을 설정합니다. 환경 변수를 설정하지 않으면 배포 환경에서는 기존 MoveGuard 분석 API를 사용합니다.

## 검증

```bash
npm run lint
npm test
```

이 브랜치는 추후 `main`에 병합하기 전 UI 구조와 기존 MVP 기능의 통합 범위를 검토하기 위한 작업본입니다.
