import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "통합 RULA 측정 | MoveGuard",
  description:
    "실시간 카메라 또는 촬영 영상으로 위험 자세를 찾고 동일한 RULA 평가표와 조사 서식을 만드는 MoveGuard 통합 분석 도구입니다.",
};

export default function VideoAnalysisPage() {
  return (
    <main className="video-analysis-page">
      <header className="topbar video-analysis-topbar">
        <a className="brand-lockup brand-link" href="/">
          <div className="brand-mark" aria-hidden="true">
            M
          </div>
          <div>
            <p className="brand-name">MoveGuard</p>
            <p className="brand-caption">통합 RULA 측정</p>
          </div>
        </a>
        <nav className="product-nav" aria-label="MoveGuard 주요 메뉴">
          <a className="active" href="/" aria-current="page">
            통합 측정
          </a>
          <a href="/dashboard">분석 대시보드</a>
        </nav>
        <div className="privacy-chip">
          <span className="privacy-dot" aria-hidden="true" />
          원본 영상 저장 없음 · 위험 프레임만 분석
        </div>
      </header>

      <iframe
        className="video-analysis-frame"
        src="/moveguard-mvp.html?embedded=1"
        title="MoveGuard 통합 RULA 측정"
        allow="camera"
      />
    </main>
  );
}
