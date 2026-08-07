import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "통합 RULA 측정 | MoveGuard",
  description:
    "실시간 카메라 또는 촬영 영상으로 위험 자세를 찾고 동일한 RULA 평가표와 조사 서식을 만드는 MoveGuard 통합 분석 도구입니다.",
};

export default function VideoAnalysisPage() {
  const analysisApiUrl = process.env.NEXT_PUBLIC_ANALYSIS_API_URL?.replace(/\/$/, "");
  const frameSrc = analysisApiUrl
    ? `/moveguard-mvp.html?embedded=1&api=${encodeURIComponent(analysisApiUrl)}`
    : "/moveguard-mvp.html?embedded=1";

  return (
    <main className="video-analysis-page">
      <header className="topbar video-analysis-topbar">
        <a className="brand-lockup brand-link" href="/">
          <div className="brand-mark" aria-hidden="true" />
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
      </header>

      <iframe
        className="video-analysis-frame"
        src={frameSrc}
        title="MoveGuard 통합 RULA 측정"
        allow="camera"
      />
    </main>
  );
}
