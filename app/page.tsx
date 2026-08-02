import type { Metadata } from "next";
import VideoAnalysisPage from "./video-analysis/page";

export const metadata: Metadata = {
  title: "통합 RULA 측정 | MoveGuard",
  description:
    "실시간 카메라 또는 촬영 영상으로 자세를 분석하고 하나의 RULA 결과표와 법정서식을 만드는 MoveGuard 통합 측정 페이지입니다.",
};

export default function Home() {
  return <VideoAnalysisPage />;
}
