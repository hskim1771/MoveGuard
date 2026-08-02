import type { Metadata } from "next";
import Dashboard from "./Dashboard";

export const metadata: Metadata = {
  title: "분석 대시보드 | MoveGuard",
  description:
    "MoveGuard 카메라 측정 기록과 위험 자세 비율, 부위별 최대 각도를 확인하는 분석 대시보드입니다.",
};

export default function DashboardPage() {
  return <Dashboard />;
}
