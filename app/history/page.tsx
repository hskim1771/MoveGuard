import type { Metadata } from "next";
import History from "./History";

export const metadata: Metadata = { title: "작업자별 분석 기록 | MoveGuard" };
export default function HistoryPage() { return <History />; }
