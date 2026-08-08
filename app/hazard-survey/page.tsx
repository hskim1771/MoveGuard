import type { Metadata } from "next";
import HazardSurveySchedule from "./HazardSurveySchedule";

export const metadata: Metadata = {
  title: "유해요인조사 일정관리 | MoveGuard",
  description: "유해요인조사 일정을 등록하고 다음 조사일까지 남은 기간을 관리합니다.",
};

export default function HazardSurveyPage() {
  return <HazardSurveySchedule />;
}
