import type { Metadata } from "next";
import PreventionEducation from "./PreventionEducation";

export const metadata: Metadata = {
  title: "예방교육관리 | MoveGuard",
  description: "근골격계질환 예방교육을 시청하고 교육 이수 현황을 관리합니다.",
};

export default function PreventionEducationPage() {
  return <PreventionEducation />;
}
