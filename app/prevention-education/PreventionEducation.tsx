"use client";

import { useMemo, useState } from "react";

type CourseStatus = "completed" | "progress" | "required";
type Course = {
  id: number;
  title: string;
  description: string;
  duration: string;
  category: string;
  progress: number;
  status: CourseStatus;
  videoId: string;
};

const STATUS_COPY: Record<CourseStatus, string> = {
  completed: "수료완료",
  progress: "수료중",
  required: "학습필요",
};

const INITIAL_COURSES: Course[] = [
  {
    id: 1,
    title: "근골격계질환 예방 기본교육",
    description: "반복 작업과 부적절한 자세로 발생하는 근골격계질환의 원인과 예방 원칙을 익힙니다.",
    duration: "12분",
    category: "필수교육",
    progress: 68,
    status: "progress",
    videoId: "3EVT9IeAX9Y",
  },
  {
    id: 2,
    title: "올바른 작업 자세와 인체공학",
    description: "작업대, 의자, 모니터를 신체 조건에 맞게 조정하고 부담을 줄이는 방법을 알아봅니다.",
    duration: "9분",
    category: "작업환경",
    progress: 100,
    status: "completed",
    videoId: "ywPvht5eBWU",
  },
  {
    id: 3,
    title: "중량물 취급과 허리 보호",
    description: "들기, 옮기기, 내려놓기 단계별 안전수칙과 허리 부담을 줄이는 동작을 학습합니다.",
    duration: "8분",
    category: "현장실무",
    progress: 0,
    status: "required",
    videoId: "3EVT9IeAX9Y",
  },
  {
    id: 4,
    title: "작업 전 5분 스트레칭",
    description: "목, 어깨, 허리와 손목의 긴장을 완화하는 짧은 작업 전 스트레칭 과정입니다.",
    duration: "5분",
    category: "건강관리",
    progress: 0,
    status: "required",
    videoId: "ywPvht5eBWU",
  },
];

export default function PreventionEducation() {
  const courses = INITIAL_COURSES;
  const [selectedId, setSelectedId] = useState(1);
  const [filter, setFilter] = useState<"all" | CourseStatus>("all");
  const selected = courses.find((course) => course.id === selectedId) ?? courses[0];
  const visibleCourses = filter === "all" ? courses : courses.filter((course) => course.status === filter);
  const completedCount = courses.filter((course) => course.status === "completed").length;
  const averageProgress = Math.round(courses.reduce((sum, course) => sum + course.progress, 0) / courses.length);

  const statusCounts = useMemo(
    () => ({
      completed: courses.filter((course) => course.status === "completed").length,
      progress: courses.filter((course) => course.status === "progress").length,
      required: courses.filter((course) => course.status === "required").length,
    }),
    [courses],
  );

  return (
    <main className="dashboard-page education-page">
      <header className="topbar dashboard-topbar">
        <a className="brand-lockup brand-link" href="/dashboard">
          <div className="brand-mark" aria-hidden="true" />
          <div><p className="brand-name">MoveGuard</p><p className="brand-caption">산업안전 관리</p></div>
        </a>
        <nav className="product-nav" aria-label="MoveGuard 주요 메뉴">
          <a href="/dashboard">대시보드</a>
          <a href="/">실시간 분석</a>
          <a href="/history">작업자 관리</a>
          <a href="/hazard-survey">일정관리</a>
          <a className="active" href="/prevention-education" aria-current="page">예방교육관리</a>
          <a href="/history">분석 기록</a>
        </nav>
      </header>

      <section className="dashboard-content education-content">
        <div className="dashboard-hero education-hero">
          <div>
            <p className="eyebrow">PREVENTION LEARNING CENTER</p>
            <h1>근골격계질환 예방교육</h1>
            <p>작업자에게 필요한 예방교육을 시청하고 이수 현황을 한눈에 관리하세요.</p>
          </div>
          <div className="education-summary-chip"><span>나의 학습률</span><strong>{averageProgress}%</strong></div>
        </div>

        <section className="education-stats" aria-label="교육 현황">
          <article><span className="education-stat-icon">✓</span><div><strong>{completedCount}<small>개</small></strong><p>수료한 교육</p></div></article>
          <article><span className="education-stat-icon is-blue">▶</span><div><strong>{statusCounts.progress}<small>개</small></strong><p>학습 중인 교육</p></div></article>
          <article><span className="education-stat-icon is-amber">!</span><div><strong>{statusCounts.required}<small>개</small></strong><p>학습이 필요한 교육</p></div></article>
          <article className="education-progress-stat"><div><span>전체 이수율</span><strong>{Math.round((completedCount / courses.length) * 100)}%</strong></div><div className="education-progress-track"><i style={{ width: `${(completedCount / courses.length) * 100}%` }} /></div><p>총 {courses.length}개 과정 중 {completedCount}개 완료</p></article>
        </section>

        <section className="education-library">
          <div className="education-library-heading">
            <div><p className="panel-kicker">LEARNING LIBRARY</p><h2>전체 교육 과정</h2></div>
            <div className="education-filters" aria-label="교육 상태 필터">
              {([['all', '전체'], ['completed', '수료완료'], ['progress', '수료중'], ['required', '학습필요']] as const).map(([value, label]) => (
                <button className={filter === value ? "active" : ""} key={value} onClick={() => setFilter(value)}>{label}</button>
              ))}
            </div>
          </div>
          <div className="education-course-grid">
            {visibleCourses.map((course, index) => (
              <button className={`education-course-card ${selected.id === course.id ? "is-selected" : ""}`} key={course.id} onClick={() => setSelectedId(course.id)}>
                <div className={`course-cover cover-${index % 4}`}>
                  <span>{course.category}</span><i>▶</i><small>{course.duration}</small>
                </div>
                <div className="course-card-body">
                  <span className={`course-status status-${course.status}`}>{STATUS_COPY[course.status]}</span>
                  <h3>{course.title}</h3>
                  <p>{course.description}</p>
                  <div className="course-progress"><div><i style={{ width: `${course.progress}%` }} /></div><span>{course.progress}%</span></div>
                </div>
              </button>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
