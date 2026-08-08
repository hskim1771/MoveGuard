"use client";

import { useEffect, useMemo, useState } from "react";

type SurveySchedule = {
  id: string;
  date: string;
  title: string;
};

const STORAGE_KEY = "moveguard-hazard-survey-schedules";
const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

function toDateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseDateKey(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day);
}

function monthLabel(date: Date) {
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
  }).format(date);
}

function fullDateLabel(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric",
    weekday: "short",
  }).format(parseDateKey(value));
}

function dayDifference(from: Date, to: Date) {
  const fromUtc = Date.UTC(from.getFullYear(), from.getMonth(), from.getDate());
  const toUtc = Date.UTC(to.getFullYear(), to.getMonth(), to.getDate());
  return Math.ceil((toUtc - fromUtc) / 86_400_000);
}

export default function HazardSurveySchedule() {
  const today = useMemo(() => new Date(), []);
  const todayKey = toDateKey(today);
  const [viewMonth, setViewMonth] = useState(
    () => new Date(today.getFullYear(), today.getMonth(), 1),
  );
  const [selectedDate, setSelectedDate] = useState(todayKey);
  const [title, setTitle] = useState("정기 유해요인조사");
  const [schedules, setSchedules] = useState<SurveySchedule[]>([]);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) setSchedules(JSON.parse(stored) as SurveySchedule[]);
    } catch {
      setSchedules([]);
    } finally {
      setIsReady(true);
    }
  }, []);

  useEffect(() => {
    if (isReady) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(schedules));
    }
  }, [isReady, schedules]);

  const calendarDays = useMemo(() => {
    const firstDay = new Date(viewMonth.getFullYear(), viewMonth.getMonth(), 1);
    const start = new Date(firstDay);
    start.setDate(1 - firstDay.getDay());
    return Array.from({ length: 42 }, (_, index) => {
      const date = new Date(start);
      date.setDate(start.getDate() + index);
      return date;
    });
  }, [viewMonth]);

  const upcoming = useMemo(
    () =>
      [...schedules]
        .filter((schedule) => schedule.date >= todayKey)
        .sort((a, b) => a.date.localeCompare(b.date)),
    [schedules, todayKey],
  );
  const nextSchedule = upcoming[0];
  const dDay = nextSchedule
    ? dayDifference(today, parseDateKey(nextSchedule.date))
    : null;

  function moveMonth(offset: number) {
    setViewMonth(
      (current) =>
        new Date(current.getFullYear(), current.getMonth() + offset, 1),
    );
  }

  function selectDay(date: Date) {
    setSelectedDate(toDateKey(date));
    setViewMonth(new Date(date.getFullYear(), date.getMonth(), 1));
  }

  function registerSchedule(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedTitle = title.trim();
    if (!selectedDate || !trimmedTitle) return;
    setSchedules((current) =>
      [
        ...current,
        { id: crypto.randomUUID(), date: selectedDate, title: trimmedTitle },
      ].sort((a, b) => a.date.localeCompare(b.date)),
    );
  }

  return (
    <main className="dashboard-page survey-page">
      <header className="topbar dashboard-topbar">
        <a className="brand-lockup brand-link" href="/dashboard">
          <div className="brand-mark" aria-hidden="true" />
          <div>
            <p className="brand-name">MoveGuard</p>
            <p className="brand-caption">산업안전 관리</p>
          </div>
        </a>
        <nav className="product-nav" aria-label="MoveGuard 주요 메뉴">
          <a href="/dashboard">대시보드</a>
          <a href="/">실시간 분석</a>
          <a href="/history">작업자 관리</a>
          <a className="active" href="/hazard-survey" aria-current="page">
            일정관리
          </a>
          <a href="/history">분석 기록</a>
        </nav>
      </header>

      <section className="dashboard-content survey-content">
        <div className="dashboard-hero survey-hero">
          <div>
            <p className="eyebrow">HAZARD SURVEY SCHEDULE</p>
            <h1>유해요인조사 일정관리</h1>
            <p>조사 일정을 한눈에 확인하고, 다음 조사일까지 남은 기간을 놓치지 마세요.</p>
          </div>
          <button className="button button-primary" onClick={() => document.getElementById("survey-title")?.focus()}>
            + 새 일정 등록
          </button>
        </div>

        <section className="survey-overview">
          <article className="survey-dday-card">
            <div className="survey-dday-copy">
              <span className="survey-dday-kicker">NEXT SURVEY</span>
              {nextSchedule ? (
                <>
                  <strong>{dDay === 0 ? "D-DAY" : `D-${dDay}`}</strong>
                  <h2>{nextSchedule.title}</h2>
                  <p>{fullDateLabel(nextSchedule.date)}</p>
                </>
              ) : (
                <>
                  <strong>D-?</strong>
                  <h2>예정된 조사가 없습니다</h2>
                  <p>캘린더에서 다음 조사 일정을 등록해 주세요.</p>
                </>
              )}
            </div>
            <div className="survey-ring" aria-hidden="true">
              <span>{nextSchedule ? Math.min(dDay ?? 0, 99) : "--"}</span>
              <small>DAYS</small>
            </div>
          </article>

          <article className="survey-stat-card">
            <span>등록된 전체 일정</span>
            <strong>{schedules.length}<small>건</small></strong>
            <p>완료 및 예정 일정 포함</p>
          </article>
          <article className="survey-stat-card">
            <span>다가오는 일정</span>
            <strong>{upcoming.length}<small>건</small></strong>
            <p>오늘 이후 예정된 조사</p>
          </article>
        </section>

        <section className="survey-workspace">
          <article className="dashboard-card survey-calendar-card">
            <div className="survey-calendar-header">
              <div>
                <p className="panel-kicker">SURVEY CALENDAR</p>
                <h2>{monthLabel(viewMonth)}</h2>
              </div>
              <div className="calendar-controls">
                <button aria-label="이전 달" onClick={() => moveMonth(-1)}>‹</button>
                <button onClick={() => setViewMonth(new Date(today.getFullYear(), today.getMonth(), 1))}>오늘</button>
                <button aria-label="다음 달" onClick={() => moveMonth(1)}>›</button>
              </div>
            </div>
            <div className="calendar-weekdays">
              {WEEKDAYS.map((day) => <span key={day}>{day}</span>)}
            </div>
            <div className="calendar-grid">
              {calendarDays.map((date) => {
                const key = toDateKey(date);
                const daySchedules = schedules.filter((schedule) => schedule.date === key);
                const isOutside = date.getMonth() !== viewMonth.getMonth();
                return (
                  <button
                    className={`${isOutside ? "is-outside" : ""} ${key === todayKey ? "is-today" : ""} ${key === selectedDate ? "is-selected" : ""}`}
                    key={key}
                    onClick={() => selectDay(date)}
                    aria-label={`${fullDateLabel(key)}${daySchedules.length ? `, 일정 ${daySchedules.length}건` : ""}`}
                  >
                    <span>{date.getDate()}</span>
                    <div className="calendar-events">
                      {daySchedules.slice(0, 2).map((schedule) => <i key={schedule.id}>{schedule.title}</i>)}
                    </div>
                  </button>
                );
              })}
            </div>
          </article>

          <aside className="dashboard-card survey-register-card">
            <p className="panel-kicker">REGISTER SCHEDULE</p>
            <h2>일정 등록</h2>
            <form onSubmit={registerSchedule}>
              <label htmlFor="survey-date">조사일</label>
              <input id="survey-date" type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} required />
              <label htmlFor="survey-title">조사명</label>
              <input id="survey-title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="조사명을 입력하세요" required />
              <button className="button button-primary" type="submit">일정 등록하기</button>
            </form>

            <div className="upcoming-list">
              <div className="upcoming-heading">
                <h3>다가오는 일정</h3>
                <span>{upcoming.length}건</span>
              </div>
              {upcoming.length ? upcoming.slice(0, 4).map((schedule) => (
                <div className="upcoming-item" key={schedule.id}>
                  <time dateTime={schedule.date}>
                    <strong>{parseDateKey(schedule.date).getDate()}</strong>
                    <span>{parseDateKey(schedule.date).getMonth() + 1}월</span>
                  </time>
                  <div><strong>{schedule.title}</strong><span>{fullDateLabel(schedule.date)}</span></div>
                  <button aria-label={`${schedule.title} 삭제`} onClick={() => setSchedules((current) => current.filter((item) => item.id !== schedule.id))}>×</button>
                </div>
              )) : <p className="upcoming-empty">등록된 예정 일정이 없습니다.</p>}
            </div>
          </aside>
        </section>
      </section>
    </main>
  );
}
