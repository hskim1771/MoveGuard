"use client";

import { useEffect, useMemo, useState } from "react";
import {
  METRIC_DANGER_AT,
  METRIC_KEYS,
  METRIC_LABELS,
  type MeasurementSession,
  type MetricKey,
} from "../../lib/measurements";

type LoadState = "loading" | "ready" | "error";

function formatDuration(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.max(0, Math.round(totalSeconds % 60));

  if (hours > 0) return `${hours}시간 ${minutes}분`;
  if (minutes > 0) return `${minutes}분 ${seconds}초`;
  return `${seconds}초`;
}

function formatDate(iso: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
}

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.round((part / total) * 100);
}

function riskForSession(session: MeasurementSession) {
  if (session.dangerSamples > 0) return "danger";
  if (session.cautionSamples > 0) return "caution";
  return "normal";
}

function riskCopy(risk: ReturnType<typeof riskForSession>) {
  if (risk === "danger") return "위험 있음";
  if (risk === "caution") return "주의 필요";
  return "안정적";
}

function mostImportantMetric(session: MeasurementSession): MetricKey {
  return METRIC_KEYS.reduce((highest, key) => {
    const currentScore =
      session.metricWarnings[key] * 1000 +
      session.metricMax[key] / METRIC_DANGER_AT[key];
    const highestScore =
      session.metricWarnings[highest] * 1000 +
      session.metricMax[highest] / METRIC_DANGER_AT[highest];
    return currentScore > highestScore ? key : highest;
  }, "neck" as MetricKey);
}

export default function Dashboard() {
  const [sessions, setSessions] = useState<MeasurementSession[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");

  async function loadSessions() {
    setLoadState("loading");
    try {
      const response = await fetch("/api/sessions", { cache: "no-store" });
      if (!response.ok) throw new Error("request failed");
      const data = (await response.json()) as {
        sessions?: MeasurementSession[];
      };
      setSessions(data.sessions ?? []);
      setLoadState("ready");
    } catch {
      setLoadState("error");
    }
  }

  useEffect(() => {
    void loadSessions();
  }, []);

  const summary = useMemo(() => {
    const totalSeconds = sessions.reduce(
      (sum, session) => sum + session.durationSeconds,
      0,
    );
    const totalSamples = sessions.reduce(
      (sum, session) => sum + session.samples,
      0,
    );
    const warningSamples = sessions.reduce(
      (sum, session) =>
        sum + session.cautionSamples + session.dangerSamples,
      0,
    );
    const maxPeople = sessions.reduce(
      (maximum, session) => Math.max(maximum, session.maxPeople),
      0,
    );
    const metricWarnings = METRIC_KEYS.reduce(
      (totals, key) => {
        totals[key] = sessions.reduce(
          (sum, session) => sum + session.metricWarnings[key],
          0,
        );
        return totals;
      },
      {} as Record<MetricKey, number>,
    );
    const mainMetric = METRIC_KEYS.reduce(
      (highest, key) =>
        metricWarnings[key] > metricWarnings[highest] ? key : highest,
      "neck" as MetricKey,
    );

    return {
      totalSeconds,
      warningRate: percent(warningSamples, totalSamples),
      maxPeople,
      mainMetric,
      mainMetricWarnings: metricWarnings[mainMetric],
    };
  }, [sessions]);

  const latest = sessions[0];

  return (
    <main className="dashboard-page">
      <header className="topbar dashboard-topbar">
        <a className="brand-lockup brand-link" href="/">
          <div className="brand-mark" aria-hidden="true">
            M
          </div>
          <div>
            <p className="brand-name">MoveGuard</p>
            <p className="brand-caption">자세 분석 MVP</p>
          </div>
        </a>
        <nav className="product-nav" aria-label="MoveGuard 주요 메뉴">
          <a className="active" href="/dashboard" aria-current="page">대시보드</a>
          <a href="/">실시간 분석</a>
          <a href="/history">작업자 관리</a>
          <a href="/history">분석 기록</a>
          <a href="/history">리포트</a>
        </nav>
        <div className="privacy-chip">
          <span className="privacy-dot" aria-hidden="true" />
          영상 저장 없음 · 수치만 저장
        </div>
      </header>

      <section className="dashboard-content">
        <div className="dashboard-hero">
          <div>
            <p className="eyebrow">POSTURE ANALYTICS</p>
            <h1>작업장 전체 대시보드</h1>
            <p>
              측정을 종료하면 시간, 위험 자세 비율, 부위별 최대 각도가 자동으로
              기록됩니다.
            </p>
          </div>
          <div className="dashboard-actions">
            <a className="button button-secondary dashboard-cta" href="/history">개인 분석 기록</a>
            <a className="button button-primary dashboard-cta" href="/">새 분석 시작</a>
          </div>
        </div>

        {(!latest || loadState !== "ready") && (
          <section className="latest-grid" aria-label="작업장 전체 위험도 빈 상태">
            <article className="dashboard-card metric-overview-card">
              <div className="dashboard-card-heading"><div><p className="panel-kicker">WORKPLACE MONITOR</p><h2>작업장 전체 부위별 위험도 모니터</h2></div></div>
              <p>분석 결과가 저장되면 부위별 위험도가 이곳에 표시됩니다.</p>
            </article>
            <article className="dashboard-card risk-overview-card">
              <div className="dashboard-card-heading"><div><p className="panel-kicker">EXPOSURE TREND</p><h2>작업장 전체 부위별 위험 노출률</h2></div></div>
              <p>최근 7일 부위별 추이는 분석 기록이 생성된 뒤 표시됩니다.</p>
            </article>
          </section>
        )}

        {loadState === "loading" && (
          <div className="dashboard-loading" aria-live="polite">
            <span className="loading-ring" aria-hidden="true" />
            <strong>측정 기록을 불러오고 있어요</strong>
          </div>
        )}

        {loadState === "error" && (
          <div className="dashboard-error" role="alert">
            <div>
              <strong>측정 기록을 불러오지 못했어요</strong>
              <p>잠시 후 다시 시도해 주세요.</p>
            </div>
            <button className="button button-primary" onClick={loadSessions}>
              다시 불러오기
            </button>
          </div>
        )}

        {loadState === "ready" && sessions.length === 0 && (
          <section className="dashboard-empty">
            <span className="empty-chart" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <p className="panel-kicker">FIRST MEASUREMENT</p>
            <h2>아직 저장된 측정 결과가 없어요</h2>
            <p>
              카메라 측정을 시작한 뒤 ‘측정 종료·저장’을 누르면 이곳에 첫
              분석 결과가 만들어집니다.
            </p>
            <a className="button button-primary dashboard-cta" href="/">
              첫 측정 시작하기
            </a>
          </section>
        )}

        {loadState === "ready" && sessions.length > 0 && latest && (
          <>
            <section className="summary-grid" aria-label="전체 측정 요약">
              <article className="summary-card">
                <span className="summary-label">오늘 분석 인원</span>
                <strong>{sessions.length}<small>회</small></strong>
                <p>저장된 측정 세션 수</p>
              </article>
              <article className="summary-card">
                <span className="summary-label">총 분석 시간</span>
                <strong className="summary-value-small">
                  {formatDuration(summary.totalSeconds)}
                </strong>
                <p>모든 세션을 합한 시간</p>
              </article>
              <article
                className={`summary-card ${summary.warningRate >= 30 ? "summary-warning" : ""}`}
              >
                <span className="summary-label">위험 이벤트</span>
                <strong>{summary.warningRate}<small>%</small></strong>
                <p>전체 인식 구간 중 위험 비율</p>
              </article>
              <article className="summary-card">
                <span className="summary-label">고위험 작업자</span>
                <strong className="summary-value-small">
                  {summary.mainMetricWarnings > 0
                    ? METRIC_LABELS[summary.mainMetric]
                    : "감지 없음"}
                </strong>
                <p>동시 인식 최대 {summary.maxPeople}명</p>
              </article>
            </section>

            <section className="latest-grid">
              <article className="dashboard-card risk-overview-card">
                <div className="dashboard-card-heading">
                  <div>
                    <p className="panel-kicker">LATEST SESSION</p>
                    <h2>최근 측정 위험 분포</h2>
                  </div>
                  <span className={`session-risk risk-label-${riskForSession(latest)}`}>
                    {riskCopy(riskForSession(latest))}
                  </span>
                </div>
                <div
                  className="risk-stack"
                  aria-label={`정상 ${percent(latest.normalSamples, latest.samples)}%, 주의 ${percent(latest.cautionSamples, latest.samples)}%, 위험 ${percent(latest.dangerSamples, latest.samples)}%`}
                >
                  <span
                    className="risk-stack-normal"
                    style={{
                      width: `${percent(latest.normalSamples, latest.samples)}%`,
                    }}
                  />
                  <span
                    className="risk-stack-caution"
                    style={{
                      width: `${percent(latest.cautionSamples, latest.samples)}%`,
                    }}
                  />
                  <span
                    className="risk-stack-danger"
                    style={{
                      width: `${percent(latest.dangerSamples, latest.samples)}%`,
                    }}
                  />
                </div>
                <div className="risk-legend">
                  <div>
                    <span className="legend-dot legend-normal" />
                    <p>정상</p>
                    <strong>
                      {percent(latest.normalSamples, latest.samples)}%
                    </strong>
                  </div>
                  <div>
                    <span className="legend-dot legend-caution" />
                    <p>주의</p>
                    <strong>
                      {percent(latest.cautionSamples, latest.samples)}%
                    </strong>
                  </div>
                  <div>
                    <span className="legend-dot legend-danger" />
                    <p>위험</p>
                    <strong>
                      {percent(latest.dangerSamples, latest.samples)}%
                    </strong>
                  </div>
                </div>
                <div className="latest-meta">
                  <span>{formatDate(latest.startedAt)}</span>
                  <span>{formatDuration(latest.durationSeconds)}</span>
                  <span>최대 {latest.maxPeople}명</span>
                  <span>신뢰도 {Math.round(latest.averageConfidence)}%</span>
                </div>
              </article>

              <article className="dashboard-card metric-overview-card">
                <div className="dashboard-card-heading">
                  <div>
                    <p className="panel-kicker">AVERAGE &amp; MAXIMUM ANGLES</p>
                    <h2>작업장 전체 부위별 위험도 모니터</h2>
                  </div>
                  <span className="estimate-chip">평균 + 최대</span>
                </div>
                <div className="metric-bars">
                  {METRIC_KEYS.map((key) => {
                    const maximum = latest.metricMax[key];
                    const average = latest.metricAverage[key];
                    const maximumRatio = Math.min(
                      100,
                      Math.round((maximum / METRIC_DANGER_AT[key]) * 100),
                    );
                    const averageRatio = Math.min(
                      100,
                      Math.round((average / METRIC_DANGER_AT[key]) * 100),
                    );
                    return (
                      <div className="metric-bar-row" key={key}>
                        <div>
                          <span>{METRIC_LABELS[key]}</span>
                          <span className="metric-angle-values">
                            <small>
                              평균 <strong>{Math.round(average)}°</strong>
                            </small>
                            <small>
                              최대 <strong>{Math.round(maximum)}°</strong>
                            </small>
                          </span>
                        </div>
                        <div className="metric-bar-track" aria-hidden="true">
                          <span
                            className={`metric-maximum ${
                              maximumRatio >= 100
                                ? "bar-danger"
                                : maximumRatio >= 65
                                  ? "bar-caution"
                                  : ""
                            }`}
                            style={{ width: `${Math.max(2, maximumRatio)}%` }}
                          />
                          <i
                            className={
                              averageRatio >= 100
                                ? "average-danger"
                                : averageRatio >= 65
                                  ? "average-caution"
                                  : "metric-average"
                            }
                            style={{ width: `${Math.max(2, averageRatio)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </article>
            </section>

            <section className="dashboard-card history-card">
              <div className="dashboard-card-heading">
                <div>
                  <p className="panel-kicker">SESSION HISTORY</p>
                  <h2>작업장 전체 부위별 위험 노출률 · 최근 7일 부위별 추이</h2>
                </div>
                <span className="history-count">최근 {sessions.length}회</span>
              </div>
              <div className="history-table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>측정 일시</th>
                      <th>측정 시간</th>
                      <th>최대 인원</th>
                      <th>주요 위험 부위</th>
                      <th>주의·위험</th>
                      <th>상태</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.map((session) => {
                      const warningRate = percent(
                        session.cautionSamples + session.dangerSamples,
                        session.samples,
                      );
                      const risk = riskForSession(session);
                      const mainMetric = mostImportantMetric(session);
                      return (
                        <tr key={session.id}>
                          <td>{formatDate(session.startedAt)}</td>
                          <td>{formatDuration(session.durationSeconds)}</td>
                          <td>{session.maxPeople}명</td>
                          <td>{METRIC_LABELS[mainMetric]}</td>
                          <td>{warningRate}%</td>
                          <td>
                            <span className={`session-risk risk-label-${risk}`}>
                              {riskCopy(risk)}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <div className="dashboard-notice">
              <span className="notice-mark" aria-hidden="true">
                i
              </span>
              <p>
                <strong>현재 MVP는 측정 세션 단위 분석입니다.</strong>
                사람 번호는 화면 위치에 따라 바뀔 수 있어 개인별 장기 추적은
                아직 포함하지 않았습니다. 각도와 위험 상태는 시험용
                추정치입니다.
              </p>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
