"use client";
import { useEffect, useMemo, useState } from "react";
import type { MeasurementSession } from "../../lib/measurements";

export default function History() {
  const [sessions,setSessions]=useState<MeasurementSession[]>([]);
  const [worker,setWorker]=useState("");
  useEffect(()=>{ fetch("/api/sessions",{cache:"no-store"}).then(r=>r.json()).then(d=>setSessions(d.sessions??[])).catch(()=>setSessions([])); },[]);
  const workers=useMemo(()=>Array.from(new Set(sessions.map(s=>s.workerName))),[sessions]);
  useEffect(()=>{ if(!worker && workers.length) setWorker(workers[0]); },[worker,workers]);
  const individual=useMemo(()=>sessions.filter(s=>s.workerName===worker),[sessions,worker]);
  return <main className="dashboard-page">
    <header className="topbar dashboard-topbar"><a className="brand-lockup brand-link" href="/dashboard"><div className="brand-mark">M</div><div><p className="brand-name">MoveGuard</p><p className="brand-caption">개인 분석 기록</p></div></a><nav className="product-nav"><a href="/dashboard">대시보드</a><a href="/">실시간 분석</a><a href="/hazard-survey">일정관리</a><a className="active" href="/history">분석 기록</a></nav></header>
    <section className="dashboard-content"><div className="dashboard-hero"><div><p className="eyebrow">WORKER HISTORY</p><h1>작업자별 분석 기록</h1><p>작업장 전체 통계와 분리된 개인별 기록입니다.</p></div><a className="button button-primary" href="/">새 분석 시작</a></div>
      <section className="dashboard-card"><label className="summary-label" htmlFor="worker-select">작업자 이름을 선택하세요</label><select id="worker-select" value={worker} onChange={e=>setWorker(e.target.value)}>{workers.length?workers.map(name=><option key={name}>{name}</option>):<option value="">저장된 작업자 없음</option>}</select></section>
      <section className="summary-grid" aria-label="개인 통계"><article className="summary-card"><span className="summary-label">개인별 분석 기록</span><strong>{individual.length}<small>회</small></strong></article><article className="summary-card"><span className="summary-label">개인별 부위별 위험도</span><strong className="summary-value-small">기록 기반</strong></article><article className="summary-card"><span className="summary-label">개인별 RULA 결과</span><strong className="summary-value-small">평가 불가</strong><p>전문가 RULA 정답이 제공되지 않았습니다.</p></article></section>
      <section className="dashboard-card history-card"><div className="dashboard-card-heading"><div><p className="panel-kicker">INDIVIDUAL ONLY</p><h2>{worker} 개인별 분석 기록</h2></div></div><div className="history-table-wrap"><table className="history-table"><thead><tr><th>분석 일시</th><th>분석 시간</th><th>위험 이벤트</th><th>관절 검출 신뢰도 · 참고값</th></tr></thead><tbody>{individual.map(s=><tr key={s.id}><td>{new Date(s.startedAt).toLocaleString("ko-KR")}</td><td>{Math.round(s.durationSeconds)}초</td><td>{s.dangerSamples}회</td><td>{Math.min(99.9,Math.floor(s.averageConfidence*10)/10)}%</td></tr>)}</tbody></table></div></section>
    </section>
  </main>;
}
