import { desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { getDb } from "../../../db";
import { measurementSessions } from "../../../db/schema";
import {
  METRIC_KEYS,
  type MeasurementSession,
  type MeasurementSessionInput,
  type MetricValues,
} from "../../../lib/measurements";

export const runtime = "edge";

function isFiniteNonNegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isMetricValues(value: unknown): value is MetricValues {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return METRIC_KEYS.every((key) => isFiniteNonNegative(record[key]));
}

function isSessionInput(value: unknown): value is MeasurementSessionInput {
  if (!value || typeof value !== "object") return false;
  const session = value as Record<string, unknown>;

  return (
    typeof session.id === "string" &&
    session.id.length > 0 &&
    session.id.length <= 100 &&
    typeof session.workerName === "string" &&
    session.workerName.length > 0 && session.workerName.length <= 100 &&
    typeof session.jobName === "string" &&
    session.jobName.length > 0 && session.jobName.length <= 200 &&
    (session.rulaScore === null || isFiniteNonNegative(session.rulaScore)) &&
    typeof session.startedAt === "string" &&
    typeof session.endedAt === "string" &&
    isFiniteNonNegative(session.durationSeconds) &&
    isFiniteNonNegative(session.maxPeople) &&
    isFiniteNonNegative(session.samples) &&
    isFiniteNonNegative(session.normalSamples) &&
    isFiniteNonNegative(session.cautionSamples) &&
    isFiniteNonNegative(session.dangerSamples) &&
    isFiniteNonNegative(session.averageConfidence) &&
    isMetricValues(session.metricMax) &&
    isMetricValues(session.metricAverage) &&
    isMetricValues(session.metricWarnings)
  );
}

function parseMetricValues(value: string): MetricValues {
  try {
    const parsed = JSON.parse(value);
    return isMetricValues(parsed) ? parsed : fallbackMetricValues();
  } catch {
    return fallbackMetricValues();
  }
}

function fallbackMetricValues(): MetricValues {
  return {
    neck: 0,
    trunk: 0,
    shoulder: 0,
    headRotation: 0,
    trunkRotation: 0,
    leftWristRotation: 0,
    rightWristRotation: 0,
  };
}

export async function GET() {
  try {
    const rows = await getDb()
      .select()
      .from(measurementSessions)
      .orderBy(desc(measurementSessions.startedAt))
      .limit(50);

    const sessions: MeasurementSession[] = rows.map((row) => ({
      ...row,
      metricMax: parseMetricValues(row.metricMax),
      metricAverage: parseMetricValues(row.metricAverage),
      metricWarnings: parseMetricValues(row.metricWarnings),
    }));

    return NextResponse.json({ sessions });
  } catch {
    return NextResponse.json(
      { message: "측정 기록을 불러오지 못했습니다." },
      { status: 503 },
    );
  }
}

export async function POST(request: Request) {
  try {
    const body: unknown = await request.json();
    if (!isSessionInput(body)) {
      return NextResponse.json(
        { message: "저장할 측정 결과의 형식이 올바르지 않습니다." },
        { status: 400 },
      );
    }

    const createdAt = new Date().toISOString();
    await getDb()
      .insert(measurementSessions)
      .values({
        ...body,
        metricMax: JSON.stringify(body.metricMax),
        metricAverage: JSON.stringify(body.metricAverage),
        metricWarnings: JSON.stringify(body.metricWarnings),
        createdAt,
      })
      .onConflictDoNothing();

    return NextResponse.json({ ok: true, createdAt }, { status: 201 });
  } catch {
    return NextResponse.json(
      { message: "측정 결과를 저장하지 못했습니다." },
      { status: 503 },
    );
  }
}
