export const METRIC_KEYS = [
  "neck",
  "trunk",
  "shoulder",
  "headRotation",
  "trunkRotation",
  "leftWristRotation",
  "rightWristRotation",
] as const;

export type MetricKey = (typeof METRIC_KEYS)[number];

export type MetricValues = Record<MetricKey, number>;

export type MeasurementSessionInput = {
  id: string;
  workerName: string;
  jobName: string;
  rulaScore: number | null;
  startedAt: string;
  endedAt: string;
  durationSeconds: number;
  maxPeople: number;
  samples: number;
  normalSamples: number;
  cautionSamples: number;
  dangerSamples: number;
  averageConfidence: number;
  metricMax: MetricValues;
  metricAverage: MetricValues;
  metricWarnings: MetricValues;
};

export type MeasurementSession = MeasurementSessionInput & {
  createdAt: string;
};

export const METRIC_LABELS: Record<MetricKey, string> = {
  neck: "목 기울기",
  trunk: "몸통 전굴",
  shoulder: "어깨 들림",
  headRotation: "머리 회전",
  trunkRotation: "몸통 회전",
  leftWristRotation: "왼손목 회전",
  rightWristRotation: "오른손목 회전",
};

export const METRIC_DANGER_AT: Record<MetricKey, number> = {
  neck: 35,
  trunk: 30,
  shoulder: 60,
  headRotation: 45,
  trunkRotation: 45,
  leftWristRotation: 70,
  rightWristRotation: 70,
};

export function emptyMetricValues(): MetricValues {
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
