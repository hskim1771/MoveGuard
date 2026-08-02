"use client";

import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  emptyMetricValues,
  METRIC_KEYS,
  type MeasurementSessionInput,
  type MetricValues,
} from "../lib/measurements";

type Phase = "idle" | "requesting" | "loading" | "running" | "error";
type Risk = "normal" | "caution" | "danger";

type Point3D = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

type Metric = {
  angle: number | null;
  label: string;
  risk: Risk;
  guide: string;
  experimental?: boolean;
};

type Metrics = {
  neck: Metric;
  trunk: Metric;
  shoulder: Metric;
  headRotation: Metric;
  trunkRotation: Metric;
  leftWristRotation: Metric;
  rightWristRotation: Metric;
  confidence: number;
  fullBody: boolean;
};

type DetectedPerson = {
  landmarks: NormalizedLandmark[];
  worldLandmarks: Point3D[];
  metrics: Metrics;
};

type ServerStatus = "disabled" | "checking" | "online" | "offline";

type RulaRisk = "low" | "medium" | "high" | "very_high" | "not_measurable";

type ServerRula = {
  final_score: number | null;
  risk_level: RulaRisk;
  scored_side: "left" | "right" | null;
  valid: boolean;
  assumptions: string[];
  issues: string[];
};

type ServerQuality = {
  status: "valid" | "low_confidence" | "not_measurable";
  pose_valid: boolean;
  hand_valid: boolean;
  occlusion_detected: boolean;
  issues: string[];
};

type ServerPersonAnalysis = {
  person_id: string;
  metrics?: Record<
    keyof Omit<Metrics, "confidence" | "fullBody">,
    {
      angle: number | null;
      risk: Risk;
      experimental: boolean;
    }
  >;
  confidence?: number;
  full_body?: boolean;
  overall_risk?: Risk;
  pose?: {
    confidence: number;
    landmarks: Point3D[];
  };
  rula?: ServerRula;
  quality?: ServerQuality;
};

type ServerAnalysis = {
  schema_version: string;
  processor: string;
  people: ServerPersonAnalysis[];
  scene_risk?: Risk;
  elapsed_ms: number;
  video_stored: boolean;
  objects?: Array<{
    detection_id: string;
    label: string;
    confidence: number;
  }>;
  quality?: ServerQuality;
  rula?: ServerRula;
};

type SessionAccumulator = {
  id: string;
  startedAt: string;
  startedAtMs: number;
  samples: number;
  normalSamples: number;
  cautionSamples: number;
  dangerSamples: number;
  confidenceTotal: number;
  maxPeople: number;
  metricMax: MetricValues;
  metricTotal: MetricValues;
  metricCounts: MetricValues;
  metricWarnings: MetricValues;
};

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const ANALYSIS_API_URL = (
  process.env.NEXT_PUBLIC_ANALYSIS_API_URL ??
  (process.env.NODE_ENV === "development"
    ? "http://127.0.0.1:8000"
    : "https://moveguard-analysis-api-4que.onrender.com")
).replace(/\/$/, "");
const ANALYSIS_API_KEY = process.env.NEXT_PUBLIC_MOVEGUARD_API_KEY ?? "";
const MAX_PEOPLE = 4;
const PERSON_COLORS = ["#32e6c0", "#58a6ff", "#c47cff", "#ff9f43"];

const EMPTY_METRICS: Metrics = {
  neck: {
    angle: null,
    label: "목 기울기",
    risk: "normal",
    guide: "귀와 어깨 중심의 기울기를 계산해요",
  },
  trunk: {
    angle: null,
    label: "몸통 전굴",
    risk: "normal",
    guide: "어깨와 골반 중심의 기울기를 계산해요",
  },
  shoulder: {
    angle: null,
    label: "어깨 들림",
    risk: "normal",
    guide: "양쪽 중 더 큰 값을 표시해요",
  },
  headRotation: {
    angle: null,
    label: "머리 회전",
    risk: "normal",
    guide: "양쪽 귀의 3D 깊이 차이로 추정해요",
  },
  trunkRotation: {
    angle: null,
    label: "몸통 회전",
    risk: "normal",
    guide: "어깨와 골반의 3D 방향을 함께 봐요",
  },
  leftWristRotation: {
    angle: null,
    label: "왼손목 회전",
    risk: "normal",
    guide: "손가락 관절점으로 추정하는 시험 기능이에요",
    experimental: true,
  },
  rightWristRotation: {
    angle: null,
    label: "오른손목 회전",
    risk: "normal",
    guide: "손가락 관절점으로 추정하는 시험 기능이에요",
    experimental: true,
  },
  confidence: 0,
  fullBody: false,
};

const RISK_ORDER: Record<Risk, number> = {
  normal: 0,
  caution: 1,
  danger: 2,
};

function midpoint(first: Point3D, second: Point3D): Point3D {
  return {
    x: (first.x + second.x) / 2,
    y: (first.y + second.y) / 2,
    z: (first.z + second.z) / 2,
    visibility: Math.min(first.visibility ?? 0, second.visibility ?? 0),
  };
}

function angleAt(first: Point3D, pivot: Point3D, third: Point3D) {
  const vectorA = { x: first.x - pivot.x, y: first.y - pivot.y };
  const vectorB = { x: third.x - pivot.x, y: third.y - pivot.y };
  const magnitudeA = Math.hypot(vectorA.x, vectorA.y);
  const magnitudeB = Math.hypot(vectorB.x, vectorB.y);

  if (!magnitudeA || !magnitudeB) return 0;

  const cosine = Math.min(
    1,
    Math.max(
      -1,
      (vectorA.x * vectorB.x + vectorA.y * vectorB.y) /
        (magnitudeA * magnitudeB),
    ),
  );

  return (Math.acos(cosine) * 180) / Math.PI;
}

function tiltFromVertical(top: Point3D, bottom: Point3D) {
  const dx = top.x - bottom.x;
  const dy = top.y - bottom.y;
  const magnitude = Math.hypot(dx, dy);

  if (!magnitude) return 0;

  const cosine = Math.min(1, Math.max(-1, -dy / magnitude));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function rotationFromPair(first: Point3D, second: Point3D) {
  const depthDifference = Math.abs(first.z - second.z);
  const visibleSeparation = Math.hypot(
    first.x - second.x,
    first.y - second.y,
  );

  if (!depthDifference && !visibleSeparation) return 0;
  return (
    (Math.atan2(depthDifference, visibleSeparation) * 180) /
    Math.PI
  );
}

function riskFor(angle: number, cautionAt: number, dangerAt: number): Risk {
  if (angle >= dangerAt) return "danger";
  if (angle >= cautionAt) return "caution";
  return "normal";
}

function highestRisk(risks: Risk[]): Risk {
  return risks.reduce(
    (highest, current) =>
      RISK_ORDER[current] > RISK_ORDER[highest] ? current : highest,
    "normal",
  );
}

function overallRisk(metrics: Metrics): Risk {
  return highestRisk([
    metrics.neck.risk,
    metrics.trunk.risk,
    metrics.shoulder.risk,
    metrics.headRotation.risk,
    metrics.trunkRotation.risk,
    metrics.leftWristRotation.risk,
    metrics.rightWristRotation.risk,
  ]);
}

function wristMetric(
  landmarks: NormalizedLandmark[],
  worldLandmarks: Point3D[],
  indexPoint: number,
  pinkyPoint: number,
  label: string,
): Metric {
  const isVisible =
    (landmarks[indexPoint].visibility ?? 0) > 0.38 &&
    (landmarks[pinkyPoint].visibility ?? 0) > 0.38;

  if (!isVisible) {
    return {
      angle: null,
      label,
      risk: "normal",
      guide: "손 전체를 카메라에 보여 주세요",
      experimental: true,
    };
  }

  const angle = rotationFromPair(
    worldLandmarks[indexPoint],
    worldLandmarks[pinkyPoint],
  );

  return {
    angle,
    label,
    risk: riskFor(angle, 45, 70),
    guide: "손가락 관절점으로 추정하는 시험 기능이에요",
    experimental: true,
  };
}

function buildMetrics(
  landmarks: NormalizedLandmark[],
  worldLandmarks: Point3D[],
): Metrics {
  const shoulderMid = midpoint(landmarks[11], landmarks[12]);
  const hipMid = midpoint(landmarks[23], landmarks[24]);
  const earMid = midpoint(landmarks[7], landmarks[8]);
  const neckAngle = tiltFromVertical(earMid, shoulderMid);
  const trunkAngle = tiltFromVertical(shoulderMid, hipMid);
  const leftShoulder = angleAt(landmarks[13], landmarks[11], landmarks[23]);
  const rightShoulder = angleAt(landmarks[14], landmarks[12], landmarks[24]);
  const shoulderAngle = Math.max(leftShoulder, rightShoulder);
  const headRotation = rotationFromPair(worldLandmarks[7], worldLandmarks[8]);
  const shoulderRotation = rotationFromPair(
    worldLandmarks[11],
    worldLandmarks[12],
  );
  const hipRotation = rotationFromPair(
    worldLandmarks[23],
    worldLandmarks[24],
  );
  const trunkRotation = (shoulderRotation + hipRotation) / 2;
  const keyIndexes = [7, 8, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28];
  const confidence =
    keyIndexes.reduce(
      (sum, index) => sum + (landmarks[index].visibility ?? 0),
      0,
    ) / keyIndexes.length;
  const fullBody = [11, 12, 23, 24, 25, 26, 27, 28].every((index) => {
    const point = landmarks[index];
    return (
      (point.visibility ?? 0) > 0.45 &&
      point.x > 0.015 &&
      point.x < 0.985 &&
      point.y > 0.015 &&
      point.y < 0.985
    );
  });

  return {
    neck: {
      angle: neckAngle,
      label: "목 기울기",
      risk: riskFor(neckAngle, 20, 35),
      guide: "시험값 20° 주의 · 35° 위험",
    },
    trunk: {
      angle: trunkAngle,
      label: "몸통 전굴",
      risk: riskFor(trunkAngle, 15, 30),
      guide: "시험값 15° 주의 · 30° 위험",
    },
    shoulder: {
      angle: shoulderAngle,
      label: "어깨 들림",
      risk: riskFor(shoulderAngle, 30, 60),
      guide: "시험값 30° 주의 · 60° 위험",
    },
    headRotation: {
      angle: headRotation,
      label: "머리 회전",
      risk: riskFor(headRotation, 25, 45),
      guide: "시험값 25° 주의 · 45° 위험",
    },
    trunkRotation: {
      angle: trunkRotation,
      label: "몸통 회전",
      risk: riskFor(trunkRotation, 25, 45),
      guide: "시험값 25° 주의 · 45° 위험",
    },
    leftWristRotation: wristMetric(
      landmarks,
      worldLandmarks,
      19,
      17,
      "왼손목 회전",
    ),
    rightWristRotation: wristMetric(
      landmarks,
      worldLandmarks,
      20,
      18,
      "오른손목 회전",
    ),
    confidence: Math.round(confidence * 100),
    fullBody,
  };
}

function riskLabel(risk: Risk) {
  if (risk === "danger") return "위험";
  if (risk === "caution") return "주의";
  return "정상";
}

function rulaRiskLabel(risk: RulaRisk) {
  if (risk === "very_high") return "즉시 개선";
  if (risk === "high") return "빠른 개선 필요";
  if (risk === "medium") return "추가 관찰 필요";
  if (risk === "low") return "낮은 위험";
  return "측정 보완 필요";
}

function phaseCopy(phase: Phase) {
  if (phase === "requesting") return "카메라 권한을 확인하고 있어요";
  if (phase === "loading") return "다중 자세 인식 모델을 준비하고 있어요";
  if (phase === "running") return "최대 4명 실시간 분석 중";
  if (phase === "error") return "연결 확인 필요";
  return "분석 대기";
}

function serverStatusCopy(status: ServerStatus) {
  if (status === "online") return "AI 분석 서버 연결됨";
  if (status === "checking") return "AI 분석 서버 확인 중";
  if (status === "offline") return "브라우저 분석으로 계속 측정 중";
  return "AI 분석 서버 배포 대기";
}

function personCenter(person: Omit<DetectedPerson, "metrics">) {
  const shoulderMid = midpoint(person.landmarks[11], person.landmarks[12]);
  const hipMid = midpoint(person.landmarks[23], person.landmarks[24]);
  return (shoulderMid.x + hipMid.x) / 2;
}

function serverPersonCenter(person: ServerPersonAnalysis) {
  const landmarks = person.pose?.landmarks;
  if (!landmarks || landmarks.length < 25) return 0;
  return (
    (landmarks[11].x +
      landmarks[12].x +
      landmarks[23].x +
      landmarks[24].x) /
    4
  );
}

function createSessionAccumulator(): SessionAccumulator {
  const now = new Date();
  return {
    id: crypto.randomUUID(),
    startedAt: now.toISOString(),
    startedAtMs: now.getTime(),
    samples: 0,
    normalSamples: 0,
    cautionSamples: 0,
    dangerSamples: 0,
    confidenceTotal: 0,
    maxPeople: 0,
    metricMax: emptyMetricValues(),
    metricTotal: emptyMetricValues(),
    metricCounts: emptyMetricValues(),
    metricWarnings: emptyMetricValues(),
  };
}

function recordSessionSample(
  accumulator: SessionAccumulator,
  people: Metrics[],
) {
  if (people.length === 0) return;

  accumulator.samples += 1;
  accumulator.maxPeople = Math.max(accumulator.maxPeople, people.length);
  accumulator.confidenceTotal +=
    people.reduce((sum, person) => sum + person.confidence, 0) / people.length;

  const risk = highestRisk(people.map((person) => overallRisk(person)));
  if (risk === "danger") accumulator.dangerSamples += 1;
  else if (risk === "caution") accumulator.cautionSamples += 1;
  else accumulator.normalSamples += 1;

  for (const key of METRIC_KEYS) {
    const angles = people
      .map((person) => person[key].angle)
      .filter((angle): angle is number => angle !== null);
    if (angles.length > 0) {
      accumulator.metricMax[key] = Math.max(
        accumulator.metricMax[key],
        ...angles,
      );
      accumulator.metricTotal[key] += angles.reduce(
        (sum, angle) => sum + angle,
        0,
      );
      accumulator.metricCounts[key] += angles.length;
    }
    if (people.some((person) => person[key].risk !== "normal")) {
      accumulator.metricWarnings[key] += 1;
    }
  }
}

function finishSession(
  accumulator: SessionAccumulator,
): MeasurementSessionInput {
  const endedAt = new Date();
  const metricAverage = emptyMetricValues();
  for (const key of METRIC_KEYS) {
    metricAverage[key] =
      accumulator.metricCounts[key] > 0
        ? Math.round(
            (accumulator.metricTotal[key] / accumulator.metricCounts[key]) *
              10,
          ) / 10
        : 0;
  }

  return {
    id: accumulator.id,
    startedAt: accumulator.startedAt,
    endedAt: endedAt.toISOString(),
    durationSeconds: Math.max(
      1,
      Math.round((endedAt.getTime() - accumulator.startedAtMs) / 1000),
    ),
    maxPeople: accumulator.maxPeople,
    samples: accumulator.samples,
    normalSamples: accumulator.normalSamples,
    cautionSamples: accumulator.cautionSamples,
    dangerSamples: accumulator.dangerSamples,
    averageConfidence:
      accumulator.samples > 0
        ? Math.round(
            (accumulator.confidenceTotal / accumulator.samples) * 10,
          ) / 10
        : 0,
    metricMax: accumulator.metricMax,
    metricAverage,
    metricWarnings: accumulator.metricWarnings,
  };
}

export default function CameraPrototype() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const frameRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const lastInferenceRef = useRef(0);
  const lastUiUpdateRef = useRef(0);
  const lastServerAnalysisRef = useRef(0);
  const serverRequestRef = useRef<AbortController | null>(null);
  const sessionRef = useRef<SessionAccumulator | null>(null);

  const [phase, setPhase] = useState<Phase>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [peopleMetrics, setPeopleMetrics] = useState<Metrics[]>([]);
  const [selectedPerson, setSelectedPerson] = useState(0);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [activeCamera, setActiveCamera] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [serverStatus, setServerStatus] = useState<ServerStatus>(
    ANALYSIS_API_URL ? "checking" : "disabled",
  );
  const [serverAnalysis, setServerAnalysis] = useState<ServerAnalysis | null>(
    null,
  );

  const hasPose = peopleMetrics.length > 0;
  const selectedPersonIndex = Math.min(
    selectedPerson,
    Math.max(peopleMetrics.length - 1, 0),
  );
  const selectedMetrics = peopleMetrics[selectedPersonIndex] ?? EMPTY_METRICS;
  const sceneRisk = hasPose
    ? highestRisk(peopleMetrics.map((metrics) => overallRisk(metrics)))
    : "normal";

  useEffect(() => {
    if (!ANALYSIS_API_URL) return;

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 3000);

    fetch(`${ANALYSIS_API_URL}/health`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then((response) => {
        if (!response.ok) throw new Error("health check failed");
        setServerStatus("online");
      })
      .catch(() => setServerStatus("offline"))
      .finally(() => window.clearTimeout(timeout));

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      serverRequestRef.current?.abort();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  function stopTracks() {
    runningRef.current = false;
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }

  async function requestLegacyAnalysis(
    people: DetectedPerson[],
    timestampMs: number,
    signal: AbortSignal,
  ) {
    const serializeLandmarks = (landmarks: Point3D[]) =>
      landmarks.map((point) => ({
        x: point.x,
        y: point.y,
        z: point.z,
        visibility: point.visibility ?? 0,
      }));
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (ANALYSIS_API_KEY) headers["x-moveguard-key"] = ANALYSIS_API_KEY;

    const response = await fetch(`${ANALYSIS_API_URL}/v1/analyze`, {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify({
        timestamp_ms: timestampMs,
        people: people.map((person, index) => ({
          person_id: `person-${index + 1}`,
          landmarks: serializeLandmarks(person.landmarks),
          world_landmarks: serializeLandmarks(person.worldLandmarks),
        })),
      }),
    });
    if (!response.ok) throw new Error("legacy analysis failed");
    return (await response.json()) as ServerAnalysis;
  }

  async function captureAnalysisFrame(video: HTMLVideoElement) {
    if (!video.videoWidth || !video.videoHeight) return null;
    const scale = Math.min(1, 960 / video.videoWidth);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.78),
    );
  }

  async function analyzeWithServer(
    video: HTMLVideoElement,
    people: DetectedPerson[],
    timestampMs: number,
  ) {
    if (!ANALYSIS_API_URL || serverRequestRef.current) return;

    const controller = new AbortController();
    serverRequestRef.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 8000);

    try {
      const image = await captureAnalysisFrame(video);
      let analysis: ServerAnalysis;
      if (image) {
        const params = new URLSearchParams({
          timestamp_ms: String(Math.round(timestampMs)),
          frame_id: `${sessionRef.current?.id ?? "live"}-${Math.round(timestampMs)}`,
        });
        if (sessionRef.current?.id) {
          params.set("measurement_id", sessionRef.current.id);
        }
        const headers: Record<string, string> = {
          "content-type": "image/jpeg",
        };
        if (ANALYSIS_API_KEY) headers["x-moveguard-key"] = ANALYSIS_API_KEY;
        const response = await fetch(
          `${ANALYSIS_API_URL}/api/analysis/posture?${params}`,
          {
            method: "POST",
            headers,
            body: image,
            signal: controller.signal,
          },
        );
        if (response.ok) {
          analysis = (await response.json()) as ServerAnalysis;
        } else if ([404, 405, 415, 503].includes(response.status)) {
          analysis = await requestLegacyAnalysis(
            people,
            timestampMs,
            controller.signal,
          );
        } else {
          throw new Error("posture analysis failed");
        }
      } else {
        analysis = await requestLegacyAnalysis(
          people,
          timestampMs,
          controller.signal,
        );
      }
      analysis.people = [...analysis.people].sort(
        (first, second) => serverPersonCenter(second) - serverPersonCenter(first),
      );
      setServerAnalysis(analysis);
      setServerStatus("online");
    } catch {
      if (!controller.signal.aborted) setServerStatus("offline");
    } finally {
      window.clearTimeout(timeout);
      if (serverRequestRef.current === controller) {
        serverRequestRef.current = null;
      }
    }
  }

  function drawPeople(
    people: DetectedPerson[],
    width: number,
    height: number,
  ) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, width, height);
    context.lineCap = "round";
    context.lineJoin = "round";

    people.forEach((person, personIndex) => {
      const color = PERSON_COLORS[personIndex % PERSON_COLORS.length];
      context.strokeStyle = color;
      context.shadowBlur = 13;
      context.shadowColor = color;
      context.lineWidth = Math.max(3, width / 290);

      for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
        const start = person.landmarks[connection.start];
        const end = person.landmarks[connection.end];
        if ((start.visibility ?? 0) < 0.42 || (end.visibility ?? 0) < 0.42) {
          continue;
        }

        context.beginPath();
        context.moveTo(start.x * width, start.y * height);
        context.lineTo(end.x * width, end.y * height);
        context.stroke();
      }

      context.shadowBlur = 7;
      for (const landmark of person.landmarks) {
        if ((landmark.visibility ?? 0) < 0.45) continue;
        context.beginPath();
        context.arc(
          landmark.x * width,
          landmark.y * height,
          Math.max(3.5, width / 210),
          0,
          Math.PI * 2,
        );
        context.fillStyle = "#f8fffd";
        context.fill();
        context.lineWidth = 2;
        context.stroke();
      }
    });

    context.shadowBlur = 0;
  }

  function runDetection() {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const now = performance.now();

    if (
      video &&
      landmarker &&
      video.readyState >= 2 &&
      video.currentTime !== lastVideoTimeRef.current &&
      now - lastInferenceRef.current >= 70
    ) {
      lastVideoTimeRef.current = video.currentTime;
      lastInferenceRef.current = now;

      try {
        const result = landmarker.detectForVideo(video, now);
        const detectedPeople = result.landmarks
          .map((landmarks, index) => {
            const worldLandmarks =
              (result.worldLandmarks[index] as Point3D[] | undefined) ??
              landmarks;
            return {
              landmarks,
              worldLandmarks,
            };
          })
          .sort((first, second) => personCenter(second) - personCenter(first))
          .map((person) => ({
            ...person,
            metrics: buildMetrics(person.landmarks, person.worldLandmarks),
          }));

        if (detectedPeople.length > 0) {
          drawPeople(detectedPeople, video.videoWidth, video.videoHeight);
        } else {
          clearCanvas();
        }

        if (now - lastUiUpdateRef.current > 140) {
          lastUiUpdateRef.current = now;
          const nextMetrics = detectedPeople.map((person) => person.metrics);
          setPeopleMetrics(nextMetrics);
          if (sessionRef.current) {
            recordSessionSample(sessionRef.current, nextMetrics);
          }
        }

        if (
          detectedPeople.length > 0 &&
          now - lastServerAnalysisRef.current > 900
        ) {
          lastServerAnalysisRef.current = now;
          void analyzeWithServer(video, detectedPeople, now);
        }
      } catch {
        setErrorMessage(
          "자세 분석을 계속할 수 없어요. 카메라를 껐다가 다시 연결해 주세요.",
        );
        setPhase("error");
        stopTracks();
        return;
      }
    }

    frameRef.current = requestAnimationFrame(runDetection);
  }

  async function prepareLandmarker() {
    if (landmarkerRef.current) return;

    const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    const options = {
      runningMode: "VIDEO" as const,
      numPoses: MAX_PEOPLE,
      minPoseDetectionConfidence: 0.52,
      minPosePresenceConfidence: 0.52,
      minTrackingConfidence: 0.52,
    };

    try {
      landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
      });
    } catch {
      landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU",
        },
      });
    }
  }

  async function refreshCameraList(stream: MediaStream) {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((device) => device.kind === "videoinput");
    const currentId = stream.getVideoTracks()[0]?.getSettings().deviceId ?? "";
    setCameras(videoDevices);
    setActiveCamera(currentId);
  }

  async function connectCamera(deviceId?: string) {
    stopTracks();
    clearCanvas();
    sessionRef.current = null;
    setPeopleMetrics([]);
    setSelectedPerson(0);
    setErrorMessage("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage(
        "이 브라우저에서는 카메라를 사용할 수 없어요. 최신 Chrome 또는 Edge에서 열어 주세요.",
      );
      setPhase("error");
      return;
    }

    try {
      setPhase("requesting");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) throw new Error("video unavailable");

      video.srcObject = stream;
      await new Promise<void>((resolve) => {
        if (video.readyState >= 1) {
          resolve();
          return;
        }
        video.onloadedmetadata = () => resolve();
      });
      await video.play();
      await refreshCameraList(stream);

      setPhase("loading");
      await prepareLandmarker();
      setPhase("running");
      runningRef.current = true;
      lastVideoTimeRef.current = -1;
      sessionRef.current = createSessionAccumulator();
      runDetection();
    } catch (error) {
      stopTracks();
      const name = error instanceof DOMException ? error.name : "";

      if (name === "NotAllowedError") {
        setErrorMessage(
          "카메라 권한이 차단됐어요. 주소창의 카메라 아이콘을 눌러 허용으로 바꿔 주세요.",
        );
      } else if (name === "NotFoundError") {
        setErrorMessage(
          "사용 가능한 카메라를 찾지 못했어요. Windows 카메라 앱에서 먼저 작동하는지 확인해 주세요.",
        );
      } else if (name === "NotReadableError") {
        setErrorMessage(
          "다른 프로그램이 카메라를 사용하고 있어요. Zoom, Teams 또는 카메라 앱을 닫고 다시 시도해 주세요.",
        );
      } else {
        setErrorMessage(
          "카메라 또는 자세 인식 모델을 불러오지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요.",
        );
      }
      setPhase("error");
    }
  }

  async function stopCamera() {
    const completed =
      sessionRef.current && sessionRef.current.samples > 0
        ? finishSession(sessionRef.current)
        : null;

    stopTracks();
    clearCanvas();
    serverRequestRef.current?.abort();
    serverRequestRef.current = null;
    setPhase("idle");
    setPeopleMetrics([]);
    setSelectedPerson(0);
    setServerAnalysis(null);
    sessionRef.current = null;

    if (!completed) {
      window.location.href = "/dashboard";
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch("/api/sessions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(completed),
      });
      if (!response.ok) throw new Error("save failed");
      window.location.href = "/dashboard";
    } catch {
      setErrorMessage(
        "측정 결과를 저장하지 못했어요. 잠시 후 카메라를 다시 연결해 측정해 주세요.",
      );
      setPhase("error");
    } finally {
      setIsSaving(false);
    }
  }

  async function handleCameraChange(deviceId: string) {
    setActiveCamera(deviceId);
    await connectCamera(deviceId);
  }

  const isBusy = phase === "requesting" || phase === "loading" || isSaving;
  const mainMetrics = [
    selectedMetrics.neck,
    selectedMetrics.trunk,
    selectedMetrics.shoulder,
  ];
  const rotationMetrics = [
    selectedMetrics.headRotation,
    selectedMetrics.trunkRotation,
    selectedMetrics.leftWristRotation,
    selectedMetrics.rightWristRotation,
  ];
  const selectedServerPerson =
    serverAnalysis?.people[selectedPersonIndex] ?? null;
  const selectedServerRula = selectedServerPerson?.rula;
  const selectedServerQuality = selectedServerPerson?.quality;
  const isV2Analysis = serverAnalysis?.processor === "moveguard-pose-analysis-v2";

  function renderMetric(metric: Metric) {
    const visibleRisk = hasPose ? metric.risk : "idle";

    return (
      <article
        className={`metric-card metric-${visibleRisk} ${metric.experimental ? "metric-experimental" : ""}`}
        key={metric.label}
      >
        <div className="metric-topline">
          <span className="metric-icon" aria-hidden="true" />
          <span className="metric-status">
            {metric.experimental
              ? metric.angle === null
                ? "손 감지 필요"
                : "시험 추정"
              : hasPose
                ? riskLabel(metric.risk)
                : "대기"}
          </span>
        </div>
        <p>{metric.label}</p>
        <strong>
          {hasPose && metric.angle !== null
            ? `${Math.round(metric.angle)}°`
            : "—"}
        </strong>
        <small>{metric.guide}</small>
      </article>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            M
          </div>
          <div>
            <p className="brand-name">MoveGuard</p>
            <p className="brand-caption">카메라 자세 인식 시험판</p>
          </div>
        </div>
        <nav className="product-nav" aria-label="MoveGuard 주요 메뉴">
          <Link className="active" href="/" aria-current="page">
            실시간 측정
          </Link>
          <Link href="/video-analysis">영상 분석</Link>
          <Link href="/dashboard">분석 대시보드</Link>
        </nav>
        <div className="privacy-chip">
          <span className="privacy-dot" aria-hidden="true" />
          영상 저장 없음 · 관절 좌표만 서버 전송
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">MULTI-PERSON CAMERA PROTOTYPE</p>
          <h1>
            한 화면에서 최대 4명의
            <br />
            자세와 회전을 확인하세요
          </h1>
          <p className="hero-copy">
            사람마다 다른 색의 관절선을 표시하고, 선택한 사람의 목·몸통·어깨
            자세와 머리·몸통·손목 회전을 실시간으로 보여 줍니다.
          </p>
        </div>
        <div className="hero-statuses">
          <div className={`phase-badge phase-${phase}`} aria-live="polite">
            <span className="phase-pulse" />
            {phaseCopy(phase)}
          </div>
          <div
            className={`server-badge server-${serverStatus}`}
            aria-live="polite"
          >
            <span />
            {serverStatusCopy(serverStatus)}
          </div>
        </div>
      </section>

      <section className="workspace">
        <div className="camera-panel">
          <div className="camera-toolbar">
            <div>
              <p className="panel-kicker">LIVE MULTI CAMERA</p>
              <h2>여러 사람 자세 분석 화면</h2>
            </div>
            <div className="toolbar-actions">
              {cameras.length > 0 && (
                <label className="camera-select-label">
                  <span>카메라</span>
                  <select
                    value={activeCamera}
                    onChange={(event) => handleCameraChange(event.target.value)}
                    disabled={isBusy}
                    aria-label="사용할 카메라 선택"
                  >
                    {cameras.map((camera, index) => (
                      <option key={camera.deviceId} value={camera.deviceId}>
                        {camera.label || `카메라 ${index + 1}`}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {phase === "running" ? (
                <button
                  className="button button-stop"
                  onClick={stopCamera}
                  disabled={isSaving}
                >
                  {isSaving ? "저장 중…" : "측정 종료·저장"}
                </button>
              ) : (
                <button
                  className="button button-primary"
                  onClick={() => connectCamera()}
                  disabled={isBusy}
                >
                  {isBusy ? "준비 중…" : "카메라 연결"}
                </button>
              )}
            </div>
          </div>

          <div className={`camera-stage risk-${sceneRisk}`}>
            <video ref={videoRef} playsInline muted />
            <canvas ref={canvasRef} aria-hidden="true" />

            {phase !== "running" && (
              <div className="camera-placeholder">
                <div className="camera-glyph" aria-hidden="true">
                  <span />
                </div>
                {phase === "error" ? (
                  <>
                    <h3>카메라 연결을 확인해 주세요</h3>
                    <p>{errorMessage}</p>
                    <button
                      className="button button-light"
                      onClick={() => connectCamera()}
                    >
                      다시 연결
                    </button>
                  </>
                ) : isBusy ? (
                  <>
                    <div className="loading-ring" aria-hidden="true" />
                    <h3>{phaseCopy(phase)}</h3>
                    <p>처음 연결할 때는 준비에 잠시 시간이 걸릴 수 있어요.</p>
                  </>
                ) : (
                  <>
                    <h3>카메라가 아직 연결되지 않았어요</h3>
                    <p>
                      위의 카메라 연결 버튼을 누르고 브라우저 권한을 허용해
                      주세요.
                    </p>
                  </>
                )}
              </div>
            )}

            {phase === "running" && !hasPose && (
              <div className="pose-guide">
                <span className="guide-frame guide-frame-wide" />
                <div>
                  <strong>사람들이 화면 안에 들어오게 해 주세요</strong>
                  <span>최대 4명 · 서로 겹치지 않으면 더 잘 인식해요</span>
                </div>
              </div>
            )}

            {phase === "running" && hasPose && (
              <>
                <div className={`live-alert alert-${sceneRisk}`}>
                  <span>{riskLabel(sceneRisk)}</span>
                  {sceneRisk === "normal"
                    ? "현재 인식된 사람들의 자세가 안정적이에요"
                    : sceneRisk === "caution"
                      ? "시험 기준을 넘은 자세가 있어요"
                      : "시험 기준상 위험 자세가 감지됐어요"}
                </div>
                <div className="people-badge">
                  {peopleMetrics.length}명 동시 분석
                </div>
              </>
            )}
          </div>

          <div className="camera-foot">
            <div>
              <span className="step-number">1</span>
              <p>
                <strong>전신이 보이게</strong>
                머리부터 발목까지 화면 안에 넣어 주세요.
              </p>
            </div>
            <div>
              <span className="step-number">2</span>
              <p>
                <strong>서로 겹치지 않게</strong>
                사람 사이에 한 걸음 정도 간격을 두면 좋아요.
              </p>
            </div>
            <div>
              <span className="step-number">3</span>
              <p>
                <strong>손바닥도 보이게</strong>
                손목 회전은 손이 가려지면 표시되지 않아요.
              </p>
            </div>
          </div>
        </div>

        <aside className="analysis-panel">
          <div className="analysis-heading">
            <div>
              <p className="panel-kicker">REAL-TIME VALUES</p>
              <h2>사람별 실시간 측정</h2>
            </div>
            <span className={`detection-chip ${hasPose ? "detected" : ""}`}>
              {hasPose ? `${peopleMetrics.length}명 인식 중` : "인식 대기"}
            </span>
          </div>

          {hasPose && (
            <div className="person-picker-wrap">
              <p>확인할 사람을 선택하세요</p>
              <div className="person-picker" role="group" aria-label="분석할 사람">
                {peopleMetrics.map((_, index) => (
                  <button
                    className={selectedPersonIndex === index ? "selected" : ""}
                    key={index}
                    onClick={() => setSelectedPerson(index)}
                    type="button"
                  >
                    <span
                      aria-hidden="true"
                      style={{ backgroundColor: PERSON_COLORS[index] }}
                    />
                    사람 {index + 1}
                  </button>
                ))}
              </div>
              <small>
                번호는 화면 왼쪽→오른쪽 순서예요. 자리를 바꾸면 번호도 바뀔 수
                있어요.
              </small>
            </div>
          )}

          <p className="metric-group-title">기울기와 어깨</p>
          <div className="metric-list">{mainMetrics.map(renderMetric)}</div>

          <div className="rotation-title">
            <p className="metric-group-title">3D 회전 추정</p>
            <span>시험 기능</span>
          </div>
          <div className="metric-list metric-list-rotation">
            {rotationMetrics.map(renderMetric)}
          </div>

          <div className="quality-card">
            <div className="quality-row">
              <span>선택한 사람 인식 신뢰도</span>
              <strong>
                {hasPose ? `${selectedMetrics.confidence}%` : "—"}
              </strong>
            </div>
            <div className="confidence-track" aria-hidden="true">
              <span
                style={{
                  width: hasPose
                    ? `${Math.max(4, selectedMetrics.confidence)}%`
                    : "0%",
                }}
              />
            </div>
            <div className="quality-row quality-secondary">
              <span>전신 프레임</span>
              <strong>
                {!hasPose
                  ? "대기"
                  : selectedMetrics.fullBody
                    ? "양호"
                    : "조정 필요"}
              </strong>
            </div>
          </div>

          <div className={`server-card server-${serverStatus}`}>
            <div className="server-card-heading">
              <div>
                <span className="server-card-dot" aria-hidden="true" />
                <strong>AI 자세·RULA 분석 서버</strong>
              </div>
              <span>{serverStatusCopy(serverStatus)}</span>
            </div>
            {serverStatus === "online" && selectedServerPerson ? (
              <>
                <p>
                  {selectedServerRula?.valid ? (
                    <>
                      선택한 사람 RULA{" "}
                      <strong>
                        {selectedServerRula.final_score}점 ·{" "}
                        {rulaRiskLabel(selectedServerRula.risk_level)}
                      </strong>
                    </>
                  ) : selectedServerPerson.overall_risk ? (
                    <>
                      기존 관절 서버 판정{" "}
                      <strong>
                        {riskLabel(selectedServerPerson.overall_risk)}
                      </strong>
                    </>
                  ) : (
                    <>
                      정밀 RULA <strong>측정 보완 필요</strong>
                    </>
                  )}
                </p>
                {isV2Analysis && (
                  <div className="server-rula-meta">
                    <span>
                      자세 {selectedServerQuality?.pose_valid ? "검출" : "확인 필요"}
                    </span>
                    <span>
                      손 {selectedServerQuality?.hand_valid ? "검출" : "확인 필요"}
                    </span>
                    <span>객체 {serverAnalysis?.objects?.length ?? 0}개</span>
                  </div>
                )}
                <small>
                  사람 {serverAnalysis?.people.length ?? 0}명 AI 재분석 ·{" "}
                  {serverAnalysis?.elapsed_ms ?? 0}ms · 원본 영상 저장 없음
                </small>
              </>
            ) : (
              <>
                <p>
                  기존 브라우저 측정은 서버와 관계없이 계속 사용할 수 있어요.
                </p>
                <small>
                  서버가 연결되면 프레임을 샘플링해 전신·손목·RULA를 다시
                  검증합니다.
                </small>
              </>
            )}
          </div>

          <div className="notice-card">
            <span className="notice-mark" aria-hidden="true">
              i
            </span>
            <p>
              <strong>회전값과 위험 기준은 시험용입니다.</strong>
              노트북 카메라 한 대로 계산한 추정치예요. 특히 손목 비틀림은
              손가락 점을 이용한 대략적인 값이며 정식 산업안전 또는 의료
              판단에 사용할 수 없습니다.
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
