from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2

from .ground_truth import (
    GroundTruthStore,
    load_ground_truth,
    write_annotation_csv,
    write_annotation_json,
)
from .metrics import build_report
from .variants import LiveVariant, build_default_variants, planned_ablation_variants


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        return None


def _metric_cell(metric: dict[str, Any]) -> Any:
    return metric.get("value") if metric.get("status") == "measured" else ""


def _metric_status(metric: dict[str, Any]) -> str:
    return str(metric.get("status", "unknown"))


def _metric_reason(metric: dict[str, Any]) -> str:
    return str(metric.get("reason", ""))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _overall_rows(report: dict[str, Any], baseline_id: str, hybrid_id: str) -> list[dict[str, Any]]:
    baseline = report["variants"][baseline_id]
    hybrid = report["variants"][hybrid_id]
    definitions = [
        (
            "Joint detection rate",
            baseline["joint_detection_overall"],
            hybrid["joint_detection_overall"],
            report["comparison"]["joint_detection_rate"],
        ),
        (
            "Joint position PCK@0.2",
            baseline["joint_position_overall"]["pck_at_0_2"],
            hybrid["joint_position_overall"]["pck_at_0_2"],
            report["comparison"]["joint_position_pck"],
        ),
        (
            "Joint angle MAE",
            baseline["joint_angles_overall"]["mae"],
            hybrid["joint_angles_overall"]["mae"],
            report["comparison"]["joint_angle_mae"],
        ),
        (
            "Posture classification accuracy",
            baseline["posture_classification"]["accuracy"],
            hybrid["posture_classification"]["accuracy"],
            report["comparison"]["posture_classification_accuracy"],
        ),
        (
            "RULA exact accuracy",
            baseline["rula"]["exact_accuracy"],
            hybrid["rula"]["exact_accuracy"],
            report["comparison"]["rula_exact_accuracy"],
        ),
        (
            "RULA ±1 accuracy",
            baseline["rula"]["within_one_accuracy"],
            hybrid["rula"]["within_one_accuracy"],
            report["comparison"]["rula_within_one_accuracy"],
        ),
        (
            "RULA risk level accuracy",
            baseline["rula"]["risk"]["accuracy"],
            hybrid["rula"]["risk"]["accuracy"],
            report["comparison"]["rula_risk_accuracy"],
        ),
    ]
    rows = []
    for name, base_metric, hybrid_metric, improvement in definitions:
        rows.append(
            {
                "metric": name,
                "baseline": _metric_cell(base_metric),
                "hybrid": _metric_cell(hybrid_metric),
                "improvement": _metric_cell(improvement),
                "unit": hybrid_metric.get("unit", ""),
                "baseline_status": _metric_status(base_metric),
                "hybrid_status": _metric_status(hybrid_metric),
                "reason": _metric_reason(hybrid_metric) or _metric_reason(base_metric),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    *,
    baseline_id: str,
    hybrid_id: str,
    ground_truth: GroundTruthStore,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_frame.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            baseline = record["results"][baseline_id]
            hybrid = record["results"][hybrid_id]
            gt_row = ground_truth.find(
                video_id=record["video_id"],
                frame_index=record["frame_index"],
                frame_id=record["frame_id"],
                person_id=hybrid.canonical.person_id or baseline.canonical.person_id,
            )
            handle.write(
                json.dumps(
                    {
                        "video_id": record["video_id"],
                        "frame_id": record["frame_id"],
                        "frame_index": record["frame_index"],
                        "timestamp_ms": record["timestamp_ms"],
                        "baseline": baseline.as_dict(),
                        "hybrid": hybrid.as_dict(),
                        "ground_truth": gt_row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    overall_rows = _overall_rows(report, baseline_id, hybrid_id)
    _write_csv(
        output_dir / "overall_metrics.csv",
        [
            "metric",
            "baseline",
            "hybrid",
            "improvement",
            "unit",
            "baseline_status",
            "hybrid_status",
            "reason",
        ],
        overall_rows,
    )

    joint_rows = []
    for joint, values in report["joint_comparison"].items():
        base_metric = values[baseline_id]
        hybrid_metric = values[hybrid_id]
        improvement = values["improvement"]
        joint_rows.append(
            {
                "joint": joint,
                "baseline_percent": _metric_cell(base_metric),
                "hybrid_percent": _metric_cell(hybrid_metric),
                "improvement_pp": _metric_cell(improvement),
                "baseline_detected": base_metric.get("detected", ""),
                "hybrid_detected": hybrid_metric.get("detected", ""),
                "expected": base_metric.get("n", ""),
            }
        )
    _write_csv(
        output_dir / "joint_detection.csv",
        [
            "joint",
            "baseline_percent",
            "hybrid_percent",
            "improvement_pp",
            "baseline_detected",
            "hybrid_detected",
            "expected",
        ],
        joint_rows,
    )

    angle_rows = []
    baseline_angles = report["variants"][baseline_id]["joint_angles"]
    hybrid_angles = report["variants"][hybrid_id]["joint_angles"]
    for angle in baseline_angles:
        base_metric = baseline_angles[angle]
        hybrid_metric = hybrid_angles[angle]
        improvement = (
            base_metric["value"] - hybrid_metric["value"]
            if base_metric["status"] == "measured" and hybrid_metric["status"] == "measured"
            else ""
        )
        angle_rows.append(
            {
                "joint_angle": angle,
                "baseline_mae_deg": _metric_cell(base_metric),
                "hybrid_mae_deg": _metric_cell(hybrid_metric),
                "improvement_deg": improvement,
                "baseline_status": _metric_status(base_metric),
                "hybrid_status": _metric_status(hybrid_metric),
            }
        )
    _write_csv(
        output_dir / "joint_angle_accuracy.csv",
        [
            "joint_angle",
            "baseline_mae_deg",
            "hybrid_mae_deg",
            "improvement_deg",
            "baseline_status",
            "hybrid_status",
        ],
        angle_rows,
    )

    performance_rows = []
    for variant_id in (baseline_id, hybrid_id):
        perf = report["variants"][variant_id]["performance"]
        performance_rows.append(
            {
                "variant": variant_id,
                "average_latency_ms": _metric_cell(perf["average_latency_ms"]),
                "p50_latency_ms": _metric_cell(perf["p50_latency_ms"]),
                "p95_latency_ms": _metric_cell(perf["p95_latency_ms"]),
                "fps": _metric_cell(perf["fps"]),
                "process_cpu_percent": _metric_cell(perf["process_cpu_percent"]),
                "memory_mb": _metric_cell(perf["memory_mb"]),
                "gpu_usage_percent": _metric_cell(perf["gpu_usage_percent"]),
                "memory_status": _metric_status(perf["memory_mb"]),
                "gpu_status": _metric_status(perf["gpu_usage_percent"]),
            }
        )
    _write_csv(
        output_dir / "performance.csv",
        [
            "variant",
            "average_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "fps",
            "process_cpu_percent",
            "memory_mb",
            "gpu_usage_percent",
            "memory_status",
            "gpu_status",
        ],
        performance_rows,
    )

    ablation_rows = [
        {
            "id": item["id"],
            "configuration": item["configuration"],
            "available": item["available"],
            "evaluated_as": item.get("evaluated_as", ""),
            "reason": item.get("reason", ""),
        }
        for item in report["ablation"]
    ]
    _write_csv(
        output_dir / "ablation.csv",
        ["id", "configuration", "available", "evaluated_as", "reason"],
        ablation_rows,
    )

    subgroup_rows = []
    for variant_id in (baseline_id, hybrid_id):
        for row in report["variants"][variant_id]["subgroups"]:
            subgroup_rows.append(
                {
                    "variant": variant_id,
                    "condition": row["condition"],
                    "value": row["value"],
                    "frames": row["frames"],
                    "joint_detection_percent": _metric_cell(row["joint_detection_rate"]),
                    "rula_exact_accuracy_percent": _metric_cell(row["rula_exact_accuracy"]),
                    "rula_status": _metric_status(row["rula_exact_accuracy"]),
                }
            )
    _write_csv(
        output_dir / "condition_analysis.csv",
        [
            "variant",
            "condition",
            "value",
            "frames",
            "joint_detection_percent",
            "rula_exact_accuracy_percent",
            "rula_status",
        ],
        subgroup_rows,
    )
    write_annotation_csv(output_dir / "ground_truth_annotation.csv", annotation_rows)
    write_annotation_json(output_dir / "ground_truth_annotation.json", annotation_rows)


def _first_sample(video_path: Path, sample_every: int) -> Any | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_index = 0
        while capture.isOpened():
            ok, frame_bgr = capture.read()
            if not ok:
                return None
            if frame_index % sample_every == 0:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    return None


def _warm_up(variant: LiveVariant, frame_rgb: Any, variant_name: str) -> dict[str, Any]:
    rss_before = _rss_mb()
    started = perf_counter()
    result = variant.analyze(
        frame_rgb,
        frame_id=f"warmup-{variant_name}",
        frame_index=-1,
        timestamp_ms=0.0,
    )
    rss_after = _rss_mb()
    return {
        "elapsed_ms": round((perf_counter() - started) * 1000, 4),
        "issues": list(result.issues),
        "excluded_from_metrics": True,
        "process_rss_after_mb": None if rss_after is None else round(rss_after, 4),
        "incremental_model_load_rss_mb": (
            None
            if rss_before is None or rss_after is None
            else round(max(0.0, rss_after - rss_before), 4)
        ),
        "memory_definition": "Incremental process RSS while lazily loading and warming this variant.",
    }


def run_experiment(
    video_paths: list[Path],
    *,
    ground_truth_path: Path | None,
    output_dir: Path,
    sample_every: int,
    max_frames: int,
) -> dict[str, Any]:
    baseline, hybrid = build_default_variants()
    baseline_id = baseline.variant_id
    hybrid_id = hybrid.variant_id
    ground_truth = load_ground_truth(ground_truth_path)

    first_frame = _first_sample(video_paths[0], sample_every)
    if first_frame is None:
        raise RuntimeError(f"No readable frame in {video_paths[0]}")
    warmup = {
        baseline_id: _warm_up(baseline, first_frame, baseline_id),
        hybrid_id: _warm_up(hybrid, first_frame, hybrid_id),
    }

    records: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    sampled_total = 0
    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_index = 0
        video_id = video_path.stem
        try:
            while capture.isOpened():
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % sample_every != 0:
                    frame_index += 1
                    continue
                if max_frames and sampled_total >= max_frames:
                    break
                timestamp_ms = 1000 * frame_index / fps if fps else float(frame_index)
                frame_id = f"{video_id}:{frame_index:06d}"
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                ordered = (baseline, hybrid) if sampled_total % 2 == 0 else (hybrid, baseline)
                results = {
                    variant.variant_id: variant.analyze(
                        frame_rgb,
                        frame_id=frame_id,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                    for variant in ordered
                }
                records.append(
                    {
                        "video_id": video_id,
                        "video_path": str(video_path.resolve()),
                        "frame_id": frame_id,
                        "frame_index": frame_index,
                        "timestamp_ms": round(timestamp_ms, 4),
                        "width_px": width,
                        "height_px": height,
                        "results": results,
                    }
                )
                annotation_rows.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "frame_index": frame_index,
                        "timestamp_ms": round(timestamp_ms, 4),
                        "person_id": "person-1",
                        "width_px": width,
                        "height_px": height,
                    }
                )
                sampled_total += 1
                frame_index += 1
        finally:
            capture.release()
        if max_frames and sampled_total >= max_frames:
            break

    if not records:
        raise RuntimeError("No frames were sampled from the input videos.")

    experiment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videos": [str(path.resolve()) for path in video_paths],
        "sample_every_n_frames": sample_every,
        "sampled_frames": len(records),
        "preprocessing": "OpenCV decode -> BGR-to-RGB only; no resize or crop",
        "execution_order": "alternated per frame to reduce fixed order bias",
        "warmup": warmup,
        "angle_engine_shared": "analysis_v2.angle_calculator.calculate_joint_angles",
        "rula_engine_shared": "analysis_v2.rula_engine.calculate_rula",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "ground_truth_path": str(ground_truth_path.resolve()) if ground_truth_path else None,
    }
    variant_status = {
        baseline_id: baseline.status(),
        hybrid_id: hybrid.status(),
    }
    report = build_report(
        records,
        baseline_id=baseline_id,
        hybrid_id=hybrid_id,
        ground_truth=ground_truth,
        experiment=experiment,
        variant_status=variant_status,
        ablation=planned_ablation_variants(),
    )
    for variant_id in (baseline_id, hybrid_id):
        memory = warmup[variant_id]["incremental_model_load_rss_mb"]
        if memory is not None:
            report["variants"][variant_id]["performance"]["memory_mb"] = {
                "status": "measured",
                "value": memory,
                "unit": "MB",
                "n": 1,
                "definition": warmup[variant_id]["memory_definition"],
            }
    write_outputs(
        output_dir,
        report,
        records,
        annotation_rows,
        baseline_id=baseline_id,
        hybrid_id=hybrid_id,
        ground_truth=ground_truth,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MoveGuard Baseline and available Hybrid on identical frames."
    )
    parser.add_argument("--video", action="append", required=True, help="Input video path; repeat for multiple videos.")
    parser.add_argument("--ground-truth", help="Optional Ground Truth CSV or JSON.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum sampled frames; 0 means all.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_every < 1:
        raise ValueError("--sample-every must be at least 1.")
    videos = [Path(value).expanduser().resolve() for value in args.video]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Video not found: {', '.join(missing)}")
    report = run_experiment(
        videos,
        ground_truth_path=(
            Path(args.ground_truth).expanduser().resolve() if args.ground_truth else None
        ),
        output_dir=Path(args.output).expanduser().resolve(),
        sample_every=args.sample_every,
        max_frames=max(0, args.max_frames),
    )
    summary = {
        "output": str(Path(args.output).expanduser().resolve()),
        "sampled_frames": report["experiment"]["sampled_frames"],
        "ground_truth_provided": report["ground_truth"]["provided"],
        "accuracy_claim_allowed": report["conclusion"]["accuracy_claim_allowed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
