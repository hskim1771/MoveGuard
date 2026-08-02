from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .canonical import JOINT_GROUPS
from .ground_truth import GroundTruthStore, integer, number, truthy
from .variants import VariantResult

HIGHER_IS_BETTER = {
    "joint_detection_rate",
    "joint_position_pck",
    "posture_classification_accuracy",
    "rula_exact_accuracy",
    "rula_within_one_accuracy",
    "rula_risk_accuracy",
    "fps",
}

ANGLE_SOURCES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "neck": (("neck", None),),
    "trunk": (("trunk", None),),
    "shoulder": (("left_upper_arm", None), ("right_upper_arm", None)),
    "elbow": (("left_lower_arm", None), ("right_lower_arm", None)),
    "wrist": (("left_wrist", None), ("right_wrist", None)),
    "knee": (("legs", "left_knee_angle"), ("legs", "right_knee_angle")),
}

GT_ANGLE_FIELDS = {
    "neck": ("neck_angle_deg",),
    "trunk": ("trunk_angle_deg",),
    "shoulder": ("left_upper_arm_angle_deg", "right_upper_arm_angle_deg"),
    "elbow": ("left_lower_arm_angle_deg", "right_lower_arm_angle_deg"),
    "wrist": ("left_wrist_angle_deg", "right_wrist_angle_deg"),
    "knee": ("left_knee_angle_deg", "right_knee_angle_deg"),
}

RULA_PARTS = {
    "neck": ("neck_score", "rula_neck_score"),
    "trunk": ("trunk_score", "rula_trunk_score"),
    "upper_arm": ("upper_arm_score", "rula_upper_arm_score"),
    "lower_arm": ("lower_arm_score", "rula_lower_arm_score"),
    "wrist": ("wrist_score", "rula_wrist_score"),
    "leg": ("leg_score", "rula_leg_score"),
}


def measured(value: float, *, unit: str, n: int, **extra: Any) -> dict[str, Any]:
    return {
        "status": "measured",
        "value": round(float(value), 4),
        "unit": unit,
        "n": int(n),
        **extra,
    }


def not_measurable(reason: str, *, n: int = 0, unit: str = "") -> dict[str, Any]:
    return {
        "status": "not_measurable",
        "value": None,
        "unit": unit,
        "n": int(n),
        "reason": reason,
    }


def _difference(
    baseline: dict[str, Any],
    hybrid: dict[str, Any],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if baseline.get("status") != "measured" or hybrid.get("status") != "measured":
        return not_measurable("Both variants must be measurable.", unit=hybrid.get("unit", ""))
    raw = hybrid["value"] - baseline["value"]
    improvement = raw if higher_is_better else -raw
    return measured(
        improvement,
        unit=hybrid.get("unit", ""),
        n=min(int(baseline.get("n", 0)), int(hybrid.get("n", 0))),
        raw_hybrid_minus_baseline=round(raw, 4),
    )


def _variant_results(records: Iterable[dict[str, Any]], variant_id: str) -> list[VariantResult]:
    return [record["results"][variant_id] for record in records]


def joint_detection_metrics(
    records: list[dict[str, Any]], variant_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results = _variant_results(records, variant_id)
    rows: dict[str, dict[str, Any]] = {}
    total_hits = 0
    total_expected = 0
    for group, joint_names in JOINT_GROUPS.items():
        expected = len(results) * len(joint_names)
        hits = sum(
            1
            for result in results
            for name in joint_names
            if result.canonical.joints[name].valid
        )
        total_hits += hits
        total_expected += expected
        rows[group] = measured(
            100 * hits / expected if expected else 0.0,
            unit="%",
            n=expected,
            detected=hits,
        )

    wrist_expected = len(results) * 2
    wrist_hits = sum(
        int(result.canonical.wrist_angle_ready[side])
        for result in results
        for side in ("left", "right")
    )
    rows["wrist_angle_ready"] = measured(
        100 * wrist_hits / wrist_expected if wrist_expected else 0.0,
        unit="%",
        n=wrist_expected,
        detected=wrist_hits,
        definition="Pose wrist plus hand axis available for wrist-angle calculation.",
    )
    overall = measured(
        100 * total_hits / total_expected if total_expected else 0.0,
        unit="%",
        n=total_expected,
        detected=total_hits,
    )
    return rows, overall


def _matched_gt(record: dict[str, Any], gt: GroundTruthStore, result: VariantResult) -> dict[str, Any] | None:
    return gt.find(
        video_id=record["video_id"],
        frame_index=record["frame_index"],
        frame_id=result.frame_id,
        person_id=result.canonical.person_id,
    )


def _gt_joint(row: dict[str, Any], name: str) -> tuple[float, float] | None:
    x = number(row.get(f"{name}_x"))
    y = number(row.get(f"{name}_y"))
    visible = truthy(row.get(f"{name}_visible"))
    if x is None or y is None or visible is False:
        return None
    return x, y


def _normalization_scale(row: dict[str, Any], width: float, height: float) -> float | None:
    neck = _gt_joint(row, "neck")
    trunk = _gt_joint(row, "trunk")
    if neck and trunk:
        scale = 2 * math.hypot((neck[0] - trunk[0]) * width, (neck[1] - trunk[1]) * height)
        if scale > 1e-6:
            return scale
    left = _gt_joint(row, "left_shoulder")
    right = _gt_joint(row, "right_shoulder")
    if left and right:
        scale = math.hypot((left[0] - right[0]) * width, (left[1] - right[1]) * height)
        if scale > 1e-6:
            return scale
    return None


def position_metrics(
    records: list[dict[str, Any]], variant_id: str, gt: GroundTruthStore
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not gt.available:
        reason = "Ground Truth joint coordinates are not provided."
        return (
            {group: not_measurable(reason) for group in JOINT_GROUPS},
            {
                "pixel_error_mean": not_measurable(reason, unit="px"),
                "normalized_joint_error_mean": not_measurable(reason),
                "pck_at_0_2": not_measurable(reason, unit="%"),
                "mpjpe": not_measurable(reason, unit="mm"),
            },
        )

    grouped_errors: dict[str, list[float]] = defaultdict(list)
    grouped_normalized: dict[str, list[float]] = defaultdict(list)
    grouped_pck: dict[str, list[float]] = defaultdict(list)
    all_errors: list[float] = []
    all_normalized: list[float] = []
    all_pck: list[float] = []
    mpjpe_mm: list[float] = []

    for record in records:
        result = record["results"][variant_id]
        row = _matched_gt(record, gt, result)
        if row is None:
            continue
        width = number(row.get("width_px")) or float(record["width_px"])
        height = number(row.get("height_px")) or float(record["height_px"])
        scale = _normalization_scale(row, width, height)
        for group, names in JOINT_GROUPS.items():
            for name in names:
                target = _gt_joint(row, name)
                if target is None:
                    continue
                prediction = result.canonical.joints[name]
                if not prediction.valid or prediction.x is None or prediction.y is None:
                    if scale is not None:
                        grouped_pck[group].append(0.0)
                        all_pck.append(0.0)
                    continue
                error = math.hypot(
                    (prediction.x - target[0]) * width,
                    (prediction.y - target[1]) * height,
                )
                grouped_errors[group].append(error)
                all_errors.append(error)
                if scale is not None:
                    normalized = error / scale
                    grouped_normalized[group].append(normalized)
                    grouped_pck[group].append(1.0 if normalized <= 0.2 else 0.0)
                    all_normalized.append(normalized)
                    all_pck.append(1.0 if normalized <= 0.2 else 0.0)

                if str(row.get("coordinate_space", "")).lower() == "world_m":
                    gt_world = tuple(number(row.get(f"{name}_world_{axis}")) for axis in "xyz")
                    pred_world = (prediction.world_x, prediction.world_y, prediction.world_z)
                    if all(value is not None for value in gt_world + pred_world):
                        mpjpe_mm.append(
                            1000
                            * math.sqrt(
                                sum(
                                    (float(pred) - float(target_value)) ** 2
                                    for pred, target_value in zip(pred_world, gt_world)
                                )
                            )
                        )

    group_rows: dict[str, dict[str, Any]] = {}
    for group in JOINT_GROUPS:
        if not grouped_errors[group]:
            group_rows[group] = not_measurable(
                "No matched visible Ground Truth coordinates.", unit="px"
            )
            continue
        group_rows[group] = measured(
            statistics.fmean(grouped_errors[group]),
            unit="px",
            n=len(grouped_errors[group]),
            normalized_error_mean=(
                round(statistics.fmean(grouped_normalized[group]), 4)
                if grouped_normalized[group]
                else None
            ),
            pck_at_0_2_percent=(
                round(100 * statistics.fmean(grouped_pck[group]), 4)
                if grouped_pck[group]
                else None
            ),
        )

    overall = {
        "pixel_error_mean": (
            measured(statistics.fmean(all_errors), unit="px", n=len(all_errors))
            if all_errors
            else not_measurable("No matched visible Ground Truth coordinates.", unit="px")
        ),
        "normalized_joint_error_mean": (
            measured(statistics.fmean(all_normalized), unit="ratio", n=len(all_normalized))
            if all_normalized
            else not_measurable("No valid torso/shoulder normalization reference.")
        ),
        "pck_at_0_2": (
            measured(100 * statistics.fmean(all_pck), unit="%", n=len(all_pck))
            if all_pck
            else not_measurable("No normalized Ground Truth joints.", unit="%")
        ),
        "mpjpe": (
            measured(statistics.fmean(mpjpe_mm), unit="mm", n=len(mpjpe_mm))
            if mpjpe_mm
            else not_measurable(
                "World-coordinate Ground Truth in meters is not provided.", unit="mm"
            )
        ),
    }
    return group_rows, overall


def _measurement_angle(result: VariantResult, name: str, auxiliary: str | None) -> float | None:
    measurement = result.measurements.get(name)
    if not measurement or not measurement.get("valid"):
        return None
    return number(measurement.get(auxiliary or "angle"))


def angle_metrics(
    records: list[dict[str, Any]], variant_id: str, gt: GroundTruthStore
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reason = "Expert or Ground Truth joint angles are not provided."
    if not gt.available:
        return (
            {group: not_measurable(reason, unit="deg") for group in ANGLE_SOURCES},
            {
                "mae": not_measurable(reason, unit="deg"),
                "rmse": not_measurable(reason, unit="deg"),
                "median_absolute_error": not_measurable(reason, unit="deg"),
            },
        )

    errors_by_group: dict[str, list[float]] = defaultdict(list)
    for record in records:
        result = record["results"][variant_id]
        row = _matched_gt(record, gt, result)
        if row is None:
            continue
        for group, prediction_sources in ANGLE_SOURCES.items():
            for (measurement_name, auxiliary), gt_field in zip(
                prediction_sources, GT_ANGLE_FIELDS[group]
            ):
                target = number(row.get(gt_field))
                prediction = _measurement_angle(result, measurement_name, auxiliary)
                if target is not None and prediction is not None:
                    errors_by_group[group].append(prediction - target)

    rows: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []
    for group in ANGLE_SOURCES:
        errors = errors_by_group[group]
        all_errors.extend(errors)
        if not errors:
            rows[group] = not_measurable(reason, unit="deg")
            continue
        absolute = [abs(error) for error in errors]
        rows[group] = measured(
            statistics.fmean(absolute),
            unit="deg",
            n=len(errors),
            rmse=round(math.sqrt(statistics.fmean(error * error for error in errors)), 4),
            median_absolute_error=round(statistics.median(absolute), 4),
        )
    if not all_errors:
        return rows, {
            "mae": not_measurable(reason, unit="deg"),
            "rmse": not_measurable(reason, unit="deg"),
            "median_absolute_error": not_measurable(reason, unit="deg"),
        }
    absolute = [abs(error) for error in all_errors]
    return rows, {
        "mae": measured(statistics.fmean(absolute), unit="deg", n=len(all_errors)),
        "rmse": measured(
            math.sqrt(statistics.fmean(error * error for error in all_errors)),
            unit="deg",
            n=len(all_errors),
        ),
        "median_absolute_error": measured(
            statistics.median(absolute), unit="deg", n=len(all_errors)
        ),
    }


def _classification_metrics(targets: list[str], predictions: list[str]) -> dict[str, Any]:
    labels = sorted(set(targets) | set(predictions))
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1
    accuracy = sum(a == p for a, p in zip(targets, predictions)) / len(targets)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": measured(100 * accuracy, unit="%", n=len(targets)),
        "precision_macro": measured(100 * statistics.fmean(precisions), unit="%", n=len(targets)),
        "recall_macro": measured(100 * statistics.fmean(recalls), unit="%", n=len(targets)),
        "f1_macro": measured(100 * statistics.fmean(f1s), unit="%", n=len(targets)),
        "confusion_matrix": matrix,
    }


def posture_metrics(
    records: list[dict[str, Any]], variant_id: str, gt: GroundTruthStore
) -> dict[str, Any]:
    targets: list[str] = []
    predictions: list[str] = []
    for record in records:
        result = record["results"][variant_id]
        row = _matched_gt(record, gt, result)
        target = str(row.get("primary_posture_label", "")).strip() if row else ""
        predicted = result.posture_labels[0] if result.posture_labels else ""
        if target and predicted:
            targets.append(target)
            predictions.append(predicted)
    if not targets:
        reason = (
            "The current MoveGuard pipeline has no posture-classification output, "
            "or Ground Truth posture labels are absent."
        )
        return {
            name: not_measurable(reason, unit="%")
            for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        } | {"confusion_matrix": {}}
    return _classification_metrics(targets, predictions)


def rula_metrics(
    records: list[dict[str, Any]], variant_id: str, gt: GroundTruthStore
) -> dict[str, Any]:
    final_targets: list[int] = []
    final_predictions: list[int | None] = []
    risk_targets: list[str] = []
    risk_predictions: list[str] = []
    part_matches: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        result = record["results"][variant_id]
        row = _matched_gt(record, gt, result)
        if row is None:
            continue
        target_score = integer(row.get("rula_final_score"))
        if target_score is not None:
            final_targets.append(target_score)
            final_predictions.append(integer(result.rula.get("final_score")))
        target_risk = str(row.get("rula_risk_level", "")).strip()
        if target_risk:
            risk_targets.append(target_risk)
            risk_predictions.append(str(result.rula.get("risk_level") or "not_measurable"))
        for part, (prediction_field, gt_field) in RULA_PARTS.items():
            target = integer(row.get(gt_field))
            if target is not None:
                part_matches[part].append(integer(result.rula.get(prediction_field)) == target)

    if final_targets:
        exact = [pred == target for pred, target in zip(final_predictions, final_targets)]
        within_one = [
            pred is not None and abs(pred - target) <= 1
            for pred, target in zip(final_predictions, final_targets)
        ]
        measurable_errors = [
            abs(pred - target)
            for pred, target in zip(final_predictions, final_targets)
            if pred is not None
        ]
        exact_metric = measured(100 * statistics.fmean(exact), unit="%", n=len(exact))
        within_one_metric = measured(
            100 * statistics.fmean(within_one), unit="%", n=len(within_one)
        )
        mae_metric = (
            measured(statistics.fmean(measurable_errors), unit="score", n=len(measurable_errors))
            if measurable_errors
            else not_measurable("No frame produced a measurable RULA score.", unit="score")
        )
    else:
        reason = "Expert RULA final scores are not provided."
        exact_metric = not_measurable(reason, unit="%")
        within_one_metric = not_measurable(reason, unit="%")
        mae_metric = not_measurable(reason, unit="score")

    if risk_targets:
        risk = _classification_metrics(risk_targets, risk_predictions)
    else:
        reason = "Expert RULA risk levels are not provided."
        risk = {
            name: not_measurable(reason, unit="%")
            for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        } | {"confusion_matrix": {}}

    body_parts = {
        part: (
            measured(100 * statistics.fmean(matches), unit="%", n=len(matches))
            if matches
            else not_measurable(f"Expert {part} RULA scores are not provided.", unit="%")
        )
        for part, matches in ((name, part_matches[name]) for name in RULA_PARTS)
    }
    return {
        "exact_accuracy": exact_metric,
        "within_one_accuracy": within_one_metric,
        "mae": mae_metric,
        "risk": risk,
        "body_part_accuracy": body_parts,
    }


def performance_metrics(results: list[VariantResult]) -> dict[str, Any]:
    if not results:
        reason = "No evaluated frames."
        return {
            "average_latency_ms": not_measurable(reason, unit="ms"),
            "p50_latency_ms": not_measurable(reason, unit="ms"),
            "p95_latency_ms": not_measurable(reason, unit="ms"),
            "fps": not_measurable(reason, unit="fps"),
            "process_cpu_percent": not_measurable(reason, unit="%"),
            "memory_mb": not_measurable("Total process memory instrumentation is unavailable.", unit="MB"),
            "gpu_usage_percent": not_measurable("GPU telemetry is unavailable.", unit="%"),
        }
    latencies = [result.latency_ms for result in results]
    cpu = [result.process_cpu_ms for result in results]
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    total_wall = sum(latencies)
    total_cpu = sum(cpu)
    average = statistics.fmean(latencies)
    return {
        "average_latency_ms": measured(average, unit="ms", n=len(results)),
        "p50_latency_ms": measured(statistics.median(latencies), unit="ms", n=len(results)),
        "p95_latency_ms": measured(ordered[p95_index], unit="ms", n=len(results)),
        "fps": measured(1000 / average if average else 0.0, unit="fps", n=len(results)),
        "process_cpu_percent": measured(
            100 * total_cpu / total_wall if total_wall else 0.0,
            unit="%",
            n=len(results),
            definition="process CPU time divided by wall time; may exceed 100% with multithreading",
        ),
        "memory_mb": not_measurable(
            "psutil/native allocator instrumentation is not installed; no value was invented.", unit="MB"
        ),
        "gpu_usage_percent": not_measurable(
            "NVML/GPU telemetry is not installed; no value was invented.", unit="%"
        ),
    }


def subgroup_metrics(
    records: list[dict[str, Any]],
    variant_id: str,
    gt: GroundTruthStore,
) -> list[dict[str, Any]]:
    if not gt.available:
        return []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result = record["results"][variant_id]
        row = _matched_gt(record, gt, result)
        if not row:
            continue
        for field in ("occlusion", "camera_angle", "distance", "lighting"):
            value = str(row.get(field, "")).strip()
            if value:
                buckets[(field, value)].append(record)
    output: list[dict[str, Any]] = []
    for (field, value), subset in sorted(buckets.items()):
        _, detection = joint_detection_metrics(subset, variant_id)
        rula = rula_metrics(subset, variant_id, gt)
        output.append(
            {
                "condition": field,
                "value": value,
                "frames": len(subset),
                "joint_detection_rate": detection,
                "rula_exact_accuracy": rula["exact_accuracy"],
            }
        )
    return output


def build_report(
    records: list[dict[str, Any]],
    *,
    baseline_id: str,
    hybrid_id: str,
    ground_truth: GroundTruthStore,
    experiment: dict[str, Any],
    variant_status: dict[str, Any],
    ablation: list[dict[str, Any]],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant_id in (baseline_id, hybrid_id):
        joint_rows, joint_overall = joint_detection_metrics(records, variant_id)
        position_rows, position_overall = position_metrics(records, variant_id, ground_truth)
        angle_rows, angle_overall = angle_metrics(records, variant_id, ground_truth)
        posture = posture_metrics(records, variant_id, ground_truth)
        rula = rula_metrics(records, variant_id, ground_truth)
        performance = performance_metrics(_variant_results(records, variant_id))
        variants[variant_id] = {
            "joint_detection": joint_rows,
            "joint_detection_overall": joint_overall,
            "joint_position": position_rows,
            "joint_position_overall": position_overall,
            "joint_angles": angle_rows,
            "joint_angles_overall": angle_overall,
            "posture_classification": posture,
            "rula": rula,
            "performance": performance,
            "subgroups": subgroup_metrics(records, variant_id, ground_truth),
        }

    baseline = variants[baseline_id]
    hybrid = variants[hybrid_id]
    comparison = {
        "joint_detection_rate": _difference(
            baseline["joint_detection_overall"],
            hybrid["joint_detection_overall"],
            higher_is_better=True,
        ),
        "joint_position_pck": _difference(
            baseline["joint_position_overall"]["pck_at_0_2"],
            hybrid["joint_position_overall"]["pck_at_0_2"],
            higher_is_better=True,
        ),
        "joint_angle_mae": _difference(
            baseline["joint_angles_overall"]["mae"],
            hybrid["joint_angles_overall"]["mae"],
            higher_is_better=False,
        ),
        "posture_classification_accuracy": _difference(
            baseline["posture_classification"]["accuracy"],
            hybrid["posture_classification"]["accuracy"],
            higher_is_better=True,
        ),
        "rula_exact_accuracy": _difference(
            baseline["rula"]["exact_accuracy"],
            hybrid["rula"]["exact_accuracy"],
            higher_is_better=True,
        ),
        "rula_within_one_accuracy": _difference(
            baseline["rula"]["within_one_accuracy"],
            hybrid["rula"]["within_one_accuracy"],
            higher_is_better=True,
        ),
        "rula_risk_accuracy": _difference(
            baseline["rula"]["risk"]["accuracy"],
            hybrid["rula"]["risk"]["accuracy"],
            higher_is_better=True,
        ),
        "average_latency_ms": _difference(
            baseline["performance"]["average_latency_ms"],
            hybrid["performance"]["average_latency_ms"],
            higher_is_better=False,
        ),
        "fps": _difference(
            baseline["performance"]["fps"],
            hybrid["performance"]["fps"],
            higher_is_better=True,
        ),
    }

    joint_comparison = {}
    for group in (*JOINT_GROUPS.keys(), "wrist_angle_ready"):
        joint_comparison[group] = {
            baseline_id: baseline["joint_detection"][group],
            hybrid_id: hybrid["joint_detection"][group],
            "improvement": _difference(
                baseline["joint_detection"][group],
                hybrid["joint_detection"][group],
                higher_is_better=True,
            ),
        }

    conclusion = {
        "accuracy_claim_allowed": ground_truth.available
        and hybrid["rula"]["exact_accuracy"]["status"] == "measured",
        "recommended_configuration": None,
        "reason": (
            "Ground Truth is incomplete, so no accuracy winner or production model is recommended."
            if not ground_truth.available
            else "Use measured accuracy and speed trade-offs; do not substitute official model AP."
        ),
    }
    return {
        "schema_version": "1.0",
        "experiment": experiment,
        "ground_truth": {
            "provided": ground_truth.available,
            "rows": len(ground_truth.rows),
            "accuracy_fields_without_ground_truth": "not_measurable",
        },
        "variant_status": variant_status,
        "variants": variants,
        "comparison": comparison,
        "joint_comparison": joint_comparison,
        "ablation": ablation,
        "conclusion": conclusion,
    }
