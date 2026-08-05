#!/usr/bin/env python3
"""Independent-test wrist and temporal stability evaluation with resume cache."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rula import calculate
from backend.temporal_stabilizer import PoseFrame, StabilizerConfig, TemporalPoseStabilizer
from backend.wrist import projected_wrist_angle
from training.train_human_pose_3d_pilot import Corrector, baseline


RAW_WRIST = {"left": (13, 15, (17, 19, 21)), "right": (14, 16, (18, 20, 22))}
JOINT_WRIST = {"left": (18, 20, (22,)), "right": (19, 21, (23,))}
GT_WRIST = {"left": (6, 7, (8, 9, 10)), "right": (12, 13, (14, 15, 16))}
HUMAN_SEGMENTS = (
    (16, 17), (1, 2), (16, 18), (18, 20), (20, 22),
    (17, 19), (19, 21), (21, 23), (1, 4), (4, 7), (2, 5), (5, 8),
)
DETECTOR = None
REFINER = None
ARCHIVES: dict[str, zipfile.ZipFile] = {}


def init_worker(model_path: str, checkpoint_path: str) -> None:
    global DETECTOR, REFINER
    DETECTOR = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
    )
    REFINER = Corrector()
    REFINER.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"])
    REFINER.eval()


def human_confidence(visibility: np.ndarray) -> np.ndarray:
    output = np.zeros(24, dtype=np.float32)
    direct = {1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28, 10: 31, 11: 32,
              16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16}
    for human, mediapipe in direct.items():
        output[human] = visibility[mediapipe]
    output[0] = visibility[[23, 24]].mean()
    output[12] = visibility[[11, 12]].mean()
    output[3] = output[6] = output[9] = min(output[0], output[12])
    output[13] = min(output[12], output[16]); output[14] = min(output[12], output[17])
    output[15] = visibility[0]
    output[22] = visibility[[17, 19, 21]].mean()
    output[23] = visibility[[18, 20, 22]].mean()
    return output


def evaluate_record(record: dict[str, object]) -> dict[str, object]:
    try:
        image_zip = ARCHIVES.setdefault(str(record["image_archive"]), zipfile.ZipFile(str(record["image_archive"])))
        label_zip = ARCHIVES.setdefault(str(record["label_archive"]), zipfile.ZipFile(str(record["label_archive"])))
        label = json.loads(label_zip.read(str(record["label_member"])))
        annotation = label["annotations"][0]
        image_info = label["images"][0]
        gt_px = np.asarray(annotation["keypoints"], dtype=np.float32)
        with Image.open(io.BytesIO(image_zip.read(str(record["image_member"])))) as source:
            source = source.convert("RGB")
            x1, y1, x2, y2 = map(float, record["bbox"])
            pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            left, top = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
            right, bottom = min(source.width, x2 + pad_x), min(source.height, y2 + pad_y)
            rgb = np.asarray(source.crop((left, top, right, bottom)))
        result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return {"id": record["id"], "group_id": record["group_id"], "status": "no_person"}
        landmarks = result.pose_landmarks[0]
        raw = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
        visibility = np.asarray([p.visibility for p in landmarks], dtype=np.float32)
        raw_features = np.asarray([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32)
        base = baseline(raw_features)
        with torch.inference_mode():
            refined = REFINER(torch.from_numpy(raw_features.reshape(1, -1)), torch.from_numpy(base[None])).numpy()[0]
        gt = gt_px.copy()
        gt[:, :2] = (gt[:, :2] - np.asarray([left, top])) / np.asarray([right - left, bottom - top])
        norm_scale = np.asarray([right - left, bottom - top], dtype=np.float32) / max(math.hypot(x2 - x1, y2 - y1), 1.0)
        frame_match = re.search(r"_(\d+)$", str(record["id"]))
        return {
            "id": record["id"], "group_id": record["group_id"], "status": "ok",
            "frame_index": int(frame_match.group(1)) if frame_match else 0,
            "image_archive": record["image_archive"], "image_member": record["image_member"],
            "camera_angle_code": image_info.get("angle"),
            "raw": raw.round(7).tolist(), "raw_confidence": visibility.round(7).tolist(),
            "joint": refined.round(7).tolist(), "joint_confidence": human_confidence(visibility).round(7).tolist(),
            "gt": gt.round(7).tolist(), "norm_scale": norm_scale.round(7).tolist(),
            "wrist_bending": int(annotation["wrist_bending"]),
        }
    except Exception as exc:
        return {"id": record.get("id"), "group_id": record.get("group_id"), "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def wrist(points: np.ndarray, confidence: np.ndarray, mapping: dict, side: str):
    elbow, wrist_index, hand = mapping[side]
    indices = (elbow, wrist_index, *hand)
    return projected_wrist_angle(
        points[elbow, :2], points[wrist_index, :2], points[list(hand), :2],
        confidence=confidence[list(indices)],
    )


def frame_rula(points: np.ndarray, confidence: np.ndarray, schema: str) -> int | None:
    if schema == "raw":
        left = (11, 13, 15, 23); right = (12, 14, 16, 24)
        nose = points[0, :2]; shoulder_mid = points[[11, 12], :2].mean(0); hip_mid = points[[23, 24], :2].mean(0)
    elif schema == "joint":
        left = (16, 18, 20, 1); right = (17, 19, 21, 2)
        nose = points[15, :2]; shoulder_mid = points[[16, 17], :2].mean(0); hip_mid = points[[1, 2], :2].mean(0)
    else:
        left = (5, 6, 7, 19); right = (11, 12, 13, 23)
        nose = points[3, :2]; shoulder_mid = points[[5, 11], :2].mean(0); hip_mid = points[[19, 23], :2].mean(0)
    selected = right if confidence[list(right[0:3])].mean() >= confidence[list(left[0:3])].mean() else left
    shoulder, elbow, wrist_index, hip = selected
    required = (shoulder, elbow, wrist_index, hip)
    if min(confidence[list(required)]) < 0.45 or not np.isfinite(points).all():
        return None
    def angle(v1, v2):
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / max(denom, 1e-9), -1, 1))))
    trunk_vector = shoulder_mid - hip_mid
    neck_angle = angle(nose - shoulder_mid, trunk_vector)
    trunk_angle = angle(trunk_vector, np.asarray([0.0, -1.0]))
    upper_angle = angle(points[elbow, :2] - points[shoulder, :2], trunk_vector)
    lower_angle = angle(points[shoulder, :2] - points[elbow, :2], points[wrist_index, :2] - points[elbow, :2])
    neck_score = 1 if neck_angle <= 10 else 2 if neck_angle <= 20 else 3
    trunk_score = 1 if trunk_angle <= 5 else 2 if trunk_angle <= 20 else 3 if trunk_angle <= 60 else 4
    upper_score = 1 if upper_angle <= 20 else 2 if upper_angle <= 45 else 3 if upper_angle <= 90 else 4
    lower_score = 1 if 60 <= lower_angle <= 100 else 2
    mapping = RAW_WRIST if schema == "raw" else JOINT_WRIST if schema == "joint" else GT_WRIST
    wrist_scores = [wrist(points, confidence, mapping, side).rula_score for side in ("left", "right")]
    wrist_score = max((score for score in wrist_scores if score is not None), default=1)
    return int(calculate({"upper_arm": upper_score, "lower_arm": lower_score, "wrist": wrist_score,
                          "wrist_twist": 1, "neck": neck_score, "trunk": trunk_score, "legs": 1})["final_score"])


def numeric_metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"valid_sample_count": 0, "mae": None, "median_absolute_error": None, "rmse": None,
                "within_5_degrees": None, "within_10_degrees": None, "within_15_degrees": None}
    array = np.asarray(values, dtype=np.float64)
    return {"valid_sample_count": len(values), "mae": float(array.mean()),
            "median_absolute_error": float(np.median(array)), "rmse": float(np.sqrt(np.mean(array ** 2))),
            "within_5_degrees": float((array <= 5).mean()), "within_10_degrees": float((array <= 10).mean()),
            "within_15_degrees": float((array <= 15).mean())}


def classification_metrics(
    truth: list[int], prediction: list[int], *, labels: tuple[int, int] = (0, 1), positive: int = 1
) -> dict[str, object]:
    return {"count": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, pos_label=positive, zero_division=0)),
            "recall": float(recall_score(truth, prediction, pos_label=positive, zero_division=0)),
            "f1": float(f1_score(truth, prediction, pos_label=positive, zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=list(labels)).tolist(),
            "labels": list(labels)}


def summarize(records: list[dict], output: Path, split: str) -> dict[str, object]:
    ok = [record for record in records if record["status"] == "ok"]
    groups = defaultdict(list)
    for record in ok:
        groups[record["group_id"]].append(record)
    methods = ("mediapipe_raw", "mediapipe_stabilized", "joint_raw", "joint_stabilized")
    angle_errors = {method: {"left": [], "right": []} for method in methods}
    position_errors = {method: [] for method in methods}
    score_truth = {method: [] for method in methods}
    score_pred = {method: [] for method in methods}
    labels, gt_classes, model_gt_classes, model_classes = [], [], [], []
    review = []
    frame_rows = []
    outlier_count = Counter(); valid_count = Counter(); total_count = Counter()
    group_comparison = []
    raw_stabilizer = TemporalPoseStabilizer(config=StabilizerConfig(max_neighbor_gap_seconds=0.20))
    joint_stabilizer = TemporalPoseStabilizer(segments=HUMAN_SEGMENTS, config=StabilizerConfig(max_neighbor_gap_seconds=0.20))

    for group_id, group in groups.items():
        group.sort(key=lambda row: row["frame_index"])
        first_index = group[0]["frame_index"]
        raw_frames = [PoseFrame(np.asarray(row["raw"]), np.asarray(row["raw_confidence"]), (row["frame_index"] - first_index) / 30.0, row["id"]) for row in group]
        joint_frames = [PoseFrame(np.asarray(row["joint"]), np.asarray(row["joint_confidence"]), (row["frame_index"] - first_index) / 30.0, row["id"]) for row in group]
        raw_stable = raw_stabilizer.stabilize(raw_frames)
        joint_stable = joint_stabilizer.stabilize(joint_frames)
        group_scores = {method: [] for method in methods}; gt_group_scores = []

        for index, row in enumerate(group):
            gt = np.asarray(row["gt"], dtype=np.float32); gt_conf = gt[:, 2] / 2.0
            raw = np.asarray(row["raw"], dtype=np.float32); raw_conf = np.asarray(row["raw_confidence"])
            joint = np.asarray(row["joint"], dtype=np.float32); joint_conf = np.asarray(row["joint_confidence"])
            variants = {
                "mediapipe_raw": (raw, raw_conf, RAW_WRIST, "raw", True, False, []),
                "mediapipe_stabilized": (raw_stable[index].keypoints, raw_stable[index].confidence, RAW_WRIST, "raw", raw_stable[index].valid, raw_stable[index].outlier, raw_stable[index].reasons),
                "joint_raw": (joint, joint_conf, JOINT_WRIST, "joint", True, False, []),
                "joint_stabilized": (joint_stable[index].keypoints, joint_stable[index].confidence, JOINT_WRIST, "joint", joint_stable[index].valid, joint_stable[index].outlier, joint_stable[index].reasons),
            }
            gt_measure = {side: wrist(gt, gt_conf, GT_WRIST, side) for side in ("left", "right")}
            gt_score = frame_rula(gt, gt_conf, "gt"); gt_group_scores.append(gt_score)
            sample_predictions = {}
            for method, (points, conf, mapping, schema, valid, outlier, reasons) in variants.items():
                total_count[method] += 1; valid_count[method] += int(valid); outlier_count[method] += int(outlier)
                model_measure = {side: wrist(points, conf, mapping, side) for side in ("left", "right")}
                sample_predictions[method] = {side: model_measure[side].to_dict() for side in ("left", "right")}
                for side in ("left", "right"):
                    if valid and gt_measure[side].valid and model_measure[side].valid:
                        error = abs(float(model_measure[side].angle) - float(gt_measure[side].angle))
                        angle_errors[method][side].append(error)
                        score_truth[method].append(int(gt_measure[side].rula_score))
                        score_pred[method].append(int(model_measure[side].rula_score))
                        pred_wrist = mapping[side][1]; gt_wrist = GT_WRIST[side][1]
                        norm_scale = np.asarray(row["norm_scale"])
                        position_errors[method].append(float(np.linalg.norm((points[pred_wrist, :2] - gt[gt_wrist, :2]) * norm_scale)))
                score = frame_rula(points, conf, schema) if valid else None
                group_scores[method].append(score)
            valid_gt_wrist = any(measure.valid for measure in gt_measure.values())
            gt_bending = int(any(measure.valid and measure.rula_score == 2 for measure in gt_measure.values()))
            joint_prediction = sample_predictions["joint_stabilized"]
            valid_model_wrist = any(joint_prediction[side]["valid"] for side in ("left", "right"))
            model_bending = int(any(joint_prediction[side]["valid"] and joint_prediction[side]["rula_score"] == 2 for side in ("left", "right")))
            label = int(row["wrist_bending"])
            if valid_gt_wrist:
                labels.append(label); gt_classes.append(gt_bending)
                if valid_model_wrist:
                    model_gt_classes.append(gt_bending); model_classes.append(model_bending)
            reasons = []
            if valid_gt_wrist and label != gt_bending: reasons.append("gt_angle_vs_dataset_label")
            if valid_gt_wrist and valid_model_wrist and model_bending != gt_bending: reasons.append("model_vs_gt_angle")
            if reasons:
                gt_angles = {side: gt_measure[side].angle for side in ("left", "right")}
                model_angles = {side: sample_predictions["joint_stabilized"][side]["angle"] for side in ("left", "right")}
                maximum_error = max((abs((model_angles[s] or 0) - (gt_angles[s] or 0)) for s in ("left", "right")), default=0)
                priority = "HIGH" if label != gt_bending and model_bending != gt_bending or maximum_error >= 30 else "MEDIUM" if maximum_error >= 15 else "LOW"
                review.append({"id": row["id"], "group_id": group_id, "image_archive": row["image_archive"], "image_member": row["image_member"],
                               "gt_keypoints": row["gt"], "gt_angles": gt_angles, "model_angles": model_angles,
                               "dataset_wrist_bending": label, "gt_bending": gt_bending, "model_bending": model_bending,
                               "model_wrist_score": max(sample_predictions["joint_stabilized"][s]["rula_score"] or 0 for s in ("left", "right")),
                               "confidence": min(sample_predictions["joint_stabilized"][s]["confidence"] for s in ("left", "right")),
                               "maximum_angle_error": maximum_error, "reason_for_review": reasons, "priority": priority})
            frame_rows.append({"id": row["id"], "group_id": group_id, "frame_index": row["frame_index"], "gt_rula": gt_score,
                               "dataset_wrist_bending": label, "gt_wrist": {s: gt_measure[s].to_dict() for s in ("left", "right")},
                               "predictions": sample_predictions,
                               "stabilizer": {"mediapipe": {"valid": raw_stable[index].valid, "outlier": raw_stable[index].outlier, "reasons": raw_stable[index].reasons},
                                              "joint": {"valid": joint_stable[index].valid, "outlier": joint_stable[index].outlier, "reasons": joint_stable[index].reasons}}})
        gt_final = max((score for score in gt_group_scores if score is not None), default=None)
        comparison = {"group_id": group_id, "gt_final": gt_final}
        for method in methods:
            valid_scores = [score for score in group_scores[method] if score is not None]
            final = max(valid_scores, default=None); comparison[method] = final
            if valid_scores and final is not None:
                comparison[f"{method}_support_frames"] = sum(score >= final - 1 for score in valid_scores)
                comparison[f"{method}_persistent"] = comparison[f"{method}_support_frames"] >= 2
        group_comparison.append(comparison)

    with (output / "frame_metrics.jsonl").open("w") as stream:
        for row in frame_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "manual_review_queue.jsonl").open("w") as stream:
        for row in sorted(review, key=lambda item: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[item["priority"]], -item["maximum_angle_error"])):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "group_rula_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in group_comparison for key in row})); writer.writeheader(); writer.writerows(group_comparison)

    selected = []
    buckets = defaultdict(list)
    for row in review: buckets[row["priority"]].append(row)
    for priority, count in (("HIGH", 100), ("MEDIUM", 70), ("LOW", 30)):
        selected.extend(sorted(buckets[priority], key=lambda item: item["id"])[:count])
    with (output / "expert_manual_review_template.csv").open("w", newline="") as stream:
        fields = ["id", "image_archive", "image_member", "priority", "gt_left_angle", "gt_right_angle", "model_left_angle", "model_right_angle",
                  "dataset_wrist_bending", "expert_left_angle", "expert_right_angle", "expert_left_rula_score", "expert_right_rula_score", "expert_final_rula", "reviewer", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in selected:
            writer.writerow({"id": row["id"], "image_archive": row["image_archive"], "image_member": row["image_member"], "priority": row["priority"],
                             "gt_left_angle": row["gt_angles"]["left"], "gt_right_angle": row["gt_angles"]["right"],
                             "model_left_angle": row["model_angles"]["left"], "model_right_angle": row["model_angles"]["right"],
                             "dataset_wrist_bending": row["dataset_wrist_bending"]})

    report: dict[str, object] = {"split": split, "evaluated_images": len(records), "detected_images": len(ok), "detection_rate": len(ok) / len(records),
        "independent_test_only": split == "test", "frame_interval_note": "filename frame index / 30 fps proxy; source FPS is not present in JSON",
        "methods": {}, "label_audit": {"gt_keypoint_angle_vs_dataset_label": classification_metrics(gt_classes, labels),
                                          "joint_stabilized_vs_gt_keypoint_angle": classification_metrics(model_gt_classes, model_classes),
                                          "agreement": sum(a == b for a, b in zip(gt_classes, labels)),
                                          "gt_vs_label_conflict": sum(a != b for a, b in zip(gt_classes, labels)),
                                          "model_vs_gt_conflict": sum(a != b for a, b in zip(model_gt_classes, model_classes)),
                                          "manual_review_required": len(review), "review_priority": dict(Counter(row["priority"] for row in review))}}
    for method in methods:
        combined = angle_errors[method]["left"] + angle_errors[method]["right"]
        report["methods"][method] = {"wrist_pck_0_10": float((np.asarray(position_errors[method]) <= 0.10).mean()),
            "wrist_pck_0_05": float((np.asarray(position_errors[method]) <= 0.05).mean()),
            "angle": {"left": numeric_metrics(angle_errors[method]["left"]), "right": numeric_metrics(angle_errors[method]["right"]), "overall": numeric_metrics(combined)},
            "rula_wrist": classification_metrics(score_truth[method], score_pred[method], labels=(1, 2), positive=2),
            "valid_rate": valid_count[method] / total_count[method], "outlier_rate": outlier_count[method] / total_count[method]}
    raw_vs_stable = []
    for model in ("mediapipe", "joint"):
        raw_name, stable_name = f"{model}_raw", f"{model}_stabilized"
        changed = [row for row in group_comparison if row.get(raw_name) != row.get(stable_name)]
        false_raw = [row for row in group_comparison if row.get(raw_name) is not None and row.get("gt_final") is not None and row[raw_name] >= 5 and row["gt_final"] <= 4]
        false_stable = [row for row in group_comparison if row.get(stable_name) is not None and row.get("gt_final") is not None and row[stable_name] >= 5 and row["gt_final"] <= 4]
        gt_risk = [row for row in group_comparison if row.get("gt_final") is not None and row["gt_final"] >= 5]
        preserved = [row for row in gt_risk if row.get(stable_name) is not None and row[stable_name] >= 5]
        raw_vs_stable.append({"model": model, "groups": len(group_comparison), "final_rula_changed": len(changed),
                              "final_rula_change_rate": len(changed) / len(group_comparison),
                              "false_high_risk_raw": len(false_raw), "false_high_risk_stabilized": len(false_stable),
                              "gt_high_risk_groups": len(gt_risk), "high_risk_preserved": len(preserved),
                              "high_risk_preservation_rate": len(preserved) / len(gt_risk) if gt_risk else None})
    report["rula_stability"] = raw_vs_stable
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "artifacts/aihub/audit/split_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=root / "experiments/wrist_rula_pipeline_validation")
    parser.add_argument("--model", type=Path, default=root / "mediapipe/models/pose_landmarker_lite.task")
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints/wrist_refiner_joint_domain_v1/best.pt")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    records = [row for row in records if row["split"] == args.split]
    if args.limit: records = records[:args.limit]
    cache_path = args.output / "independent_test_pose_cache.jsonl"
    completed = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line); completed[row["id"]] = row
    pending = [row for row in records if row["id"] not in completed]
    with get_context("spawn").Pool(args.workers, init_worker, (str(args.model), str(args.checkpoint))) as pool, cache_path.open("a") as stream:
        for index, result in enumerate(pool.imap_unordered(evaluate_record, pending, chunksize=8), 1):
            completed[result["id"]] = result; stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                stream.flush(); print(json.dumps({"phase": "independent_test_pose_cache", "completed": len(completed), "total": len(records), "percent": round(len(completed) / len(records) * 100, 2)}), flush=True)
    ordered = [completed[row["id"]] for row in records]
    print(json.dumps(summarize(ordered, args.output, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
