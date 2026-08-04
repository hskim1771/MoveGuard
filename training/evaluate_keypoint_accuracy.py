#!/usr/bin/env python3
"""Evaluate MediaPipe 2-D keypoints against AI-Hub's independent test labels.

AI-Hub's 27-point order comes from the released construction guideline's
"키포인트 위치(정면)" figure. The crown point is excluded because MediaPipe
has no anatomical equivalent. Neck, spine, and pelvis-center are derived from
the same surrounding joints. Source ZIPs are opened read-only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

JOINTS = (
    "left_ear", "right_ear", "nose", "neck", "left_shoulder", "left_elbow",
    "left_wrist", "left_index", "left_pinky", "left_thumb", "right_shoulder",
    "right_elbow", "right_wrist", "right_index", "right_pinky", "right_thumb",
    "spine", "pelvis_center", "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
)
# AI-Hub indices corresponding to JOINTS; index 0 (head crown) is excluded.
AIHUB_INDICES = tuple(range(1, 27))
MEDIAPIPE_INDICES = {
    "left_ear": 7, "right_ear": 8, "nose": 0,
    "left_shoulder": 11, "left_elbow": 13, "left_wrist": 15,
    "left_index": 19, "left_pinky": 17, "left_thumb": 21,
    "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
    "right_index": 20, "right_pinky": 18, "right_thumb": 22,
    "left_hip": 23, "left_knee": 25, "left_ankle": 27, "left_foot": 31,
    "right_hip": 24, "right_knee": 26, "right_ankle": 28, "right_foot": 32,
}

_detector = None
_archives: dict[str, zipfile.ZipFile] = {}


def _init_worker(model_path: str, min_confidence: float) -> None:
    global _detector
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    _detector = vision.PoseLandmarker.create_from_options(options)


def _predicted_points(landmarks: list[object]) -> dict[str, np.ndarray]:
    points = {name: np.asarray([landmarks[index].x, landmarks[index].y]) for name, index in MEDIAPIPE_INDICES.items()}
    points["neck"] = (points["left_shoulder"] + points["right_shoulder"]) / 2
    points["pelvis_center"] = (points["left_hip"] + points["right_hip"]) / 2
    points["spine"] = (points["neck"] + points["pelvis_center"]) / 2
    return points


def _evaluate(record: dict) -> dict[str, object]:
    try:
        image_zip = _archives.setdefault(record["image_archive"], zipfile.ZipFile(record["image_archive"]))
        label_zip = _archives.setdefault(record["label_archive"], zipfile.ZipFile(record["label_archive"]))
        label = json.loads(label_zip.read(record["label_member"]))
        ground_truth = np.asarray(label["annotations"][0]["keypoints"], dtype=float)
        with Image.open(io.BytesIO(image_zip.read(record["image_member"]))) as source:
            source = source.convert("RGB")
            x1, y1, x2, y2 = map(float, record["bbox"])
            pad_x, pad_y = (x2-x1)*.15, (y2-y1)*.15
            left, top = max(0., x1-pad_x), max(0., y1-pad_y)
            right, bottom = min(float(source.width), x2+pad_x), min(float(source.height), y2+pad_y)
            crop = source.crop((left, top, right, bottom))
            rgb = np.asarray(crop)
        result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return {"id": record["id"], "status": "no_person"}
        predicted = _predicted_points(result.pose_landmarks[0])
        crop_size = np.asarray([right-left, bottom-top])
        gt_normalized = (ground_truth[:, :2] - np.asarray([left, top])) / crop_size
        bbox_diagonal = float(np.hypot(x2-x1, y2-y1))
        errors = []
        for name, gt_index in zip(JOINTS, AIHUB_INDICES):
            pixel_error = float(np.linalg.norm((predicted[name] - gt_normalized[gt_index]) * crop_size))
            errors.append({"joint": name, "pixel_error": pixel_error, "normalized_error": pixel_error/max(bbox_diagonal, 1.), "visibility": int(ground_truth[gt_index, 2])})
        return {"id": record["id"], "status": "ok", "errors": errors}
    except Exception as exc:
        return {"id": record.get("id"), "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _metrics(rows: list[dict], visible_only: bool) -> dict[str, object]:
    values: dict[str, list[dict]] = {joint: [] for joint in JOINTS}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for item in row["errors"]:
            if not visible_only or item["visibility"] == 2:
                values[item["joint"]].append(item)
    flat = [item for items in values.values() for item in items]
    summarize = lambda items: {
        "count": len(items),
        "mean_pixel_error": float(np.mean([x["pixel_error"] for x in items])) if items else None,
        "mean_normalized_error": float(np.mean([x["normalized_error"] for x in items])) if items else None,
        "pck_0_05": float(np.mean([x["normalized_error"] <= .05 for x in items])) if items else None,
        "pck_0_10": float(np.mean([x["normalized_error"] <= .10 for x in items])) if items else None,
        "pck_0_20": float(np.mean([x["normalized_error"] <= .20 for x in items])) if items else None,
    }
    return {"overall": summarize(flat), "per_joint": {joint: summarize(items) for joint, items in values.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("mediapipe/models/pose_landmarker_lite.task"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2)//2)))
    parser.add_argument("--min-confidence", type=float, default=.3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "keypoint_errors.jsonl"
    done: dict[str, dict] = {}
    if detail_path.exists():
        for line in detail_path.open(encoding="utf-8"):
            row = json.loads(line); done[str(row["id"])] = row
    records = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    records = [r for r in records if r["split"] == args.split and r["id"] not in done]
    if args.limit is not None:
        records = records[:max(0, args.limit-len(done))]
    started = time.perf_counter()
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.workers, _init_worker, (str(args.model), args.min_confidence)) as pool, detail_path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(pool.imap_unordered(_evaluate, records, chunksize=8), 1):
            done[str(row["id"])] = row; out.write(json.dumps(row, ensure_ascii=False)+"\n")
            if index % 100 == 0:
                out.flush(); print(json.dumps({"processed_this_run": index, "completed": len(done), "elapsed_seconds": time.perf_counter()-started}), flush=True)
    rows = list(done.values()); total=len(rows); detected=sum(r.get("status")=="ok" for r in rows)
    report = {
        "study": "MediaPipe vs AI-Hub 2-D keypoint agreement", "split": args.split,
        "evaluated_images": total, "detected_images": detected,
        "detection_rate": detected/total if total else None,
        "normalization": "Euclidean pixel error divided by AI-Hub person bbox diagonal",
        "mapping": dict(zip(JOINTS, AIHUB_INDICES)),
        "excluded": {"aihub_index_0": "head crown has no direct MediaPipe equivalent"},
        "all_annotated": _metrics(rows, False), "visible_only_aihub_v2": _metrics(rows, True),
        "failure_counts": {status:sum(r.get("status")==status for r in rows) for status in ("no_person", "error")},
        "seconds_this_run": time.perf_counter()-started,
    }
    (args.output/"keypoint_accuracy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
