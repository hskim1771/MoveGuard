#!/usr/bin/env python3
"""Stream AI-Hub images from ZIPs through MediaPipe Pose with resume support."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import mediapipe as mp
import numpy as np
import psutil
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ANGLE_TRIPLETS = {
    "left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
    "left_shoulder": (23, 11, 13), "right_shoulder": (24, 12, 14),
    "left_hip": (11, 23, 25), "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27), "right_knee": (24, 26, 28),
}


def angle(points: np.ndarray, triple: tuple[int, int, int]) -> float:
    a, b, c = (points[i, :3] for i in triple)
    u, v = a - b, c - b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denom, -1, 1))))


def features(landmarks: list[object]) -> tuple[list[float], dict[str, float]]:
    pts = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in landmarks], dtype=np.float32)
    hip = (pts[23, :3] + pts[24, :3]) / 2
    scale = max(float(np.linalg.norm(pts[11, :2] - pts[12, :2])), 1e-6)
    pts[:, :3] = (pts[:, :3] - hip) / scale
    derived = {name: angle(pts, triple) for name, triple in ANGLE_TRIPLETS.items()}
    shoulder_mid = (pts[11, :3] + pts[12, :3]) / 2
    derived["trunk_inclination"] = math.degrees(math.atan2(abs(float(shoulder_mid[0])), abs(float(shoulder_mid[1])) + 1e-6))
    derived["visibility_mean"] = float(pts[:, 3].mean())
    return pts.flatten().astype(float).tolist() + list(derived.values()), derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("mediapipe/models/pose_landmarker_lite.task"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--use-label-bbox", action="store_true", help="Crop the AI-Hub annotated person bbox in memory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "features.jsonl"
    checkpoint_path = args.output / "checkpoint.json"
    report_path = args.output / "preprocess_report.json"
    completed: set[str] = set()
    if feature_path.exists():
        with feature_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    completed.add(str(json.loads(line)["id"]))
                except Exception:
                    pass

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    pending = [r for r in records if str(r["id"]) not in completed]
    if args.limit is not None:
        pending = pending[: max(0, args.limit - len(completed))]
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=args.min_confidence,
        min_pose_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )
    archives: dict[str, zipfile.ZipFile] = {}
    process = psutil.Process(os.getpid())
    started = time.perf_counter()
    success = failures = full_body = 0
    reasons: dict[str, int] = {}
    peak_rss = process.memory_info().rss
    try:
        with vision.PoseLandmarker.create_from_options(opts) as landmarker, feature_path.open("a", encoding="utf-8") as out:
            for index, record in enumerate(pending, 1):
                reason = None
                try:
                    archive = str(record["image_archive"])
                    zf = archives.setdefault(archive, zipfile.ZipFile(archive))
                    with Image.open(io.BytesIO(zf.read(str(record["image_member"])))) as image:
                        image = image.convert("RGB")
                        if args.use_label_bbox and isinstance(record.get("bbox"), list) and len(record["bbox"]) == 4:
                            x1,y1,x2,y2 = map(float, record["bbox"])
                            # Dataset bbox is x1,y1,x2,y2; add 15% context without touching the source.
                            pad_x=(x2-x1)*.15; pad_y=(y2-y1)*.15
                            image=image.crop((max(0,x1-pad_x),max(0,y1-pad_y),min(image.width,x2+pad_x),min(image.height,y2+pad_y)))
                        rgb = np.asarray(image)
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    if not result.pose_landmarks:
                        reason = "no_person"
                    else:
                        landmarks = result.pose_landmarks[0]
                        vector, derived = features(landmarks)
                        success += 1
                        if all(landmarks[i].visibility >= 0.5 for i in (11, 12, 23, 24, 25, 26, 27, 28)):
                            full_body += 1
                        out.write(json.dumps({
                            "id": record["id"], "split": record["split"], "group_id": record["group_id"],
                            "labels": record["labels"], "features": vector, "angles": derived,
                            "detection_confidence": derived["visibility_mean"], "status": "ok",
                        }, ensure_ascii=False) + "\n")
                except Exception as exc:
                    reason = type(exc).__name__
                if reason:
                    failures += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    out.write(json.dumps({"id": record["id"], "split": record["split"], "status": "failed", "reason": reason}, ensure_ascii=False) + "\n")
                out.flush()
                peak_rss = max(peak_rss, process.memory_info().rss)
                elapsed = time.perf_counter() - started
                checkpoint_path.write_text(json.dumps({
                    "processed": len(completed) + index, "success": success, "failures": failures,
                    "elapsed_seconds": elapsed, "last_id": record["id"],
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for zf in archives.values():
            zf.close()
    elapsed = time.perf_counter() - started
    processed = success + failures
    report = {
        "processed": processed, "detected": success, "failed": failures,
        "min_confidence": args.min_confidence,
        "used_label_bbox": args.use_label_bbox,
        "person_detection_rate": success / processed if processed else 0,
        "full_body_detection_rate": full_body / processed if processed else 0,
        "seconds": elapsed, "images_per_second": processed / elapsed if elapsed else 0,
        "peak_rss_mb": peak_rss / 1024 / 1024, "failure_reasons": reasons,
        "estimated_full_hours": (108000 / (processed / elapsed) / 3600) if processed and elapsed else None,
        "resume_command": f"python training/preprocess_mediapipe.py --manifest {args.manifest} --output {args.output}",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
