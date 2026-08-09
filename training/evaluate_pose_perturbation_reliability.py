#!/usr/bin/env python3
"""MediaPipe test-retest reliability under small image perturbations.

The original inference is used as the reference. This measures repeatability,
not accuracy against an expert or ground truth.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageEnhance
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluate_wrist_temporal_pipeline import JOINT_WRIST, RAW_WRIST, frame_rula, human_confidence, wrist
from training.train_human_pose_3d_pilot import Corrector, baseline

DETECTOR = None
REFINER = None
ARCHIVES: dict[str, zipfile.ZipFile] = {}
VARIANTS = ("original", "brightness_80", "brightness_120", "jpeg_60", "center_crop_5")


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
    if checkpoint_path:
        REFINER = Corrector()
        REFINER.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"])
        REFINER.eval()


def transform(source: Image.Image, variant: str) -> Image.Image:
    if variant == "original":
        return source
    if variant == "brightness_80":
        return ImageEnhance.Brightness(source).enhance(0.8)
    if variant == "brightness_120":
        return ImageEnhance.Brightness(source).enhance(1.2)
    if variant == "jpeg_60":
        stream = io.BytesIO()
        source.save(stream, format="JPEG", quality=60)
        return Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
    if variant == "center_crop_5":
        dx, dy = int(source.width * 0.05), int(source.height * 0.05)
        return source.crop((dx, dy, source.width - dx, source.height - dy)).resize(source.size)
    raise ValueError(variant)


def infer(image: Image.Image) -> dict[str, object]:
    result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
    if not result.pose_landmarks:
        return {"detected": False}
    landmarks = result.pose_landmarks[0]
    points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    confidence = np.asarray([p.visibility for p in landmarks], dtype=np.float32)
    schema, mapping = "raw", RAW_WRIST
    if REFINER is not None:
        features = np.column_stack([points, confidence]).astype(np.float32)
        with torch.inference_mode():
            points = REFINER(torch.from_numpy(features.reshape(1, -1)), torch.from_numpy(baseline(features)[None])).numpy()[0]
        confidence = human_confidence(confidence); schema, mapping = "joint", JOINT_WRIST
    measurements = {side: wrist(points, confidence, mapping, side) for side in ("left", "right")}
    return {
        "detected": True,
        "points": points.tolist(),
        "confidence": confidence.tolist(),
        "rula": frame_rula(points, confidence, schema),
        "wrist": {side: measurements[side].to_dict() for side in measurements},
    }


def evaluate(row: dict[str, object]) -> dict[str, object]:
    archive = ARCHIVES.setdefault(str(row["image_archive"]), zipfile.ZipFile(str(row["image_archive"])))
    with Image.open(io.BytesIO(archive.read(str(row["image_member"])))) as image:
        source = image.convert("RGB")
        return {"id": row["id"], "group_id": row["group_id"], "variants": {name: infer(transform(source, name)) for name in VARIANTS}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/wrist_rula_pipeline_validation/independent_test_pose_cache.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "mediapipe/models/pose_landmarker_lite.task")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/pose_perturbation_reliability/reliability_report.json")
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for line in args.cache.read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "ok":
            groups[row["group_id"]].append(row)
    selected = []
    for rows in groups.values():
        rows.sort(key=lambda item: item["frame_index"])
        indices = np.linspace(0, len(rows) - 1, min(args.per_group, len(rows)), dtype=int)
        selected.extend(rows[int(index)] for index in indices)

    with get_context("spawn").Pool(args.workers, init_worker, (str(args.model), str(args.checkpoint) if args.checkpoint else "")) as pool:
        records = list(pool.imap_unordered(evaluate, selected, chunksize=2))

    metrics = {}
    for variant in VARIANTS[1:]:
        comparable = 0
        detected_both = 0
        landmark_errors = []
        angle_errors = []
        wrist_score_match = []
        rula_match = []
        for row in records:
            original, changed = row["variants"]["original"], row["variants"][variant]
            comparable += int(original["detected"])
            if not original["detected"] or not changed["detected"]:
                continue
            detected_both += 1
            a, b = np.asarray(original["points"]), np.asarray(changed["points"])
            landmark_errors.extend(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).tolist())
            if original["rula"] is not None and changed["rula"] is not None:
                rula_match.append(int(original["rula"] == changed["rula"]))
            for side in ("left", "right"):
                ow, cw = original["wrist"][side], changed["wrist"][side]
                if ow["valid"] and cw["valid"]:
                    angle_errors.append(abs(float(ow["angle"]) - float(cw["angle"])))
                    wrist_score_match.append(int(ow["rula_score"] == cw["rula_score"]))
        metrics[variant] = {
            "reference_detected": comparable,
            "detected_both": detected_both,
            "detection_retention": detected_both / comparable if comparable else None,
            "landmark_self_mae_normalized": float(np.mean(landmark_errors)) if landmark_errors else None,
            "wrist_angle_repeat_mae_deg": float(np.mean(angle_errors)) if angle_errors else None,
            "wrist_score_exact_repeatability": float(np.mean(wrist_score_match)) if wrist_score_match else None,
            "frame_rula_exact_repeatability": float(np.mean(rula_match)) if rula_match else None,
            "valid_wrist_comparisons": len(wrist_score_match),
            "valid_frame_rula_comparisons": len(rula_match),
        }
    wrist_total = sum(item["valid_wrist_comparisons"] for item in metrics.values())
    rula_total = sum(item["valid_frame_rula_comparisons"] for item in metrics.values())
    detection_total = sum(item["reference_detected"] for item in metrics.values())
    report = {
        "definition": "same-image perturbation repeatability; not expert RULA accuracy",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sample_images": len(records),
        "groups": len(groups),
        "variants": metrics,
        "overall": {
            "detection_retention": sum(item["detected_both"] for item in metrics.values()) / detection_total,
            "wrist_score_exact_repeatability": sum(item["wrist_score_exact_repeatability"] * item["valid_wrist_comparisons"] for item in metrics.values()) / wrist_total,
            "frame_rula_exact_repeatability": sum(item["frame_rula_exact_repeatability"] * item["valid_frame_rula_comparisons"] for item in metrics.values()) / rula_total,
            "wrist_angle_repeat_mae_deg": sum(item["wrist_angle_repeat_mae_deg"] * item["valid_wrist_comparisons"] for item in metrics.values()) / wrist_total,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
