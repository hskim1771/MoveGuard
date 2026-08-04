#!/usr/bin/env python3
"""Apply frozen Human Pose wrist refiners to the independent industrial test set."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from train_human_pose_3d_pilot import Corrector, baseline

STRICT = {
    "pelvis_center": (0, 18), "left_hip": (1, 19), "right_hip": (2, 23),
    "spine": (6, 17), "left_knee": (4, 20), "right_knee": (5, 24),
    "left_ankle": (7, 21), "right_ankle": (8, 25), "left_foot": (10, 22),
    "right_foot": (11, 26), "neck": (12, 4), "left_shoulder": (16, 5),
    "right_shoulder": (17, 11), "left_elbow": (18, 6), "right_elbow": (19, 12),
    "left_wrist": (20, 7), "right_wrist": (21, 13),
}
MODELS = {}
DETECTOR = None
ARCHIVES = {}


def joint_angle(a, b, c) -> float:
    u, v = a-b, c-b
    return math.degrees(math.acos(float(np.clip(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9), -1, 1))))


def pose_angles(points: np.ndarray, industrial: bool) -> dict[str, float]:
    if industrial:
        pelvis, neck, head = points[18], points[4], points[0]
        ls, le, lw = points[5], points[6], points[7]
        rs, re, rw = points[11], points[12], points[13]
        lh, rh = points[[8, 9, 10]].mean(0), points[[14, 15, 16]].mean(0)
        lhip, rhip = points[19], points[23]
    else:
        pelvis, neck, head = points[0], points[12], points[15]
        ls, le, lw, lh = points[16], points[18], points[20], points[22]
        rs, re, rw, rh = points[17], points[19], points[21], points[23]
        lhip, rhip = points[1], points[2]
    trunk = math.degrees(math.atan2(abs(float(neck[0]-pelvis[0])), abs(float(neck[1]-pelvis[1]))+1e-9))
    return {
        "left_shoulder": joint_angle(lhip, ls, le), "right_shoulder": joint_angle(rhip, rs, re),
        "left_elbow": joint_angle(ls, le, lw), "right_elbow": joint_angle(rs, re, rw),
        "left_wrist": joint_angle(le, lw, lh), "right_wrist": joint_angle(re, rw, rh),
        "neck_proxy": joint_angle(pelvis, neck, head), "trunk": trunk,
    }


def angle_bin(name: str, angle: float) -> int:
    thresholds = {
        "left_shoulder": (20, 45, 90), "right_shoulder": (20, 45, 90),
        "left_elbow": (60, 100), "right_elbow": (60, 100),
        "left_wrist": (165, 195), "right_wrist": (165, 195),
        "neck_proxy": (10, 20), "trunk": (5, 20, 60),
    }[name]
    return sum(angle >= x for x in thresholds)


def init_worker(model_path: str, checkpoints: dict[str, str]):
    global DETECTOR, MODELS
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.3, min_pose_presence_confidence=.3,
    )
    DETECTOR = vision.PoseLandmarker.create_from_options(opts)
    for name, path in checkpoints.items():
        model = Corrector(); model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        model.eval(); MODELS[name] = model


def evaluate(record: dict) -> dict:
    try:
        iz = ARCHIVES.setdefault(record["image_archive"], zipfile.ZipFile(record["image_archive"]))
        lz = ARCHIVES.setdefault(record["label_archive"], zipfile.ZipFile(record["label_archive"]))
        label = json.loads(lz.read(record["label_member"]))
        gt_px = np.asarray(label["annotations"][0]["keypoints"], np.float32)
        with Image.open(io.BytesIO(iz.read(record["image_member"]))) as source:
            source = source.convert("RGB"); x1, y1, x2, y2 = map(float, record["bbox"])
            px, py = (x2-x1)*.15, (y2-y1)*.15
            left, top = max(0., x1-px), max(0., y1-py)
            right, bottom = min(source.width, x2+px), min(source.height, y2+py)
            rgb = np.asarray(source.crop((left, top, right, bottom)))
        result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return {"id": record["id"], "status": "no_person"}
        lm = np.asarray([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]], np.float32)
        base = baseline(lm); predictions = {"mediapipe": base}
        with torch.no_grad():
            x = torch.from_numpy(lm.reshape(1, -1)); b = torch.from_numpy(base[None])
            for name, model in MODELS.items(): predictions[name] = model(x, b).numpy()[0]
        crop_size = np.asarray([right-left, bottom-top], np.float32)
        gt = (gt_px[:, :2] - np.asarray([left, top])) / crop_size
        diag = max(float(np.hypot(x2-x1, y2-y1)), 1.)
        errors, angle_errors = [], []
        gt_angles = pose_angles(gt, True)
        for model_name, pred in predictions.items():
            for joint, (pi, gi) in STRICT.items():
                pixel = float(np.linalg.norm((pred[pi] - gt[gi]) * crop_size))
                errors.append({"model": model_name, "joint": joint, "pixel_error": pixel,
                               "normalized_error": pixel/diag, "visibility": int(gt_px[gi, 2])})
            pred_angles = pose_angles(pred, False)
            for part, actual in gt_angles.items():
                guessed = pred_angles[part]
                angle_errors.append({"model": model_name, "part": part, "absolute_error": abs(guessed-actual),
                                     "same_bin": angle_bin(part, guessed) == angle_bin(part, actual)})
        return {"id": record["id"], "status": "ok", "errors": errors, "angle_errors": angle_errors}
    except Exception as exc:
        return {"id": record.get("id"), "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def summarize(rows: list[dict]) -> list[dict]:
    values = defaultdict(list)
    for row in rows:
        for x in row.get("errors", []): values[(x["model"], x["joint"])].append(x)
    output = []
    for (model, joint), items in sorted(values.items()):
        norm = np.asarray([x["normalized_error"] for x in items]); pixel = np.asarray([x["pixel_error"] for x in items])
        output.append({"model": model, "joint": joint, "valid_gt_count": len(items), "prediction_count": len(items),
                       "mean_pixel_error": float(pixel.mean()), "mean_normalized_error": float(norm.mean()),
                       "pck_0_05": float((norm <= .05).mean()), "pck_0_10": float((norm <= .10).mean()),
                       "confidence": "not stored; do not interpret PCK as confidence"})
    for model in sorted({x["model"] for x in output}):
        items = [e for r in rows for e in r.get("errors", []) if e["model"] == model]
        norm = np.asarray([x["normalized_error"] for x in items]); pixel = np.asarray([x["pixel_error"] for x in items])
        output.append({"model": model, "joint": "OVERALL_STRICT_COMMON", "valid_gt_count": len(items), "prediction_count": len(items),
                       "mean_pixel_error": float(pixel.mean()), "mean_normalized_error": float(norm.mean()),
                       "pck_0_05": float((norm <= .05).mean()), "pck_0_10": float((norm <= .10).mean()),
                       "confidence": "not stored; do not interpret PCK as confidence"})
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=root/"artifacts/aihub/audit/split_manifest.jsonl")
    p.add_argument("--output", type=Path, default=root/"results/wrist_refiner_large_scale_comparison")
    p.add_argument("--model", type=Path, default=root/"mediapipe/models/pose_landmarker_lite.task")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--limit", type=int); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "refiner_5k": root/"checkpoints/human_pose_3d_v1/best.pt",
        "refiner_20k": root/"checkpoints/wrist_refiner_20k_v1/best.pt",
        "refiner_50k": root/"checkpoints/wrist_refiner_50k_v1/best.pt",
    }
    missing = [str(x) for x in checkpoints.values() if not x.exists()]
    if missing: raise FileNotFoundError(f"Train models before transfer evaluation: {missing}")
    records = [json.loads(x) for x in args.manifest.read_text().splitlines() if x]
    records = [x for x in records if x["split"] == "test"]
    if args.limit: records = records[:args.limit]
    detail = args.output/"industrial_refiner_errors.jsonl"; done = {}
    if detail.exists():
        for line in detail.read_text().splitlines():
            row=json.loads(line); done[row["id"]]=row
    pending = [x for x in records if x["id"] not in done]
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.workers, init_worker, (str(args.model), {k:str(v) for k,v in checkpoints.items()})) as pool, detail.open("a") as out:
        for n, row in enumerate(pool.imap_unordered(evaluate, pending, chunksize=8), 1):
            done[row["id"]]=row; out.write(json.dumps(row, ensure_ascii=False)+"\n")
            if n % 100 == 0: out.flush(); print(json.dumps({"phase":"industrial_transfer","completed":len(done),"total":len(records)}), flush=True)
    rows=list(done.values()); table=summarize(rows)
    with (args.output/"joint_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    wrists=[x for x in table if x["joint"] in ("left_wrist","right_wrist")]
    with (args.output/"industrial_wrist_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=wrists[0].keys());w.writeheader();w.writerows(wrists)
    report={"evaluated_images":len(rows),"detected_images":sum(x.get("status")=="ok" for x in rows),
            "detection_rate":sum(x.get("status")=="ok" for x in rows)/len(rows),
            "normalization":"pixel error / AI-Hub annotated person bbox diagonal","strict_common_joint_mapping":STRICT,
            "metrics":table,"failure_counts":{s:sum(x.get("status")==s for x in rows) for s in ("no_person","error")}}
    (args.output/"industrial_transfer_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    angles=[]
    grouped=defaultdict(list)
    for row in rows:
        for item in row.get("angle_errors",[]): grouped[(item["model"],item["part"])].append(item)
    for (model,part),items in sorted(grouped.items()):
        err=np.asarray([x["absolute_error"] for x in items])
        angles.append({"model":model,"body_part":part,"count":len(items),"mae_degrees":float(err.mean()),
                       "median_absolute_error":float(np.median(err)),"p90_absolute_error":float(np.percentile(err,90)),
                       "rula_like_bin_agreement":float(np.mean([x["same_bin"] for x in items])),
                       "note":"2D proxy; wrist twist/pronation/supination is not evaluated"})
    with (args.output/"rula_angle_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=angles[0].keys());w.writeheader();w.writerows(angles)


if __name__ == "__main__": main()
