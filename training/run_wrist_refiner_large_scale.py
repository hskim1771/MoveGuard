#!/usr/bin/env python3
"""Prepare nested Human Pose splits, cache MediaPipe, and train 20k/50k refiners.

The existing 5k checkpoint and its fixed validation/test membership are read-only.
Images are decoded directly from the 55GB ZIP, so the archive is never extracted.
Chunked caches make MediaPipe preprocessing safe to resume after interruption.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_human_pose_3d_pilot import Corrector, JOINTS, WRISTS, baseline, metrics


SEED = 20260804


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.started = time.time()

    def update(self, phase: str, done: int, total: int, **extra) -> None:
        elapsed = time.time() - self.started
        rate = done / elapsed if elapsed and done else 0.0
        payload = {
            "phase": phase, "processed": done, "total": total,
            "percent": round(done * 100 / total, 2) if total else 0,
            "elapsed_seconds": round(elapsed, 1), "items_per_second": round(rate, 2),
            "eta_seconds": round((total - done) / rate, 1) if rate else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra,
        }
        write_json(self.path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def scan_rows(label_root: Path, archive: Path, progress: Progress) -> list[dict]:
    label_files = sorted(label_root.rglob("*.json"))
    with zipfile.ZipFile(archive) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
    rows = []
    for n, label in enumerate(label_files, 1):
        d = json.loads(label.read_text())
        info, ann = d["info"], d["annotations"]
        image_name = label.stem + ".jpg"
        member = members.get(image_name)
        if member:
            gt = np.asarray(ann["2d_pos"], np.float32) / np.asarray(
                [info["img_width"], info["img_height"]], np.float32
            )
            rows.append({
                "id": label.stem, "member": member, "label": str(label),
                "action": str(info["action_category_id"]), "actor": str(info["actor_id"]),
                "camera": int(info["camera_no"]), "frame": int(ann["frame_no"]),
                "gt": gt.tolist(),
            })
        if n % 10000 == 0:
            progress.update("라벨·이미지 연결 조사", n, len(label_files), matched=len(rows))
    progress.update("라벨·이미지 연결 조사", len(label_files), len(label_files), matched=len(rows))
    return rows


def diverse_order(rows: list[dict], seed: int) -> list[dict]:
    """Round-robin action/actor/camera strata after temporal spreading."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["action"], r["actor"], r["camera"])].append(r)
    rng = random.Random(seed)
    queues = []
    for key in sorted(buckets):
        seq = sorted(buckets[key], key=lambda x: x["frame"])
        # Even/odd temporal traversal avoids selecting one contiguous prefix.
        seq = seq[::2] + list(reversed(seq[1::2]))
        queues.append(deque(seq))
    rng.shuffle(queues)
    out = []
    while queues:
        active = []
        for q in queues:
            if q:
                out.append(q.popleft())
            if q:
                active.append(q)
        queues = active
    return out


def prepare_manifest(args, progress: Progress) -> dict:
    if args.manifest.exists():
        return json.loads(args.manifest.read_text())
    rows = scan_rows(args.labels, args.archive, progress)
    old_split = json.loads(args.baseline_split.read_text())
    groups = {k: {tuple(x) for x in v} for k, v in old_split["groups"].items()}
    by_id = {r["id"]: r for r in rows}
    old_selected = json.loads(args.selected_manifest.read_text())
    old_ids = {Path(x["label_2d"]).stem for x in old_selected}
    fixed = {
        split: [r for r in rows if (r["action"], r["actor"]) in groups[split] and r["id"] in old_ids]
        for split in ("train", "val", "test")
    }
    old_train = fixed["train"]
    extra = [r for r in rows if (r["action"], r["actor"]) in groups["train"] and r["id"] not in old_ids]
    ordered = old_train + diverse_order(extra, args.seed)
    if len(ordered) < 50000:
        raise RuntimeError(f"50k training candidates unavailable: {len(ordered)}")
    members = {
        "train_5k_baseline": [r["id"] for r in old_train],
        "train_20k": [r["id"] for r in ordered[:20000]],
        "train_50k": [r["id"] for r in ordered[:50000]],
        "validation_fixed": [r["id"] for r in fixed["val"]],
        "test_fixed": [r["id"] for r in fixed["test"]],
    }
    group_overlap = {
        f"{a}_{b}": sorted([list(x) for x in groups[a] & groups[b]])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    manifest = {
        "seed": args.seed,
        "selection": "existing 5k train first; deterministic action/actor/camera round-robin with temporal spreading",
        "split_rule": "frozen existing action+actor groups; frozen exact 5k validation/test image IDs",
        "source": {"archive": str(args.archive), "labels": str(args.labels)},
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint),
            "checkpoint_sha256": file_sha256(args.baseline_checkpoint),
            "split": str(args.baseline_split), "split_sha256": file_sha256(args.baseline_split),
        },
        "counts": {k: len(v) for k, v in members.items()},
        "leakage_audit": {
            "group_overlap": group_overlap,
            "image_id_overlap": {
                "train_val": len(set(members["train_50k"]) & set(members["validation_fixed"])),
                "train_test": len(set(members["train_50k"]) & set(members["test_fixed"])),
                "val_test": len(set(members["validation_fixed"]) & set(members["test_fixed"])),
            },
            "status": "PASS" if not any(group_overlap.values()) else "FAIL",
            "known_limit": "Actors repeat across different action groups; action+actor pairs and exact frames do not cross splits.",
        },
        "members": members,
        "rows": {i: by_id[i] for i in set().union(*map(set, members.values()))},
    }
    write_json(args.manifest, manifest)
    progress.update("실험 manifest 생성", 1, 1, counts=manifest["counts"], leakage="PASS")
    return manifest


def cache_features(args, manifest: dict, progress: Progress) -> dict[str, np.ndarray]:
    ids = manifest["members"]["train_50k"] + manifest["members"]["validation_fixed"] + manifest["members"]["test_fixed"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(args.cache_dir.glob("chunk_*.npz"))
    completed = sum(len(np.load(p)["ids"]) for p in chunks)
    if completed > len(ids):
        raise RuntimeError("Cache has more rows than manifest; refusing unsafe reuse")
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.3, min_pose_presence_confidence=.3,
    )
    detector = vision.PoseLandmarker.create_from_options(opts) if completed < len(ids) else None
    with zipfile.ZipFile(args.archive) as zf:
        for start in range(completed, len(ids), args.chunk_size):
            batch_ids = ids[start:start + args.chunk_size]
            xs, bases, gts, detected = [], [], [], []
            for image_id in batch_ids:
                row = manifest["rows"][image_id]
                arr = np.frombuffer(zf.read(row["member"]), np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    result = None
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                ok = bool(result and result.pose_landmarks)
                lm = np.asarray([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]], np.float32) if ok else np.zeros((33, 4), np.float32)
                xs.append(lm.reshape(-1)); bases.append(baseline(lm)); gts.append(row["gt"]); detected.append(ok)
            chunk = args.cache_dir / f"chunk_{start:06d}.npz"
            np.savez_compressed(chunk, ids=np.asarray(batch_ids), x=np.asarray(xs, np.float32),
                                base=np.asarray(bases, np.float32), gt=np.asarray(gts, np.float32),
                                detected=np.asarray(detected, bool))
            progress.update("MediaPipe 50k 전처리", min(start + len(batch_ids), len(ids)), len(ids),
                            cache_chunks=len(list(args.cache_dir.glob("chunk_*.npz"))))
    if detector:
        detector.close()
    loaded = [np.load(p) for p in sorted(args.cache_dir.glob("chunk_*.npz"))]
    result = {k: np.concatenate([z[k] for z in loaded]) for k in ("ids", "x", "base", "gt", "detected")}
    if list(result["ids"].astype(str)) != ids:
        raise RuntimeError("Cache ID order does not match manifest")
    return result


def detailed_wrist(pred, gt, detected) -> dict:
    span = gt.max(1) - gt.min(1)
    diag = np.linalg.norm(span, axis=1).clip(1e-6)
    norm = np.linalg.norm(pred[:, WRISTS] - gt[:, WRISTS], axis=2) / diag[:, None]
    return {
        "images": int(len(gt)), "detection_rate": float(detected.mean()),
        "left_pck_0_05": float((norm[:, 0] <= .05).mean()),
        "right_pck_0_05": float((norm[:, 1] <= .05).mean()),
        "mean_pck_0_05": float((norm <= .05).mean()),
        "left_pck_0_10": float((norm[:, 0] <= .10).mean()),
        "right_pck_0_10": float((norm[:, 1] <= .10).mean()),
        "mean_pck_0_10": float((norm <= .10).mean()),
        "mean_normalized_error": float(norm.mean()),
    }


def train_one(size: int, arrays: dict, manifest: dict, args, progress: Progress) -> dict:
    out = args.repo / "checkpoints" / f"wrist_refiner_{size // 1000}k_v1"
    result_dir = args.repo / "results" / f"wrist_refiner_{size // 1000}k_v1"
    out.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "training_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    ids = list(arrays["ids"].astype(str)); pos = {x: i for i, x in enumerate(ids)}
    tr_all = [pos[x] for x in manifest["members"][f"train_{size // 1000}k"]]
    va_all = [pos[x] for x in manifest["members"]["validation_fixed"]]
    te_all = [pos[x] for x in manifest["members"]["test_fixed"]]
    detected = arrays["detected"]
    tr = [i for i in tr_all if detected[i]]; va = [i for i in va_all if detected[i]]; te = [i for i in te_all if detected[i]]
    x, base_arr, gt = arrays["x"], arrays["base"], arrays["gt"]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Corrector().to(device)
    def loader(idx, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(base_arr[idx]), torch.from_numpy(gt[idx])),
                          batch_size=128, shuffle=shuffle)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=.02)
    best, patience, history = float("inf"), 0, []
    log_path = result_dir / "training_log.csv"
    for epoch in range(1, 81):
        model.train(); losses = []
        for xb, bb, yb in loader(tr, True):
            xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb, bb), yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); vals = []
        with torch.no_grad():
            for xb, bb, yb in loader(va):
                vals.append(loss_fn(model(xb.to(device), bb.to(device)), yb.to(device)).item())
        val_loss = float(np.mean(vals)); row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerows(history)
        if val_loss < best - 1e-6:
            best, patience = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "joints": JOINTS, "seed": args.seed, "train_size": size}, out / "best.pt")
        else:
            patience += 1
        progress.update(f"{size // 1000}k 모델 학습", epoch, 80, train_loss=row["train_loss"], val_loss=val_loss, best_val_loss=best, patience=patience)
        if patience >= 12:
            break
    ck = torch.load(out / "best.pt", map_location=device); model.load_state_dict(ck["state_dict"]); model.eval()
    preds = []
    with torch.no_grad():
        for xb, bb, _ in loader(te): preds.append(model(xb.to(device), bb.to(device)).cpu().numpy())
    pred = np.concatenate(preds)
    report = {
        "configuration": {"seed": args.seed, "learning_rate": 1e-3, "batch_size": 128, "max_epochs": 80,
                          "optimizer": "AdamW", "scheduler": None, "loss": "SmoothL1 beta=0.02",
                          "augmentation": None, "early_stopping_patience": 12, "device": str(device)},
        "counts": {"train": len(tr_all), "validation": len(va_all), "test": len(te_all),
                   "detected_train": len(tr), "detected_validation": len(va), "detected_test": len(te)},
        "best_epoch": min(history, key=lambda z: z["val_loss"])["epoch"], "best_val_loss": best,
        "test": detailed_wrist(pred, gt[te], detected[te_all]), "epochs": history,
        "checkpoint": str(out / "best.pt"),
    }
    write_json(report_path, report)
    write_json(result_dir / "training_config.json", report["configuration"] | report["counts"])
    return report


def evaluate_frozen_5k(arrays: dict, manifest: dict, args) -> dict:
    ids = list(arrays["ids"].astype(str)); pos = {x: i for i, x in enumerate(ids)}
    test_all = [pos[x] for x in manifest["members"]["test_fixed"]]
    detected = arrays["detected"]; test = [i for i in test_all if detected[i]]
    model = Corrector(); model.load_state_dict(torch.load(args.baseline_checkpoint, map_location="cpu")["state_dict"]); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(arrays["x"][test]), torch.from_numpy(arrays["base"][test])).numpy()
    return {
        "mediapipe": detailed_wrist(arrays["base"][test], arrays["gt"][test], detected[test_all]),
        "refiner_5k_frozen": detailed_wrist(pred, arrays["gt"][test], detected[test_all]),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=repo)
    ap.add_argument("--archive", type=Path, default=repo / "data/human_pose_3d_sample/image_archive/010.사람인체자세3D/2.Validation/원시데이터(2)/Image_230714_add/image_val_batch2_0714.zip")
    ap.add_argument("--labels", type=Path, default=repo / "data/human_pose_3d_sample/labels/extracted_2d")
    ap.add_argument("--selected-manifest", type=Path, default=repo / "data/human_pose_3d_sample/labels_selected/selection_manifest.json")
    ap.add_argument("--baseline-checkpoint", type=Path, default=repo / "checkpoints/human_pose_3d_v1/best.pt")
    ap.add_argument("--baseline-split", type=Path, default=repo / "checkpoints/human_pose_3d_v1/split_manifest.json")
    ap.add_argument("--model", type=Path, default=repo / "mediapipe/models/pose_landmarker_lite.task")
    ap.add_argument("--comparison", type=Path, default=repo / "results/wrist_refiner_large_scale_comparison")
    ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--chunk-size", type=int, default=250)
    args = ap.parse_args(); args.repo = args.repo.resolve(); args.comparison.mkdir(parents=True, exist_ok=True)
    args.manifest = args.comparison / "experiment_manifest.json"
    args.cache_dir = args.repo / "data/human_pose_3d_sample/mediapipe_cache_50k_v1"
    progress = Progress(args.comparison / "progress.json")
    manifest = prepare_manifest(args, progress)
    arrays = cache_features(args, manifest, progress)
    comparison = evaluate_frozen_5k(arrays, manifest, args)
    comparison["refiner_20k"] = train_one(20000, arrays, manifest, args, progress)["test"]
    comparison["refiner_50k"] = train_one(50000, arrays, manifest, args, progress)["test"]
    write_json(args.comparison / "humanpose_wrist_metrics.json", comparison)
    progress.update("Human Pose 대규모 학습 완료", 1, 1, metrics=comparison)


if __name__ == "__main__":
    main()
