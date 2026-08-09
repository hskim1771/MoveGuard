#!/usr/bin/env python3
"""Select a clean, diverse subset from the existing industrial train split.

The source split and source data are read-only. Selection metadata is written
to a new experiment directory; validation and test records are never selected.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CRITICAL_MP = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
SEGMENTS = ((16, 18), (18, 20), (17, 19), (19, 21), (1, 4), (4, 7), (2, 5), (5, 8))


def load_chunks(path: Path) -> dict[str, np.ndarray]:
    chunks = [np.load(file) for file in sorted(path.glob("chunk_*.npz"))]
    keys = ("ids", "split", "x", "base", "gt", "mask", "norm_scale")
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}


def frame_number(identifier: str) -> int:
    match = re.search(r"_(\d+)$", identifier)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/human_pose_3d_sample/industrial_raw_cache_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/aihub/audit/split_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/industrial_clean_aug_v1")
    parser.add_argument("--mean-confidence", type=float, default=0.65)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--max-mean-error", type=float, default=0.25)
    parser.add_argument("--min-frame-gap", type=int, default=2)
    args = parser.parse_args()

    arrays = load_chunks(args.cache)
    metadata = {row["id"]: row for row in map(json.loads, args.manifest.read_text().splitlines())}
    candidates = []
    rejection = Counter()
    train_indices = np.flatnonzero(arrays["split"].astype(str) == "train")

    for index in train_indices:
        identifier = str(arrays["ids"][index])
        row = metadata.get(identifier)
        if not row or row.get("split") != "train":
            rejection["manifest_or_split"] += 1
            continue
        landmarks = arrays["x"][index].reshape(33, 4)
        base, gt, mask = arrays["base"][index], arrays["gt"][index], arrays["mask"][index]
        norm_scale = arrays["norm_scale"][index]
        confidence = landmarks[list(CRITICAL_MP), 3]
        mean_confidence, min_confidence = float(confidence.mean()), float(confidence.min())
        if mean_confidence < args.mean_confidence:
            rejection["low_mean_confidence"] += 1
            continue
        if min_confidence < args.min_confidence:
            rejection["low_critical_confidence"] += 1
            continue
        inside = ((gt >= -0.05) & (gt <= 1.05)).all(axis=1)
        if not bool(inside[mask].all()):
            rejection["gt_outside_crop"] += 1
            continue
        errors = np.linalg.norm((base - gt) * norm_scale[None, :], axis=1)
        mean_error = float(errors[mask].mean())
        if mean_error > args.max_mean_error:
            rejection["pose_or_label_disagreement"] += 1
            continue
        lengths = np.asarray([np.linalg.norm(gt[a] - gt[b]) for a, b in SEGMENTS])
        if not np.isfinite(lengths).all() or float(lengths.min()) < 0.015:
            rejection["degenerate_segment"] += 1
            continue
        side_ratios = lengths[::2] / np.maximum(lengths[1::2], 1e-6)
        if bool(((side_ratios < 0.30) | (side_ratios > 3.33)).any()):
            rejection["implausible_left_right_ratio"] += 1
            continue
        quality = 0.55 * mean_confidence + 0.25 * min_confidence + 0.20 * max(0.0, 1.0 - mean_error / args.max_mean_error)
        candidates.append({
            "id": identifier,
            "group_id": row["group_id"],
            "frame_index": frame_number(identifier),
            "labels": row["labels"],
            "quality_score": quality,
            "mean_critical_confidence": mean_confidence,
            "min_critical_confidence": min_confidence,
            "mean_normalized_disagreement": mean_error,
        })

    # Remove near-identical consecutive frames but prefer the higher-quality one.
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_group[row["group_id"]].append(row)
    selected = []
    for rows in by_group.values():
        rows.sort(key=lambda item: (-item["quality_score"], item["frame_index"]))
        kept_frames = []
        for row in rows:
            if any(abs(row["frame_index"] - frame) < args.min_frame_gap for frame in kept_frames):
                rejection["near_duplicate_frame"] += 1
                continue
            kept_frames.append(row["frame_index"])
            selected.append(row)
    selected.sort(key=lambda item: item["id"])

    original_labels = Counter(metadata[str(arrays["ids"][index])]["labels"]["wrist_bending"] for index in train_indices if str(arrays["ids"][index]) in metadata)
    selected_labels = Counter(row["labels"]["wrist_bending"] for row in selected)
    report = {
        "source": "industrial train split only",
        "source_train_count": len(train_indices),
        "quality_candidates_before_dedup": len(candidates),
        "selected_count": len(selected),
        "selection_rate": len(selected) / len(train_indices),
        "thresholds": {
            "mean_critical_confidence": args.mean_confidence,
            "min_critical_confidence": args.min_confidence,
            "max_mean_normalized_disagreement": args.max_mean_error,
            "min_frame_gap": args.min_frame_gap,
        },
        "rejection_reasons": dict(rejection),
        "wrist_bending_distribution": {
            "source": dict(original_labels),
            "selected": dict(selected_labels),
        },
        "split_protection": {"train_selected": len(selected), "validation_selected": 0, "test_selected": 0},
        "augmentation_plan": {
            "online_only": True,
            "copies_per_epoch": "random; source files are not duplicated",
            "allowed": ["brightness", "contrast", "jpeg", "small_noise_blur", "rotation_3deg", "translation_scale_5pct", "small_occlusion"],
            "horizontal_flip": "only with left/right keypoint and label remapping",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "clean_train_manifest.jsonl").open("w") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
