#!/usr/bin/env python3
"""Measure RULA frame-selection repeatability without expert score labels.

This is a resampling reliability test, not an expert-agreement accuracy test.
It repeatedly removes a random 20% of frames from each existing independent
test group and checks whether the selected group score remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluate_wrist_temporal_pipeline import frame_rula


def aggregate(scores: list[int], method: str) -> float | None:
    if not scores:
        return None
    ordered = sorted(scores, reverse=True)
    if method == "max":
        return float(ordered[0])
    top = ordered[: min(3, len(ordered))]
    if method == "top3_median":
        return float(np.median(top))
    if method == "top3_mean":
        return float(np.mean(top))
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "experiments/wrist_rula_pipeline_validation/independent_test_pose_cache.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/rula_repeatability/reliability_report.json",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sample-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    groups: dict[str, list[int]] = defaultdict(list)
    detected = 0
    for line in args.cache.read_text().splitlines():
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        detected += 1
        points = np.asarray(row["raw"], dtype=np.float32)
        confidence = np.asarray(row["raw_confidence"], dtype=np.float32)
        score = frame_rula(points, confidence, "raw")
        if score is not None:
            groups[row["group_id"]].append(score)

    rng = np.random.default_rng(args.seed)
    methods = ("max", "top3_median", "top3_mean")
    agreement = {method: [] for method in methods}
    deviations = {method: [] for method in methods}
    per_group = []

    for group_id, scores in sorted(groups.items()):
        sample_size = max(3, int(round(len(scores) * args.sample_ratio)))
        sample_size = min(sample_size, len(scores))
        references = {method: aggregate(scores, method) for method in methods}
        group_matches = {method: 0 for method in methods}
        group_deviation = {method: [] for method in methods}
        for _ in range(args.repeats):
            indices = rng.choice(len(scores), size=sample_size, replace=False)
            sampled = [scores[int(index)] for index in indices]
            for method in methods:
                value = aggregate(sampled, method)
                exact = value == references[method]
                group_matches[method] += int(exact)
                group_deviation[method].append(abs(float(value) - float(references[method])))
                agreement[method].append(int(exact))
                deviations[method].append(group_deviation[method][-1])
        per_group.append(
            {
                "group_id": group_id,
                "frames": len(scores),
                "sampled_frames": sample_size,
                "full_scores": references,
                "exact_repeatability": {
                    method: group_matches[method] / args.repeats for method in methods
                },
                "mean_absolute_score_change": {
                    method: float(np.mean(group_deviation[method])) for method in methods
                },
            }
        )

    report = {
        "definition": "80% frame resampling repeatability; not expert RULA accuracy",
        "source": "fixed independent AI-Hub industrial test split MediaPipe raw cache",
        "detected_images": detected,
        "groups": len(groups),
        "repeats_per_group": args.repeats,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "methods": {
            method: {
                "exact_repeatability": float(np.mean(agreement[method])),
                "mean_absolute_score_change": float(np.mean(deviations[method])),
                "changed_runs": int(len(agreement[method]) - sum(agreement[method])),
                "total_runs": len(agreement[method]),
            }
            for method in methods
        },
        "per_group": per_group,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
