#!/usr/bin/env python3
"""Compare MoveGuard RULA with an AI-Hub human-label-derived reference.

This is a posture-only agreement study, not expert final-RULA validation.  The
AI-Hub JSON has human-reviewed posture/adjustment labels but no final RULA,
force, load, repetition, or muscle-use ground truth.  Missing inputs are fixed
identically for both paths and the independent test split is evaluated once.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rula import calculate


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a[:2] - b[:2], c[:2] - b[:2]
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator == 0:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denominator, -1, 1))))


def moveguard_components(points: np.ndarray) -> dict[str, int]:
    """Mirror the browser baseline's 2-D angle and threshold rules."""
    left_visibility = float(points[[11, 13, 15, 23], 3].mean())
    right_visibility = float(points[[12, 14, 16, 24], 3].mean())
    shoulder, elbow, wrist, hip = (12, 14, 16, 24) if right_visibility >= left_visibility else (11, 13, 15, 23)
    shoulder_mid = (points[11, :3] + points[12, :3]) / 2
    hip_mid = (points[23, :3] + points[24, :3]) / 2
    trunk_angle = math.degrees(math.atan2(abs(float(shoulder_mid[0] - hip_mid[0])), abs(float(shoulder_mid[1] - hip_mid[1])) + 1e-9))
    neck_angle = angle(hip_mid, shoulder_mid, points[0])
    neck_angle = abs(180.0 - neck_angle)
    upper_arm_angle = angle(points[hip], points[shoulder], points[elbow])
    elbow_angle = angle(points[shoulder], points[elbow], points[wrist])
    return {
        "neck": 1 if neck_angle <= 10 else 2 if neck_angle <= 20 else 3,
        "trunk": 1 if trunk_angle <= 5 else 2 if trunk_angle <= 20 else 3 if trunk_angle <= 60 else 4,
        "upper_arm": 1 if upper_arm_angle <= 20 else 2 if upper_arm_angle <= 45 else 3 if upper_arm_angle <= 90 else 4,
        "lower_arm": 1 if 60 <= elbow_angle <= 100 else 2,
        "wrist": 1,
        "wrist_twist": 1,
    }


def reference_components(labels: dict[str, int]) -> dict[str, int]:
    """Convert official AI-Hub posture enums/flags to posture-only RULA inputs."""
    neck = labels["neck"] + 1
    neck += int(bool(labels.get("neck_twist"))) + int(bool(labels.get("neck_bending")))

    # The released JSON has only upright vs >=20-degree bent trunk posture.
    trunk = 1 if labels["trunk"] == 0 else 3
    trunk += int(bool(labels.get("trunk_twist"))) + int(bool(labels.get("trunk_bending")))

    upper_arm = labels["upperarm"] + 1
    upper_arm += int(bool(labels.get("upperarm_abduction"))) + int(bool(labels.get("upperarm_raised")))
    upper_arm -= int(bool(labels.get("upperarm_supported")))

    lower_arm = {0: 2, 1: 1, 2: 2}[labels["lowerarm"]]
    lower_arm += int(bool(labels.get("lowerarm_outbody")))

    wrist = 1 + int(bool(labels.get("wrist_bending")))
    wrist_twist = 2 if labels.get("wrist_twisted_end") else 1
    legs = 1 if labels.get("legs_supported") else 2
    return {
        "neck": max(1, min(6, neck)), "trunk": max(1, min(6, trunk)),
        "upper_arm": max(1, min(4, upper_arm)), "lower_arm": max(1, min(3, lower_arm)),
        "wrist": max(1, min(4, wrist)), "wrist_twist": wrist_twist, "legs": legs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    reference_final: list[int] = []
    predicted_final: list[int] = []
    reference_action: list[int] = []
    predicted_action: list[int] = []
    component_pairs = {k: ([], []) for k in ("neck", "trunk", "upper_arm", "lower_arm")}

    for line in args.features.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("status") != "ok" or record.get("split") != args.split:
            continue
        points = np.asarray(record["features"][:165], dtype=float).reshape(33, 5)
        ref = reference_components(record["labels"])
        pred = moveguard_components(points)
        # Non-visual/user inputs are held equal so this isolates automatic pose scoring.
        pred.update({"wrist": 1, "wrist_twist": 1, "legs": ref["legs"]})
        fixed = {"muscle_a": 0, "force_a": 0, "muscle_b": 0, "force_b": 0}
        ref_result = calculate({**ref, **fixed})
        pred_result = calculate({**pred, **fixed})
        reference_final.append(int(ref_result["final_score"]))
        predicted_final.append(int(pred_result["final_score"]))
        reference_action.append(int(ref_result["action_level"]))
        predicted_action.append(int(pred_result["action_level"]))
        for key, pair in component_pairs.items():
            pair[0].append(ref[key]); pair[1].append(pred[key])

    ref_array, pred_array = np.asarray(reference_final), np.asarray(predicted_final)
    report = {
        "study_name": "AI-Hub human-label-derived posture-only RULA agreement",
        "split": args.split, "evaluated_frames": len(reference_final),
        "assumptions": {
            "force_load": 0, "muscle_repetition": 0,
            "moveguard_wrist": "neutral", "legs_support": "same AI-Hub human label in both paths",
            "claim_limit": "Not expert final-RULA validation; AI-Hub has no expert final score.",
        },
        "final_score": {
            "exact_agreement": float(np.mean(ref_array == pred_array)),
            "within_one_agreement": float(np.mean(np.abs(ref_array - pred_array) <= 1)),
            "mean_absolute_error": float(np.mean(np.abs(ref_array - pred_array))),
            "weighted_cohen_kappa": float(cohen_kappa_score(ref_array, pred_array, weights="quadratic")),
            "confusion_matrix": confusion_matrix(ref_array, pred_array, labels=list(range(1, 8))).tolist(),
        },
        "action_level": {
            "agreement": float(accuracy_score(reference_action, predicted_action)),
            "weighted_cohen_kappa": float(cohen_kappa_score(reference_action, predicted_action, weights="quadratic")),
            "confusion_matrix": confusion_matrix(reference_action, predicted_action, labels=[1, 2, 3, 4]).tolist(),
        },
        "components": {
            key: {"exact_agreement": float(accuracy_score(ref, pred)), "confusion_matrix": confusion_matrix(ref, pred).tolist()}
            for key, (ref, pred) in component_pairs.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
