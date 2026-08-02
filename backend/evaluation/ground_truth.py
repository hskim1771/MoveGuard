from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import JOINT_NAMES

IDENTITY_FIELDS = [
    "video_id",
    "frame_id",
    "frame_index",
    "timestamp_ms",
    "person_id",
    "width_px",
    "height_px",
    "coordinate_space",
]
METADATA_FIELDS = [
    "split",
    "annotator_id",
    "reviewed_by",
    "occlusion",
    "camera_angle",
    "distance",
    "lighting",
    "notes",
]
JOINT_FIELDS = [
    field
    for joint in JOINT_NAMES
    for field in (
        f"{joint}_x",
        f"{joint}_y",
        f"{joint}_z",
        f"{joint}_visible",
        f"{joint}_world_x",
        f"{joint}_world_y",
        f"{joint}_world_z",
    )
]
ANGLE_FIELDS = [
    "neck_angle_deg",
    "trunk_angle_deg",
    "left_upper_arm_angle_deg",
    "right_upper_arm_angle_deg",
    "left_lower_arm_angle_deg",
    "right_lower_arm_angle_deg",
    "left_wrist_angle_deg",
    "right_wrist_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
]
POSTURE_FIELDS = ["primary_posture_label", "posture_labels"]
RULA_FIELDS = [
    "rula_final_score",
    "rula_risk_level",
    "rula_neck_score",
    "rula_trunk_score",
    "rula_upper_arm_score",
    "rula_lower_arm_score",
    "rula_wrist_score",
    "rula_leg_score",
    "rula_scored_side",
]
GROUND_TRUTH_FIELDS = (
    IDENTITY_FIELDS
    + METADATA_FIELDS
    + JOINT_FIELDS
    + ANGLE_FIELDS
    + POSTURE_FIELDS
    + RULA_FIELDS
)


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value: Any) -> int | None:
    parsed = number(value)
    return None if parsed is None else int(parsed)


def truthy(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "visible"}:
        return True
    if normalized in {"0", "false", "no", "n", "occluded"}:
        return False
    return None


@dataclass(frozen=True)
class GroundTruthStore:
    rows: tuple[dict[str, Any], ...]

    def find(
        self,
        *,
        video_id: str,
        frame_index: int,
        frame_id: str,
        person_id: str | None,
    ) -> dict[str, Any] | None:
        candidates = [
            row
            for row in self.rows
            if str(row.get("video_id", "")) == video_id
            and integer(row.get("frame_index")) == frame_index
        ]
        if not candidates:
            candidates = [
                row for row in self.rows if str(row.get("frame_id", "")) == frame_id
            ]
        if person_id:
            person_match = [
                row
                for row in candidates
                if str(row.get("person_id", "")) == person_id
            ]
            if person_match:
                return person_match[0]
        return candidates[0] if candidates else None

    @property
    def available(self) -> bool:
        return bool(self.rows)


def _flatten_json_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {field: item.get(field, "") for field in IDENTITY_FIELDS}
    metadata = item.get("metadata") or {}
    for field in METADATA_FIELDS:
        row[field] = metadata.get(field, item.get(field, ""))
    for joint, values in (item.get("joints") or {}).items():
        if joint not in JOINT_NAMES or not isinstance(values, dict):
            continue
        for suffix in ("x", "y", "z", "visible", "world_x", "world_y", "world_z"):
            row[f"{joint}_{suffix}"] = values.get(suffix, "")
    for name, value in (item.get("angles_deg") or {}).items():
        row[f"{name}_angle_deg"] = value
    posture = item.get("posture") or {}
    row["primary_posture_label"] = posture.get("primary", "")
    labels = posture.get("labels", "")
    row["posture_labels"] = (
        ";".join(str(label) for label in labels)
        if isinstance(labels, list)
        else labels
    )
    rula = item.get("rula") or {}
    for source, target in {
        "final_score": "rula_final_score",
        "risk_level": "rula_risk_level",
        "neck_score": "rula_neck_score",
        "trunk_score": "rula_trunk_score",
        "upper_arm_score": "rula_upper_arm_score",
        "lower_arm_score": "rula_lower_arm_score",
        "wrist_score": "rula_wrist_score",
        "leg_score": "rula_leg_score",
        "scored_side": "rula_scored_side",
    }.items():
        row[target] = rula.get(source, "")
    return row


def load_ground_truth(path: str | Path | None) -> GroundTruthStore:
    if path is None:
        return GroundTruthStore(())
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Ground Truth file not found: {resolved}")
    if resolved.suffix.lower() == ".csv":
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            return GroundTruthStore(tuple(csv.DictReader(handle)))
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        items = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("Ground Truth JSON must be a list or contain a frames list.")
        return GroundTruthStore(tuple(_flatten_json_row(item) for item in items))
    raise ValueError("Ground Truth must be CSV or JSON.")


def write_annotation_csv(
    path: str | Path,
    sampled_frames: Iterable[dict[str, Any]],
) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUND_TRUTH_FIELDS)
        writer.writeheader()
        for frame in sampled_frames:
            row = {field: "" for field in GROUND_TRUTH_FIELDS}
            row.update(frame)
            row["person_id"] = row.get("person_id") or "person-1"
            row["coordinate_space"] = "image_normalized"
            row["split"] = row.get("split") or "test"
            writer.writerow(row)


def write_annotation_json(path: str | Path, sampled_frames: Iterable[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in sampled_frames:
        frames.append(
            {
                **frame,
                "person_id": frame.get("person_id") or "person-1",
                "coordinate_space": "image_normalized",
                "metadata": {
                    "split": "test",
                    "annotator_id": "",
                    "reviewed_by": "",
                    "occlusion": "",
                    "camera_angle": "",
                    "distance": "",
                    "lighting": "",
                    "notes": "",
                },
                "joints": {
                    name: {
                        "x": None,
                        "y": None,
                        "z": None,
                        "visible": None,
                        "world_x": None,
                        "world_y": None,
                        "world_z": None,
                    }
                    for name in JOINT_NAMES
                },
                "angles_deg": {name.removesuffix("_angle_deg"): None for name in ANGLE_FIELDS},
                "posture": {"primary": None, "labels": []},
                "rula": {
                    "final_score": None,
                    "risk_level": None,
                    "neck_score": None,
                    "trunk_score": None,
                    "upper_arm_score": None,
                    "lower_arm_score": None,
                    "wrist_score": None,
                    "leg_score": None,
                    "scored_side": None,
                },
            }
        )
    resolved.write_text(
        json.dumps({"schema_version": "1.0", "frames": frames}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

