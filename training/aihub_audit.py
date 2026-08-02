#!/usr/bin/env python3
"""Audit MoveGuard AI-Hub ZIP archives and build leakage-safe manifests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
POSTURE_FIELDS = ("neck", "trunk", "upperarm", "lowerarm", "legs")
FLAG_FIELDS = (
    "neck_twist", "neck_bending", "trunk_twist", "trunk_bending",
    "trunk_supported", "legs_supported", "upperarm_abduction",
    "upperarm_raised", "upperarm_supported", "upperarm_across",
    "lowerarm_outbody", "wrist_bending", "wrist_twisted",
    "wrist_twisted_end",
)


def norm_member(name: str) -> str:
    return PurePosixPath(name.lstrip("/")).name


def group_id(filename: str) -> str:
    """Keep worker/video/continuous frames together; drop only frame suffix."""
    stem = Path(filename).stem
    return re.sub(r"_\d+$", "", stem)


def stable_split(groups: list[str], seed: int) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = round(n * 0.70)
    val_end = train_end + round(n * 0.15)
    return {
        group: "train" if i < train_end else "validation" if i < val_end else "test"
        for i, group in enumerate(shuffled)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--skip-crc", action="store_true", help="Skip ZIP CRC when it was already completed separately")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    archives = sorted(args.data_root.rglob("*.zip"))
    image_index: dict[str, tuple[str, str]] = {}
    json_index: dict[str, tuple[str, str]] = {}
    duplicate_names: list[dict[str, str]] = []
    corrupt_archives: list[str] = []

    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as zf:
                if not args.skip_crc:
                    bad = zf.testzip()
                    if bad:
                        corrupt_archives.append(f"{archive}:{bad}")
                for member in zf.namelist():
                    name = norm_member(member)
                    suffix = Path(name).suffix.lower()
                    target = json_index if suffix == ".json" else image_index if suffix in IMAGE_SUFFIXES else None
                    if target is None:
                        continue
                    key = Path(name).stem
                    if key in target:
                        duplicate_names.append({"stem": key, "first": target[key][0], "second": str(archive)})
                    else:
                        target[key] = (str(archive), member)
        except zipfile.BadZipFile:
            corrupt_archives.append(str(archive))

    paired = sorted(image_index.keys() & json_index.keys())
    image_only = sorted(image_index.keys() - json_index.keys())
    json_only = sorted(json_index.keys() - image_index.keys())
    label_counts = {field: collections.Counter() for field in POSTURE_FIELDS + FLAG_FIELDS}
    workers: collections.Counter[str] = collections.Counter()
    factories: collections.Counter[str] = collections.Counter()
    processes: collections.Counter[str] = collections.Counter()
    invalid_json: list[dict[str, str]] = []
    rula_fields: collections.Counter[str] = collections.Counter()
    records: list[dict[str, object]] = []

    json_archives: dict[str, zipfile.ZipFile] = {}
    try:
        for stem in paired:
            image_archive, image_member = image_index[stem]
            label_archive, label_member = json_index[stem]
            try:
                if label_archive not in json_archives:
                    json_archives[label_archive] = zipfile.ZipFile(label_archive)
                zf = json_archives[label_archive]
                data = json.loads(zf.read(label_member))
                image = (data.get("images") or [{}])[0]
                ann = (data.get("annotations") or [{}])[0]
                posture = ann.get("posture") or {}
                for field in POSTURE_FIELDS:
                    label_counts[field][str(posture.get(field, "missing"))] += 1
                for field in FLAG_FIELDS:
                    label_counts[field][str(ann.get(field, "missing"))] += 1
                workers[str(ann.get("worker", "missing"))] += 1
                factories[str(ann.get("factory", "missing"))] += 1
                processes[str(ann.get("process", "missing"))] += 1
                for key in ann:
                    if "rula" in key.lower():
                        rula_fields[key] += 1
                keypoints = ann.get("keypoints")
                records.append({
                    "id": stem,
                    "file_name": image.get("file_name", Path(image_member).name),
                    "image_archive": image_archive,
                    "image_member": image_member,
                    "label_archive": label_archive,
                    "label_member": label_member,
                    "group_id": group_id(stem),
                    "worker": ann.get("worker"),
                    "factory": ann.get("factory"),
                    "process": ann.get("process"),
                    "keypoint_count": len(keypoints) if isinstance(keypoints, list) else 0,
                    "bbox": ann.get("bbox"),
                    "image_width": image.get("width"),
                    "image_height": image.get("height"),
                    "labels": {**{f: posture.get(f) for f in POSTURE_FIELDS}, **{f: ann.get(f) for f in FLAG_FIELDS}},
                })
            except Exception as exc:  # keep auditing after individual failures
                invalid_json.append({"id": stem, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        for zf in json_archives.values():
            zf.close()

    splits = stable_split(sorted({str(r["group_id"]) for r in records}), args.seed)
    split_counts: collections.Counter[str] = collections.Counter()
    manifest_path = args.output / "split_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for record in records:
            record["split"] = splits[str(record["group_id"])]
            split_counts[str(record["split"])] += 1
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "data_root": str(args.data_root.resolve()),
        "archives": [str(p.resolve()) for p in archives],
        "seed": args.seed,
        "image_count": len(image_index),
        "json_count": len(json_index),
        "paired_count": len(paired),
        "image_without_label_count": len(image_only),
        "label_without_image_count": len(json_only),
        "duplicate_name_count": len(duplicate_names),
        "invalid_json_count": len(invalid_json),
        "corrupt_archive_count": len(corrupt_archives),
        "group_count": len(splits),
        "split_counts": dict(split_counts),
        "label_counts": {k: dict(v) for k, v in label_counts.items()},
        "workers": dict(workers),
        "factories": dict(factories),
        "processes": dict(processes),
        "expert_rula_fields": dict(rula_fields),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "examples": {
            "image_without_label": image_only[:20],
            "label_without_image": json_only[:20],
            "duplicates": duplicate_names[:20],
            "invalid_json": invalid_json[:20],
            "corrupt_archives": corrupt_archives[:20],
        },
    }
    (args.output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
