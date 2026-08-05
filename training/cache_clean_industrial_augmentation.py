#!/usr/bin/env python3
"""Build a separate photometric-augmentation cache for clean industrial data."""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from train_human_pose_3d_pilot import baseline

ROOT = Path(__file__).resolve().parents[1]
STRICT = {0:18, 1:19, 2:23, 4:20, 5:24, 6:17, 7:21, 8:25, 10:22, 11:26, 12:4, 16:5, 17:11, 18:6, 19:12, 20:7, 21:13}
DETECTOR = None
ARCHIVES: dict[str, zipfile.ZipFile] = {}


def init_worker(model: str) -> None:
    global DETECTOR
    DETECTOR = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=.3, min_pose_presence_confidence=.3))


def augment(image: Image.Image, seed: int, variant: int) -> tuple[Image.Image, str]:
    if variant == 0:
        return image, "original_clean"
    rng = random.Random(seed + variant * 1000003)
    if variant == 1:
        output = ImageEnhance.Brightness(image).enhance(rng.uniform(.80, 1.20))
        output = ImageEnhance.Contrast(output).enhance(rng.uniform(.85, 1.15))
        return output, "brightness_contrast"
    stream = io.BytesIO()
    image.filter(ImageFilter.GaussianBlur(rng.uniform(0, .55))).save(stream, format="JPEG", quality=rng.randint(55, 90))
    return Image.open(io.BytesIO(stream.getvalue())).convert("RGB"), "jpeg_blur"


def process(record: dict[str, object]) -> list[dict[str, object]]:
    try:
        image_zip = ARCHIVES.setdefault(str(record["image_archive"]), zipfile.ZipFile(str(record["image_archive"])))
        label_zip = ARCHIVES.setdefault(str(record["label_archive"]), zipfile.ZipFile(str(record["label_archive"])))
        label = json.loads(label_zip.read(str(record["label_member"])))
        keypoints = np.asarray(label["annotations"][0]["keypoints"], np.float32)
        with Image.open(io.BytesIO(image_zip.read(str(record["image_member"])))) as source:
            original_width, original_height = source.size
            source.draft("RGB", (960, 540)); source = source.convert("RGB")
            sx, sy = source.width / original_width, source.height / original_height
            keypoints[:, :2] *= np.asarray([sx, sy], np.float32)
            x1, y1, x2, y2 = map(float, record["bbox"]); x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
            px, py = (x2 - x1) * .15, (y2 - y1) * .15
            left, top = max(0., x1 - px), max(0., y1 - py)
            right, bottom = min(source.width, x2 + px), min(source.height, y2 + py)
            crop = source.crop((left, top, right, bottom))
        gt27 = (keypoints[:, :2] - np.asarray([left, top], np.float32)) / np.asarray([right-left, bottom-top], np.float32)
        norm_scale = np.asarray([right-left, bottom-top], np.float32) / max(float(np.hypot(x2-x1, y2-y1)), 1.)
        rows = []
        stable_seed = sum((index + 1) * ord(char) for index, char in enumerate(str(record["id"])))
        for variant in range(3):
            image, variant_name = augment(crop, stable_seed, variant)
            result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
            if not result.pose_landmarks:
                continue
            lm = np.asarray([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]], np.float32)
            base = baseline(lm); target = base.copy(); mask = np.zeros(24, bool)
            for human, aihub in STRICT.items(): target[human] = gt27[aihub]; mask[human] = True
            target[22] = gt27[[8, 9, 10]].mean(axis=0); target[23] = gt27[[14, 15, 16]].mean(axis=0)
            mask[22] = mask[23] = True
            rows.append({"id": f'{record["id"]}__aug{variant}', "source_id": record["id"], "variant": variant_name,
                         "x": lm.reshape(-1), "base": base, "gt": target, "mask": mask, "norm_scale": norm_scale})
        return rows
    except Exception as exc:
        return [{"id": record.get("id"), "error": f"{type(exc).__name__}: {exc}"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=ROOT / "experiments/industrial_clean_aug_v1/clean_train_manifest.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/aihub/audit/split_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/human_pose_3d_sample/industrial_clean_aug_cache_v1")
    parser.add_argument("--model", type=Path, default=ROOT / "mediapipe/models/pose_landmarker_lite.task")
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    selected = {row["id"] for row in map(json.loads, args.selection.read_text().splitlines())}
    records = [row for row in map(json.loads, args.manifest.read_text().splitlines()) if row["id"] in selected and row["split"] == "train"]
    done = set()
    for file in args.output.glob("chunk_*.npz"): done.update(np.load(file)["source_ids"].astype(str))
    pending = [row for row in records if row["id"] not in done]
    buffer = []; failures = args.output / "failures.jsonl"; started = time.time()
    with get_context("spawn").Pool(args.workers, init_worker, (str(args.model),)) as pool, failures.open("a") as fail:
        for count, result_rows in enumerate(pool.imap_unordered(process, pending, chunksize=4), 1):
            for row in result_rows:
                if "error" in row: fail.write(json.dumps(row, ensure_ascii=False) + "\n")
                else: buffer.append(row)
            if len(buffer) >= args.chunk_size:
                chunk_index = len(list(args.output.glob("chunk_*.npz")))
                np.savez_compressed(args.output / f"chunk_{chunk_index:04d}.npz",
                    ids=np.asarray([row["id"] for row in buffer]), source_ids=np.asarray([row["source_id"] for row in buffer]),
                    variant=np.asarray([row["variant"] for row in buffer]), x=np.asarray([row["x"] for row in buffer], np.float32),
                    base=np.asarray([row["base"] for row in buffer], np.float32), gt=np.asarray([row["gt"] for row in buffer], np.float32),
                    mask=np.asarray([row["mask"] for row in buffer], bool), norm_scale=np.asarray([row["norm_scale"] for row in buffer], np.float32)); buffer = []
            if count % 100 == 0:
                rate = count / max(time.time() - started, 1e-6)
                print(json.dumps({"completed_sources": len(done)+count, "total_sources": len(records), "percent": round((len(done)+count)/len(records)*100, 2), "rate": round(rate, 2)}), flush=True)
    if buffer:
        chunk_index = len(list(args.output.glob("chunk_*.npz")))
        np.savez_compressed(args.output / f"chunk_{chunk_index:04d}.npz",
            ids=np.asarray([row["id"] for row in buffer]), source_ids=np.asarray([row["source_id"] for row in buffer]), variant=np.asarray([row["variant"] for row in buffer]),
            x=np.asarray([row["x"] for row in buffer], np.float32), base=np.asarray([row["base"] for row in buffer], np.float32), gt=np.asarray([row["gt"] for row in buffer], np.float32),
            mask=np.asarray([row["mask"] for row in buffer], bool), norm_scale=np.asarray([row["norm_scale"] for row in buffer], np.float32))
    print(json.dumps({"source_count": len(records), "variants_requested": len(records)*3, "output": str(args.output)}), flush=True)


if __name__ == "__main__": main()
