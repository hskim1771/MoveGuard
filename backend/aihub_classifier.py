"""AI Hub 71439 EfficientNet posture classification.

The downloaded AI Hub package contains five independent EfficientNet-B3
classifiers.  They classify posture categories, but do not contain the wrist
classifier or enough angle detail to calculate a complete RULA score.  The
returned RULA hints are therefore comparison-only and never replace the
browser's angle-based RULA result.
"""

from __future__ import annotations

import os
import threading
import gc
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any


MODEL_CLASS_COUNTS = {
    "NECK": 3,
    "TRUNK": 2,
    "UPPERARM": 4,
    "LOWERARM": 3,
    "LEGS": 6,
}

# Class order is confirmed by the AI Hub training script
# (make_postureDataset.py) and the official posture-labeling chart.  The
# original annotations are used as zero-based list indexes and then written to
# folders such as NECK_0, NECK_1, and NECK_2.
CLASS_SEMANTICS: dict[str, tuple[dict[str, Any], ...]] = {
    "NECK": (
        {
            "label_ko": "뒤로 젖힘",
            "rula_base_score": 4,
            "rula_base_score_range": None,
            "note": "RULA 목 신전 자세에 해당합니다.",
        },
        {
            "label_ko": "똑바로",
            "rula_base_score": 1,
            "rula_base_score_range": None,
            "note": "목 중립 자세로 해석합니다.",
        },
        {
            "label_ko": "앞으로 굽힘",
            "rula_base_score": None,
            "rula_base_score_range": [2, 3],
            "note": "AI Hub 클래스만으로 10~20도와 20도 초과를 구분할 수 없습니다.",
        },
    ),
    "TRUNK": (
        {
            "label_ko": "곧바로 편 자세",
            "rula_base_score": 1,
            "rula_base_score_range": None,
            "note": "몸통 중립 자세로 해석합니다.",
        },
        {
            "label_ko": "상체를 앞으로 20도 이상 굽힘",
            "rula_base_score": None,
            "rula_base_score_range": [3, 4],
            "note": "AI Hub 클래스만으로 20~60도와 60도 초과를 구분할 수 없습니다.",
        },
    ),
    "UPPERARM": (
        {
            "label_ko": "위팔이 몸 앞으로",
            "rula_base_score": None,
            "rula_base_score_range": [2, 3],
            "note": "앞쪽 들림 각도의 세부 범위는 MediaPipe 각도로 확인해야 합니다.",
        },
        {
            "label_ko": "위팔이 몸통과 일직선",
            "rula_base_score": 1,
            "rula_base_score_range": None,
            "note": "위팔 중립 자세로 해석합니다.",
        },
        {
            "label_ko": "위팔이 몸 뒤로",
            "rula_base_score": 2,
            "rula_base_score_range": None,
            "note": "위팔 신전 자세로 해석합니다.",
        },
        {
            "label_ko": "위팔이 몸 위로",
            "rula_base_score": 4,
            "rula_base_score_range": None,
            "note": "어깨 높이보다 크게 든 자세로 해석합니다.",
        },
    ),
    "LOWERARM": (
        {
            "label_ko": "양손을 어깨 아래로 내림",
            "rula_base_score": None,
            "rula_base_score_range": None,
            "note": "RULA 아래팔 점수는 팔꿈치 각도로 계산해야 합니다.",
        },
        {
            "label_ko": "한 손만 어깨 위로 올림",
            "rula_base_score": None,
            "rula_base_score_range": None,
            "note": "RULA 아래팔 점수는 팔꿈치 각도로 계산해야 합니다.",
        },
        {
            "label_ko": "양손 모두 어깨 위로 올림",
            "rula_base_score": None,
            "rula_base_score_range": None,
            "note": "RULA 아래팔 점수는 팔꿈치 각도로 계산해야 합니다.",
        },
    ),
    "LEGS": (
        {
            "label_ko": "앉은 자세",
            "rula_base_score": None,
            "rula_base_score_range": [1, 2],
            "note": "발과 다리의 지지 여부를 함께 확인해야 합니다.",
        },
        {
            "label_ko": "두 다리를 펴고 섬",
            "rula_base_score": 1,
            "rula_base_score_range": None,
            "note": "양발이 지지되고 균형 잡힌 경우로 해석합니다.",
        },
        {
            "label_ko": "한 다리로 섬",
            "rula_base_score": 2,
            "rula_base_score_range": None,
            "note": "불균형한 다리 자세로 해석합니다.",
        },
        {
            "label_ko": "두 다리를 구부려 섬",
            "rula_base_score": 2,
            "rula_base_score_range": None,
            "note": "안정적인 지지 여부를 현장에서 다시 확인해야 합니다.",
        },
        {
            "label_ko": "한 다리로 구부려 섬",
            "rula_base_score": 2,
            "rula_base_score_range": None,
            "note": "불균형한 다리 자세로 해석합니다.",
        },
        {
            "label_ko": "무릎을 꿇는 자세",
            "rula_base_score": 2,
            "rula_base_score_range": None,
            "note": "불안정하거나 제한된 다리 자세로 해석합니다.",
        },
    ),
}

LIMITATIONS = [
    "AI Hub 제공 패키지에는 손목 자세 분류 모델이 없습니다.",
    "목·몸통·위팔의 일부 클래스는 RULA 각도 구간을 모두 구분하지 못합니다.",
    "비틀림·옆굽힘·지지 여부 등 부가정보는 별도 확인이 필요합니다.",
    "AI Hub 분류 결과는 기존 MediaPipe RULA 결과를 덮어쓰지 않고 비교용으로만 사용합니다.",
]

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class AihubClassifierError(RuntimeError):
    """Base error for the AI Hub classifier."""


class InvalidImageError(AihubClassifierError):
    """The request body is not a supported image."""


class ModelUnavailableError(AihubClassifierError):
    """The local AI Hub model runtime cannot be loaded."""


_model_lock = threading.Lock()
_runtime_loaded_once = False


def _default_model_dir() -> Path:
    backend_root = Path(__file__).resolve().parent
    return backend_root / "aihub_models"


def model_directory() -> Path:
    return Path(
        os.getenv("AIHUB_MODEL_DIR", str(_default_model_dir()))
    ).expanduser().resolve()


def classifier_status() -> dict[str, Any]:
    models_dir = model_directory()
    missing_models = [
        f"{model_name}.onnx"
        for model_name in MODEL_CLASS_COUNTS
        if not (models_dir / f"{model_name}.onnx").is_file()
    ]
    return {
        "configured": not missing_models,
        "loaded": _runtime_loaded_once,
        "runtime": "onnxruntime",
        "modelFormat": "onnx",
        "modelsDirectory": str(models_dir),
        "missingModels": missing_models,
        "comparisonOnly": True,
    }


def class_semantic(model_name: str, class_index: int) -> dict[str, Any]:
    try:
        semantic = CLASS_SEMANTICS[model_name][class_index]
    except (KeyError, IndexError) as error:
        raise ValueError(
            f"Unknown AI Hub class: {model_name}_{class_index}"
        ) from error
    return dict(semantic)


def _load_dependencies():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise ModelUnavailableError(
            "AI Hub ONNX runtime dependencies are missing."
        ) from error
    return np, ort, Image, UnidentifiedImageError


def _validate_models(models_dir: Path) -> None:
    missing = [
        models_dir / f"{model_name}.onnx"
        for model_name in MODEL_CLASS_COUNTS
        if not (models_dir / f"{model_name}.onnx").is_file()
    ]
    if missing:
        raise ModelUnavailableError(
            "Missing AI Hub ONNX models: "
            + ", ".join(path.name for path in missing)
        )


def _session_options(ort_module):
    options = ort_module.SessionOptions()
    options.intra_op_num_threads = max(
        1,
        int(os.getenv("AIHUB_ONNX_THREADS", "1")),
    )
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = (
        ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return options


def _prepare_image(
    image_bytes: bytes,
    np_module,
    image_class,
    unidentified_image_error,
):
    if not image_bytes:
        raise InvalidImageError("Image body is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Image exceeds the 15 MB request limit.")

    try:
        with image_class.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    "Image dimensions exceed the 40-megapixel limit."
                )
            image = source.convert("RGB")
    except (unidentified_image_error, OSError) as error:
        raise InvalidImageError(
            "Request body is not a readable JPEG, PNG, or WebP image."
        ) from error

    image = image.resize((300, 300), image_class.Resampling.BILINEAR)
    array = np_module.asarray(image, dtype=np_module.float32) / 255.0
    mean = np_module.asarray([0.485, 0.456, 0.406], dtype=np_module.float32)
    std = np_module.asarray([0.229, 0.224, 0.225], dtype=np_module.float32)
    array = (array - mean) / std
    array = np_module.transpose(array, (2, 0, 1))
    return np_module.expand_dims(array, axis=0).astype(
        np_module.float32,
        copy=False,
    )


def classify_image(image_bytes: bytes) -> dict[str, Any]:
    global _runtime_loaded_once

    started_at = perf_counter()
    models_dir = model_directory()
    np, ort, Image, UnidentifiedImageError = _load_dependencies()
    _validate_models(models_dir)
    tensor = _prepare_image(
        image_bytes,
        np,
        Image,
        UnidentifiedImageError,
    )

    predictions: dict[str, Any] = {}
    with _model_lock:
        for model_name in MODEL_CLASS_COUNTS:
            session = ort.InferenceSession(
                (models_dir / f"{model_name}.onnx").as_posix(),
                sess_options=_session_options(ort),
                providers=["CPUExecutionProvider"],
            )
            logits = session.run(["logits"], {"image": tensor})[0][0]
            stabilized = logits - np.max(logits)
            exponentials = np.exp(stabilized)
            probabilities = (exponentials / np.sum(exponentials)).tolist()

            class_index = int(np.argmax(probabilities))
            confidence = float(probabilities[class_index])
            semantic = class_semantic(model_name, class_index)
            predictions[model_name.lower()] = {
                "class_id": f"{model_name}_{class_index}",
                "class_index": class_index,
                "label_ko": semantic["label_ko"],
                "confidence": round(confidence, 6),
                "confidence_percent": round(confidence * 100, 1),
                "probabilities": [
                    round(float(probability), 6)
                    for probability in probabilities
                ],
                "rula_base_score": semantic["rula_base_score"],
                "rula_base_score_range": semantic["rula_base_score_range"],
                "note": semantic["note"],
            }
            del session, logits
            gc.collect()

        _runtime_loaded_once = True

    return {
        "schema_version": "1.0",
        "processor": "aihub-71439-efficientnet-b3",
        "source": "AI Hub dataset 71439 model package",
        "rula_mode": "comparison_only",
        "predictions": predictions,
        "limitations": LIMITATIONS,
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
        "image_stored": False,
    }
