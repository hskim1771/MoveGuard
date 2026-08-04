"""Runtime joint-domain correction for MediaPipe Pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


HUMAN_TO_MEDIAPIPE = {
    1: 23,
    2: 24,
    4: 25,
    5: 26,
    7: 27,
    8: 28,
    10: 31,
    11: 32,
    16: 11,
    17: 12,
    18: 13,
    19: 14,
    20: 15,
    21: 16,
}


@dataclass(frozen=True)
class RefinedLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


class Corrector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(33 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 24 * 2),
        )

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + self.net(features).reshape(-1, 24, 2) * 0.10


def _baseline(landmarks: np.ndarray) -> np.ndarray:
    left_hand = landmarks[[17, 19, 21], :2].mean(0)
    right_hand = landmarks[[18, 20, 22], :2].mean(0)
    pelvis = landmarks[[23, 24], :2].mean(0)
    neck = landmarks[[11, 12], :2].mean(0)
    spine1 = pelvis * 0.75 + neck * 0.25
    spine2 = (pelvis + neck) / 2
    spine3 = pelvis * 0.25 + neck * 0.75
    return np.asarray(
        [
            pelvis, landmarks[23, :2], landmarks[24, :2], spine1,
            landmarks[25, :2], landmarks[26, :2], spine2,
            landmarks[27, :2], landmarks[28, :2], spine3,
            landmarks[31, :2], landmarks[32, :2], neck,
            (neck + landmarks[11, :2]) / 2,
            (neck + landmarks[12, :2]) / 2, landmarks[0, :2],
            landmarks[11, :2], landmarks[12, :2], landmarks[13, :2],
            landmarks[14, :2], landmarks[15, :2], landmarks[16, :2],
            left_hand, right_hand,
        ],
        dtype=np.float32,
    )


class PoseRefiner:
    """Apply the frozen joint-domain model without changing MediaPipe detection."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = Corrector()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, landmarks: Sequence[object]) -> list[RefinedLandmark]:
        raw = np.asarray(
            [[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32
        )
        normalized, bounds = self._person_normalized(raw)
        base = _baseline(normalized)
        predicted = self.model(
            torch.from_numpy(normalized.reshape(1, -1)),
            torch.from_numpy(base[None]),
        ).numpy()[0]
        left, top, width, height = bounds
        output = [
            RefinedLandmark(
                x=float(p.x), y=float(p.y), z=float(p.z),
                visibility=float(p.visibility), presence=float(p.presence),
            )
            for p in landmarks
        ]
        for human_index, mediapipe_index in HUMAN_TO_MEDIAPIPE.items():
            point = predicted[human_index]
            original = output[mediapipe_index]
            output[mediapipe_index] = RefinedLandmark(
                x=float(np.clip(left + point[0] * width, -0.25, 1.25)),
                y=float(np.clip(top + point[1] * height, -0.25, 1.25)),
                z=original.z,
                visibility=original.visibility,
                presence=original.presence,
            )
        return output

    @staticmethod
    def _person_normalized(raw: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        visible = raw[:, 3] >= 0.3
        points = raw[visible, :2] if visible.any() else raw[:, :2]
        minimum, maximum = points.min(0), points.max(0)
        span = np.maximum(maximum - minimum, 1e-3)
        left, top = minimum - span * 0.15
        right, bottom = maximum + span * 0.15
        width, height = np.maximum([right - left, bottom - top], 1e-3)
        normalized = raw.copy()
        normalized[:, 0] = (raw[:, 0] - left) / width
        normalized[:, 1] = (raw[:, 1] - top) / height
        normalized[:, 2] = raw[:, 2] / width
        return normalized, (float(left), float(top), float(width), float(height))
