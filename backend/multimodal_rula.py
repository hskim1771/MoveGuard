"""Runtime inference for MoveGuard's image + MediaPipe posture model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from backend.rula import calculate
from training.evaluate_rula_agreement import reference_components
from training.preprocess_mediapipe import features as pose_features

CATEGORICAL = {"neck": 3, "trunk": 2, "upperarm": 4, "lowerarm": 3, "legs": 6}
BINARY = (
    "neck_twist", "neck_bending", "trunk_twist", "trunk_bending", "trunk_supported",
    "legs_supported", "upperarm_abduction", "upperarm_raised", "upperarm_supported",
    "upperarm_across", "lowerarm_outbody", "wrist_bending", "wrist_twisted", "wrist_twisted_end",
)


class MultiModalRula(nn.Module):
    def __init__(self, pose_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = (3, 16, 32, 64, 128)
        for source, target in zip(channels, channels[1:]):
            layers += [nn.Conv2d(source, target, 3, stride=2, padding=1), nn.BatchNorm2d(target), nn.SiLU()]
        self.image = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 128), nn.SiLU())
        self.pose = nn.Sequential(nn.Linear(pose_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.2), nn.Linear(256, 128), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(256, 192), nn.SiLU(), nn.Dropout(.25))
        self.categorical = nn.ModuleDict({key: nn.Linear(192, count) for key, count in CATEGORICAL.items()})
        self.binary = nn.Linear(192, len(BINARY))

    def forward(self, image: torch.Tensor, pose: torch.Tensor):
        fused = self.fusion(torch.cat((self.image(image), self.pose(pose)), dim=1))
        return {key: head(fused) for key, head in self.categorical.items()}, self.binary(fused)


class RulaPosturePredictor:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.mean = np.asarray(checkpoint["pose_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["pose_std"], dtype=np.float32)
        self.image_size = int(checkpoint["image_size"])
        self.model = MultiModalRula(len(self.mean))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.validation = checkpoint.get("validation", {})

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, landmarks: list[object], *, muscle: int = 0, force: int = 0, legs_unsupported: bool = False) -> dict[str, object]:
        vector, _ = pose_features(landmarks)
        pixels = self._person_crop(rgb, landmarks)
        image = torch.from_numpy(pixels.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        pose = torch.from_numpy((np.asarray(vector, dtype=np.float32) - self.mean) / self.std).unsqueeze(0)
        categorical_logits, binary_logits = self.model(image, pose)
        labels: dict[str, int] = {}
        confidence: dict[str, float] = {}
        for key, logits in categorical_logits.items():
            probabilities = torch.softmax(logits, dim=1)[0]
            labels[key] = int(probabilities.argmax())
            confidence[key] = float(probabilities.max())
        probabilities = torch.sigmoid(binary_logits)[0]
        for index, key in enumerate(BINARY):
            labels[key] = int(probabilities[index] >= .5)
            confidence[key] = float(max(probabilities[index], 1 - probabilities[index]))
        if legs_unsupported:
            labels["legs_supported"] = 0
        components = reference_components(labels)
        result = calculate({**components, "muscle_a": muscle, "force_a": force, "muscle_b": muscle, "force_b": force})
        return {"labels": labels, "label_confidence": confidence, "components": components, **result}

    def _person_crop(self, rgb: np.ndarray, landmarks: list[object]) -> np.ndarray:
        height, width = rgb.shape[:2]
        visible = [(p.x * width, p.y * height) for p in landmarks if p.visibility >= .3]
        if not visible:
            visible = [(p.x * width, p.y * height) for p in landmarks]
        points = np.asarray(visible)
        x1, y1 = points.min(axis=0); x2, y2 = points.max(axis=0)
        pad_x, pad_y = max((x2 - x1) * .15, 1), max((y2 - y1) * .15, 1)
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
        crop = rgb[top:bottom, left:right]
        if crop.size == 0:
            crop = rgb
        return cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
