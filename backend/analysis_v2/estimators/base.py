from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..domain import (
    EstimatorStatus,
    HandDetection,
    ObjectDetection,
    PoseDetection,
)


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, image_rgb: Any) -> list[PoseDetection]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> EstimatorStatus:
        raise NotImplementedError


class HandEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        image_rgb: Any,
        pose: PoseDetection,
    ) -> dict[str, HandDetection]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> EstimatorStatus:
        raise NotImplementedError


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image_rgb: Any) -> list[ObjectDetection]:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> EstimatorStatus:
        raise NotImplementedError
