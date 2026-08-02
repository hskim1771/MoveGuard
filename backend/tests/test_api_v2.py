import unittest
from unittest.mock import patch

import cv2
import numpy as np
from analysis_v2.api import router
from analysis_v2.domain import EstimatorStatus
from analysis_v2.estimators.base import HandEstimator, ObjectDetector, PoseEstimator
from analysis_v2.pipeline import PostureAnalysisPipeline
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.helpers import request_payload


class UnusedPoseEstimator(PoseEstimator):
    def estimate(self, image_rgb):
        raise AssertionError("JSON landmark API must not rerun pose inference")

    def status(self):
        return EstimatorStatus("UnusedPoseEstimator", True, False)


class UnusedHandEstimator(HandEstimator):
    def estimate(self, image_rgb, pose):
        raise AssertionError("JSON landmark API must not rerun hand inference")

    def status(self):
        return EstimatorStatus("UnusedHandEstimator", True, False)


class EmptyObjectDetector(ObjectDetector):
    def detect(self, image_rgb):
        return []

    def status(self):
        return EstimatorStatus("EmptyObjectDetector", False, False)


class EmptyPoseEstimator(PoseEstimator):
    def estimate(self, image_rgb):
        return []

    def status(self):
        return EstimatorStatus("EmptyPoseEstimator", True, False)


class PostureApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = FastAPI()
        app.include_router(router)
        cls.client = TestClient(app)
        cls.pipeline = PostureAnalysisPipeline(
            UnusedPoseEstimator(), UnusedHandEstimator(), EmptyObjectDetector()
        )
        cls.image_pipeline = PostureAnalysisPipeline(
            EmptyPoseEstimator(), UnusedHandEstimator(), EmptyObjectDetector()
        )

    def test_json_landmark_request_returns_v2_contract(self) -> None:
        with patch("analysis_v2.api.get_pipeline", return_value=self.pipeline):
            response = self.client.post(
                "/api/analysis/posture",
                json=request_payload(),
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], "2.0")
        self.assertEqual(payload["frame_id"], "test-frame")
        self.assertTrue(payload["rula"]["valid"])
        self.assertFalse(payload["video_stored"])

    def test_video_body_is_rejected_with_frame_sampling_guidance(self) -> None:
        with patch("analysis_v2.api.get_pipeline", return_value=self.pipeline):
            response = self.client.post(
                "/api/analysis/posture",
                content=b"not-a-frame",
                headers={"content-type": "video/mp4"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertIn("sampled image frames", response.json()["detail"])

    def test_jpeg_frame_returns_normal_not_measurable_contract(self) -> None:
        encoded, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(encoded)

        with patch("analysis_v2.api.get_pipeline", return_value=self.image_pipeline):
            response = self.client.post(
                "/api/analysis/posture?frame_id=browser-frame",
                content=jpeg.tobytes(),
                headers={"content-type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["frame_id"], "browser-frame")
        self.assertEqual(payload["rula"]["risk_level"], "not_measurable")
        self.assertFalse(payload["video_stored"])


if __name__ == "__main__":
    unittest.main()
