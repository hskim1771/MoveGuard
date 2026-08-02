import unittest

from analysis_v2.domain import EstimatorStatus, ObjectDetection
from analysis_v2.estimators.base import HandEstimator, ObjectDetector, PoseEstimator
from analysis_v2.pipeline import PostureAnalysisPipeline

from tests.helpers import make_hand, make_pose


class FakePoseEstimator(PoseEstimator):
    def __init__(self, poses):
        self.poses = poses

    def estimate(self, image_rgb):
        return self.poses

    def status(self):
        return EstimatorStatus("FakePoseEstimator", True, True)


class FakeHandEstimator(HandEstimator):
    def estimate(self, image_rgb, pose):
        return {"left": make_hand("left"), "right": make_hand("right")}

    def status(self):
        return EstimatorStatus("FakeHandEstimator", True, True)


class FakeObjectDetector(ObjectDetector):
    def __init__(self, detections=None):
        self.detections = detections or []

    def detect(self, image_rgb):
        return self.detections

    def status(self):
        return EstimatorStatus("FakeObjectDetector", True, True)


class PipelineTests(unittest.TestCase):
    def test_person_absent_returns_normal_not_measurable_status(self) -> None:
        pipeline = PostureAnalysisPipeline(
            FakePoseEstimator([]), FakeHandEstimator(), FakeObjectDetector()
        )

        result = pipeline.analyze_image(object())

        self.assertEqual(result["quality"]["status"], "not_measurable")
        self.assertEqual(result["rula"]["issues"], ["person_not_detected"])
        self.assertEqual(result["objects"], [])

    def test_pipeline_exports_auditable_frame_fields(self) -> None:
        detection = ObjectDetection(
            "object-1", "box", 0.9, (0.35, 0.50, 0.45, 0.62)
        )
        pipeline = PostureAnalysisPipeline(
            FakePoseEstimator([make_pose()]),
            FakeHandEstimator(),
            FakeObjectDetector([detection]),
        )

        result = pipeline.analyze_image(
            object(),
            worker_id="worker-1",
            measurement_id="measurement-1",
            frame_id="frame-1",
            timestamp_ms=1234,
        )

        self.assertEqual(result["frame_id"], "frame-1")
        self.assertEqual(result["worker_id"], "worker-1")
        self.assertTrue(result["rula"]["valid"])
        self.assertEqual(result["objects"][0]["label"], "box")
        self.assertEqual(result["object_relationships"][0]["closest_joint"], "left_wrist")
        self.assertIn("left_upper_arm", result["measurements"])


if __name__ == "__main__":
    unittest.main()
