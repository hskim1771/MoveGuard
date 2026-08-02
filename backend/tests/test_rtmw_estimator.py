import unittest

import numpy as np

from analysis_v2.estimators.rtmw_pose import RTMWPoseEstimator, map_rtmw_wholebody_person


class RTMWMappingTests(unittest.TestCase):
    def test_coco_wholebody_maps_to_shared_pose_and_hands(self):
        keypoints = [[float(index), float(index * 2)] for index in range(133)]
        scores = [0.9] * 133

        pose, hands = map_rtmw_wholebody_person(
            keypoints,
            scores,
            width=200,
            height=400,
            person_id="person-1",
        )

        self.assertEqual(len(pose.landmarks), 33)
        self.assertAlmostEqual(pose.landmarks[11].x, 5 / 200)
        self.assertAlmostEqual(pose.landmarks[11].y, 10 / 400)
        self.assertEqual(len(hands["left"].landmarks), 21)
        self.assertAlmostEqual(hands["left"].landmarks[0].x, 91 / 200)
        self.assertAlmostEqual(hands["right"].landmarks[9].x, 121 / 200)

    def test_low_confidence_hand_is_not_exposed_as_angle_ready(self):
        keypoints = [[10.0, 20.0] for _ in range(133)]
        scores = [0.9] * 133
        scores[91] = 0.1

        _, hands = map_rtmw_wholebody_person(
            keypoints,
            scores,
            width=100,
            height=100,
            person_id="person-1",
        )

        self.assertNotIn("left", hands)
        self.assertIn("right", hands)

    def test_empty_person_detector_does_not_fall_back_to_whole_image_pose(self):
        class EmptyDetector:
            def __call__(self, image):
                return np.empty((0, 5), dtype=np.float32)

        class PoseMustNotRun:
            def __call__(self, image, *, bboxes):
                raise AssertionError("Pose inference must not run without a person bbox")

        estimator = RTMWPoseEstimator()
        estimator._model = type(
            "FakeWholebody",
            (),
            {"det_model": EmptyDetector(), "pose_model": PoseMustNotRun()},
        )()

        result = estimator.estimate(np.full((64, 64, 3), 255, dtype=np.uint8))

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
