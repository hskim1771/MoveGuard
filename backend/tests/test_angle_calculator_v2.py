import unittest

from analysis_v2.angle_calculator import calculate_joint_angles

from tests.helpers import make_hand, make_pose


class AngleCalculatorTests(unittest.TestCase):
    def test_neutral_upper_arm_is_zero_not_legacy_180(self) -> None:
        measurements = calculate_joint_angles(
            make_pose(),
            {"left": make_hand("left"), "right": make_hand("right")},
        )

        self.assertTrue(measurements["left_upper_arm"].valid)
        self.assertIsNotNone(measurements["left_upper_arm"].angle)
        self.assertIsNotNone(measurements["right_upper_arm"].angle)
        self.assertLess(measurements["left_upper_arm"].angle, 5.0)  # type: ignore[operator]
        self.assertLess(measurements["right_upper_arm"].angle, 5.0)  # type: ignore[operator]

    def test_hand_world_landmarks_produce_wrist_measurement(self) -> None:
        measurements = calculate_joint_angles(
            make_pose(),
            {"left": make_hand("left"), "right": make_hand("right")},
        )

        self.assertTrue(measurements["left_wrist"].valid)
        self.assertIsNotNone(measurements["left_wrist"].angle)
        self.assertAlmostEqual(measurements["left_wrist"].angle, 0.0, places=5)
        self.assertEqual(measurements["left_wrist"].auxiliary["method"], "world_3d")
        self.assertFalse(measurements["left_wrist_twist"].valid)

    def test_low_confidence_landmark_is_not_scored(self) -> None:
        pose = make_pose()
        points = list(pose.landmarks)
        points[11] = type(points[11])(
            points[11].x,
            points[11].y,
            points[11].z,
            0.1,
        )
        pose = type(pose)(
            person_id=pose.person_id,
            landmarks=tuple(points),
            world_landmarks=tuple(points),
            confidence=pose.confidence,
        )

        measurements = calculate_joint_angles(pose, {})

        self.assertEqual(measurements["trunk"].status, "low_confidence")
        self.assertIsNone(measurements["trunk"].angle)


if __name__ == "__main__":
    unittest.main()
