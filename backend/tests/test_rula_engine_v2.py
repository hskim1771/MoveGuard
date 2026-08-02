import unittest

from analysis_v2.angle_calculator import calculate_joint_angles
from analysis_v2.rula_engine import RulaOptions, calculate_rula

from tests.helpers import make_hand, make_pose


class RulaEngineTests(unittest.TestCase):
    def test_returns_complete_rula_when_pose_and_hand_are_valid(self) -> None:
        measurements = calculate_joint_angles(
            make_pose(),
            {"left": make_hand("left"), "right": make_hand("right")},
        )

        result = calculate_rula(measurements, RulaOptions())

        self.assertTrue(result["valid"])
        self.assertEqual(result["final_score"], 2)
        self.assertEqual(result["risk_level"], "low")
        self.assertIn("wrist_twist_mid_range_assumed_single_camera", result["assumptions"])

    def test_missing_selected_wrist_returns_not_measurable(self) -> None:
        measurements = calculate_joint_angles(make_pose(), {})

        result = calculate_rula(measurements)

        self.assertFalse(result["valid"])
        self.assertIsNone(result["final_score"])
        self.assertEqual(result["risk_level"], "not_measurable")


if __name__ == "__main__":
    unittest.main()
