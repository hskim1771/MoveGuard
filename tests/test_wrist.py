import unittest

from backend.wrist import projected_wrist_angle


class WristMeasurementTest(unittest.TestCase):
    def test_neutral_is_zero_deviation(self):
        result = projected_wrist_angle(
            (0, 0), (1, 0), ((2, 0), (2.1, 0), (1.9, 0)), confidence=(1, 1, 1, 1, 1)
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.angle, 0.0)
        self.assertEqual(result.posture, "neutral")
        self.assertEqual(result.rula_score, 1)

    def test_bent_wrist_without_camera_orientation_is_unknown_direction(self):
        result = projected_wrist_angle(
            (0, 0), (1, 0), ((1, 1), (1.1, 1), (0.9, 1)), confidence=(1, 1, 1, 1, 1)
        )
        self.assertAlmostEqual(result.angle, 90.0)
        self.assertEqual(result.posture, "unknown")
        self.assertEqual(result.rula_score, 2)

    def test_orientation_can_separate_flexion_and_extension(self):
        flexion = projected_wrist_angle(
            (0, 0), (1, 0), ((1, 1),), confidence=(1, 1, 1), orientation_sign=1)
        extension = projected_wrist_angle(
            (0, 0), (1, 0), ((1, -1),), confidence=(1, 1, 1), orientation_sign=1)
        self.assertEqual(flexion.posture, "flexion")
        self.assertEqual(extension.posture, "extension")

    def test_low_confidence_is_invalid(self):
        result = projected_wrist_angle(
            (0, 0), (1, 0), ((2, 0),), confidence=(1, 0.2, 1))
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "low_confidence")


if __name__ == "__main__":
    unittest.main()
