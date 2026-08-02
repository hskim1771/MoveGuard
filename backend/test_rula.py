import unittest

import rula


def neutral_landmarks():
    points = [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
    points[7] = (-0.1, -0.2, 0.0, 1.0)
    points[8] = (0.1, -0.2, 0.0, 1.0)
    points[11] = (-0.2, 0.0, 0.0, 1.0)
    points[12] = (0.2, 0.0, 0.0, 1.0)
    points[13] = (-0.2, 0.5, 0.0, 1.0)
    points[14] = (0.2, 0.5, 0.0, 1.0)
    points[15] = (-0.6, 0.5, 0.0, 1.0)
    points[16] = (0.6, 0.5, 0.0, 1.0)
    points[17] = (-0.8, 0.5, 0.0, 1.0)
    points[18] = (0.8, 0.5, 0.0, 1.0)
    points[19] = (-0.8, 0.5, 0.0, 1.0)
    points[20] = (0.8, 0.5, 0.0, 1.0)
    points[23] = (-0.2, 1.0, 0.0, 1.0)
    points[24] = (0.2, 1.0, 0.0, 1.0)
    points[25] = (-0.2, 1.4, 0.0, 1.0)
    points[26] = (0.2, 1.4, 0.0, 1.0)
    points[27] = (-0.2, 1.8, 0.0, 1.0)
    points[28] = (0.2, 1.8, 0.0, 1.0)
    return points


class RulaTests(unittest.TestCase):
    def test_neutral_hanging_arm_is_zero_degrees(self):
        angles = rula.compute_frame_angles(neutral_landmarks())
        self.assertAlmostEqual(angles["upperArmAngle"], 0.0)
        self.assertEqual(rula.upper_arm_score(angles["upperArmAngle"]), 1)

    def test_neutral_posture_is_grand_score_two(self):
        angles = rula.compute_frame_angles(neutral_landmarks())
        result = rula.compute_rula(angles, 0, False, False, False)
        self.assertEqual(result["scoreA"], 1)
        self.assertEqual(result["scoreB"], 2)
        self.assertEqual(result["grandScore"], 2)

    def test_rula_adjustments_are_applied(self):
        self.assertEqual(rula.neck_score(5, rotated=True), 2)
        self.assertEqual(rula.neck_score(5, extended=True), 4)
        self.assertEqual(rula.trunk_score(10, side_bent=True), 3)
        self.assertEqual(rula.upper_arm_score(10, True, True, False), 3)
        self.assertEqual(rula.upper_arm_score(10, True, True, True), 2)
        self.assertEqual(rula.lower_arm_score(90, True), 2)
        self.assertEqual(rula.wrist_score(20, True), 4)

    def test_force_score_distinguishes_intermittent_and_repeated_load(self):
        self.assertEqual(rula.force_score(0, True), 0)
        self.assertEqual(rula.force_score(5, False), 1)
        self.assertEqual(rula.force_score(5, True), 2)
        self.assertEqual(rula.force_score(11, False), 3)

    def test_table_a_covers_every_adjusted_posture_score(self):
        self.assertEqual(len(rula.TABLE_A), 6)
        for upper_arm in range(1, 7):
            for lower_arm in range(1, 4):
                for wrist in range(1, 5):
                    for twist in range(1, 3):
                        value = rula.TABLE_A[upper_arm - 1][lower_arm - 1][wrist][twist - 1]
                        self.assertIn(value, range(1, 10))

    def test_single_frame_spike_is_not_selected(self):
        normal = rula.compute_frame_angles(neutral_landmarks())
        spike = {**normal, "neckAngle": 90, "trunkAngle": 90, "upperArmAngle": 150}
        candidates = [{"angles": normal, "image": index} for index in range(4)]
        candidates.insert(2, {"angles": spike, "image": "spike"})
        selected = rula.select_representative_frame(candidates)
        self.assertNotEqual(selected["image"], "spike")
        self.assertLess(selected["angles"]["trunkAngle"], 20)


if __name__ == "__main__":
    unittest.main()
