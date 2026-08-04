from pathlib import Path
from types import SimpleNamespace
import unittest

from backend.pose_refiner import PoseRefiner


class PoseRefinerTest(unittest.TestCase):
    def test_checkpoint_produces_33_finite_landmarks(self):
        checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/wrist_refiner_joint_domain_v1/best.pt"
        model = PoseRefiner(checkpoint)
        landmarks = [
            SimpleNamespace(
                x=0.25 + (index % 5) * 0.1,
                y=0.15 + (index // 5) * 0.08,
                z=-0.05,
                visibility=0.9,
                presence=0.95,
            )
            for index in range(33)
        ]
        refined = model.predict(landmarks)
        self.assertEqual(len(refined), 33)
        self.assertTrue(all(-0.25 <= point.x <= 1.25 for point in refined))
        self.assertTrue(all(-0.25 <= point.y <= 1.25 for point in refined))


if __name__ == "__main__":
    unittest.main()
