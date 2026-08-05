import unittest

import numpy as np

from backend.temporal_stabilizer import PoseFrame, StabilizerConfig, TemporalPoseStabilizer


def frame(points, confidence=None, timestamp=0.0):
    values = np.asarray(points, dtype=np.float32)
    return PoseFrame(
        values,
        np.ones(len(values), dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32),
        timestamp,
    )


class TemporalPoseStabilizerTest(unittest.TestCase):
    def setUp(self):
        self.base = np.asarray([[0.3 + i * 0.01, 0.3 + i * 0.005] for i in range(33)], dtype=np.float32)
        self.config = StabilizerConfig(max_neighbor_gap_seconds=0.5)

    def test_replaces_isolated_low_confidence_wrist_spike(self):
        spike = self.base.copy(); spike[15] = [1.1, 0.1]
        confidence = np.ones(33); confidence[15] = 0.2
        result = TemporalPoseStabilizer(config=self.config).stabilize([
            frame(self.base, timestamp=0.0), frame(spike, confidence, 0.1), frame(self.base, timestamp=0.2)
        ])
        self.assertIn(15, result[1].outlier_joints)
        np.testing.assert_allclose(result[1].keypoints[15], self.base[15], atol=1e-5)

    def test_preserves_sustained_fast_motion(self):
        frames = []
        for index, offset in enumerate((0.0, 0.15, 0.30)):
            moved = self.base.copy(); moved[15, 0] += offset
            frames.append(frame(moved, timestamp=index * 0.1))
        result = TemporalPoseStabilizer(config=self.config).stabilize(frames)
        self.assertNotIn(15, result[1].outlier_joints)
        self.assertAlmostEqual(float(result[1].keypoints[15, 0]), float(self.base[15, 0] + 0.15), places=5)

    def test_sparse_samples_do_not_use_temporal_interpolation(self):
        spike = self.base.copy(); spike[15] = [1.1, 0.1]
        confidence = np.ones(33); confidence[15] = 0.2
        result = TemporalPoseStabilizer(config=self.config).stabilize([
            frame(self.base, timestamp=0.0), frame(spike, confidence, 6.0), frame(self.base, timestamp=12.0)
        ])
        self.assertNotIn("isolated_temporal_jump", result[1].reasons)

    def test_marks_nonfinite_frame_invalid(self):
        broken = self.base.copy(); broken[15, 0] = np.nan
        result = TemporalPoseStabilizer(config=self.config).stabilize([frame(broken)])
        self.assertFalse(result[0].valid)


if __name__ == "__main__":
    unittest.main()
