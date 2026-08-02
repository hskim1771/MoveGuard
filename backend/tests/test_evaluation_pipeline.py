import unittest

from analysis_v2.domain import EstimatorStatus
from analysis_v2.estimators.base import HandEstimator, PoseEstimator
from evaluation.canonical import canonicalize
from evaluation.ground_truth import GroundTruthStore
from evaluation.fusion import fuse_pose_detections, moveguard_specialist_profile
from evaluation.metrics import build_report
from evaluation.variants import LiveVariant, NoHandEstimator, planned_ablation_variants

from tests.helpers import make_hand, make_pose


class StaticPoseEstimator(PoseEstimator):
    def estimate(self, image_rgb):
        return [make_pose()]

    def status(self):
        return EstimatorStatus("StaticPoseEstimator", True, True)


class StaticHandEstimator(HandEstimator):
    def estimate(self, image_rgb, pose):
        return {"left": make_hand("left"), "right": make_hand("right")}

    def status(self):
        return EstimatorStatus("StaticHandEstimator", True, True)


def make_variants():
    baseline = LiveVariant(
        variant_id="baseline_v1",
        label="baseline",
        pose_estimator=StaticPoseEstimator(),
        hand_estimator=NoHandEstimator(),
        pose_source="mediapipe_pose",
        hand_source=None,
    )
    hybrid = LiveVariant(
        variant_id="hybrid_v2_available",
        label="hybrid",
        pose_estimator=StaticPoseEstimator(),
        hand_estimator=StaticHandEstimator(),
        pose_source="mediapipe_pose",
        hand_source="hand_model",
    )
    return baseline, hybrid


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        baseline, hybrid = make_variants()
        self.baseline = baseline.analyze(
            object(), frame_id="clip:000000", frame_index=0, timestamp_ms=0
        )
        self.hybrid = hybrid.analyze(
            object(), frame_id="clip:000000", frame_index=0, timestamp_ms=0
        )
        self.record = {
            "video_id": "clip",
            "frame_id": "clip:000000",
            "frame_index": 0,
            "timestamp_ms": 0,
            "width_px": 1000,
            "height_px": 1000,
            "results": {
                "baseline_v1": self.baseline,
                "hybrid_v2_available": self.hybrid,
            },
        }

    def build(self, ground_truth):
        return build_report(
            [self.record],
            baseline_id="baseline_v1",
            hybrid_id="hybrid_v2_available",
            ground_truth=ground_truth,
            experiment={"sampled_frames": 1},
            variant_status={},
            ablation=planned_ablation_variants(),
        )

    def test_no_ground_truth_never_creates_accuracy(self):
        report = self.build(GroundTruthStore(()))

        self.assertFalse(report["conclusion"]["accuracy_claim_allowed"])
        self.assertEqual(
            report["variants"]["hybrid_v2_available"]["rula"]["exact_accuracy"]["status"],
            "not_measurable",
        )
        self.assertEqual(
            report["comparison"]["joint_position_pck"]["status"],
            "not_measurable",
        )

    def test_hybrid_wrist_source_and_angle_readiness_are_auditable(self):
        left_wrist = self.hybrid.canonical.joints["left_wrist"]

        self.assertEqual(left_wrist.source_model, "hand_model")
        self.assertTrue(self.hybrid.canonical.wrist_angle_ready["left"])
        self.assertFalse(self.baseline.canonical.wrist_angle_ready["left"])
        report = self.build(GroundTruthStore(()))
        improvement = report["joint_comparison"]["wrist_angle_ready"]["improvement"]
        self.assertEqual(improvement["status"], "measured")
        self.assertEqual(improvement["value"], 100.0)

    def test_ground_truth_enables_position_angle_and_rula_metrics(self):
        row = {
            "video_id": "clip",
            "frame_id": "clip:000000",
            "frame_index": "0",
            "person_id": "worker-1",
            "width_px": "1000",
            "height_px": "1000",
            "coordinate_space": "image_normalized",
            "rula_final_score": str(self.hybrid.rula["final_score"]),
            "rula_risk_level": self.hybrid.rula["risk_level"],
        }
        for name, joint in self.hybrid.canonical.joints.items():
            if joint.x is not None and joint.y is not None:
                row[f"{name}_x"] = str(joint.x)
                row[f"{name}_y"] = str(joint.y)
                row[f"{name}_visible"] = "1"
        for measurement_name, measurement in self.hybrid.measurements.items():
            if measurement.get("valid"):
                row[f"{measurement_name}_angle_deg"] = str(measurement["angle"])
        row["left_knee_angle_deg"] = str(
            self.hybrid.measurements["legs"]["left_knee_angle"]
        )
        row["right_knee_angle_deg"] = str(
            self.hybrid.measurements["legs"]["right_knee_angle"]
        )

        report = self.build(GroundTruthStore((row,)))
        hybrid = report["variants"]["hybrid_v2_available"]

        self.assertEqual(hybrid["joint_position_overall"]["pck_at_0_2"]["value"], 100.0)
        self.assertEqual(hybrid["joint_angles_overall"]["mae"]["value"], 0.0)
        self.assertEqual(hybrid["rula"]["exact_accuracy"]["value"], 100.0)

    def test_joint_fusion_uses_declared_specialist_priority(self):
        mediapipe = make_pose(confidence=0.8)
        vitpose = make_pose(confidence=0.95)
        fused = fuse_pose_detections(
            {"mediapipe_pose": mediapipe, "vitpose_h": vitpose},
            moveguard_specialist_profile(),
        )

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused.source_by_index[11], "vitpose_h")
        self.assertEqual(fused.canonical_source_overrides["left_shoulder"], "vitpose_h")

    def test_joint_fusion_rejects_every_source_below_threshold(self):
        fused = fuse_pose_detections(
            {"mediapipe_pose": make_pose(confidence=0.2)},
            moveguard_specialist_profile(),
        )

        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertNotIn(11, fused.source_by_index)
        self.assertEqual(fused.pose.landmarks[11].confidence, 0.0)

    def test_hand_replacement_keeps_hand_model_provenance(self):
        canonical = canonicalize(
            "clip:000000",
            make_pose(),
            {"left": make_hand("left")},
            pose_source="fused_pose",
            hand_source="rtmpose_hand5",
            joint_source_overrides={"left_wrist": "vitpose_h"},
        )

        self.assertEqual(
            canonical.joints["left_wrist"].source_model, "rtmpose_hand5"
        )


if __name__ == "__main__":
    unittest.main()
