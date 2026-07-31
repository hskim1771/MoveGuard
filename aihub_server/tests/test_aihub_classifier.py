import unittest

from app.aihub_classifier import (
    CLASS_SEMANTICS,
    MODEL_CLASS_COUNTS,
    class_semantic,
)


class AihubClassMappingTests(unittest.TestCase):
    def test_every_model_class_has_semantics(self) -> None:
        mapped_counts = {
            model_name: len(classes)
            for model_name, classes in CLASS_SEMANTICS.items()
        }
        self.assertEqual(mapped_counts, MODEL_CLASS_COUNTS)

    def test_confirmed_class_order(self) -> None:
        self.assertEqual(class_semantic("NECK", 0)["label_ko"], "뒤로 젖힘")
        self.assertEqual(class_semantic("NECK", 1)["label_ko"], "똑바로")
        self.assertEqual(
            class_semantic("TRUNK", 1)["label_ko"],
            "상체를 앞으로 20도 이상 굽힘",
        )
        self.assertEqual(
            class_semantic("UPPERARM", 3)["label_ko"],
            "위팔이 몸 위로",
        )
        self.assertEqual(
            class_semantic("LEGS", 5)["label_ko"],
            "무릎을 꿇는 자세",
        )

    def test_ambiguous_classes_are_not_given_false_precision(self) -> None:
        neck_flexion = class_semantic("NECK", 2)
        self.assertIsNone(neck_flexion["rula_base_score"])
        self.assertEqual(neck_flexion["rula_base_score_range"], [2, 3])

        lower_arm = class_semantic("LOWERARM", 0)
        self.assertIsNone(lower_arm["rula_base_score"])
        self.assertIsNone(lower_arm["rula_base_score_range"])


if __name__ == "__main__":
    unittest.main()
