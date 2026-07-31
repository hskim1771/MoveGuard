import unittest

from app.analyzer import analyze_request
from app.models import AnalyzeRequest


def sample_landmarks() -> list[dict[str, float]]:
    points = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99}
        for _ in range(33)
    ]
    points[7] = {"x": 0.47, "y": 0.2, "z": 0.0, "visibility": 0.99}
    points[8] = {"x": 0.53, "y": 0.2, "z": 0.0, "visibility": 0.99}
    points[11] = {"x": 0.42, "y": 0.35, "z": 0.0, "visibility": 0.99}
    points[12] = {"x": 0.58, "y": 0.35, "z": 0.0, "visibility": 0.99}
    points[13] = {"x": 0.42, "y": 0.55, "z": 0.0, "visibility": 0.99}
    points[14] = {"x": 0.58, "y": 0.55, "z": 0.0, "visibility": 0.99}
    points[17] = {"x": 0.40, "y": 0.72, "z": 0.0, "visibility": 0.99}
    points[18] = {"x": 0.60, "y": 0.72, "z": 0.0, "visibility": 0.99}
    points[19] = {"x": 0.44, "y": 0.72, "z": 0.0, "visibility": 0.99}
    points[20] = {"x": 0.56, "y": 0.72, "z": 0.0, "visibility": 0.99}
    points[23] = {"x": 0.44, "y": 0.62, "z": 0.0, "visibility": 0.99}
    points[24] = {"x": 0.56, "y": 0.62, "z": 0.0, "visibility": 0.99}
    points[25] = {"x": 0.44, "y": 0.78, "z": 0.0, "visibility": 0.99}
    points[26] = {"x": 0.56, "y": 0.78, "z": 0.0, "visibility": 0.99}
    points[27] = {"x": 0.44, "y": 0.94, "z": 0.0, "visibility": 0.99}
    points[28] = {"x": 0.56, "y": 0.94, "z": 0.0, "visibility": 0.99}
    return points


class AnalyzeRequestTests(unittest.TestCase):
    def test_returns_one_complete_person_analysis(self) -> None:
        landmarks = sample_landmarks()
        payload = AnalyzeRequest.model_validate(
            {
                "timestamp_ms": 1000,
                "people": [
                    {
                        "person_id": "person-1",
                        "landmarks": landmarks,
                        "world_landmarks": landmarks,
                    }
                ],
            }
        )

        result = analyze_request(payload)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].person_id, "person-1")
        self.assertEqual(set(result[0].metrics), {
            "neck",
            "trunk",
            "shoulder",
            "headRotation",
            "trunkRotation",
            "leftWristRotation",
            "rightWristRotation",
        })
        self.assertEqual(result[0].confidence, 99)
        self.assertTrue(result[0].full_body)


if __name__ == "__main__":
    unittest.main()
