from __future__ import annotations

from analysis_v2.domain import HandDetection, Landmark, PoseDetection


def make_pose(confidence: float = 0.99) -> PoseDetection:
    points = [Landmark(0.5, 0.5, 0.0, confidence) for _ in range(33)]
    configured = {
        0: (0.50, 0.16),
        7: (0.46, 0.20),
        8: (0.54, 0.20),
        11: (0.40, 0.30),
        12: (0.60, 0.30),
        13: (0.40, 0.45),
        14: (0.60, 0.45),
        15: (0.40, 0.56),
        16: (0.60, 0.56),
        23: (0.42, 0.60),
        24: (0.58, 0.60),
        25: (0.43, 0.78),
        26: (0.57, 0.78),
        27: (0.44, 0.95),
        28: (0.56, 0.95),
    }
    for index, (x, y) in configured.items():
        points[index] = Landmark(x, y, 0.0, confidence)
    return PoseDetection(
        person_id="worker-1",
        landmarks=tuple(points),
        world_landmarks=tuple(points),
        confidence=confidence,
    )


def make_hand(side: str, confidence: float = 0.98) -> HandDetection:
    x = 0.40 if side == "left" else 0.60
    image_points = [Landmark(x, 0.60, 0.0, confidence) for _ in range(21)]
    image_points[0] = Landmark(x, 0.56, 0.0, confidence)
    image_points[5] = Landmark(x - 0.03, 0.62, 0.0, confidence)
    image_points[9] = Landmark(x, 0.66, 0.0, confidence)
    image_points[17] = Landmark(x + 0.03, 0.62, 0.0, confidence)

    world_points = [Landmark(0.0, 0.06, 0.0, confidence) for _ in range(21)]
    world_points[0] = Landmark(0.0, 0.0, 0.0, confidence)
    world_points[5] = Landmark(-0.03, 0.06, 0.0, confidence)
    world_points[9] = Landmark(0.0, 0.10, 0.0, confidence)
    world_points[17] = Landmark(0.03, 0.06, 0.0, confidence)
    return HandDetection(
        side=side,  # type: ignore[arg-type]
        landmarks=tuple(image_points),
        world_landmarks=tuple(world_points),
        confidence=confidence,
    )


def landmark_payload(point: Landmark) -> dict[str, float]:
    return {
        "x": point.x,
        "y": point.y,
        "z": point.z,
        "confidence": point.confidence,
    }


def request_payload() -> dict:
    pose = make_pose()
    left = make_hand("left")
    right = make_hand("right")
    return {
        "frame_id": "test-frame",
        "worker_id": "worker-1",
        "measurement_id": "measurement-1",
        "pose": {
            "person_id": pose.person_id,
            "landmarks": [landmark_payload(point) for point in pose.landmarks],
            "world_landmarks": [
                landmark_payload(point) for point in pose.world_landmarks or ()
            ],
            "confidence": pose.confidence,
        },
        "hands": {
            "left": {
                "landmarks": [landmark_payload(point) for point in left.landmarks],
                "world_landmarks": [
                    landmark_payload(point) for point in left.world_landmarks or ()
                ],
                "confidence": left.confidence,
            },
            "right": {
                "landmarks": [landmark_payload(point) for point in right.landmarks],
                "world_landmarks": [
                    landmark_payload(point) for point in right.world_landmarks or ()
                ],
                "confidence": right.confidence,
            },
        },
    }
