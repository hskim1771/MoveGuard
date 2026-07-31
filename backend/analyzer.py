import math

from models import (
    AnalyzeRequest,
    Landmark,
    MetricResult,
    PersonAnalysis,
    PersonPose,
    Risk,
)


RISK_ORDER: dict[Risk, int] = {
    "normal": 0,
    "caution": 1,
    "danger": 2,
}


def midpoint(first: Landmark, second: Landmark) -> Landmark:
    return Landmark(
        x=(first.x + second.x) / 2,
        y=(first.y + second.y) / 2,
        z=(first.z + second.z) / 2,
        visibility=min(first.visibility, second.visibility),
    )


def angle_at(first: Landmark, pivot: Landmark, third: Landmark) -> float:
    vector_a = (first.x - pivot.x, first.y - pivot.y)
    vector_b = (third.x - pivot.x, third.y - pivot.y)
    magnitude_a = math.hypot(*vector_a)
    magnitude_b = math.hypot(*vector_b)
    if not magnitude_a or not magnitude_b:
        return 0

    cosine = max(
        -1,
        min(
            1,
            (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1])
            / (magnitude_a * magnitude_b),
        ),
    )
    return math.degrees(math.acos(cosine))


def tilt_from_vertical(top: Landmark, bottom: Landmark) -> float:
    delta_x = top.x - bottom.x
    delta_y = top.y - bottom.y
    magnitude = math.hypot(delta_x, delta_y)
    if not magnitude:
        return 0

    cosine = max(-1, min(1, -delta_y / magnitude))
    return math.degrees(math.acos(cosine))


def rotation_from_pair(first: Landmark, second: Landmark) -> float:
    depth_difference = abs(first.z - second.z)
    visible_separation = math.hypot(first.x - second.x, first.y - second.y)
    if not depth_difference and not visible_separation:
        return 0
    return math.degrees(math.atan2(depth_difference, visible_separation))


def risk_for(angle: float, caution_at: float, danger_at: float) -> Risk:
    if angle >= danger_at:
        return "danger"
    if angle >= caution_at:
        return "caution"
    return "normal"


def highest_risk(risks: list[Risk]) -> Risk:
    return max(risks, key=RISK_ORDER.get, default="normal")


def metric(
    angle: float,
    caution_at: float,
    danger_at: float,
    *,
    experimental: bool = False,
) -> MetricResult:
    return MetricResult(
        angle=round(angle, 2),
        risk=risk_for(angle, caution_at, danger_at),
        experimental=experimental,
    )


def wrist_metric(
    landmarks: list[Landmark],
    world_landmarks: list[Landmark],
    index_point: int,
    pinky_point: int,
) -> MetricResult:
    is_visible = (
        landmarks[index_point].visibility > 0.38
        and landmarks[pinky_point].visibility > 0.38
    )
    if not is_visible:
        return MetricResult(angle=None, risk="normal", experimental=True)

    return metric(
        rotation_from_pair(
            world_landmarks[index_point],
            world_landmarks[pinky_point],
        ),
        45,
        70,
        experimental=True,
    )


def analyze_person(person: PersonPose) -> PersonAnalysis:
    landmarks = person.landmarks
    world_landmarks = person.world_landmarks or landmarks

    shoulder_mid = midpoint(landmarks[11], landmarks[12])
    hip_mid = midpoint(landmarks[23], landmarks[24])
    ear_mid = midpoint(landmarks[7], landmarks[8])

    neck_angle = tilt_from_vertical(ear_mid, shoulder_mid)
    trunk_angle = tilt_from_vertical(shoulder_mid, hip_mid)
    left_shoulder = angle_at(landmarks[13], landmarks[11], landmarks[23])
    right_shoulder = angle_at(landmarks[14], landmarks[12], landmarks[24])
    shoulder_angle = max(left_shoulder, right_shoulder)
    head_rotation = rotation_from_pair(world_landmarks[7], world_landmarks[8])
    shoulder_rotation = rotation_from_pair(
        world_landmarks[11], world_landmarks[12]
    )
    hip_rotation = rotation_from_pair(world_landmarks[23], world_landmarks[24])
    trunk_rotation = (shoulder_rotation + hip_rotation) / 2

    metrics = {
        "neck": metric(neck_angle, 20, 35),
        "trunk": metric(trunk_angle, 15, 30),
        "shoulder": metric(shoulder_angle, 30, 60),
        "headRotation": metric(head_rotation, 25, 45),
        "trunkRotation": metric(trunk_rotation, 25, 45),
        "leftWristRotation": wrist_metric(
            landmarks, world_landmarks, 19, 17
        ),
        "rightWristRotation": wrist_metric(
            landmarks, world_landmarks, 20, 18
        ),
    }

    key_indexes = [7, 8, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
    confidence = round(
        sum(landmarks[index].visibility for index in key_indexes)
        / len(key_indexes)
        * 100
    )
    full_body = all(
        landmarks[index].visibility > 0.45
        and 0.015 < landmarks[index].x < 0.985
        and 0.015 < landmarks[index].y < 0.985
        for index in [11, 12, 23, 24, 25, 26, 27, 28]
    )
    overall_risk = highest_risk([item.risk for item in metrics.values()])

    return PersonAnalysis(
        person_id=person.person_id,
        metrics=metrics,
        confidence=confidence,
        full_body=full_body,
        overall_risk=overall_risk,
    )


def analyze_request(payload: AnalyzeRequest) -> list[PersonAnalysis]:
    return [analyze_person(person) for person in payload.people]
