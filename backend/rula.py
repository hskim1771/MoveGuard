"""
RULA scoring logic — moveguard_mvp.html의 TABLE_A/B/C, 채점 함수를 그대로 이식.
frontend(JS)와 반드시 같은 값을 내야 하므로, 저 파일을 고치면 여기도 같이 고칠 것.
"""
import math
import statistics

# Table A[upperArm-1][lowerArm-1][wrist] = [T1값, T2값]
TABLE_A = [
    [{1: [1, 2], 2: [2, 2], 3: [2, 3], 4: [3, 3]},
     {1: [2, 2], 2: [2, 2], 3: [3, 3], 4: [3, 3]},
     {1: [2, 3], 2: [3, 3], 3: [3, 3], 4: [4, 4]}],
    [{1: [2, 3], 2: [3, 3], 3: [3, 4], 4: [4, 4]},
     {1: [3, 3], 2: [3, 4], 3: [4, 4], 4: [4, 5]},
     {1: [3, 4], 2: [4, 4], 3: [4, 4], 4: [5, 5]}],
    [{1: [3, 3], 2: [4, 4], 3: [4, 4], 4: [5, 5]},
     {1: [3, 4], 2: [4, 4], 3: [4, 4], 4: [5, 5]},
     {1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]}],
    [{1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
     {1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
     {1: [4, 4], 2: [4, 5], 3: [5, 5], 4: [6, 6]}],
    [{1: [5, 5], 2: [5, 5], 3: [5, 6], 4: [6, 7]},
     {1: [5, 6], 2: [6, 6], 3: [6, 7], 4: [7, 7]},
     {1: [6, 6], 2: [6, 7], 3: [7, 7], 4: [7, 8]}],
    [{1: [7, 7], 2: [7, 7], 3: [7, 8], 4: [8, 9]},
     {1: [8, 8], 2: [8, 8], 3: [8, 9], 4: [9, 9]},
     {1: [9, 9], 2: [9, 9], 3: [9, 9], 4: [9, 9]}],
]

# Table B[neck-1][trunk-1] = [leg1값, leg2값]
TABLE_B = [
    [[2, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
]

# Table C[WA-1][NTL-1], WA 1-8, NTL 1-7
TABLE_C = [
    [1, 2, 3, 4, 5, 6, 7],
    [2, 2, 2, 3, 4, 5, 5],
    [3, 3, 3, 3, 4, 5, 6],
    [3, 3, 4, 4, 4, 5, 6],
    [4, 4, 4, 4, 5, 6, 7],
    [4, 4, 4, 5, 6, 6, 7],
    [5, 5, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
]

ACTION_LEVELS = [
    {"max": 2, "label": "조치수준 1 — 문제 없음", "desc": "현재 자세는 허용 가능한 수준입니다."},
    {"max": 4, "label": "조치수준 2 — 추가 관찰 필요", "desc": "조만간 개선을 권장합니다."},
    {"max": 6, "label": "조치수준 3 — 빠른 시일 내 개선 필요", "desc": "작업 방식 개선을 가까운 시일 내 계획하세요."},
    {"max": 99, "label": "조치수준 4 — 즉시 개선 필요", "desc": "즉각적인 작업 개선 조치가 필요합니다."},
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def angle_between(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(value * value for value in v1))
    m2 = math.sqrt(sum(value * value for value in v2))
    if m1 == 0 or m2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos))


def vec(a, b):
    return tuple(b[index] - a[index] for index in range(min(len(a), len(b), 3)))


def neck_score(a, rotated=False, side_bent=False, extended=False):
    if extended:
        base = 4
    elif a <= 10:
        base = 1
    elif a <= 20:
        base = 2
    else:
        base = 3
    return clamp(base + int(rotated or side_bent), 1, 6)


def trunk_score(a, rotated=False, side_bent=False):
    if a <= 5:
        base = 1
    elif a <= 20:
        base = 2
    elif a <= 60:
        base = 3
    else:
        base = 4
    return clamp(base + int(rotated or side_bent), 1, 6)


def upper_arm_score(a, shoulder_raised=False, arm_abducted=False, arm_supported=False):
    if a <= 20:
        base = 1
    elif a <= 45:
        base = 2
    elif a <= 90:
        base = 3
    else:
        base = 4
    return clamp(
        base + int(shoulder_raised) + int(arm_abducted) - int(arm_supported),
        1,
        6,
    )


def lower_arm_score(a, out_to_side=False):
    base = 1 if 60 <= a <= 100 else 2
    return clamp(base + int(out_to_side), 1, 3)


def wrist_score(a, deviated=False):
    if a <= 5:
        return 1
    base = 2 if a <= 15 else 3
    return clamp(base + int(deviated), 1, 4)


def force_score(kg, static_or_repetitive):
    if kg < 2:
        return 0
    if kg <= 10:
        return 2 if static_or_repetitive else 1
    return 3


def _xyz(point):
    return point[:3] if len(point) >= 4 else (point[0], point[1], 0.0)


def _visibility(point):
    return point[3] if len(point) >= 4 else point[2]


def _midpoint(a, b):
    return tuple((first + second) / 2 for first, second in zip(_xyz(a), _xyz(b)))


def _pair_rotation(a, b):
    first, second = _xyz(a), _xyz(b)
    depth = abs(first[2] - second[2])
    visible = math.hypot(first[0] - second[0], first[1] - second[1])
    return math.degrees(math.atan2(depth, visible)) if depth or visible else 0.0


def compute_frame_angles(lm, world_lm=None):
    """Calculate RULA posture angles, preferring 3D world landmarks when available."""
    left_vis = sum(_visibility(lm[index]) for index in (11, 13, 15, 23)) / 4
    right_vis = sum(_visibility(lm[index]) for index in (12, 14, 16, 24)) / 4
    full_body_indexes = (11, 12, 23, 24, 25, 26, 27, 28)
    full_body_confidence = sum(_visibility(lm[index]) for index in full_body_indexes) / len(full_body_indexes)
    full_body_in_frame = all(
        _visibility(lm[index]) >= 0.4
        and 0.015 <= lm[index][0] <= 0.985
        and 0.015 <= lm[index][1] <= 0.985
        for index in full_body_indexes
    )
    if right_vis >= left_vis:
        sh_i, el_i, wr_i, index_i, pinky_i, hip_i = 12, 14, 16, 20, 18, 24
    else:
        sh_i, el_i, wr_i, index_i, pinky_i, hip_i = 11, 13, 15, 19, 17, 23

    points = world_lm if world_lm and len(world_lm) >= 33 else lm
    shoulder_mid = _midpoint(points[11], points[12])
    hip_mid = _midpoint(points[23], points[24])
    ear_mid = _midpoint(points[7], points[8])
    hand_mid = _midpoint(points[index_i], points[pinky_i])

    vertical = (0, -1, 0)
    trunk_vec = vec(hip_mid, shoulder_mid)
    trunk_angle = angle_between(trunk_vec, vertical)

    neck_vec = vec(shoulder_mid, ear_mid)
    neck_angle = angle_between(neck_vec, trunk_vec)

    # Both vectors point down from the shoulder, so a neutral hanging arm is 0 degrees.
    upper_arm_angle = angle_between(
        vec(_xyz(points[sh_i]), _xyz(points[el_i])),
        vec(_xyz(points[sh_i]), _xyz(points[hip_i])),
    )
    elbow_angle = angle_between(
        vec(_xyz(points[el_i]), _xyz(points[sh_i])),
        vec(_xyz(points[el_i]), _xyz(points[wr_i])),
    )
    wrist_joint_angle = angle_between(
        vec(_xyz(points[wr_i]), _xyz(points[el_i])),
        vec(_xyz(points[wr_i]), hand_mid),
    )
    wrist_angle = abs(180 - wrist_joint_angle)

    shoulder_rotation = _pair_rotation(points[11], points[12])
    head_rotation = abs(_pair_rotation(points[7], points[8]) - shoulder_rotation)
    hip_rotation = _pair_rotation(points[23], points[24])
    trunk_rotation = abs(shoulder_rotation - hip_rotation)
    wrist_rotation = abs(_pair_rotation(points[index_i], points[pinky_i]) - shoulder_rotation)

    return {
        "neckAngle": neck_angle, "trunkAngle": trunk_angle,
        "upperArmAngle": upper_arm_angle, "elbowAngle": elbow_angle,
        "wristAngle": wrist_angle,
        "neckRotated": head_rotation >= 25,
        "neckSideBent": False,
        "trunkRotated": trunk_rotation >= 20,
        "trunkSideBent": False,
        "wristTwisted": wrist_rotation >= 45,
        "confidence": max(left_vis, right_vis),
        "fullBodyConfidence": full_body_confidence,
        "fullBodyInFrame": full_body_in_frame,
    }


def posture_score(angles):
    return (
        neck_score(angles["neckAngle"], angles.get("neckRotated", False), angles.get("neckSideBent", False))
        + trunk_score(angles["trunkAngle"], angles.get("trunkRotated", False), angles.get("trunkSideBent", False))
        + upper_arm_score(angles["upperArmAngle"])
        + lower_arm_score(angles["elbowAngle"])
        + wrist_score(angles.get("wristAngle", 0))
    )


def select_representative_frame(candidates, percentile=0.9):
    """Suppress one-frame landmark spikes, then select a persistent high-risk frame."""
    if not candidates:
        return None

    angle_keys = ("neckAngle", "trunkAngle", "upperArmAngle", "elbowAngle", "wristAngle")
    flag_keys = ("neckRotated", "neckSideBent", "trunkRotated", "trunkSideBent", "wristTwisted")
    stabilized = []
    for index, candidate in enumerate(candidates):
        window = candidates[max(0, index - 1): min(len(candidates), index + 2)]
        angles = dict(candidate["angles"])
        for key in angle_keys:
            angles[key] = statistics.median(item["angles"].get(key, 0) for item in window)
        for key in flag_keys:
            angles[key] = sum(bool(item["angles"].get(key, False)) for item in window) > len(window) / 2
        stabilized.append({**candidate, "angles": angles, "posture_score": posture_score(angles)})

    ordered = sorted(stabilized, key=lambda item: (item["posture_score"], item["angles"]["confidence"]))
    selected_index = max(0, min(len(ordered) - 1, math.floor((len(ordered) - 1) * percentile)))
    return ordered[selected_index]


def compute_rula(
    worst_frame_angles,
    weight_kg,
    static_force,
    muscle_static,
    legs_unsupported,
    shoulder_raised=False,
    arm_abducted=False,
    arm_supported=False,
    neck_extended=False,
    neck_side_bent=False,
    trunk_side_bent=False,
    lower_arm_out=False,
    wrist_deviated=False,
):
    """
    worst_frame_angles: compute_frame_angles()의 결과 (가장 위험한 프레임)
    나머지는 프론트에서 넘어오는 보충 입력값 (하중, 근육사용, 다리지지 여부)
    """
    n_s = neck_score(
        worst_frame_angles["neckAngle"],
        worst_frame_angles.get("neckRotated", False),
        worst_frame_angles.get("neckSideBent", False) or neck_side_bent,
        neck_extended,
    )
    t_s = trunk_score(
        worst_frame_angles["trunkAngle"],
        worst_frame_angles.get("trunkRotated", False),
        worst_frame_angles.get("trunkSideBent", False) or trunk_side_bent,
    )
    ua_s = upper_arm_score(
        worst_frame_angles["upperArmAngle"],
        shoulder_raised,
        arm_abducted,
        arm_supported,
    )
    la_s = lower_arm_score(worst_frame_angles["elbowAngle"], lower_arm_out)
    wrist_s = wrist_score(worst_frame_angles.get("wristAngle", 0), wrist_deviated)
    wrist_twist = 2 if worst_frame_angles.get("wristTwisted", False) else 1
    leg_s = 2 if legs_unsupported else 1

    f_s = force_score(weight_kg, static_force)
    m_s = 1 if muscle_static else 0

    table_a_val = TABLE_A[ua_s - 1][la_s - 1][wrist_s][wrist_twist - 1]
    score_a = clamp(table_a_val + m_s + f_s, 1, 8)

    table_b_val = TABLE_B[n_s - 1][t_s - 1][leg_s - 1]
    score_b = clamp(table_b_val + m_s + f_s, 1, 7)

    grand = TABLE_C[score_a - 1][score_b - 1]
    level = next(l for l in ACTION_LEVELS if grand <= l["max"])

    return {
        "neckScore": n_s, "trunkScore": t_s, "upperArmScore": ua_s, "lowerArmScore": la_s,
        "wristScore": wrist_s, "wristTwist": wrist_twist, "legScore": leg_s,
        "forceScore": f_s, "muscleScore": m_s,
        "scoreA": score_a, "scoreB": score_b,
        "grandScore": grand, "actionLevel": level,
        "angles": worst_frame_angles,
    }
