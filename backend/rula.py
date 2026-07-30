"""
RULA scoring logic — moveguard_mvp.html의 TABLE_A/B/C, 채점 함수를 그대로 이식.
frontend(JS)와 반드시 같은 값을 내야 하므로, 저 파일을 고치면 여기도 같이 고칠 것.
"""
import math

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
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1, m2 = math.hypot(*v1), math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos))


def vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def neck_score(a):
    if a <= 10:
        return 1
    if a <= 20:
        return 2
    return 3


def trunk_score(a):
    if a <= 5:
        return 1
    if a <= 20:
        return 2
    if a <= 60:
        return 3
    return 4


def upper_arm_score(a):
    if a <= 20:
        return 1
    if a <= 45:
        return 2
    if a <= 90:
        return 3
    return 4


def lower_arm_score(a):
    return 1 if 60 <= a <= 100 else 2


def force_score(kg, static_or_repetitive):
    if kg < 2:
        return 1 if static_or_repetitive else 0
    if kg <= 10:
        return 2
    return 3


def compute_frame_angles(lm):
    """lm: MediaPipe 33-point landmark list, 각 원소는 (x, y, visibility)"""
    left_vis = (lm[11][2] + lm[13][2] + lm[15][2] + lm[23][2]) / 4
    right_vis = (lm[12][2] + lm[14][2] + lm[16][2] + lm[24][2]) / 4
    if right_vis >= left_vis:
        sh, el, wr = lm[12], lm[14], lm[16]
    else:
        sh, el, wr = lm[11], lm[13], lm[15]
    nose = lm[0]
    shoulder_mid = ((lm[11][0] + lm[12][0]) / 2, (lm[11][1] + lm[12][1]) / 2)
    hip_mid = ((lm[23][0] + lm[24][0]) / 2, (lm[23][1] + lm[24][1]) / 2)

    vertical = (0, -1)
    trunk_vec = vec(hip_mid, shoulder_mid)
    trunk_angle = angle_between(trunk_vec, vertical)

    neck_vec = vec(shoulder_mid, nose[:2])
    neck_angle = angle_between(neck_vec, trunk_vec)

    upper_arm_vec = vec(sh[:2], el[:2])
    upper_arm_angle = angle_between(upper_arm_vec, trunk_vec)

    elbow_angle = angle_between(vec(el[:2], sh[:2]), vec(el[:2], wr[:2]))

    return {
        "neckAngle": neck_angle, "trunkAngle": trunk_angle,
        "upperArmAngle": upper_arm_angle, "elbowAngle": elbow_angle,
        "confidence": max(left_vis, right_vis),
    }


def compute_rula(worst_frame_angles, weight_kg, static_force, muscle_static, legs_unsupported):
    """
    worst_frame_angles: compute_frame_angles()의 결과 (가장 위험한 프레임)
    나머지는 프론트에서 넘어오는 보충 입력값 (하중, 근육사용, 다리지지 여부)
    """
    n_s = neck_score(worst_frame_angles["neckAngle"])
    t_s = trunk_score(worst_frame_angles["trunkAngle"])
    ua_s = upper_arm_score(worst_frame_angles["upperArmAngle"])
    la_s = lower_arm_score(worst_frame_angles["elbowAngle"])
    wrist_s = 1       # 손목: 카메라만으로 측정 불가 → 중립 고정
    wrist_twist = 1   # T1 고정
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
