from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import JointMeasurement

# These lookup values intentionally match the current MoveGuard MVP tables so
# the new endpoint remains comparable with the existing report flow.
TABLE_A = [
    [
        {1: [1, 2], 2: [2, 2], 3: [2, 3], 4: [3, 3]},
        {1: [2, 2], 2: [2, 2], 3: [3, 3], 4: [3, 3]},
        {1: [2, 3], 2: [3, 3], 3: [3, 3], 4: [4, 4]},
    ],
    [
        {1: [2, 3], 2: [3, 3], 3: [3, 4], 4: [4, 4]},
        {1: [3, 3], 2: [3, 4], 3: [4, 4], 4: [4, 5]},
        {1: [3, 4], 2: [4, 4], 3: [4, 4], 4: [5, 5]},
    ],
    [
        {1: [3, 3], 2: [4, 4], 3: [4, 4], 4: [5, 5]},
        {1: [3, 4], 2: [4, 4], 3: [4, 4], 4: [5, 5]},
        {1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
    ],
    [
        {1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
        {1: [4, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
        {1: [4, 4], 2: [4, 5], 3: [5, 5], 4: [6, 6]},
    ],
]

TABLE_B = [
    [[2, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
]

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


@dataclass(frozen=True)
class RulaOptions:
    weight_kg: float = 0.0
    static_force: bool = False
    muscle_static: bool = False
    legs_unsupported: bool | None = None
    wrist_twist_at_end_range: bool | None = None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def neck_score(angle: float) -> int:
    if angle <= 10:
        return 1
    if angle <= 20:
        return 2
    return 3


def trunk_score(angle: float) -> int:
    if angle <= 5:
        return 1
    if angle <= 20:
        return 2
    if angle <= 60:
        return 3
    return 4


def upper_arm_score(angle: float) -> int:
    if angle <= 20:
        return 1
    if angle <= 45:
        return 2
    if angle <= 90:
        return 3
    return 4


def lower_arm_score(angle: float) -> int:
    return 1 if 60 <= angle <= 100 else 2


def wrist_score(angle: float, deviation_angle: float | None) -> int:
    if angle <= 5:
        score = 1
    elif angle <= 15:
        score = 2
    else:
        score = 3
    if deviation_angle is not None and deviation_angle > 10:
        score += 1
    return _clamp(score, 1, 4)


def force_score(options: RulaOptions) -> int:
    if options.weight_kg < 2:
        return 1 if options.static_force else 0
    if options.weight_kg <= 10:
        return 2
    return 3


def risk_level(final_score: int) -> str:
    if final_score <= 2:
        return "low"
    if final_score <= 4:
        return "medium"
    if final_score <= 6:
        return "high"
    return "very_high"


def _invalid_result(
    partial_scores: dict[str, int | None],
    issues: list[str],
    assumptions: list[str],
) -> dict[str, Any]:
    return {
        **partial_scores,
        "score_a": None,
        "score_b": None,
        "final_score": None,
        "risk_level": "not_measurable",
        "scored_side": None,
        "valid": False,
        "issues": issues,
        "assumptions": assumptions,
    }


def calculate_rula(
    measurements: dict[str, JointMeasurement],
    options: RulaOptions | None = None,
) -> dict[str, Any]:
    options = options or RulaOptions()
    assumptions: list[str] = []
    issues: list[str] = []

    required_center = ("neck", "trunk", "legs")
    for name in required_center:
        measurement = measurements.get(name)
        if measurement is None or not measurement.valid:
            issues.append(f"{name}:{measurement.reason if measurement else 'missing'}")

    neck = measurements.get("neck")
    trunk = measurements.get("trunk")
    legs = measurements.get("legs")
    n_score = neck_score(neck.angle) if neck and neck.valid else None  # type: ignore[arg-type]
    t_score = trunk_score(trunk.angle) if trunk and trunk.valid else None  # type: ignore[arg-type]

    inferred_support = legs.auxiliary.get("support_inferred") if legs else None
    if options.legs_unsupported is not None:
        leg_score = 2 if options.legs_unsupported else 1
    elif legs and legs.valid and inferred_support is not None:
        leg_score = 1 if inferred_support else 2
        assumptions.append("leg_support_inferred_from_knee_symmetry")
    else:
        leg_score = None

    side_candidates: list[dict[str, Any]] = []
    for side in ("left", "right"):
        upper = measurements.get(f"{side}_upper_arm")
        lower = measurements.get(f"{side}_lower_arm")
        wrist = measurements.get(f"{side}_wrist")
        if not upper or not upper.valid:
            issues.append(f"{side}_upper_arm:{upper.reason if upper else 'missing'}")
            continue
        if not lower or not lower.valid:
            issues.append(f"{side}_lower_arm:{lower.reason if lower else 'missing'}")
            continue
        upper_score = upper_arm_score(upper.angle)  # type: ignore[arg-type]
        lower_score = lower_arm_score(lower.angle)  # type: ignore[arg-type]
        wrist_value = None
        if wrist and wrist.valid:
            wrist_value = wrist_score(
                wrist.angle,  # type: ignore[arg-type]
                wrist.auxiliary.get("deviation_angle"),  # type: ignore[arg-type]
            )
        side_candidates.append(
            {
                "side": side,
                "upper": upper_score,
                "lower": lower_score,
                "wrist": wrist_value,
                "pose_severity": upper_score + lower_score,
            }
        )

    partial_scores: dict[str, int | None] = {
        "neck_score": n_score,
        "trunk_score": t_score,
        "upper_arm_score": None,
        "lower_arm_score": None,
        "wrist_score": None,
        "leg_score": leg_score,
    }

    if issues and not side_candidates:
        return _invalid_result(partial_scores, issues, assumptions)
    if n_score is None or t_score is None or leg_score is None:
        return _invalid_result(partial_scores, issues, assumptions)

    # RULA is evaluated on the more severe side. If pose severity ties, prefer
    # the side whose hand is actually measurable rather than inventing a wrist.
    side_candidates.sort(
        key=lambda item: (item["pose_severity"], item["wrist"] is not None),
        reverse=True,
    )
    selected = side_candidates[0]
    partial_scores.update(
        {
            "upper_arm_score": selected["upper"],
            "lower_arm_score": selected["lower"],
            "wrist_score": selected["wrist"],
        }
    )
    if selected["wrist"] is None:
        issues.append(f"{selected['side']}_wrist:not_measurable")
        return _invalid_result(partial_scores, issues, assumptions)

    if options.wrist_twist_at_end_range is None:
        wrist_twist_score = 1
        assumptions.append("wrist_twist_mid_range_assumed_single_camera")
    else:
        wrist_twist_score = 2 if options.wrist_twist_at_end_range else 1

    muscle_score = 1 if options.muscle_static else 0
    load_score = force_score(options)
    table_a = TABLE_A[selected["upper"] - 1][selected["lower"] - 1][
        selected["wrist"]
    ][wrist_twist_score - 1]
    score_a = _clamp(table_a + muscle_score + load_score, 1, 8)
    table_b = TABLE_B[n_score - 1][t_score - 1][leg_score - 1]
    score_b = _clamp(table_b + muscle_score + load_score, 1, 7)
    final_score = TABLE_C[score_a - 1][score_b - 1]

    return {
        **partial_scores,
        "wrist_twist_score": wrist_twist_score,
        "muscle_score": muscle_score,
        "force_score": load_score,
        "score_a": score_a,
        "score_b": score_b,
        "final_score": final_score,
        "risk_level": risk_level(final_score),
        "scored_side": selected["side"],
        "valid": True,
        "issues": issues,
        "assumptions": assumptions,
    }
