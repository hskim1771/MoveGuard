"""RULA Table A/B/C calculation; no learned or fabricated RULA score."""

TABLE_A = [
 [[(1,2),(2,2),(2,3),(3,3)],[(2,2),(2,2),(3,3),(3,3)],[(2,3),(3,3),(3,3),(4,4)]],
 [[(2,3),(3,3),(3,4),(4,4)],[(3,3),(3,4),(4,4),(4,5)],[(3,4),(4,4),(4,4),(5,5)]],
 [[(3,3),(4,4),(4,4),(5,5)],[(3,4),(4,4),(4,4),(5,5)],[(4,4),(4,4),(4,5),(5,5)]],
 [[(4,4),(4,4),(4,5),(5,5)],[(4,4),(4,4),(4,5),(5,5)],[(4,4),(4,5),(5,5),(6,6)]],
]
TABLE_B = [
 [(2,3),(2,3),(3,4),(5,5),(6,6),(7,7)],[(2,3),(2,3),(4,5),(5,5),(6,7),(7,7)],
 [(3,3),(3,4),(4,5),(5,6),(6,7),(7,7)],[(5,5),(5,6),(6,7),(7,7),(7,7),(8,8)],
 [(7,7),(7,7),(7,8),(8,8),(8,8),(8,8)],[(8,8),(8,8),(8,8),(8,9),(9,9),(9,9)],
]
TABLE_C = [[1,2,3,4,5,6,7],[2,2,2,3,4,5,5],[3,3,3,3,4,5,6],[3,3,4,4,4,5,6],[4,4,4,4,5,6,7],[4,4,4,5,6,6,7],[5,5,5,6,6,7,7],[5,5,6,6,7,7,7]]


def calculate(scores: dict[str, int]) -> dict[str, int | str]:
    """Calculate official lookup result from explicit component scores."""
    ua = max(1, min(4, scores["upper_arm"]))
    la = max(1, min(3, scores["lower_arm"]))
    wrist = max(1, min(4, scores["wrist"]))
    twist = max(1, min(2, scores["wrist_twist"]))
    neck = max(1, min(6, scores["neck"]))
    trunk = max(1, min(6, scores["trunk"]))
    legs = max(1, min(2, scores["legs"]))
    table_a = TABLE_A[ua-1][la-1][wrist-1][twist-1]
    table_b = TABLE_B[neck-1][trunk-1][legs-1]
    score_a = min(8, table_a + scores.get("muscle_a", 0) + scores.get("force_a", 0))
    score_b = min(7, table_b + scores.get("muscle_b", 0) + scores.get("force_b", 0))
    final = TABLE_C[score_a-1][score_b-1]
    action = 1 if final <= 2 else 2 if final <= 4 else 3 if final <= 6 else 4
    return {"table_a": table_a, "table_b": table_b, "score_a": score_a, "score_b": score_b, "final_score": final, "action_level": action}
