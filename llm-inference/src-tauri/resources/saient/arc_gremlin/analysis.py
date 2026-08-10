from __future__ import annotations

from typing import Any

from .legacy import _to_grid, apply_hypothesis, load_task


def _grid_match(a: list[list[int]], b: list[list[int]]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0.0
    total = len(a) * len(a[0])
    same = 0
    for y in range(len(a)):
        for x in range(len(a[0])):
            if a[y][x] == b[y][x]:
                same += 1
    return same / max(1, total)


def classify_failure(task_path: str, hypothesis: dict[str, Any] | None, score: float) -> dict[str, Any]:
    task = load_task(task_path)
    train = task.get("train", [])
    if not train:
        return {"category": "unknown", "reason": "no_train_pairs"}

    if hypothesis is None:
        return {"category": "no_solution", "reason": "no_hypothesis"}

    per_pair: list[dict[str, Any]] = []
    no_op_count = 0
    shape_mismatch = 0
    matched_total = 0.0

    for pair in train:
        if isinstance(pair, dict):
            inp = _to_grid(pair.get("input"))
            out = _to_grid(pair.get("output"))
        else:
            inp = _to_grid(pair[0])
            out = _to_grid(pair[1])
        pred = apply_hypothesis(inp, hypothesis, target_grid=out)

        if pred == inp and out != inp:
            no_op_count += 1

        if len(pred) != len(out) or (pred and out and len(pred[0]) != len(out[0])):
            shape_mismatch += 1
            pair_match = 0.0
        else:
            pair_match = _grid_match(pred, out)
            matched_total += pair_match

        per_pair.append(
            {
                "input_shape": [len(inp), len(inp[0]) if inp else 0],
                "output_shape": [len(out), len(out[0]) if out else 0],
                "pred_shape": [len(pred), len(pred[0]) if pred else 0],
                "pair_match": round(pair_match, 4),
            }
        )

    n = max(1, len(per_pair))
    avg_pair_match = matched_total / n

    category = "wrong_transformation"
    reason = "default"
    likely_missing_op = "template_match"
    confidence = 0.35
    if float(score) >= 0.999:
        category = "solved"
        reason = "perfect_match"
        likely_missing_op = "none"
        confidence = 1.0
    elif no_op_count >= max(1, n // 2):
        category = "no_op_loop"
        reason = "predictions_equal_inputs"
        likely_missing_op = "template_match"
        confidence = 0.8
    elif shape_mismatch >= max(1, n // 2):
        category = "wrong_transformation"
        reason = "shape_mismatch"
        likely_missing_op = "append_head_rows"
        confidence = 0.72
    elif 0.0 < avg_pair_match < 0.95:
        category = "partial_match"
        reason = "nonzero_overlap_but_not_solved"
        likely_missing_op = "rotate_90"
        confidence = 0.67
    elif float(score) >= 0.95:
        category = "overfit_rule"
        reason = "high_train_score_without_perfect_generalization"
        likely_missing_op = "template_match"
        confidence = 0.7

    if avg_pair_match <= 0.05 and shape_mismatch == 0:
        likely_missing_op = "template_match"
        confidence = max(confidence, 0.75)

    if shape_mismatch and avg_pair_match > 0.5:
        likely_missing_op = "append_tail_rows"
        confidence = max(confidence, 0.7)

    return {
        "failure_type": category,
        "category": category,
        "reason": reason,
        "likely_missing_op": likely_missing_op,
        "confidence": round(float(confidence), 3),
        "avg_pair_match": round(avg_pair_match, 4),
        "pairs": per_pair,
    }


def detect_plateau(score_progression: list[float], window: int = 20, eps: float = 1e-9) -> dict[str, Any]:
    if not score_progression:
        return {"plateau": False, "window": window, "delta": 0.0}
    if len(score_progression) < window:
        return {"plateau": False, "window": window, "delta": 0.0}

    tail = score_progression[-window:]
    delta = float(max(tail) - min(tail))
    return {
        "plateau": bool(delta <= eps),
        "window": int(window),
        "delta": delta,
        "best_tail": float(max(tail)),
        "last_tail": float(tail[-1]),
    }
