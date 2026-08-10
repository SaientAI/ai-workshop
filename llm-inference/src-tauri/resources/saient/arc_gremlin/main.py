from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from hashlib import sha256
from typing import Any

from .analysis import classify_failure, detect_plateau
from .config import seed_for_task, set_seed
from .legacy import OPS
from .legacy import run_arc_reasoning as _legacy_run_arc_reasoning

STRATEGIES = ("exploration_boost", "reset_priors", "noise_injection")


def _state_signature(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return "none"
    payload = {
        "beliefs": state.get("beliefs", {}),
        "arc_last_good_ops": state.get("arc_last_good_ops", []),
        "arc_op_stats": state.get("arc_op_stats", {}),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()


def _boost_ops(state: dict[str, Any], ops: list[str], amount: int = 2) -> None:
    op_stats = state.setdefault("arc_op_stats", {op: {"attempts": 0, "successes": 0} for op in OPS})
    for op in ops:
        st = op_stats.setdefault(op, {"attempts": 0, "successes": 0})
        st["attempts"] = int(st.get("attempts", 0)) + amount
        st["successes"] = int(st.get("successes", 0)) + amount


def _enforce_missing_op(state: dict[str, Any], failure: dict[str, Any]) -> None:
    op = str(failure.get("likely_missing_op", "") or "")
    conf = float(failure.get("confidence", 0.0) or 0.0)
    if not op or op == "none" or op not in OPS:
        state["arc_forced_op"] = ""
        state["arc_forced_op_prob"] = 0.0
        return
    # Turn hint into force when confidence is strong.
    if conf > 0.6:
        _boost_ops(state, [op], amount=18)
        state["arc_forced_op"] = op
        state["arc_forced_op_prob"] = 0.7
    else:
        state["arc_forced_op"] = op
        state["arc_forced_op_prob"] = 0.3


def _apply_strategy_pressure(state: dict[str, Any], strategy: dict[str, Any]) -> None:
    name = str(strategy.get("strategy", "exploration_boost"))
    op_stats = state.setdefault("arc_op_stats", {op: {"attempts": 0, "successes": 0} for op in OPS})
    rnd = random.Random(int(strategy.get("parameters", {}).get("seed", 0)))
    if name == "reset_priors":
        # Hard reset towards neutral priors.
        for op in OPS:
            st = op_stats.setdefault(op, {"attempts": 0, "successes": 0})
            st["attempts"] = max(0, int(round(float(st.get("attempts", 0)) * 0.5)))
            st["successes"] = max(0, int(round(float(st.get("successes", 0)) * 0.5)))
        state["arc_last_good_ops"] = []
        return

    if name == "noise_injection":
        # Small randomized perturbations of op priors.
        for op in OPS:
            st = op_stats.setdefault(op, {"attempts": 0, "successes": 0})
            st["attempts"] = max(0, int(st.get("attempts", 0)) + rnd.randint(-2, 2))
            st["successes"] = max(0, int(st.get("successes", 0)) + rnd.randint(-1, 1))
        return

    # Default exploration_boost anti-dominance pressure.
    entries = []
    for op, st in op_stats.items():
        attempts = max(0, int(st.get("attempts", 0)))
        successes = max(0, int(st.get("successes", 0)))
        rate = successes / max(1, attempts)
        entries.append((rate, attempts, op))
    entries.sort(reverse=True)

    # Penalize dominant rules.
    for _, _, op in entries[:3]:
        st = op_stats[op]
        st["successes"] = int(round(max(0.0, float(st.get("successes", 0)) * 0.7)))

    # Reward novelty by lowering attempt counts on underexplored ops.
    by_attempts = sorted(((int(st.get("attempts", 0)), op) for op, st in op_stats.items()))
    for _, op in by_attempts[:5]:
        st = op_stats[op]
        st["attempts"] = max(0, int(st.get("attempts", 0)) - 3)

    # Inject light noise into priors to avoid immediate re-collapse.
    for op in OPS:
        if rnd.random() < 0.15:
            st = op_stats.setdefault(op, {"attempts": 0, "successes": 0})
            st["attempts"] = max(0, int(st.get("attempts", 0)) - 1)


def _likely_ops_from_failure(failure: dict[str, Any]) -> list[str]:
    m = {
        "rotate_90": ["rotate_90", "flip_h", "flip_v"],
        "append_head_rows": ["append_head_rows", "append_tail_rows", "crop_to_bbox"],
        "append_tail_rows": ["append_tail_rows", "append_head_rows", "crop_to_bbox"],
        "template_match": ["template_match", "extract_repeating_unit", "transpose"],
        "none": [],
    }
    return m.get(str(failure.get("likely_missing_op", "template_match")), ["template_match"])


def _plateau_length(score_progression: list[float], eps: float = 1e-9) -> int:
    if not score_progression:
        return 0
    last = float(score_progression[-1])
    n = 0
    for v in reversed(score_progression):
        if abs(float(v) - last) <= eps:
            n += 1
        else:
            break
    return n


def _plateau_bucket(n: int) -> str:
    if n < 10:
        return "short"
    if n < 30:
        return "medium"
    return "long"


def _context_key(context: dict[str, Any]) -> str:
    return f"{context.get('failure_type', 'unknown')}|{context.get('plateau_bucket', 'na')}"


def _weighted_mean(items: list[dict[str, Any]], decay: float = 0.985) -> float:
    if not items:
        return 0.0
    wsum = 0.0
    xsum = 0.0
    for idx, it in enumerate(items):
        # Newer samples get larger weight.
        age = len(items) - 1 - idx
        w = decay**age
        x = float(it.get("weighted_delta", it.get("delta", 0.0)))
        wsum += w
        xsum += w * x
    return xsum / max(1e-9, wsum)


def _recompute_strategy_effectiveness(state: dict[str, Any]) -> None:
    outcomes = list(state.get("strategy_outcomes", []))
    overall: dict[str, dict[str, Any]] = {}
    by_context: dict[str, dict[str, dict[str, Any]]] = {}
    usage = state.setdefault("strategy_usage", {"overall": {}, "by_context": {}})

    for sname in STRATEGIES:
        rows = [o for o in outcomes if str(o.get("strategy")) == sname]
        overall[sname] = {
            "mean_delta": _weighted_mean(rows),
            "count": len(rows),
            "usage_count": int((usage.get("overall") or {}).get(sname, 0)),
        }

    for o in outcomes:
        ckey = _context_key(o.get("context") or {})
        sname = str(o.get("strategy", "exploration_boost"))
        by_context.setdefault(ckey, {})
        rows = [x for x in outcomes if _context_key(x.get("context") or {}) == ckey and str(x.get("strategy")) == sname]
        usage_ctx = (((usage.get("by_context") or {}).get(ckey, {})).get(sname, 0))
        by_context[ckey][sname] = {
            "mean_delta": _weighted_mean(rows),
            "count": len(rows),
            "usage_count": int(usage_ctx),
        }

    state["strategy_effectiveness"] = {"overall": overall, "by_context": by_context}


def _record_strategy_outcome(state: dict[str, Any], outcome: dict[str, Any]) -> None:
    outcomes = state.setdefault("strategy_outcomes", [])
    outcomes.append(outcome)
    state["strategy_outcomes"] = outcomes[-512:]
    # Track usage for anti-dominance policy scoring.
    usage = state.setdefault("strategy_usage", {"overall": {}, "by_context": {}})
    sname = str(outcome.get("strategy", "exploration_boost"))
    ckey = _context_key(outcome.get("context") or {})
    usage["overall"][sname] = int((usage.get("overall") or {}).get(sname, 0)) + 1
    usage.setdefault("by_context", {}).setdefault(ckey, {})
    usage["by_context"][ckey][sname] = int(usage["by_context"][ckey].get(sname, 0)) + 1

    _recompute_strategy_effectiveness(state)
    policy = state.setdefault("strategy_policy", {"epsilon": 0.2, "epsilon_decay": 0.995, "epsilon_min": 0.05})
    eps = float(policy.get("epsilon", 0.2))
    eps_decay = float(policy.get("epsilon_decay", 0.995))
    eps_min = float(policy.get("epsilon_min", 0.05))
    policy["epsilon"] = max(eps_min, eps * eps_decay)


def _is_strategy_temporarily_ineffective(state: dict[str, Any], strategy: str) -> bool:
    eff = state.get("strategy_effectiveness", {})
    overall = eff.get("overall", {}) if isinstance(eff, dict) else {}
    row = overall.get(strategy, {}) if isinstance(overall, dict) else {}
    cnt = int(row.get("count", 0))
    mean = float(row.get("mean_delta", 0.0))
    # Enough evidence and no positive contribution -> suppress selection pressure.
    return cnt >= 3 and mean <= 0.0


def _select_best_strategy(state: dict[str, Any], context: dict[str, Any], seed: int) -> tuple[str, dict[str, Any]]:
    forced = os.environ.get("ARC_FORCE_STRATEGY", "").strip()
    if forced in STRATEGIES:
        return forced, {"mode": "forced", "epsilon": 0.0, "context_key": _context_key(context)}

    rnd = random.Random(seed)
    policy_state = state.setdefault("strategy_policy", {"epsilon": 0.2, "epsilon_decay": 0.995, "epsilon_min": 0.05})
    epsilon = float(policy_state.get("epsilon", 0.2))
    epsilon = max(0.01, min(0.8, epsilon))

    # Exploration vs exploitation.
    if rnd.random() < epsilon:
        choices = list(STRATEGIES)
        weights = []
        for sname in choices:
            weights.append(0.2 if _is_strategy_temporarily_ineffective(state, sname) else 1.0)
        chosen = rnd.choices(choices, weights=weights, k=1)[0]
        policy = {"mode": "explore", "epsilon": epsilon}
        return chosen, policy

    eff = state.get("strategy_effectiveness", {})
    ckey = _context_key(context)
    by_ctx = (eff.get("by_context") or {}).get(ckey, {})
    overall = eff.get("overall", {})

    best_name = STRATEGIES[0]
    best_score = -1e9
    for sname in STRATEGIES:
        ctx_mean = float((by_ctx.get(sname) or {}).get("mean_delta", 0.0))
        ctx_count = int((by_ctx.get(sname) or {}).get("count", 0))
        ctx_usage = int((by_ctx.get(sname) or {}).get("usage_count", 0))
        ov_mean = float((overall.get(sname) or {}).get("mean_delta", 0.0))
        ov_count = int((overall.get(sname) or {}).get("count", 0))
        ov_usage = int((overall.get(sname) or {}).get("usage_count", 0))
        # Context-first with fallback to global.
        score = (ctx_mean * (1.0 if ctx_count > 0 else 0.0)) + (0.6 * ov_mean)
        # Light confidence bonus for data-rich strategies.
        score += 0.01 * min(20, ctx_count + ov_count)
        # Anti-dominance: penalize overused strategies.
        usage_penalty = 1.0 + (0.2 * ctx_usage) + (0.05 * ov_usage)
        score = score / usage_penalty
        if _is_strategy_temporarily_ineffective(state, sname):
            score *= 0.2
        if score > best_score:
            best_score = score
            best_name = sname
    policy = {"mode": "exploit", "epsilon": epsilon, "context_key": ckey}
    return best_name, policy


def _strategy_parameters(name: str, seed: int) -> dict[str, Any]:
    defaults = {
        "exploration_boost": {"mutation_rate": 0.8, "novelty_weight": 1.5, "seed": seed},
        "reset_priors": {"mutation_rate": 0.3, "novelty_weight": 1.0, "seed": seed},
        "noise_injection": {"mutation_rate": 0.6, "novelty_weight": 1.2, "seed": seed},
    }
    return dict(defaults.get(name, defaults["exploration_boost"]))


def _probe_alternative_strategy(
    task_path: str,
    iterations: int,
    shifted_seed: int,
    base_state: dict[str, Any],
    selected: str,
    context: dict[str, Any],
    score_before: float,
    probe_p: float = 0.35,
) -> dict[str, Any] | None:
    rnd = random.Random(shifted_seed ^ 0xA11CE)
    if rnd.random() > probe_p:
        return None

    alternatives = [s for s in STRATEGIES if s != selected]
    if not alternatives:
        return None
    alt = rnd.choice(alternatives)

    probe_state = deepcopy(base_state)
    strategy = {
        "strategy": alt,
        "reason": "regret_probe",
        "parameters": _strategy_parameters(alt, shifted_seed + 17),
    }
    _apply_strategy_pressure(probe_state, strategy)
    probe = _legacy_run_arc_reasoning(
        task_path=task_path,
        iterations=iterations,
        seed=(shifted_seed + 17) & 0xFFFFFFFF,
        state=probe_state,
    )
    score_after = float(probe.get("score", 0.0))
    return {
        "strategy": alt,
        "score_before": score_before,
        "score_after": score_after,
        "delta": score_after - score_before,
        "context": context,
    }


def _update_task_performance(state: dict[str, Any], task_id: str, score: float) -> dict[str, Any]:
    perf = state.setdefault("task_performance", {})
    rec = perf.setdefault(task_id, {"scores": [], "capability_limited": False, "reason": ""})
    scores = list(rec.get("scores", []))
    scores.append(float(score))
    scores = scores[-30:]
    rec["scores"] = scores
    cap = False
    reason = ""
    if len(scores) >= 4:
        mx = max(scores)
        mn = min(scores)
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        # Flat low-variance plateau below perfect score => likely capability ceiling.
        if mx < 0.999 and (mx - mn) <= 0.01 and var <= 1e-4:
            cap = True
            reason = "no_improvement_low_variance"
    rec["capability_limited"] = cap
    rec["reason"] = reason
    return rec


def run_arc_reasoning(task_path: str, iterations: int = 80, seed: int | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    s = state if state is not None else {}
    s.setdefault("strategy_policy", {"epsilon": 0.2, "epsilon_decay": 0.995, "epsilon_min": 0.05})
    if seed is None:
        run_seed = seed_for_task(task_path)
    else:
        run_seed = int(seed)
        set_seed(run_seed)

    # Loop detection based on belief + op-prior signature.
    task_id = os.path.basename(task_path)
    task_contexts = s.setdefault("task_context", {})
    s["arc_current_context"] = str(task_contexts.get(task_id, "unknown"))

    sig = _state_signature(s)
    seen = s.setdefault("arc_seen_state_hashes", [])
    loop_detected = sig in seen
    seen.append(sig)
    s["arc_seen_state_hashes"] = seen[-128:]

    strategy_shift: dict[str, Any] | None = None
    if loop_detected:
        strategy_shift = {
            "strategy": "escape_mode",
            "reason": "loop_detected",
            "parameters": {"mutation_rate": 1.0, "novelty_weight": 2.0, "seed": run_seed},
        }
        _apply_strategy_pressure(s, strategy_shift)
        s["arc_last_good_ops"] = []

    result = _legacy_run_arc_reasoning(task_path=task_path, iterations=iterations, seed=run_seed, state=s)
    score_before = float(result.get("score", 0.0))
    result["failure_analysis"] = classify_failure(
        task_path=task_path,
        hypothesis=result.get("hypothesis"),
        score=score_before,
    )
    task_contexts[task_id] = str(result["failure_analysis"].get("failure_type", "unknown"))

    progression = [float(x) for x in result.get("score_progression", [])]
    plateau = detect_plateau(progression, window=min(20, max(5, len(progression))))
    result["plateau"] = plateau
    result["strategy_shift_triggered"] = False
    result["strategy_shift"] = strategy_shift

    # Actionable failure pressure: bias operator priors by likely missing operation.
    likely_ops = _likely_ops_from_failure(result["failure_analysis"])
    if likely_ops:
        _boost_ops(s, likely_ops, amount=2)
    _enforce_missing_op(s, result["failure_analysis"])

    task_perf = _update_task_performance(s, task_id, score_before)
    result["task_capability"] = {
        "capability_limited": bool(task_perf.get("capability_limited", False)),
        "reason": str(task_perf.get("reason", "")),
        "recent_scores": list(task_perf.get("scores", []))[-10:],
    }

    # Record loop-escape outcome after first run.
    if strategy_shift and str(strategy_shift.get("strategy")) == "escape_mode":
        prev_score = float(s.get("arc_last_score", score_before))
        conf = float(result.get("failure_analysis", {}).get("confidence", 0.5))
        loop_outcome = {
            "strategy": "escape_mode",
            "score_before": prev_score,
            "score_after": score_before,
            "delta": score_before - prev_score,
            "weighted_delta": (score_before - prev_score) * conf,
            "task_id": task_id,
            "context": {
                "failure_type": str(result["failure_analysis"].get("failure_type", "unknown")),
                "plateau_length": _plateau_length(progression),
                "plateau_bucket": _plateau_bucket(_plateau_length(progression)),
            },
            "reason": "loop_detected",
            "shift_seed": run_seed,
            "confidence": conf,
        }
        _record_strategy_outcome(s, loop_outcome)
        result["strategy_outcome"] = loop_outcome

    # Plateau recovery: shift search seed and continue for a short extra pass.
    if (
        bool(plateau.get("plateau"))
        and iterations >= 20
        and score_before < 0.999
        and not bool(task_perf.get("capability_limited", False))
    ):
        context = {
            "failure_type": str(result["failure_analysis"].get("failure_type", "unknown")),
            "plateau_length": _plateau_length(progression),
            "plateau_bucket": _plateau_bucket(_plateau_length(progression)),
        }
        selected, policy = _select_best_strategy(s, context=context, seed=run_seed)
        strategy_shift = {
            "strategy": selected,
            "reason": "plateau_no_progress",
            "parameters": _strategy_parameters(selected, run_seed),
            "policy": policy,
        }
        probe_base_state = deepcopy(s)
        _apply_strategy_pressure(s, strategy_shift)
        shifted_seed = (run_seed + 1009) & 0xFFFFFFFF
        shifted = _legacy_run_arc_reasoning(
            task_path=task_path,
            iterations=max(10, iterations // 2),
            seed=shifted_seed,
            state=s,
        )
        score_after = float(shifted.get("score", 0.0))
        conf = float(result.get("failure_analysis", {}).get("confidence", 0.5))
        outcome = {
            "strategy": selected,
            "score_before": score_before,
            "score_after": score_after,
            "delta": score_after - score_before,
            "weighted_delta": (score_after - score_before) * conf,
            "task_id": task_id,
            "context": context,
            "reason": "plateau_no_progress",
            "shift_seed": shifted_seed,
            "confidence": conf,
        }

        # Regret probe: occasionally evaluate an alternative strategy in a shadow rollout.
        alt = _probe_alternative_strategy(
            task_path=task_path,
            iterations=max(10, iterations // 2),
            shifted_seed=shifted_seed,
            base_state=probe_base_state,
            selected=selected,
            context=context,
            score_before=score_before,
        )
        if alt is not None:
            outcome["regret"] = max(0.0, float(alt.get("delta", 0.0)) - float(outcome["delta"]))
            outcome["alternative_probe"] = alt

        _record_strategy_outcome(s, outcome)
        result["strategy_policy"] = policy
        result["strategy_outcome"] = outcome

        if score_after > score_before:
            shifted["failure_analysis"] = classify_failure(
                task_path=task_path,
                hypothesis=shifted.get("hypothesis"),
                score=score_after,
            )
            shifted["plateau"] = plateau
            shifted["strategy_shift_triggered"] = True
            shifted["strategy_shift"] = strategy_shift
            shifted["strategy_policy"] = policy
            shifted["strategy_outcome"] = outcome
            shifted["shift_seed"] = shifted_seed
            shifted["loop_detected"] = loop_detected
            s.setdefault("arc_strategy_history", []).append(shifted["strategy_shift"])
            return shifted
        result["strategy_shift_triggered"] = True
        result["strategy_shift"] = strategy_shift
        result["strategy_policy"] = policy
        result["shift_seed"] = shifted_seed

    result["loop_detected"] = loop_detected
    result["strategy_effectiveness"] = s.get("strategy_effectiveness", {})
    result["strategy_policy_state"] = s.get("strategy_policy", {})
    if strategy_shift:
        s.setdefault("arc_strategy_history", []).append(strategy_shift)
        s["arc_strategy_history"] = s["arc_strategy_history"][-64:]
    s["arc_last_score"] = float(result.get("score", 0.0))

    return result
