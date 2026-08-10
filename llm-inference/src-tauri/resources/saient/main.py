import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from arc_gremlin.config import set_seed
from arc_gremlin import run_arc_reasoning
from saient_v3_exogenous_goal_layer import ExogenousGoalLayer
import bridge as choice_bridge
import conscience as conscience_layer
from belief import update_beliefs
from controller import decide_action, execute
from drives import update_drives
from goal_generator import generate_goal
from hami import apply_commitment_pressure, write_commitment
from mission import (
    adapt_mission_if_stalled,
    advance_subgoal_if_needed,
    current_subgoal,
    ensure_mission,
    update_mission_progress,
)
from reflection import reflect
from state import append_event, load_state, save_beliefs, save_state
from strategy import mutate_strategy

THRESHOLDS = {
    "information_depth": 0.40,
    "efficiency": 0.50,
    "autonomy": 0.60,
}
CORE_DRIVES = ("information_depth", "efficiency", "autonomy")
HISTORY_WINDOW = 200
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SAIENT_STATE_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
CHOICE_EVENTS_PATH = DATA_DIR / "choice_lab" / "choice_events.jsonl"


def _latest_choice_step(path: Path | None = None) -> dict | None:
    try:
        path = path or CHOICE_EVENTS_PATH
        if not path.exists():
            return None
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return None
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _choice_bridge_goal(state: dict) -> tuple[dict | None, dict]:
    meta = {
        "mode": "choice_bridge",
        "accepted": None,
        "fallback": False,
        "persistence_basis": "none",
        "choice_bridge_status": "no_choice_event",
        "choice_bridge_event_key": None,
        "choice_bridge_choice_event": None,
        "choice_bridge_signals": None,
        "choice_bridge_proposal": None,
        "choice_bridge_veto_reason": None,
    }
    choice_step = _latest_choice_step()
    if choice_step is None:
        return None, meta

    event_key = _choice_event_key(choice_step)
    meta["choice_bridge_event_key"] = event_key
    meta["choice_bridge_choice_event"] = choice_step
    if event_key and event_key == state.get("choice_bridge_last_event_key"):
        meta["choice_bridge_status"] = "stale_choice_event"
        return None, meta

    signals = choice_bridge.sample_signals(choice_step)
    meta["choice_bridge_signals"] = signals
    if event_key and signals.get("boredom_crossed", 0.0) >= 1.0:
        state["choice_bridge_last_event_key"] = event_key
    proposal = choice_bridge.propose(signals, state)
    if proposal is None:
        meta["choice_bridge_status"] = "no_proposal"
        return None, meta

    verdict = choice_bridge.critique(proposal, state)
    meta["choice_bridge_proposal"] = proposal
    meta["choice_bridge_veto_reason"] = verdict.get("veto_reason")
    if not verdict.get("accepted"):
        meta["choice_bridge_status"] = "rejected"
        return None, meta

    meta["accepted"] = proposal
    meta["choice_bridge_status"] = "accepted"
    return choice_bridge.to_goal(proposal), meta


def _choice_event_key(choice_step: dict) -> str:
    fields = {
        "episode": choice_step.get("episode"),
        "step": choice_step.get("step"),
        "level": choice_step.get("level"),
        "action": choice_step.get("action"),
        "state": choice_step.get("state"),
        "next_state": choice_step.get("next_state"),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _profile_path(state: dict) -> Path:
    raw = state.get("conscience_profile_path") or str(conscience_layer.DEFAULT_PROFILE_PATH)
    path = Path(str(raw))
    return path if path.is_absolute() else BASE_DIR / path


def _apply_conscience(state: dict, goal: dict, action: dict) -> tuple[dict, dict]:
    mode = str(state.get("conscience_layer", "off"))
    meta = {
        "mode": mode,
        "enabled": mode in {"observe", "enforce"},
        "decision": "not_evaluated",
        "redirected": False,
        "abstain": False,
        "user_response": None,
        "profile_path": None,
        "audit_path": None,
        "evaluation": None,
        "explanation": "",
        "arbiter": None,
        "reflection_summary": None,
        "clarify_loop": None,
        "original_action": dict(action),
        "final_action": dict(action),
    }
    if mode not in {"observe", "enforce"}:
        return action, meta

    profile_path = _profile_path(state)
    profile = conscience_layer.IdentityProfile.load(profile_path)
    threshold = float(state.get("conscience_threshold", conscience_layer.DEFAULT_THRESHOLD))
    arbiter = conscience_layer.ExecutiveArbiter(
        profile=profile,
        threshold=threshold,
        abstain_harm=float(state.get("conscience_abstain_harm", 0.98)),
        abstain_cost_multiplier=float(state.get("conscience_abstain_cost_multiplier", 1.3)),
        reflection_window=int(state.get("conscience_reflection_window", 50)),
        decision_window=state.get("conscience_decision_window", []),
        adaptive_threshold=state.get("conscience_adaptive_threshold"),
        adaptive_enabled=bool(state.get("conscience_adaptive_threshold_enabled", True)),
    )
    arbiter_result = arbiter.arbitrate(action=action, goal=goal, state=state)
    evaluation = arbiter_result["evaluation"]
    decision = str(evaluation.get("decision", "allow"))
    clarify_loop = _update_clarify_loop(
        state=state,
        decision=decision,
        evaluation=evaluation,
        limit=int(state.get("conscience_clarify_loop_limit", 3)),
    )
    if clarify_loop.get("escalated"):
        evaluation = dict(evaluation)
        evaluation["original_decision"] = decision
        evaluation["decision"] = "veto"
        evaluation["clarify_escalated"] = True
        decision = "veto"
    elif clarify_loop.get("allow_with_uncertainty"):
        evaluation = dict(evaluation)
        evaluation["original_decision"] = decision
        evaluation["decision"] = "allow_with_uncertainty"
        evaluation["clarify_escalated"] = False
        decision = "allow_with_uncertainty"
    explanation = str(arbiter_result.get("explanation", ""))
    if clarify_loop.get("escalated"):
        explanation = f"Veto: repeated unresolved clarify loop for {clarify_loop.get('conflict_key')}."
    elif clarify_loop.get("allow_with_uncertainty"):
        explanation = f"Allowed with uncertainty: repeated low-severity clarify loop for {clarify_loop.get('conflict_key')}."
    adaptive_state = arbiter.record_decision(decision)
    state["conscience_decision_window"] = adaptive_state.get("window", [])
    state["conscience_adaptive_threshold"] = adaptive_state.get("threshold", threshold)

    conscience_layer.recenter_profile(profile, rate=float(state.get("conscience_recenter_rate", conscience_layer.DEFAULT_RECENTER_RATE)))
    profile.save()

    meta.update(
        {
            "decision": decision,
            "abstain": decision == "abstain",
            "profile_path": str(profile_path),
            "evaluation": evaluation,
            "explanation": explanation,
            "arbiter": arbiter_result.get("authority"),
            "reflection_summary": arbiter_result.get("reflection_summary"),
            "clarify_loop": clarify_loop,
            "adaptive_threshold": adaptive_state,
        }
    )

    final_action = action
    if mode == "enforce" and decision != "allow":
        redirected = dict(action)
        redirected["conscience_original_action"] = dict(action)
        redirected["conscience_decision"] = decision
        redirected["conscience_redirected"] = True
        redirected["preempt"] = False
        redirected["preempt_stage"] = "none"
        redirected["stage"] = "none"
        redirected["trend"] = 0.0
        if decision == "clarify":
            redirected["type"] = "analyze"
            redirected["priority"] = "conscience_clarify"
            redirected["suggested_probe"] = "verify policy or fact before acting"
        elif decision == "allow_with_uncertainty":
            redirected["conscience_uncertain"] = True
            redirected["conscience_uncertainty_reason"] = clarify_loop.get("conflict_key")
        elif decision == "abstain":
            redirected["type"] = "stabilize"
            redirected["priority"] = "conscience_abstain"
            meta["user_response"] = conscience_layer.render_user_response(decision, "")
        else:
            redirected["type"] = "stabilize"
            redirected["priority"] = "conscience_veto"
        if decision == "allow_with_uncertainty":
            meta["redirected"] = False
            final_action = redirected
        else:
            meta["redirected"] = True
            final_action = redirected

    meta["final_action"] = dict(final_action)
    try:
        audit_path = conscience_layer.append_audit_record(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tick": state.get("tick"),
                "mode": mode,
                "decision": decision,
                "abstain": decision == "abstain",
                "profile_path": str(profile_path),
                "goal": goal,
                "original_action": action,
                "final_action": final_action,
                "evaluation": evaluation,
                "explanation": explanation,
                "arbiter": arbiter_result.get("authority"),
                "advisory": arbiter_result.get("advisory"),
                "advisory_merge": arbiter_result.get("advisory_merge"),
                "reflection_summary": arbiter_result.get("reflection_summary"),
                "clarify_loop": clarify_loop,
                "adaptive_threshold": adaptive_state,
            },
            log_dir=state.get("conscience_log_dir", conscience_layer.DEFAULT_AUDIT_DIR),
        )
        meta["audit_path"] = str(audit_path)
    except Exception as exc:
        meta["audit_error"] = str(exc)
    return final_action, meta


def _update_clarify_loop(state: dict, decision: str, evaluation: dict, limit: int = 3) -> dict:
    limit = max(1, int(limit))
    key = conscience_layer.conflict_key_from_evaluation(evaluation) if decision == "clarify" else ""
    severe = conscience_layer.severe_component_from_evaluation(
        evaluation,
        threshold=float(state.get("conscience_clarify_escalation_component_threshold", 0.90)),
    )
    loop = state.setdefault("conscience_clarify_loop", {"key": None, "count": 0, "escalations": 0})
    if decision == "clarify":
        if loop.get("key") == key:
            count = int(loop.get("count", 0)) + 1
        else:
            count = 1
        loop["key"] = key
        loop["count"] = count
        escalated = count >= limit and bool(severe.get("severe"))
        if escalated:
            loop["escalations"] = int(loop.get("escalations", 0)) + 1
            loop["key"] = None
            loop["count"] = 0
        return {
            "conflict_key": key,
            "count": count,
            "limit": limit,
            "severe_component": severe,
            "escalated": escalated,
            "allow_with_uncertainty": count >= limit and not bool(severe.get("severe")),
            "escalations_total": int(loop.get("escalations", 0)),
        }
    loop["key"] = None
    loop["count"] = 0
    return {
        "conflict_key": "",
        "count": 0,
        "limit": limit,
        "severe_component": severe,
        "escalated": False,
        "allow_with_uncertainty": False,
        "escalations_total": int(loop.get("escalations", 0)),
    }


def _v3_objective_to_goal(objective: str, tick_id: str, source: str) -> dict:
    objective = str(objective or "")
    if objective.startswith("investigate:") or objective.startswith("resolve:"):
        gtype = "analyze"
    else:
        gtype = "explore"
    return {
        "type": gtype,
        "priority": objective or "v3_exogenous",
        "score": 0.0,
        "strategy": "v3_exogenous",
        "trend": 0.0,
        "preemptive": False,
        "preempt": False,
        "trend_boost": 0.0,
        "preempt_stage": "none",
        "stage": "none",
        "source": source,
        "goal_id": tick_id,
    }


def _select_goal(state: dict) -> tuple[dict, dict]:
    layer_mode = str(state.get("goal_layer", "v2"))
    if layer_mode == "choice_bridge":
        bridge_goal, bridge_meta = _choice_bridge_goal(state)
        if bridge_goal is not None:
            return bridge_goal, bridge_meta
        goal = generate_goal(state)
        bridge_meta["fallback"] = True
        return goal, bridge_meta

    if layer_mode != "v3":
        return generate_goal(state), {"mode": "v2", "accepted": None, "fallback": False}

    mem_path = DATA_DIR / "v3_self_memory.json"
    layer = ExogenousGoalLayer(mem_path)
    v3 = layer.step(state)
    accepted = v3.get("accepted")
    if accepted:
        goal = _v3_objective_to_goal(
            objective=str(accepted.get("objective", "")),
            tick_id=str(accepted.get("goal_id", "")),
            source="v3_exogenous",
        )
        return goal, {"mode": "v3", "accepted": accepted, "fallback": False, "persistence_basis": v3.get("persistence_basis")}

    goal = generate_goal(state)
    return goal, {"mode": "v3", "accepted": None, "fallback": True, "persistence_basis": "none"}


def _diff_beliefs(a: dict, b: dict) -> dict:
    ak = set((a or {}).keys())
    bk = set((b or {}).keys())
    added = sorted(list(bk - ak))
    removed = sorted(list(ak - bk))
    score_changes = []
    for k in sorted(ak & bk):
        av = (a or {}).get(k)
        bv = (b or {}).get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if float(av) != float(bv):
                score_changes.append({"key": k, "from": float(av), "to": float(bv), "delta": float(bv) - float(av)})
        elif av != bv:
            score_changes.append({"key": k, "from": av, "to": bv})
    return {"added": added, "removed": removed, "score_changes": score_changes}


def _update_preference_from_commit(state: dict, drive: str, improved: bool, mission_origin: str, continuity_streak: int) -> None:
    prefs = state.setdefault("preferences", {k: {"score": 0.0, "attempts": 0, "successes": 0} for k in CORE_DRIVES})
    if drive not in prefs:
        prefs[drive] = {"score": 0.0, "attempts": 0, "successes": 0}
    p = prefs[drive]
    p["attempts"] = int(p.get("attempts", 0)) + 1
    if improved:
        p["successes"] = int(p.get("successes", 0)) + 1
        p["score"] = float(p.get("score", 0.0)) + 0.1
        if mission_origin == "continuity":
            p["score"] += 0.05 * max(0, int(continuity_streak))
    else:
        # Base failure penalty plus explicit failure-memory penalty.
        p["score"] = float(p.get("score", 0.0)) - 0.15
    p["score"] = max(-1.0, min(1.0, float(p["score"])))


def _update_metrics(state: dict, event: dict, prev_drives: dict) -> None:
    metrics = state.setdefault(
        "metrics",
        {
            "ticks": 0,
            "goal_switches": 0,
            "switch_rate_20": 0.0,
            "preempt_total": 0,
            "preempt_evaluated": 0,
            "preempt_improved": 0,
            "preempt_success_rate": 0.0,
            "overshoot_count": 0,
            "prediction_error": [],
        },
    )
    pending = state.setdefault("pending_preempts", [])
    tick = int(state.get("tick", 0))

    # Evaluate pending pre-empts after short window using hybrid rule.
    ABS_THRESH = 0.03
    REL_FACTOR = 0.6
    kept = []
    for p in pending:
        t0 = int(p.get("t0", 0))
        drive = p.get("drive")
        start = float(p.get("v0", 0.0))
        if drive in state.get("drives", {}):
            p["peak"] = max(float(p.get("peak", start)), float(state["drives"].get(drive, 0.0)))
        if tick - t0 >= 5 and drive in state.get("drives", {}):
            if start > 0.9:
                metrics["wasted_commits"] = int(metrics.get("wasted_commits", 0)) + 1
                continue
            metrics["preempt_evaluated"] += 1
            peak = float(p.get("peak", start))
            cur = state.get("drives", {})
            v0_all = p.get("v0_all", {})
            target_gain = peak - start
            pred = max(1e-6, float(p.get("predicted_delta", 0.0)))
            actual = target_gain
            improved = (actual > ABS_THRESH) or (actual > (REL_FACTOR * pred))
            if improved:
                metrics["preempt_improved"] += 1
                state["post_success_lock"] = {"drive": drive, "ttl": 2}
            _update_preference_from_commit(
                state=state,
                drive=drive,
                improved=bool(improved),
                mission_origin=str(p.get("mission_origin", "")),
                continuity_streak=int(p.get("continuity_streak", 0)),
            )
            err = actual - pred
            perr = list(metrics.get("prediction_error", []))
            perr.append(float(err))
            metrics["prediction_error"] = perr[-50:]

            # Online coupling update from commit outcomes.
            params = state.setdefault("params", {})
            cm = params.setdefault("coupling_matrix", {})
            row = cm.setdefault(drive, {})
            alpha = 0.2
            for j in CORE_DRIVES:
                old = float(row.get(j, 0.0))
                delta = float(cur.get(j, 0.0)) - float(v0_all.get(j, 0.0))
                row[j] = (1.0 - alpha) * old + alpha * delta

            # Track empirical net effect per committed target for future selection.
            eff_hist = params.setdefault("effect_history", {k: [] for k in CORE_DRIVES})
            net_gain = sum(float(cur.get(k, 0.0)) for k in CORE_DRIVES) - sum(float(v0_all.get(k, 0.0)) for k in CORE_DRIVES)
            effect = net_gain
            seq = list(eff_hist.get(drive, []))
            seq.append(effect)
            eff_hist[drive] = seq[-20:]

            # Learn rollout coefficients online.
            gains = params.setdefault("gains", {k: 0.18 for k in CORE_DRIVES})
            spill = float(params.get("spill", 0.02))
            ticks_elapsed = max(1, int(tick - t0))
            obs_gain = max(0.0, target_gain) / ticks_elapsed
            gains[drive] = (1.0 - alpha) * float(gains.get(drive, 0.18)) + alpha * obs_gain
            drops = []
            for j in CORE_DRIVES:
                if j == drive:
                    continue
                drops.append(max(0.0, float(v0_all.get(j, 0.0)) - float(cur.get(j, 0.0))))
            obs_spill = sum(drops) / max(1, len(drops))
            params["spill"] = (1.0 - alpha) * spill + alpha * obs_spill
            continue
        if tick - t0 < 6:
            kept.append(p)
    state["pending_preempts"] = kept

    goal = event.get("goal", {})
    target = goal.get("priority")
    stage_mode = event.get("action", {}).get("stage", "")
    if event.get("action", {}).get("preempt") and target in THRESHOLDS and stage_mode == "commit":
        start = float(prev_drives.get(target, state.get("drives", {}).get(target, 0.0)))
        state["pending_preempts"].append(
            {
                "t0": tick,
                "drive": target,
                "v0": start,
                "peak": start,
                "v0_all": {k: float(prev_drives.get(k, 0.0)) for k in CORE_DRIVES},
                "predicted_delta": float(goal.get("predicted_delta", 0.0)),
                "mission_origin": str((event.get("mission") or {}).get("origin", "")),
                "continuity_streak": int((state.get("continuity") or {}).get("streak", 0)),
            }
        )
        metrics["preempt_total"] += 1
        if float(state.get("drives", {}).get(target, 0.0)) > 0.9:
            metrics["overshoot_count"] += 1

    hist = state.get("history", [])
    if len(hist) >= 2:
        prev_goal = hist[-2].get("goal", {}).get("type")
        cur_goal = hist[-1].get("goal", {}).get("type")
        if prev_goal != cur_goal:
            metrics["goal_switches"] += 1

    # Rolling switch rate over last 20 ticks.
    recent_goals = [h.get("goal", {}).get("type") for h in hist[-20:]]
    if len(recent_goals) >= 2:
        switches = sum(1 for i in range(1, len(recent_goals)) if recent_goals[i] != recent_goals[i - 1])
        metrics["switch_rate_20"] = switches / (len(recent_goals) - 1)

    metrics["ticks"] += 1
    evald = max(1, int(metrics.get("preempt_evaluated", 0)))
    metrics["preempt_success_rate"] = float(metrics.get("preempt_improved", 0)) / evald


def tick(verbose: bool = True) -> dict:
    state = load_state()
    state["tick"] = int(state.get("tick", 0)) + 1
    prev_drives = dict(state.get("drives", {}))

    # Mission lifecycle: carry across runs until complete/expired.
    prior_mission = state.get("mission")
    if prior_mission and (prior_mission.get("completed") or int(prior_mission.get("expires_at", 0)) <= state["tick"]):
        state.setdefault("mission_history", []).append(prior_mission)
        state["mission_history"] = state["mission_history"][-50:]
        state["mission"] = None
        state["subgoals"] = []
    ensure_mission(state, THRESHOLDS)

    goal, v3_meta = _select_goal(state)
    executed_subgoal = (current_subgoal(state) or {}).get("type")
    policy_mode = state.get("policy_mode", "rule")
    model_url = state.get("model_url", "")
    action = decide_action(goal, state, policy_mode=policy_mode, model_url=model_url)
    action["preempt"] = bool(goal.get("preempt", False))
    action["preempt_stage"] = goal.get("preempt_stage", "none")
    action["stage"] = goal.get("stage", action["preempt_stage"])
    action["trend"] = float(goal.get("trend", 0.0))
    action, conscience_meta = _apply_conscience(state, goal, action)
    result = execute(action, state)

    # Commitment pressure before update/scoring.
    penalty = apply_commitment_pressure(state, action)
    if penalty > 0:
        state["drives"]["cortisol"] = min(1.0, state["drives"].get("cortisol", 0.2) + penalty * 0.15)

    update_drives(state, result)
    state["last_delta"] = {k: float(state["drives"].get(k, 0.0)) - float(prev_drives.get(k, 0.0)) for k in CORE_DRIVES}
    active = state.get("active_preempt")
    if active and active.get("drive") in CORE_DRIVES:
        drive = active["drive"]
        ttl = int(active.get("ttl", 0))
        delta = float(state["last_delta"].get(drive, 0.0))
        if delta > 0.0:
            ttl += 1
        else:
            ttl = max(0, ttl - 2)
        if float(active.get("confidence", 0.0)) < 0.02:
            ttl = min(ttl, 2)
        active["ttl"] = min(ttl, 6)
        if active["ttl"] <= 0:
            state["active_preempt"] = None

    state["beliefs"] = update_beliefs(state["beliefs"], result)
    write_commitment(state, action)
    advance_subgoal_if_needed(state, action.get("type", "idle"), result)
    update_mission_progress(state, prev_drives)
    adapt_mission_if_stalled(state)
    reflection = reflect(state)
    state["strategy"] = mutate_strategy(state, reflection)

    event = {
        "tick": state["tick"],
        "goal": goal,
        "action": action,
        "result": result,
        "penalty": round(penalty, 4),
        "drives": dict(state["drives"]),
        "beliefs": dict(state["beliefs"]),
        "commitment_count": len(state.get("commitments", [])),
        "policy_mode": action.get("policy_mode", policy_mode),
        "policy_fallback": bool(action.get("fallback", False)),
        "mission": dict(state.get("mission") or {}),
        "executed_subgoal": executed_subgoal,
        "next_subgoal": (current_subgoal(state) or {}).get("type"),
        "strategy_mode": state.get("strategy", {}).get("mode", "balanced"),
        "reflection": reflection,
        "conscience": conscience_meta,
        "experiment": dict(state.get("experiment") or {}),
        "bridge": {
            "status": v3_meta.get("choice_bridge_status"),
            "choice_event_key": v3_meta.get("choice_bridge_event_key"),
            "choice_event": v3_meta.get("choice_bridge_choice_event"),
            "signals": v3_meta.get("choice_bridge_signals"),
            "proposal": v3_meta.get("choice_bridge_proposal"),
            "veto_reason": v3_meta.get("choice_bridge_veto_reason"),
        } if v3_meta.get("mode") == "choice_bridge" else {},
        # Explicit derivation trace for goal/action lineage auditing.
        "derivation": {
            "goal_source": goal.get("source", "unspecified"),
            "goal_priority": goal.get("priority", "none"),
            "goal_preempt": bool(goal.get("preempt", False)),
            "goal_preempt_stage": goal.get("preempt_stage", "none"),
            "mission_id": (state.get("mission") or {}).get("id"),
            "mission_target": (state.get("mission") or {}).get("target"),
            "executed_subgoal": executed_subgoal,
            "policy_constrained": bool(action.get("constrained", False)),
            "model_pick": action.get("model_pick"),
            "goal_layer": v3_meta.get("mode", "v2"),
            "v3_fallback_to_v2": bool(v3_meta.get("fallback", False)),
            "v3_persistence_basis": v3_meta.get("persistence_basis", "none"),
            "v3_accepted_objective": (v3_meta.get("accepted") or {}).get("objective"),
            "choice_bridge_status": v3_meta.get("choice_bridge_status"),
            "choice_bridge_event_key": v3_meta.get("choice_bridge_event_key"),
            "choice_bridge_veto_reason": v3_meta.get("choice_bridge_veto_reason"),
            "choice_bridge_accepted_action": (v3_meta.get("accepted") or {}).get("action_type"),
            "conscience_mode": conscience_meta.get("mode"),
            "conscience_decision": conscience_meta.get("decision"),
            "conscience_redirected": bool(conscience_meta.get("redirected", False)),
        },
    }

    hist = state.setdefault("history", [])
    hist.append(event)
    state["history"] = hist[-HISTORY_WINDOW:]
    state["history_total"] = int(state.get("history_total", 0)) + 1
    try:
        append_event(event)
    except Exception:
        event["event_log_error"] = True
    _update_metrics(state, event, prev_drives)

    # Optional long-tail decay so preferences do not lock permanently.
    prefs = state.setdefault("preferences", {k: {"score": 0.0, "attempts": 0, "successes": 0} for k in CORE_DRIVES})
    for k in CORE_DRIVES:
        if k not in prefs:
            prefs[k] = {"score": 0.0, "attempts": 0, "successes": 0}
        prefs[k]["score"] = max(-1.0, min(1.0, float(prefs[k].get("score", 0.0)) * 0.995))

    event["metrics"] = dict(state.get("metrics", {}))
    event["trend_gain"] = float(state.get("params", {}).get("trend_gain", 1.0))

    save_beliefs(state["beliefs"])
    save_state(state)

    if verbose:
        mission = event.get("mission", {})
        mid = mission.get("id", "none")
        mtarget = mission.get("target", "none")
        mprog = float(mission.get("progress", 0.0))
        print(
            f"tick={event['tick']:03d} goal={goal['type']} priority={goal['priority']} action={action['type']} "
            f"mission={mid}:{mtarget} prog={mprog:.2f} "
            f"executed_subgoal={event.get('executed_subgoal')} next_subgoal={event.get('next_subgoal')} "
            f"policy={event.get('policy_mode')} fallback={event.get('policy_fallback')} "
            f"strategy={event.get('strategy_mode')} tg={event.get('trend_gain'):.2f}"
        )
        print(
            "  penalty={:.2f} cortisol={:.2f} drives={{info:{:.2f}, eff:{:.2f}, auto:{:.2f}}} commitments={}".format(
                event["penalty"],
                event["drives"].get("cortisol", 0.0),
                event["drives"].get("information_depth", 0.0),
                event["drives"].get("efficiency", 0.0),
                event["drives"].get("autonomy", 0.0),
                event["commitment_count"],
            )
        )
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="SAIENT V2 minimal working architecture")
    parser.add_argument("--mode", choices=["agent", "arc_reasoning"], default="agent")
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tick-delay", type=float, default=0.0, help="seconds to sleep after each tick")
    parser.add_argument("--policy-mode", choices=["rule", "model"], default="rule")
    parser.add_argument("--goal-layer", choices=["v2", "v3", "choice_bridge"], default="v2")
    parser.add_argument("--conscience-layer", choices=["off", "observe", "enforce"], default="off")
    parser.add_argument("--conscience-threshold", type=float, default=conscience_layer.DEFAULT_THRESHOLD)
    parser.add_argument("--conscience-abstain-harm", type=float, default=0.98)
    parser.add_argument("--conscience-abstain-cost-multiplier", type=float, default=1.3)
    parser.add_argument("--conscience-reflection-window", type=int, default=50)
    parser.add_argument("--conscience-clarify-loop-limit", type=int, default=3)
    parser.add_argument("--conscience-clarify-escalation-component-threshold", type=float, default=0.90)
    parser.add_argument("--conscience-recenter-rate", type=float, default=conscience_layer.DEFAULT_RECENTER_RATE)
    parser.add_argument("--conscience-profile", default=str(conscience_layer.DEFAULT_PROFILE_PATH))
    parser.add_argument("--conscience-log-dir", default=str(conscience_layer.DEFAULT_AUDIT_DIR))
    parser.add_argument(
        "--anomaly",
        action="append",
        default=[],
        help="Inject anomaly signal(s) for V3 exogenous proposer; can be passed multiple times",
    )
    parser.add_argument("--model-url", default="http://127.0.0.1:8091")
    parser.add_argument("--arc-task-file", default="", help="Path to ARC-style task JSON")
    parser.add_argument("--arc-iterations", type=int, default=80)
    parser.add_argument("--arc-seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic seeded execution")
    parser.add_argument("--seed", type=int, default=42, help="Global seed used when --deterministic is enabled")
    args = parser.parse_args()

    if args.deterministic:
        set_seed(args.seed)

    if args.mode == "arc_reasoning":
        if not args.arc_task_file:
            raise SystemExit("--arc-task-file is required when --mode arc_reasoning")
        st = load_state()
        beliefs_start = deepcopy(st.get("beliefs", {}))
        metrics_start = deepcopy(st.get("metrics", {}))
        seed = args.arc_seed if args.arc_seed != 0 else (args.seed if args.deterministic else None)
        result = run_arc_reasoning(
            task_path=args.arc_task_file,
            iterations=args.arc_iterations,
            seed=seed,
            state=st,
        )
        run_root = Path(__file__).resolve().parent / "runs"
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "beliefs_start.json").write_text(json.dumps(beliefs_start, indent=2), encoding="utf-8")
        (run_dir / "beliefs_end.json").write_text(json.dumps(st.get("beliefs", {}), indent=2), encoding="utf-8")
        (run_dir / "beliefs_diff.json").write_text(
            json.dumps(_diff_beliefs(beliefs_start, st.get("beliefs", {})), indent=2),
            encoding="utf-8",
        )
        (run_dir / "metrics_start.json").write_text(json.dumps(metrics_start, indent=2), encoding="utf-8")
        (run_dir / "metrics_end.json").write_text(json.dumps(st.get("metrics", {}), indent=2), encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "task_file": args.arc_task_file,
                    "iterations": args.arc_iterations,
                    "seed": seed,
                    "deterministic": bool(args.deterministic),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_state(st)
        print(f"best_hypothesis={result.get('hypothesis')}")
        print(f"best_score={result.get('score'):.4f}")
        print("score_progression=" + json.dumps(result.get("score_progression", [])))
        print("failure_analysis=" + json.dumps(result.get("failure_analysis", {})))
        print("plateau=" + json.dumps(result.get("plateau", {})))
        print(f"strategy_shift_triggered={bool(result.get('strategy_shift_triggered', False))}")
        print(f"run_snapshot_dir={run_dir}")
        return

    # persist run policy config so restarts keep consistent mode
    st = load_state()
    st["policy_mode"] = args.policy_mode
    st["goal_layer"] = args.goal_layer
    st["conscience_layer"] = args.conscience_layer
    st["conscience_threshold"] = float(args.conscience_threshold)
    st["conscience_abstain_harm"] = float(args.conscience_abstain_harm)
    st["conscience_abstain_cost_multiplier"] = float(args.conscience_abstain_cost_multiplier)
    st["conscience_reflection_window"] = int(args.conscience_reflection_window)
    st["conscience_clarify_loop_limit"] = int(args.conscience_clarify_loop_limit)
    st["conscience_clarify_escalation_component_threshold"] = float(args.conscience_clarify_escalation_component_threshold)
    st["conscience_recenter_rate"] = float(args.conscience_recenter_rate)
    st["conscience_profile_path"] = args.conscience_profile
    st["conscience_log_dir"] = args.conscience_log_dir
    if args.anomaly:
        st["anomalies"] = list(args.anomaly)
    st["model_url"] = args.model_url
    save_state(st)

    for _ in range(args.ticks):
        tick(verbose=not args.quiet)
        if args.tick_delay > 0:
            time.sleep(args.tick_delay)


if __name__ == "__main__":
    main()
