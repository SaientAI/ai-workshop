import random
from trend import detect_trend

THRESHOLDS = {
    "information_depth": 0.40,
    "efficiency": 0.50,
    "autonomy": 0.60,
}

TARGETS = ["information_depth", "efficiency", "autonomy"]
MAPPING = {
    "information_depth": "explore",
    "efficiency": "optimize",
    "autonomy": "self_direct",
}
ACTION_TO_TARGET = {
    "explore": "information_depth",
    "optimize": "efficiency",
    "self_direct": "autonomy",
}
FAST_DROP = -0.018
MIN_COMMIT_TREND = -0.02
MIN_DELTA = 0.05
MIN_DEFICIT = 0.2
CORE_DRIVES = ("information_depth", "efficiency", "autonomy")


def _active_preempt_payload(drive: str, trend: float) -> dict:
    confidence = abs(float(trend))
    ttl = 2 if confidence < 0.02 else 5
    return {"drive": drive, "ttl": ttl, "confidence": confidence}


def _goal(payload: dict, source: str) -> dict:
    out = dict(payload)
    out["source"] = source
    return out


def persistent_drop_for(k: str, history: list, eps: float = -0.005, n: int = 3) -> bool:
    vals = [float(h.get("drives", {}).get(k, 0.0)) for h in history][- (n + 1):]
    if len(vals) < n + 1:
        return False
    return all((vals[i + 1] - vals[i]) < eps for i in range(len(vals) - 1))


def predict_delta(vals: list[float], horizon: int = 4) -> float:
    # simple linear fit over last 5 points
    if len(vals) < 5:
        return 0.0
    y = vals[-5:]
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    xm = sum(x) / 5.0
    ym = sum(y) / 5.0
    num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    den = sum((xi - xm) ** 2 for xi in x) or 1.0
    slope = num / den
    return slope * horizon


def _clamp_drives(d: dict) -> None:
    for k in CORE_DRIVES:
        d[k] = max(0.05, min(1.0, float(d.get(k, 0.5))))


def volatility(k: str, history: list) -> float:
    vals = [float(h.get("drives", {}).get(k, 0.0)) for h in history][-6:]
    if len(vals) < 6:
        return 0.0
    return sum(abs(vals[i + 1] - vals[i]) for i in range(5)) / 5.0


def simulate_commit(state: dict, target: str, steps: int = 5) -> float:
    sim = {k: float(state.get("drives", {}).get(k, 0.5)) for k in CORE_DRIVES}
    params = state.get("params", {})
    gains = params.get("gains", {})
    decay = float(params.get("decay", 0.015))
    spill = float(params.get("spill", 0.02))
    for _ in range(steps):
        for k in CORE_DRIVES:
            if k == target:
                sim[k] += max(0.08, float(gains.get(target, 0.18))) * 2.0
            else:
                sim[k] += spill
        for k in CORE_DRIVES:
            sim[k] -= decay
        _clamp_drives(sim)
    return sum(sim.values())


def simulate_wait(state: dict, steps: int = 5) -> float:
    sim = {k: float(state.get("drives", {}).get(k, 0.5)) for k in CORE_DRIVES}
    decay = float(state.get("params", {}).get("decay", 0.015))
    for _ in range(steps):
        for k in CORE_DRIVES:
            sim[k] -= decay
        _clamp_drives(sim)
    return sum(sim.values())


def generate_goal(state: dict) -> dict:
    drives = state["drives"]
    beliefs = state["beliefs"]
    history = state.get("history", [])
    subgoals = state.get("subgoals", [])
    strategy = state.get("strategy", {"mode": "balanced"})
    params = state.get("params", {})
    effect_history = params.setdefault("effect_history", {k: [] for k in CORE_DRIVES})
    trends = detect_trend(state)
    trend_gain = float(params.get("trend_gain", 1.0))
    base_thresh = -0.03
    thresh = base_thresh * (0.8 if trend_gain < 1.0 else 1.0)
    state.setdefault("mission_budget", 1.0)
    preempt_stage = state.setdefault("preempt_stage", {})
    preempt_lock = state.setdefault("preempt_lock", {"drive": None, "ttl": 0})
    post_lock = state.get("post_success_lock")
    if post_lock and int(post_lock.get("ttl", 0)) > 0 and post_lock.get("drive") in TARGETS:
        k = post_lock["drive"]
        state["post_success_lock"]["ttl"] = int(post_lock.get("ttl", 0)) - 1
        return {
            "type": MAPPING[k],
            "priority": k,
            "locked": True,
            "preempt": False,
            "preemptive": False,
            "preempt_stage": "none",
            "stage": "none",
            "trend": round(float(trends.get(k, 0.0)), 4),
            "trend_boost": 0.0,
            "predicted_delta": 0.0,
            "source": "post_success_lock",
            "strategy": strategy.get("mode", "balanced"),
            "score": 0.0,
        }
    if int(state.get("preempt_refractory", 0)) > 0:
        state["preempt_refractory"] = int(state.get("preempt_refractory", 0)) - 1
        allow_preempt = False
    else:
        allow_preempt = True
    active_preempt = state.get("active_preempt")
    if active_preempt:
        active_preempt["ttl"] = int(active_preempt.get("ttl", 0)) - 1
        if active_preempt["ttl"] <= 0:
            state["active_preempt"] = None
            active_preempt = None

    if drives.get("cortisol", 0.0) > 0.75:
        return _goal(
            {"type": "stabilize", "priority": "cortisol", "preempt": False, "preempt_stage": "none"},
            "stress_guard",
        )

    # Decay preempt stages.
    for drive, st in list(preempt_stage.items()):
        st["ttl"] = int(st.get("ttl", 0)) - 1
        if st["ttl"] <= 0:
            preempt_stage.pop(drive, None)

    # Hard preempt override when fast drop is real.
    trend_targets = {k: float(trends.get(k, 0.0)) for k in TARGETS if k in trends}
    candidates = [
        k
        for k, t in trend_targets.items()
        if t < FAST_DROP and persistent_drop_for(k, history) and volatility(k, history) <= 0.03
    ]
    wait_score = simulate_wait(state, steps=5)
    best_score = float("-inf")
    worst_drive = None
    for cand in candidates:
        commit_score = simulate_commit(state, cand, steps=5)
        hist = effect_history.get(cand, [])
        avg_effect = (sum(hist) / len(hist)) if hist else 0.0
        score = (commit_score - wait_score) + avg_effect * 0.5
        if score > best_score:
            best_score = score
            worst_drive = cand
    worst_t = float(trend_targets.get(worst_drive, 0.0)) if worst_drive else 0.0
    k = active_preempt.get("drive") if active_preempt and active_preempt.get("drive") in TARGETS else worst_drive
    k_t = float(trend_targets.get(k, 0.0)) if k else 0.0

    # Commit follow-through: active commit ignores competing drives.
    if active_preempt and active_preempt.get("drive") in TARGETS:
        ak = active_preempt["drive"]
        st = preempt_stage.get(ak, {})
        if st.get("mode") == "commit":
            return _goal(
                {
                    "type": MAPPING[ak],
                    "priority": ak,
                    "preempt": True,
                    "preemptive": True,
                    "preempt_stage": "commit",
                    "stage": "commit",
                    "trend": round(float(trend_targets.get(ak, 0.0)), 4),
                    "trend_boost": 0.0,
                    "strategy": strategy.get("mode", "balanced"),
                    "score": 0.0,
                },
                "active_preempt_followthrough",
            )

    has_deficit = bool(k and float(drives.get(k, 0.5)) < (1.0 - MIN_DEFICIT))
    is_real_drop = bool(
        k
        and allow_preempt
        and (k_t < FAST_DROP)
        and has_deficit
        and persistent_drop_for(k, history)
        and (best_score >= MIN_DELTA)
    )
    if is_real_drop:
        preempt_lock["drive"] = k
        preempt_lock["ttl"] = 2

    if allow_preempt and int(preempt_lock.get("ttl", 0)) > 0 and preempt_lock.get("drive") in TARGETS:
        k = preempt_lock["drive"]
        t = float(trend_targets.get(k, 0.0))
        stage = preempt_stage.get(k, {"mode": None, "ttl": 0})
        if stage.get("mode") != "probe":
            goal_type = "analyze"
            preempt_stage[k] = {"mode": "probe", "ttl": 2, "trigger_t": t}
            stage_mode = "probe"
            source = "gate"
            predicted_delta = 0.0
        else:
            trigger_t = float(stage.get("trigger_t", -0.01))
            if t < (trigger_t * 0.8) or int(stage.get("ttl", 0)) <= 0:
                vals = [float(h.get("drives", {}).get(k, 0.0)) for h in history]
                delta_wait = predict_delta(vals, horizon=4)
                B = 0.18 * 2.0
                delta_act = delta_wait + B
                MARGIN = 0.04
                stay_probe = False
                if delta_act < delta_wait + MARGIN:
                    stay_probe = True
                if t > MIN_COMMIT_TREND:
                    stay_probe = True
                # No-commit zone: don't commit when target drive is already high.
                if float(drives.get(k, 0.5)) > (1.0 - MIN_DEFICIT):
                    stay_probe = True

                if not stay_probe:
                    goal_type = MAPPING[k]
                    preempt_stage[k] = {"mode": "commit", "ttl": 2}
                    state["active_preempt"] = _active_preempt_payload(k, t)
                    state["preempt_refractory"] = 3
                    stage_mode = "commit"
                    source = "gate"
                    predicted_delta = max(0.0, delta_act - delta_wait)
                    recent_effects = params.get("effect_history", {}).get(k, [0.05])
                    avg_effect = (sum(recent_effects) / len(recent_effects)) if recent_effects else 0.05
                    scale = max(0.1, (0.5 + avg_effect / 0.1))
                    predicted_delta *= scale
                    predicted_delta = min(predicted_delta, 0.12)
                else:
                    # Exploration can probe, but not force low-quality commit.
                    if random.random() < 0.03 and abs(t) >= 0.025:
                        goal_type = MAPPING[k]
                        preempt_stage[k] = {"mode": "commit", "ttl": 2}
                        state["active_preempt"] = _active_preempt_payload(k, t)
                        state["preempt_refractory"] = 3
                        stage_mode = "commit"
                        source = "explore"
                        predicted_delta = max(0.0, delta_act - delta_wait)
                        recent_effects = params.get("effect_history", {}).get(k, [0.05])
                        avg_effect = (sum(recent_effects) / len(recent_effects)) if recent_effects else 0.05
                        scale = max(0.1, (0.5 + avg_effect / 0.1))
                        predicted_delta *= scale
                        predicted_delta = min(predicted_delta, 0.12)
                    else:
                        goal_type = "analyze"
                        preempt_stage[k] = {"mode": "probe", "ttl": 2, "trigger_t": trigger_t}
                        stage_mode = "probe"
                        source = "gate"
                        predicted_delta = 0.0
            else:
                goal_type = "analyze"
                preempt_stage[k] = {"mode": "probe", "ttl": max(1, int(stage.get("ttl", 0))), "trigger_t": trigger_t}
                stage_mode = "probe"
                source = "gate"
                predicted_delta = 0.0
        preempt_lock["ttl"] = int(preempt_lock.get("ttl", 0)) - 1
        return {
            "type": goal_type,
            "priority": k,
            "preempt": True,
            "preemptive": True,
            "preempt_stage": stage_mode,
            "stage": stage_mode,
            "trend": round(t, 4),
            "trend_boost": 0.0,
            "predicted_delta": round(float(predicted_delta), 4),
            "source": source,
            "strategy": strategy.get("mode", "balanced"),
            "score": 0.0,
        }

    # Mission candidate (apply as bias, not domination).
    mission_target = None
    mission_action = None
    mission_priority = "mission_subgoal"
    for sg in subgoals:
        if sg.get("done", False):
            continue
        sub_type = sg.get("type", "explore")
        mission_target = ACTION_TO_TARGET.get(sub_type, "information_depth")
        mission_action = sub_type
        low_eff = drives.get("efficiency", 0.5) < 0.30
        high_stress = drives.get("cortisol", 0.0) > 0.45
        last3 = [h.get("action", {}).get("type", "") for h in history[-3:]]
        analyzed_recently = any(a == "analyze" for a in last3)
        if beliefs.get("caution_weight", 0.5) > 0.70 and sub_type in {"explore", "self_direct"}:
            if low_eff:
                mission_target = "efficiency"
                mission_action = "optimize"
                mission_priority = "mission_recovery"
            elif high_stress and (not analyzed_recently) and random.random() < 0.22:
                mission_action = "analyze"
                mission_priority = "mission_guard"
        break

    # History-derived failure memory biases away from repeated failure.
    recent = history[-20:]
    fail_counts = {"explore": 0, "optimize": 0, "self_direct": 0}
    for h in recent:
        atype = h.get("action", {}).get("type", "")
        ok = h.get("result", {}).get("success", True)
        if atype in fail_counts and not ok:
            fail_counts[atype] += 1

    mode = strategy.get("mode", "balanced")
    if mode == "exploratory":
        noise_min, noise_max = 0.10, 0.40
        urgency_mult = 1.00
    elif mode == "aggressive":
        noise_min, noise_max = 0.00, 0.10
        urgency_mult = 1.45
    else:
        noise_min, noise_max = 0.00, 0.20
        urgency_mult = 1.00

    scores: dict[str, float] = {}
    boost_by_target: dict[str, float] = {}
    is_preempt_by_target: dict[str, bool] = {}
    for target in TARGETS:
        drive_value = float(drives.get(target, 0.5))
        t = float(trends.get(target, 0.0))
        is_preempt = t < thresh
        boost = min(0.5, abs(t) * (2.0 + trend_gain)) if is_preempt else 0.0
        preempt_bonus = 0.35 if is_preempt else 0.0
        urgency = (1.0 - drive_value) * urgency_mult + boost
        if t < 0:
            urgency += 0.07
        noise = random.uniform(noise_min, noise_max)
        conflict = sum(abs(drive_value - float(drives.get(other, 0.5))) for other in TARGETS) / len(TARGETS)
        action = MAPPING[target]
        fail_bias = min(0.45, fail_counts.get(action, 0) * 0.08)
        scores[target] = urgency + noise + conflict + preempt_bonus - fail_bias
        boost_by_target[target] = boost
        is_preempt_by_target[target] = is_preempt

    # Mission budget influence, no monopoly.
    mission_budget = float(state.get("mission_budget", 1.0))
    if mission_target in scores:
        scores[mission_target] += 0.2 * mission_budget

    # Cooldown + hysteresis.
    cooldown = state.setdefault("cooldown", {"drive": None, "ttl": 0})
    if int(cooldown.get("ttl", 0)) > 0:
        cooldown["ttl"] = int(cooldown.get("ttl", 0)) - 1
    hold_drive = cooldown.get("drive")
    if int(cooldown.get("ttl", 0)) > 0 and hold_drive in scores:
        scores[hold_drive] += 0.06

    last_target = state.get("last_target")
    if last_target in scores:
        for key in scores:
            if key != last_target:
                scores[key] -= 0.12
        scores[last_target] += 0.08

    if beliefs.get("exploration_bias", 0.5) > 0.65:
        scores["information_depth"] += 0.06

    # Anti-limit-cycle fallback.
    last_actions = [h.get("action", {}).get("type", "") for h in history[-4:]]
    if len(last_actions) == 4 and len(set(last_actions)) == 1:
        forced = {
            "explore": "efficiency",
            "optimize": "autonomy",
            "self_direct": "information_depth",
            "analyze": "information_depth",
        }.get(last_actions[-1], random.choice(TARGETS))
        state["last_target"] = forced
        state["cooldown"] = {"drive": forced, "ttl": 2}
        return _goal(
            {
                "type": MAPPING.get(forced, "explore"),
                "priority": forced,
                "score": round(scores.get(forced, 0.0), 4),
                "strategy": mode,
                "trend": round(float(trends.get(forced, 0.0)), 4),
                "preemptive": bool(is_preempt_by_target.get(forced, False)),
                "preempt": bool(is_preempt_by_target.get(forced, False)),
                "trend_boost": round(float(boost_by_target.get(forced, 0.0)), 4),
                "preempt_stage": "commit" if is_preempt_by_target.get(forced, False) else "none",
            },
            "anti_cycle_forced_switch",
        )

    target = max(scores, key=scores.get)
    state["last_target"] = target
    state["cooldown"] = {"drive": target, "ttl": 2}
    selected_is_mission = bool(mission_target == target and mission_action)

    if selected_is_mission:
        state["mission_budget"] = max(0.1, mission_budget * 0.85)
        goal_type = mission_action
        priority = mission_priority
        stage_mode = "none"
        preempt = False
    else:
        state["mission_budget"] = min(1.0, mission_budget + 0.05)
        goal_type = MAPPING.get(target, "idle")
        priority = target
        preempt = bool(is_preempt_by_target.get(target, False))
        t = float(trends.get(target, 0.0))
        if t < thresh:
            stage = preempt_stage.get(target, {"mode": None, "ttl": 0})
            if stage.get("mode") != "probe":
                goal_type = "analyze"
                preempt_stage[target] = {"mode": "probe", "ttl": 2, "trigger_t": t}
                stage_mode = "probe"
            else:
                goal_type = MAPPING.get(target, "idle")
                preempt_stage[target] = {"mode": "commit", "ttl": 2}
                stage_mode = "commit"
                state["active_preempt"] = _active_preempt_payload(target, t)
                state["cooldown"] = {"drive": target, "ttl": 3}
        else:
            stage_mode = "none"

    return _goal(
        {
            "type": goal_type,
            "priority": priority,
            "score": round(scores[target], 4),
            "strategy": mode,
            "trend": round(float(trends.get(target, 0.0)), 4),
            "preemptive": bool(is_preempt_by_target.get(target, False)),
            "preempt": preempt,
            "trend_boost": round(float(boost_by_target.get(target, 0.0)), 4),
            "preempt_stage": stage_mode,
            "stage": stage_mode,
        },
        "mission_bias" if selected_is_mission else "drive_score_selection",
    )
