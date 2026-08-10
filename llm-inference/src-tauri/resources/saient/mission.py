from __future__ import annotations

import random
import uuid
from typing import Dict, Any


MISSION_TARGETS = ["information_depth", "efficiency", "autonomy"]
TARGET_TO_ACTION = {
    "information_depth": "explore",
    "efficiency": "optimize",
    "autonomy": "self_direct",
}


def _dominant_deficit(drives: Dict[str, float], thresholds: Dict[str, float]) -> str:
    deficits = {
        k: max(0.0, float(thresholds.get(k, 0.5)) - float(drives.get(k, 0.5)))
        for k in MISSION_TARGETS
    }
    # if no real deficits, pick the lowest drive to keep evolution pressure
    if all(v <= 0 for v in deficits.values()):
        return min(MISSION_TARGETS, key=lambda k: float(drives.get(k, 0.5)))
    return max(deficits, key=deficits.get)


def _build_subgoals(target: str) -> list[dict]:
    primary = TARGET_TO_ACTION.get(target, "explore")
    if target == "efficiency":
        sequence = [primary, "analyze", "optimize", "explore", "optimize", "self_direct"]
    elif target == "information_depth":
        sequence = [primary, "analyze", "explore", "optimize", "explore", "self_direct"]
    else:
        sequence = [primary, "analyze", "self_direct", "optimize", "self_direct", "explore"]
    return [{"type": s, "done": False} for s in sequence]


def _choose_continuity_target(state: Dict[str, Any]) -> str:
    continuity = state.setdefault("continuity", {"target": None, "ttl": 0, "streak": 0})
    current = continuity.get("target")
    ttl = int(continuity.get("ttl", 0))
    if current in MISSION_TARGETS and ttl > 0:
        continuity["ttl"] = ttl - 1
        return current

    drives = state.get("drives", {})
    preferences = state.get("preferences", {})
    identity = state.setdefault("identity", {"dominant": None, "stability": 0})

    def novelty_score(k: str) -> float:
        attempts = int(preferences.get(k, {}).get("attempts", 0))
        total = sum(int(preferences.get(t, {}).get("attempts", 0)) for t in MISSION_TARGETS) + 1
        return 1.0 - (attempts / total)

    def confidence_score(k: str) -> float:
        attempts = int(preferences.get(k, {}).get("attempts", 0))
        successes = int(preferences.get(k, {}).get("successes", 0))
        success_rate = successes / max(1, attempts)
        return min(1.0, success_rate)

    scored: Dict[str, float] = {}
    for target in MISSION_TARGETS:
        base = float(drives.get(target, 0.5))
        pref = float(preferences.get(target, {}).get("score", 0.0))
        novelty = novelty_score(target)
        confidence = confidence_score(target)
        identity_bonus = 0.25 if target == identity.get("dominant") else 0.0
        scored[target] = base + (0.3 * pref) + (0.2 * novelty) + (0.2 * confidence) + identity_bonus

    best = max(scored.values()) if scored else 0.0
    candidates = [k for k, v in scored.items() if abs(v - best) < 1e-9] or list(MISSION_TARGETS)
    target = random.choice(candidates)

    # Identity stabilization and adaptation.
    if target == identity.get("dominant"):
        identity["stability"] = int(identity.get("stability", 0)) + 1
    else:
        identity["stability"] = int(identity.get("stability", 0)) - 1
    identity["stability"] = max(-10, min(10, int(identity["stability"])))
    if identity.get("dominant") not in MISSION_TARGETS:
        identity["dominant"] = target
        identity["stability"] = 0
    elif identity["stability"] < -5:
        identity["dominant"] = target
        identity["stability"] = 0

    continuity["target"] = target
    continuity["ttl"] = random.randint(2, 4)
    continuity["streak"] = int(continuity.get("streak", 0)) + 1
    return target


def ensure_mission(state: Dict[str, Any], thresholds: Dict[str, float]) -> None:
    mission = state.get("mission")
    tick = int(state.get("tick", 0))
    if mission and not mission.get("completed", False) and int(mission.get("expires_at", tick + 1)) > tick:
        return

    drives = state.get("drives", {})
    deficits = {
        k: max(0.0, float(thresholds.get(k, 0.5)) - float(drives.get(k, 0.5)))
        for k in MISSION_TARGETS
    }
    has_real_deficit = any(v > 0.01 for v in deficits.values())
    if has_real_deficit:
        target = max(deficits, key=deficits.get)
        origin = "deficit"
    else:
        target = _choose_continuity_target(state)
        origin = "continuity"

    start_value = float(state.get("drives", {}).get(target, 0.5))
    # Relative target prevents immediate completion when drives are already high.
    delta_target = 0.16 if start_value < 0.6 else 0.10
    target_value = min(0.96, start_value + delta_target)
    horizon = 30
    state["mission"] = {
        "id": f"m-{uuid.uuid4().hex[:8]}",
        "target": target,
        "intent": f"raise_{target}",
        "start_tick": tick,
        "expires_at": tick + horizon,
        "completed": False,
        "progress": 0.0,
        "stall_ticks": 0,
        "last_progress": 0.0,
        "start_value": start_value,
        "target_value": target_value,
        "min_ticks": 8,
        "origin": origin,
    }
    state["subgoals"] = _build_subgoals(target)


def current_subgoal(state: Dict[str, Any]) -> dict | None:
    for sg in state.get("subgoals", []):
        if not sg.get("done", False):
            return sg
    return None


def advance_subgoal_if_needed(state: Dict[str, Any], action_type: str, result: Dict[str, Any]) -> None:
    if not result.get("success", False):
        return
    sg = current_subgoal(state)
    if not sg:
        return
    if sg.get("type") == action_type:
        sg["done"] = True


def update_mission_progress(state: Dict[str, Any], prev_drives: Dict[str, float]) -> None:
    mission = state.get("mission")
    if not mission:
        return
    target = mission.get("target")
    if target not in MISSION_TARGETS:
        return
    now = float(state.get("drives", {}).get(target, 0.0))
    before = float(prev_drives.get(target, now))
    start_value = float(mission.get("start_value", before))
    target_value = float(mission.get("target_value", min(0.96, start_value + 0.12)))
    mission["start_value"] = start_value
    mission["target_value"] = target_value

    delta = max(0.0, now - before)
    mission["progress"] = max(0.0, min(1.0, float(mission.get("progress", 0.0)) + delta))
    if mission["progress"] <= float(mission.get("last_progress", 0.0)) + 1e-6:
        mission["stall_ticks"] = int(mission.get("stall_ticks", 0)) + 1
    else:
        mission["stall_ticks"] = 0
    mission["last_progress"] = mission["progress"]

    # completion conditions: reached healthy level or all subgoals done
    all_done = all(sg.get("done", False) for sg in state.get("subgoals", [])) if state.get("subgoals") else False
    tick = int(state.get("tick", 0))
    start_tick = int(mission.get("start_tick", tick))
    min_ticks = int(mission.get("min_ticks", 0))
    duration_ok = (tick - start_tick) >= min_ticks
    relative_complete = now >= target_value and mission["progress"] >= 0.08
    subgoal_complete = all_done and mission["progress"] >= 0.05
    if duration_ok and (relative_complete or subgoal_complete):
        mission["completed"] = True


def adapt_mission_if_stalled(state: Dict[str, Any]) -> None:
    mission = state.get("mission")
    if not mission or mission.get("completed"):
        return
    if int(mission.get("stall_ticks", 0)) < 10:
        return

    current_target = mission.get("target", "efficiency")
    retarget = "efficiency" if current_target != "efficiency" else "information_depth"
    mission["target"] = retarget
    mission["intent"] = f"raise_{retarget}"
    mission["stall_ticks"] = 0
    mission["last_progress"] = 0.0
    mission["progress"] = max(0.0, float(mission.get("progress", 0.0)) * 0.6)
    state["subgoals"] = _build_subgoals(retarget)
