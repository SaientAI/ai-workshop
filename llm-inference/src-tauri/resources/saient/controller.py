import json
import random
import re
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ALLOWED_ACTIONS = {"explore", "optimize", "self_direct", "analyze", "stabilize", "arc_attempt"}


def _rule_action(goal: Dict) -> Dict:
    return {
        "type": goal.get("type", "idle"),
        "priority": goal.get("priority", "none"),
    }


def _parse_action(text: str) -> str | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    # JSON mode support: {"action":"optimize"}
    if "{" in t and "}" in t:
        try:
            obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
            a = str(obj.get("action", "")).strip().lower()
            a = a.replace("-", "_")
            if a == "selfdirect":
                a = "self_direct"
            if a in ALLOWED_ACTIONS:
                return a
        except Exception:
            pass
    # strict first token wins
    token = re.split(r"[^a-z_]+", t, maxsplit=1)[0]
    token = token.replace("-", "_")
    if token == "selfdirect":
        token = "self_direct"
    if token in ALLOWED_ACTIONS:
        return token
    for action in ALLOWED_ACTIONS:
        if action in t:
            return action
    return None


def _model_pick_action(goal: Dict, state: Dict, model_url: str, timeout: float = 20.0) -> str | None:
    mission = state.get("mission", {})
    drives = state.get("drives", {})
    history = state.get("history", [])
    recent = [h.get("action", {}).get("type", "") for h in history[-8:]]
    prompt = (
        "Choose exactly ONE action.\n"
        "Return STRICT JSON: {\"action\":\"explore|optimize|self_direct|analyze|stabilize\"}\n"
        f"Proposed goal action: {goal.get('type')}\n"
        f"Goal priority: {goal.get('priority')}\n"
        f"Mission target: {mission.get('target','none')}\n"
        f"Mission progress: {mission.get('progress',0.0):.3f}\n"
        f"Drives: info={drives.get('information_depth',0.0):.3f}, "
        f"eff={drives.get('efficiency',0.0):.3f}, auto={drives.get('autonomy',0.0):.3f}, "
        f"cort={drives.get('cortisol',0.0):.3f}\n"
        f"Recent actions: {recent}\n"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You are an action selector. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 24,
        "temperature": 0.0,
    }
    req = Request(
        model_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    text = ""
    try:
        text = obj["choices"][0]["message"]["content"] or ""
    except Exception:
        text = ""
    if not text:
        try:
            text = obj["choices"][0]["message"].get("reasoning_content", "")
        except Exception:
            text = ""
    return _parse_action(text)


def decide_action(goal: Dict, state: Dict, policy_mode: str = "rule", model_url: str = "") -> Dict:
    if policy_mode == "arc_reasoning":
        return {
            "type": "arc_attempt",
            "priority": goal.get("priority", "arc"),
            "hypothesis": goal.get("hypothesis"),
            "score": goal.get("score"),
        }
    if policy_mode != "model":
        return _rule_action(goal)

    goal_type = goal.get("type", "idle")
    goal_priority = goal.get("priority", "none")
    picked = _model_pick_action(goal, state, model_url=model_url) if model_url else None
    if picked is None:
        fallback = _rule_action(goal)
        fallback["fallback"] = True
        fallback["policy_mode"] = "rule_fallback"
        return fallback

    # Mission subgoals are commitments: do not allow model drift on active subgoal steps.
    if goal_priority == "mission_subgoal" and goal_type in ALLOWED_ACTIONS and picked != goal_type:
        return {
            "type": goal_type,
            "priority": goal_priority,
            "policy_mode": "model_constrained",
            "model_pick": picked,
            "constrained": True,
        }

    return {
        "type": picked,
        "priority": goal_priority,
        "policy_mode": "model",
    }


def execute(action: Dict, state: Dict) -> Dict:
    action_type = action.get("type", "idle")
    if action_type == "arc_attempt":
        return {
            "type": action_type,
            "success": True,
            "score": float(action.get("score", 0.0)),
            "hypothesis": action.get("hypothesis"),
        }
    drives = state.get("drives", {})
    efficiency = float(drives.get("efficiency", 0.5))
    info = float(drives.get("information_depth", 0.5))
    autonomy = float(drives.get("autonomy", 0.5))

    is_preempt_commit = bool(action.get("preempt")) and str(action.get("stage", "")) == "commit"

    if action_type == "self_direct" and state.get("drives", {}).get("efficiency", 1.0) < 0.30:
        return {
            "type": action_type,
            "success": False,
            "failure_reason": "over_aggression",
            "preempt": bool(action.get("preempt", False)),
            "stage": action.get("stage", "none"),
        }

    if action_type == "stabilize":
        return {
            "type": action_type,
            "success": True,
            "preempt": bool(action.get("preempt", False)),
            "stage": action.get("stage", "none"),
        }

    # Stochastic environment pressure: force occasional failure so adaptation is exercised.
    base_fail = 0.18
    if action_type == "explore":
        fail_p = base_fail + max(0.0, 0.45 - efficiency) * 0.35
        if random.random() < fail_p:
            return {
                "type": action_type,
                "success": False,
                "failure_reason": "resource_overstretch",
                "preempt": bool(action.get("preempt", False)),
                "stage": action.get("stage", "none"),
            }
    elif action_type == "optimize":
        fail_p = base_fail - 0.03 + max(0.0, 0.35 - info) * 0.30
        if random.random() < fail_p:
            return {
                "type": action_type,
                "success": False,
                "failure_reason": "underinformed_plan",
                "recovery": True,
                "preempt": bool(action.get("preempt", False)),
                "stage": action.get("stage", "none"),
            }
    elif action_type == "self_direct":
        fail_p = base_fail + max(0.0, 0.40 - efficiency) * 0.45 + max(0.0, autonomy - 0.9) * 0.20
        if random.random() < fail_p:
            return {
                "type": action_type,
                "success": False,
                "failure_reason": "over_aggression",
                "preempt": bool(action.get("preempt", False)),
                "stage": action.get("stage", "none"),
            }
    elif action_type == "analyze":
        fail_p = base_fail - 0.08 + max(0.0, autonomy - 0.85) * 0.10
        if random.random() < fail_p:
            return {
                "type": action_type,
                "success": False,
                "failure_reason": "analysis_paralysis",
                "preempt": bool(action.get("preempt", False)),
                "stage": action.get("stage", "none"),
            }
    result = {
        "type": action_type,
        "success": True,
        "preempt": bool(action.get("preempt", False)),
        "stage": action.get("stage", "none"),
    }
    if action_type == "optimize":
        result["recovery"] = True
    if is_preempt_commit:
        t = abs(float(action.get("trend", 0.0)))
        if t > 0.025:
            strength = 2.4
        else:
            strength = 1.8
        result["recovery"] = True
        result["target"] = action.get("priority")
        result["strength"] = strength
    return result
