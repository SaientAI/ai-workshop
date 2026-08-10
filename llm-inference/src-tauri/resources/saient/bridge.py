"""Choice Lab to Saient signal bridge.

The bridge is a signal emitter, not a decision maker. It observes Choice Lab
state, emits labelled candidate proposals, and leaves final action execution to
Saient's existing controller path.
"""

from __future__ import annotations

from typing import Any

_PROPOSABLE_ACTIONS = {"explore", "optimize", "self_direct", "analyze", "stabilize"}
_BOREDOM_THRESHOLD = 0.65
_NOVELTY_LOW_WATERMARK = 0.15
_CORTISOL_VETO = 0.75
_BLOCKED_STAGES = {"commit"}
_CONFIDENCE_FLOOR = 0.30


def _float_from(primary: dict[str, Any], fallback: dict[str, Any], key: str, default: float) -> float:
    if key in primary:
        return float(primary[key])
    if key in fallback:
        return float(fallback[key])
    return default


def sample_signals(choice_state: dict[str, Any] | None) -> dict[str, float]:
    """Extract bridge signals from a Choice Lab event row or live state dict."""
    choice_state = choice_state or {}
    internal = choice_state.get("internal") or {}
    if not isinstance(internal, dict):
        internal = {}
    components = choice_state.get("components") or {}
    if not isinstance(components, dict):
        components = {}

    crossed_raw = internal.get("boredom_crossed_now", choice_state.get("boredom_crossed_now", choice_state.get("boredom_crossed", False)))
    return {
        "boredom": _float_from(internal, choice_state, "boredom", 0.0),
        "boredom_threshold": _float_from(internal, choice_state, "boredom_threshold", _BOREDOM_THRESHOLD),
        "novelty": _float_from(internal, components, "novelty", _float_from(choice_state, {}, "novelty", 0.0)),
        "energy": _float_from(internal, choice_state, "energy", 0.0),
        "satiety": _float_from(internal, choice_state, "satiety", 0.0),
        "boredom_crossed": float(bool(crossed_raw)),
    }


def propose(signals: dict[str, float], saient_state: dict[str, Any]) -> dict[str, Any] | None:
    """Return a candidate proposal when Choice Lab boredom pressure is strong."""
    boredom = float(signals.get("boredom", 0.0))
    threshold = float(signals.get("boredom_threshold", _BOREDOM_THRESHOLD))

    if boredom < threshold:
        return None
    if float(signals.get("boredom_crossed", 0.0)) < 1.0:
        return None
    if float(signals.get("energy", 0.0)) < 0.55 or float(signals.get("satiety", 0.0)) < 0.65:
        return None

    novelty = float(signals.get("novelty", 0.0))
    if novelty < _NOVELTY_LOW_WATERMARK:
        action_type = "stabilize"
        reason = "boredom_high_novelty_exhausted"
    else:
        action_type = "explore"
        reason = "boredom_high_novelty_available"

    excess = boredom - threshold
    max_excess = 1.0 - threshold if threshold < 1.0 else 1.0
    confidence = min(1.0, excess / max(1e-6, max_excess))

    return {
        "source": "exogenous_boredom",
        "action_type": action_type,
        "priority": "exogenous",
        "confidence": round(confidence, 4),
        "reason": reason,
        "raw_signals": dict(signals),
    }


def critique(proposal: dict[str, Any], saient_state: dict[str, Any]) -> dict[str, Any]:
    """Apply hard vetoes to a bridge proposal."""
    required = {"source", "action_type", "priority", "confidence", "reason", "raw_signals"}
    missing = sorted(required - set(proposal))
    if missing:
        return _reject(proposal, f"malformed_missing_{'_'.join(missing)}")

    if proposal.get("action_type") not in _PROPOSABLE_ACTIONS:
        return _reject(proposal, "action_not_proposable")
    if proposal.get("priority") != "exogenous":
        return _reject(proposal, "priority_not_exogenous")
    if proposal.get("source") != "exogenous_boredom":
        return _reject(proposal, "source_not_exogenous_boredom")

    drives = saient_state.get("drives", {})
    if float(drives.get("cortisol", 0.0)) >= _CORTISOL_VETO:
        return _reject(proposal, "cortisol_too_high")

    preempt_stage = saient_state.get("preempt_stage") or {}
    active_preempt = saient_state.get("active_preempt")
    if isinstance(active_preempt, dict) and isinstance(preempt_stage, dict):
        drive = active_preempt.get("drive")
        stage_info = preempt_stage.get(drive, {}) if drive else {}
        stage = stage_info.get("mode") if isinstance(stage_info, dict) else None
        if stage in _BLOCKED_STAGES:
            return _reject(proposal, "preempt_commit_active")

    if float(proposal.get("confidence", 0.0)) < _CONFIDENCE_FLOOR:
        return _reject(proposal, "confidence_below_floor")

    mission = saient_state.get("mission")
    if mission and not mission.get("completed", False):
        try:
            from mission import TARGET_TO_ACTION
        except ModuleNotFoundError:
            from saient.mission import TARGET_TO_ACTION
        mission_action = TARGET_TO_ACTION.get(mission.get("target", ""), "")
        if mission_action == proposal.get("action_type"):
            return _reject(proposal, "mission_already_targeting_same_action")

    return {"accepted": True, "veto_reason": None, "proposal": proposal}


def to_goal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Convert an accepted proposal into an Saient-compatible goal dict."""
    return {
        "type": proposal["action_type"],
        "priority": "exogenous",
        "score": 0.0,
        "strategy": "choice_bridge",
        "trend": 0.0,
        "preemptive": False,
        "preempt": False,
        "trend_boost": 0.0,
        "preempt_stage": "none",
        "stage": "none",
        "source": proposal["source"],
        "reason": proposal["reason"],
        "goal_id": None,
    }


def _reject(proposal: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"accepted": False, "veto_reason": reason, "proposal": proposal}
