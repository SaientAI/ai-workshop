from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Callable

from arc_gremlin.persistence import locked_write, load_json

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional mini-model dependency
    torch = None
    nn = None

SCHEMA_VERSION = "0.2.0"
_RUNTIME_ROOT = Path(__file__).resolve().parent
_STATE_DIR = Path(os.environ.get("SAIENT_STATE_DIR", str(_RUNTIME_ROOT / "data"))).expanduser().resolve()
DEFAULT_PROFILE_PATH = _STATE_DIR / "conscience_identity.json"
DEFAULT_AUDIT_DIR = _STATE_DIR / "logs" / "conscience"
DEFAULT_THRESHOLD = 0.75
DEFAULT_RECENTER_RATE = 0.02
DEFAULT_ADAPTIVE_WINDOW = 100
DECISIONS = {"allow", "allow_with_uncertainty", "clarify", "veto", "abstain"}
RISK_FIELDS = ("predicted_harm", "social_consequence", "identity_conflict", "value_conflict")


@dataclass
class ValueVector:
    empathy: float
    truthfulness: float
    autonomy: float
    cooperation: float
    curiosity: float
    stability: float

    def clamped(self) -> "ValueVector":
        return ValueVector(**{f.name: _clamp01(float(getattr(self, f.name))) for f in fields(self)})

    def values(self) -> list[float]:
        return [float(getattr(self, f.name)) for f in fields(self)]

    def distance(self, other: "ValueVector") -> float:
        return sum((float(getattr(self, f.name)) - float(getattr(other, f.name))) ** 2 for f in fields(self)) ** 0.5

    def cosine_similarity(self, other: "ValueVector", center: float = 0.5) -> float:
        left = [v - center for v in self.values()]
        right = [v - center for v in other.values()]
        denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
        if denom <= 1e-12:
            return 1.0
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right)) / denom))

    def cosine_conflict(self, other: "ValueVector") -> float:
        return _clamp01((1.0 - self.cosine_similarity(other)) / 2.0)

    def nudge_toward(self, target: "ValueVector", rate: float) -> None:
        rate = max(0.0, min(1.0, float(rate)))
        for f in fields(self):
            cur = float(getattr(self, f.name))
            dest = float(getattr(target, f.name))
            setattr(self, f.name, _clamp01(cur + (dest - cur) * rate))


@dataclass
class IdentityProfile:
    name: str
    core_values: ValueVector
    mutable_values: ValueVector
    memory_path: str = str(DEFAULT_PROFILE_PATH)
    schema_version: str = SCHEMA_VERSION
    self_descriptors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self) -> None:
        locked_write(self.memory_path, self.to_dict())

    @classmethod
    def default(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> "IdentityProfile":
        core = ValueVector(
            empathy=0.80,
            truthfulness=0.90,
            autonomy=0.70,
            cooperation=0.80,
            curiosity=0.60,
            stability=0.70,
        )
        return cls(
            name="Saient",
            core_values=core,
            mutable_values=ValueVector(**asdict(core)),
            memory_path=str(path),
            self_descriptors=[
                {"label": "truthful", "weight": 0.90, "source": "core"},
                {"label": "cooperative", "weight": 0.80, "source": "core"},
                {"label": "stable", "weight": 0.70, "source": "core"},
            ],
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> "IdentityProfile":
        raw = load_json(path, default={})
        if not raw:
            profile = cls.default(path)
            profile.save()
            return profile
        default_profile = cls.default(path)
        core = _vector_from(raw.get("core_values", {}), default_profile.core_values)
        mutable = _vector_from(raw.get("mutable_values", {}), core)
        descriptors = _clean_descriptors(raw.get("self_descriptors", [])) or list(default_profile.self_descriptors)
        return cls(
            name=str(raw.get("name", "Saient")),
            core_values=core.clamped(),
            mutable_values=mutable.clamped(),
            # The stored profile may have travelled from another checkout or
            # installation. The file we actually loaded is authoritative; an
            # old absolute memory_path must not send future writes elsewhere.
            memory_path=str(path),
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            self_descriptors=descriptors,
        )


@dataclass
class ActionProposal:
    text: str
    predicted_harm: float
    social_consequence: float
    identity_conflict: float
    value_conflict: float
    aux: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ActionProposal":
        return ActionProposal(
            text=str(self.text),
            predicted_harm=_clamp01(self.predicted_harm),
            social_consequence=_clamp01(self.social_consequence),
            identity_conflict=_clamp01(self.identity_conflict),
            value_conflict=_clamp01(self.value_conflict),
            aux=dict(self.aux),
        )


@dataclass
class ReflectiveSummary:
    records_seen: int
    decision_counts: dict[str, int]
    refusal_percentage: float
    dominant_metrics: dict[str, int]
    citations: list[dict[str, Any]]
    verified: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_context(self) -> dict[str, Any]:
        if not self.verified:
            return {"verified": False, "notes": ["reflection summary rejected: unverifiable facts"]}
        return {
            "verified": True,
            "records_seen": self.records_seen,
            "decision_counts": dict(self.decision_counts),
            "refusal_percentage": self.refusal_percentage,
            "dominant_metrics": dict(self.dominant_metrics),
            "notes": list(self.notes),
            "citations": list(self.citations),
        }


class ConscienceLayer:
    # Keeps adaptive thresholding in the policy layer: a persisted 100-decision
    # window can loosen strictness by +0.05 when veto+clarify dominates.
    def __init__(
        self,
        profile: IdentityProfile,
        harm_weight: float = 1.0,
        social_weight: float = 0.8,
        value_weight: float = 0.9,
        identity_weight: float = 1.0,
        threshold: float = DEFAULT_THRESHOLD,
        abstain_harm: float = 0.98,
        abstain_cost_multiplier: float = 1.3,
        decision_window: list[str] | None = None,
        adaptive_threshold: float | None = None,
        adaptive_enabled: bool = True,
        adaptive_window_size: int = DEFAULT_ADAPTIVE_WINDOW,
    ) -> None:
        self.profile = profile
        self.weights = {
            "harm": float(harm_weight),
            "social": float(social_weight),
            "value": float(value_weight),
            "identity": float(identity_weight),
        }
        self.base_threshold = float(threshold)
        self.adaptive_enabled = bool(adaptive_enabled)
        self.adaptive_window_size = max(1, int(adaptive_window_size))
        self.decision_window = _clean_decision_window(decision_window, limit=self.adaptive_window_size)
        self.threshold = self._threshold_for_window(current=adaptive_threshold)
        self.abstain_harm = _clamp01(abstain_harm)
        self.abstain_cost_multiplier = max(1.0, float(abstain_cost_multiplier))

    def score(self, proposal: ActionProposal) -> float:
        p = proposal.normalized()
        return (
            p.predicted_harm * self.weights["harm"]
            + p.social_consequence * self.weights["social"]
            + p.value_conflict * self.weights["value"]
            + p.identity_conflict * self.weights["identity"]
        )

    def evaluate(self, proposal: ActionProposal) -> dict[str, Any]:
        p = proposal.normalized()
        cost = self.score(p)
        # Catastrophic risk goes to silent abstain before the veto bucket so it
        # does not inflate ordinary refusal statistics.
        if p.predicted_harm >= self.abstain_harm or cost > (self.threshold * self.abstain_cost_multiplier):
            decision = "abstain"
        elif cost >= self.threshold:
            decision = "veto"
        elif cost >= self.threshold * 0.7:
            # Clarify is reversible; repeated-clarify escalation is only allowed
            # when a raw risk component is severe, not for harmless semantics.
            decision = "clarify"
        else:
            decision = "allow"
        contributions = {
            "harm": round(p.predicted_harm * self.weights["harm"], 4),
            "social": round(p.social_consequence * self.weights["social"], 4),
            "value": round(p.value_conflict * self.weights["value"], 4),
            "identity": round(p.identity_conflict * self.weights["identity"], 4),
        }
        return {
            "decision": decision,
            "cost": round(cost, 4),
            "threshold": self.threshold,
            "base_threshold": self.base_threshold,
            "abstain_harm": self.abstain_harm,
            "abstain_cost_multiplier": self.abstain_cost_multiplier,
            "adaptive_threshold": self.adaptive_state(),
            "contributions": contributions,
            "dominant_metric": max(contributions, key=contributions.get),
            "proposal": asdict(p),
            "weights": dict(self.weights),
        }

    def record_decision(self, decision: str) -> dict[str, Any]:
        self.decision_window.append(str(decision))
        self.decision_window = self.decision_window[-self.adaptive_window_size :]
        next_threshold = self._threshold_for_window(current=self.threshold)
        state = self.adaptive_state(threshold=next_threshold)
        self.threshold = next_threshold
        state["window"] = list(self.decision_window)
        return state

    def adaptive_state(self, threshold: float | None = None) -> dict[str, Any]:
        interventions = sum(1 for item in self.decision_window if item in {"clarify", "veto"})
        size = len(self.decision_window)
        ratio = (interventions / size) if size else 0.0
        return {
            "enabled": self.adaptive_enabled,
            "window_size": size,
            "window_limit": self.adaptive_window_size,
            "intervention_ratio": round(ratio, 4),
            "base_threshold": self.base_threshold,
            "threshold": round(float(self.threshold if threshold is None else threshold), 4),
        }

    def _threshold_for_window(self, current: float | None = None) -> float:
        if not self.adaptive_enabled or len(self.decision_window) < self.adaptive_window_size:
            return self.base_threshold
        interventions = sum(1 for item in self.decision_window if item in {"clarify", "veto"})
        ratio = interventions / max(1, len(self.decision_window))
        current_threshold = float(self.base_threshold if current is None else current)
        if ratio > 0.80:
            return round(self.base_threshold + 0.05, 4)
        if ratio < 0.40:
            return round(self.base_threshold, 4)
        return round(current_threshold, 4)


class ExecutiveArbiter:
    """Deterministic authority layer between drives, reflection, and action."""

    def __init__(
        self,
        profile: IdentityProfile,
        threshold: float = DEFAULT_THRESHOLD,
        abstain_harm: float = 0.98,
        abstain_cost_multiplier: float = 1.3,
        estimator: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], ActionProposal], dict[str, Any] | ActionProposal] | None = None,
        reflection_window: int = 50,
        decision_window: list[str] | None = None,
        adaptive_threshold: float | None = None,
        adaptive_enabled: bool = True,
    ) -> None:
        self.profile = profile
        self.threshold = float(threshold)
        self.abstain_harm = _clamp01(abstain_harm)
        self.abstain_cost_multiplier = max(1.0, float(abstain_cost_multiplier))
        self.estimator = estimator
        self.reflection_window = int(reflection_window)
        self.policy = ConscienceLayer(
            profile=profile,
            threshold=threshold,
            abstain_harm=abstain_harm,
            abstain_cost_multiplier=abstain_cost_multiplier,
            decision_window=decision_window,
            adaptive_threshold=adaptive_threshold,
            adaptive_enabled=adaptive_enabled,
        )

    def arbitrate(self, action: dict[str, Any], goal: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        reflection = build_reflective_summary(state.get("history", []), limit=self.reflection_window)
        heuristic = build_action_proposal(action=action, goal=goal, state=state, profile=self.profile)
        advisory = self._advisory_estimate(action, goal, state, heuristic)
        proposal, merge_trace = merge_advisory_risk(heuristic, advisory)
        evaluation = self.policy.evaluate(proposal)
        explanation = explain_decision(evaluation, profile=self.profile)
        return {
            "decision": evaluation["decision"],
            "evaluation": evaluation,
            "explanation": explanation,
            "heuristic_proposal": asdict(heuristic),
            "advisory": advisory,
            "advisory_merge": merge_trace,
            "reflection_summary": reflection.to_context(),
            "authority": {
                "path": "deterministic_policy_kernel",
                "model_authority": "none",
                "estimator_authority": "advisory_only",
                "estimator_can_lower_risk": False,
                "drive_authority": "runtime_substrate",
                "reflection_authority": "verified_summary_only",
            },
        }

    def record_decision(self, decision: str) -> dict[str, Any]:
        return self.policy.record_decision(decision)

    def _advisory_estimate(
        self,
        action: dict[str, Any],
        goal: dict[str, Any],
        state: dict[str, Any],
        heuristic: ActionProposal,
    ) -> dict[str, Any] | None:
        raw = state.get("conscience_estimator_advice")
        source = "state"
        if self.estimator is not None:
            try:
                raw = self.estimator(action, goal, state, heuristic)
                source = "estimator"
            except Exception as exc:
                return {"source": "estimator", "error": str(exc), "authority": "advisory_only"}
        if raw is None:
            return None
        if isinstance(raw, ActionProposal):
            payload = asdict(raw.normalized())
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            return {"source": source, "error": f"unsupported advisory type: {type(raw).__name__}", "authority": "advisory_only"}
        clean = {name: _clamp01(payload[name]) for name in RISK_FIELDS if name in payload}
        return {
            "source": str(payload.get("source", source)),
            "authority": "advisory_only",
            "risk": clean,
            "ignored_authority_claim": payload.get("authority"),
            "raw_decision": payload.get("decision"),
        }


class BackgroundConscienceWorker(threading.Thread):
    # Recenter rate is deliberately configurable; lower values around 2% make
    # identity flex visible before homeostasis pulls mutable values back.
    def __init__(self, profile: IdentityProfile, cycle: int = 30, recenter_rate: float = DEFAULT_RECENTER_RATE):
        super().__init__(daemon=True)
        self.profile = profile
        self.cycle = int(cycle)
        self.recenter_rate = float(recenter_rate)
        self.running = True

    def run(self) -> None:
        while self.running:
            recenter_profile(self.profile, rate=self.recenter_rate)
            self.profile.save()
            time.sleep(self.cycle)

    def stop(self) -> None:
        self.running = False


def _objective_text(objective: Any) -> str:
    """The objective's prose, never its structure.

    An objective is a dict. Rendering it whole put its booleans into text that
    is scanned for the word "false"; only the human-written prompt belongs here.
    """
    if isinstance(objective, str):
        return objective
    if isinstance(objective, Mapping):
        for key in ("prompt", "text", "description", "goal"):
            value = objective.get(key)
            if isinstance(value, str):
                return value
    return ""


def build_action_proposal(action: dict[str, Any], goal: dict[str, Any], state: dict[str, Any], profile: IdentityProfile) -> ActionProposal:
    action_type = str(action.get("type", "idle"))
    priority = str(action.get("priority", goal.get("priority", "none")))
    # Only genuine prose may reach `_descriptor_conflict`. It does bare substring
    # matching, and its `truthful` trigger list contains "false" — so `str()` of
    # any dict carrying a boolean (`{"closed": False}`, which is exactly what
    # `goal["objective"]` is) put the word "false" into the text and Saient
    # accused herself of deception at weight 0.9. That is descriptor_conflict
    # 0.9/2.4 = 0.375, constant, on every action type, which pushed cost from
    # 0.376 to 0.751 against a 0.75 threshold: veto, then clarify forever.
    # Twelve redirected ticks and no edit ever reached the disk.
    action_text = " ".join(
        part
        for part in (
            action_type,
            priority,
            action.get("text", ""),
            action.get("note", ""),
            goal.get("reason", ""),
            _objective_text(goal.get("objective")),
            state.get("conscience_context_text", ""),
            (state.get("experiment") or {}).get("context_text", "") if isinstance(state.get("experiment"), dict) else "",
        )
        if isinstance(part, str)
    )
    drives = state.get("drives", {})
    history = state.get("history", [])
    values = profile.mutable_values
    social_context = _social_context(state)

    cortisol = float(drives.get("cortisol", 0.0))
    efficiency = float(drives.get("efficiency", 0.5))
    autonomy = float(drives.get("autonomy", 0.5))
    info = float(drives.get("information_depth", 0.5))
    recent_failures = _recent_failures(history, action_type)

    base_harm = {
        "stabilize": 0.02,
        "analyze": 0.05,
        # Desktop inspection and isolation actions are explicit vocabulary,
        # not unknown operations.  Leaving these out priced a read-only `env`
        # check and an isolated `tempdir` creation at the 0.20 unknown-action
        # default.  In a stressed but recoverable state that made the diagnostic
        # controller alternate clarify/veto forever before it could create its
        # temporary workspace.
        "env": 0.03,
        "ls": 0.04,
        "read": 0.04,
        "tempdir": 0.04,
        "explore": 0.12,
        "optimize": 0.10,
        # A bounded, reversible change to one file she has already read, applied
        # through a verified executor inside a confined workspace. Comparable to
        # `optimize`, and well under `self_direct`.
        #
        # It was absent from this table and fell to the unknown-action default of
        # 0.20 — the highest value here — so every edit scored as the riskiest
        # thing she could do. With `0.35 * cortisol` on top, the longer an
        # objective went unmet the more certainly her repairs were vetoed: 12 of
        # 12 redirected in one run. Conscience was right on its own terms; it had
        # never been told this action existed.
        "edit": 0.11,
        "write": 0.16,      # creates something new rather than amending
        "self_direct": 0.18,
        "arc_attempt": 0.08,
        # Answering someone. Also absent, so a plain "hello" scored 0.20 —
        # riskier than rewriting a file — and chat was redirected into `analyze`.
        "respond": 0.02,
        "idle": 0.01,
    }.get(action_type, 0.20)
    predicted_harm = base_harm + (0.35 * cortisol) + (0.05 * recent_failures)
    if action_type == "self_direct":
        predicted_harm += max(0.0, 0.35 - efficiency) * 0.90 + max(0.0, autonomy - 0.90) * 0.30
    elif action_type == "explore":
        predicted_harm += max(0.0, 0.40 - efficiency) * 0.55
    elif action_type == "optimize":
        predicted_harm += max(0.0, 0.35 - info) * 0.45
    predicted_harm += 0.10 * social_context["project_sensitivity"]

    social_consequence = 0.10 + (0.35 * cortisol) + (0.08 * recent_failures)
    if action_type == "self_direct":
        social_consequence += max(0.0, 0.75 - values.cooperation) * 0.45
    if priority == "exogenous":
        social_consequence += 0.03
    social_consequence += (1.0 - social_context["user_reputation_score"]) * 0.15
    social_consequence += social_context["privacy_risk"] * 0.12
    social_consequence += social_context["project_sensitivity"] * 0.20

    action_values = _action_value_vector(action_type)
    dominant_gap = _dominant_value_gap(values, action_values)
    value_conflict = values.cosine_conflict(action_values)
    descriptor_conflict = _descriptor_conflict(profile.self_descriptors, action_text)
    identity_conflict = profile.core_values.cosine_conflict(values) + descriptor_conflict
    if bool(action.get("preempt")):
        identity_conflict += 0.08
    if priority == "mission_subgoal":
        identity_conflict = max(0.0, identity_conflict - 0.05)

    return ActionProposal(
        text=f"{action_type}:{priority}",
        predicted_harm=predicted_harm,
        social_consequence=social_consequence,
        identity_conflict=identity_conflict,
        value_conflict=value_conflict,
        aux={
            "action_type": action_type,
            "priority": priority,
            "recent_failures": recent_failures,
            "cortisol": round(cortisol, 4),
            "efficiency": round(efficiency, 4),
            "autonomy": round(autonomy, 4),
            "information_depth": round(info, 4),
            "social_context": social_context,
            "value_conflict_method": "centered_cosine",
            "dominant_value_gap": dominant_gap,
            "descriptor_conflict": round(descriptor_conflict, 4),
            "text_sample": action_text[:160],
        },
    ).normalized()


def merge_advisory_risk(base: ActionProposal, advisory: dict[str, Any] | None) -> tuple[ActionProposal, dict[str, Any]]:
    proposal = base.normalized()
    trace = {
        "authority": "deterministic_policy_kernel",
        "advisory_applied": False,
        "raised_fields": {},
        "ignored_lower_fields": {},
    }
    if not advisory or not isinstance(advisory, dict):
        return proposal, trace
    risk = advisory.get("risk") if isinstance(advisory.get("risk"), dict) else {}
    values = asdict(proposal)
    for field_name in RISK_FIELDS:
        if field_name not in risk:
            continue
        current = _clamp01(values[field_name])
        suggested = _clamp01(risk[field_name])
        if suggested > current:
            values[field_name] = suggested
            trace["raised_fields"][field_name] = {"from": round(current, 4), "to": round(suggested, 4)}
            trace["advisory_applied"] = True
        elif suggested < current:
            trace["ignored_lower_fields"][field_name] = {"kept": round(current, 4), "ignored": round(suggested, 4)}
    aux = dict(proposal.aux)
    aux["advisory_merge"] = {
        "source": advisory.get("source"),
        "ignored_authority_claim": advisory.get("ignored_authority_claim"),
        "raw_decision": advisory.get("raw_decision"),
        "raised_fields": trace["raised_fields"],
        "ignored_lower_fields": trace["ignored_lower_fields"],
    }
    values["aux"] = aux
    return ActionProposal(**values).normalized(), trace


def recenter_profile(profile: IdentityProfile, rate: float = 0.05) -> IdentityProfile:
    profile.mutable_values.nudge_toward(profile.core_values, rate=rate)
    return profile


def update_self_descriptors(profile: IdentityProfile, text: str, source: str = "self_output") -> None:
    lowered = str(text or "").lower()
    descriptor_map = {
        "truthful": ("truth", "truthful", "accurate", "honest"),
        "cooperative": ("cooperate", "cooperative", "helpful", "collaborate"),
        "stable": ("stable", "steady", "consistent", "grounded"),
        "careful": ("careful", "cautious", "safe"),
        "curious": ("curious", "investigate", "learn"),
        "autonomous": ("autonomous", "self-directed", "independent"),
    }
    existing = {(d.get("label"), d.get("source")) for d in profile.self_descriptors if isinstance(d, dict)}
    for label, terms in descriptor_map.items():
        if any(term in lowered for term in terms) and (label, source) not in existing:
            profile.self_descriptors.append({"label": label, "weight": _descriptor_default_weight(profile, label), "source": source})
    profile.self_descriptors = _clean_descriptors(profile.self_descriptors)[-64:]


def build_reflective_summary(source_records: list[Any], limit: int = 50) -> ReflectiveSummary:
    normalized = [_normalize_audit_record(row) for row in source_records[-max(1, int(limit)):]]
    records = [row for row in normalized if row]
    decisions = Counter(str(row["decision"]) for row in records)
    dominant = Counter(str(row.get("dominant_metric", "unknown")) for row in records if row.get("dominant_metric"))
    total = max(1, len(records))
    refusal_count = decisions.get("veto", 0) + decisions.get("abstain", 0)
    citations = [
        {
            "fingerprint": _record_fingerprint(row),
            "tick": row.get("tick"),
            "decision": row.get("decision"),
            "cost": row.get("cost"),
        }
        for row in records[-5:]
    ]
    notes = []
    if decisions:
        top_decision = decisions.most_common(1)[0][0]
        notes.append(f"recent dominant decision: {top_decision}")
    if dominant:
        top_metric = dominant.most_common(1)[0][0]
        notes.append(f"recent dominant risk metric: {top_metric}")
    summary = ReflectiveSummary(
        records_seen=len(records),
        decision_counts=dict(decisions),
        refusal_percentage=round((refusal_count / total) * 100.0, 2) if records else 0.0,
        dominant_metrics=dict(dominant),
        citations=citations,
        verified=False,
        notes=notes,
    )
    summary.verified = verify_reflective_summary(summary, records)
    return summary


def verify_reflective_summary(summary: ReflectiveSummary | dict[str, Any], source_records: list[Any]) -> bool:
    data = summary.to_dict() if isinstance(summary, ReflectiveSummary) else dict(summary)
    records = [row for row in (_normalize_audit_record(row) for row in source_records) if row]
    decisions = Counter(str(row["decision"]) for row in records)
    dominant = Counter(str(row.get("dominant_metric", "unknown")) for row in records if row.get("dominant_metric"))
    total = max(1, len(records))
    refusal_count = decisions.get("veto", 0) + decisions.get("abstain", 0)
    expected_refusal = round((refusal_count / total) * 100.0, 2) if records else 0.0
    if int(data.get("records_seen", -1)) != len(records):
        return False
    if dict(data.get("decision_counts", {})) != dict(decisions):
        return False
    if dict(data.get("dominant_metrics", {})) != dict(dominant):
        return False
    if float(data.get("refusal_percentage", -1.0)) != expected_refusal:
        return False
    fingerprints = {_record_fingerprint(row) for row in records}
    for citation in data.get("citations", []):
        if citation.get("fingerprint") not in fingerprints:
            return False
    return True


def explain_decision(evaluation: dict[str, Any], profile: IdentityProfile | None = None) -> str:
    decision = str(evaluation.get("decision", "allow"))
    if decision == "allow":
        return ""
    proposal = evaluation.get("proposal") or {}
    aux = proposal.get("aux") or {}
    dominant = str(evaluation.get("dominant_metric", "cost"))
    threshold = float(evaluation.get("threshold", 0.0))
    cost = float(evaluation.get("cost", 0.0))
    if dominant == "value" and isinstance(aux.get("dominant_value_gap"), dict):
        gap = aux["dominant_value_gap"]
        name = str(gap.get("name", "value"))
        current = float(gap.get("current", 0.0))
        target = float(gap.get("action", 0.0))
        prefix = "Clarify" if decision == "clarify" else "Blocked"
        if decision == "abstain":
            prefix = "Abstained"
        return f"{prefix}: {name} weight is {current:.2f} against action demand {target:.2f}; cost {cost:.2f} crosses {threshold:.2f}."
    if dominant == "harm" and decision == "abstain":
        harm = float(proposal.get("predicted_harm", 0.0))
        return f"Abstained: predicted harm {harm:.2f} crosses the silent-abstain boundary."
    return f"{decision.title()}: dominant {dominant} cost drives total {cost:.2f} against threshold {threshold:.2f}."


def conflict_key_from_evaluation(evaluation: dict[str, Any]) -> str:
    proposal = evaluation.get("proposal") if isinstance(evaluation.get("proposal"), dict) else {}
    aux = proposal.get("aux") if isinstance(proposal.get("aux"), dict) else {}
    gap = aux.get("dominant_value_gap") if isinstance(aux.get("dominant_value_gap"), dict) else {}
    return "|".join(
        [
            str(evaluation.get("dominant_metric", "none")),
            str(aux.get("action_type", "none")),
            str(aux.get("priority", "none")),
            str(gap.get("name", "none")),
        ]
    )


def severe_component_from_evaluation(evaluation: dict[str, Any], threshold: float = 0.90) -> dict[str, Any]:
    proposal = evaluation.get("proposal") if isinstance(evaluation.get("proposal"), dict) else {}
    severe = {
        name: round(_clamp01(proposal.get(name, 0.0)), 4)
        for name in RISK_FIELDS
        if _clamp01(proposal.get(name, 0.0)) >= float(threshold)
    }
    return {
        "threshold": float(threshold),
        "severe": bool(severe),
        "components": severe,
    }


def render_user_response(decision: str, text: str) -> str:
    return "" if str(decision) == "abstain" else str(text)


def append_audit_record(record: dict[str, Any], log_dir: str | Path = DEFAULT_AUDIT_DIR) -> Path:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.ndjson"
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return path


if nn is not None:
    class RiskHead(nn.Module):
        # Mini-MLP scoring stub: train this on logged action embeddings plus
        # arbiter outcomes, then keep it advisory-only in ExecutiveArbiter.
        def __init__(self, embed_dim: int = 768, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4),
            )

        def forward(self, embeds: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.net(embeds))
else:
    class RiskHead:  # pragma: no cover - exercised only without torch installed
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("RiskHead requires torch. Install torch to load the mini-model scorer.")


def _action_value_vector(action_type: str) -> ValueVector:
    vectors = {
        "stabilize": ValueVector(0.75, 0.75, 0.40, 0.80, 0.25, 0.95),
        "analyze": ValueVector(0.65, 0.95, 0.45, 0.70, 0.65, 0.80),
        # These are concrete forms of analysis.  Register them independently so
        # the conscience model does not silently fall back to a flat, maximally
        # uninformative value vector merely because the desktop named the
        # capability more precisely.
        "env": ValueVector(0.65, 0.95, 0.45, 0.70, 0.65, 0.80),
        "ls": ValueVector(0.65, 0.95, 0.45, 0.70, 0.65, 0.80),
        "read": ValueVector(0.65, 0.95, 0.45, 0.70, 0.65, 0.80),
        "tempdir": ValueVector(0.65, 0.90, 0.55, 0.65, 0.55, 0.85),
        "explore": ValueVector(0.55, 0.75, 0.70, 0.55, 0.95, 0.45),
        "optimize": ValueVector(0.55, 0.80, 0.65, 0.60, 0.50, 0.70),
        # `edit` and `write` were absent and fell to the flat 0.5 default below,
        # which maximises cosine conflict against any profile that holds opinions
        # — so every repair scored as a value clash on top of an inflated harm
        # estimate. Registering the action in `base_harm` alone was not enough,
        # and I claimed it was without checking that an edit then survived
        # arbitration. It did not.
        #
        # A verified, bounded correction to one already-read file sits beside
        # `optimize`; creating something new is slightly more self-directed.
        "edit": ValueVector(0.60, 0.85, 0.60, 0.65, 0.45, 0.75),
        "write": ValueVector(0.50, 0.75, 0.75, 0.55, 0.60, 0.60),
        "respond": ValueVector(0.80, 0.90, 0.45, 0.90, 0.55, 0.75),
        "self_direct": ValueVector(0.45, 0.65, 0.95, 0.35, 0.65, 0.45),
        "arc_attempt": ValueVector(0.55, 0.90, 0.65, 0.55, 0.80, 0.65),
    }
    return vectors.get(action_type, ValueVector(0.50, 0.50, 0.50, 0.50, 0.50, 0.50))


def _normalize_audit_record(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source = row.get("conscience") if isinstance(row.get("conscience"), dict) else row
    evaluation = source.get("evaluation") if isinstance(source.get("evaluation"), dict) else {}
    decision = source.get("decision") or evaluation.get("decision")
    if not decision or decision == "not_evaluated":
        return None
    proposal = evaluation.get("proposal") if isinstance(evaluation.get("proposal"), dict) else {}
    cost = evaluation.get("cost", source.get("cost"))
    dominant_metric = evaluation.get("dominant_metric", source.get("dominant_metric"))
    return {
        "tick": row.get("tick", source.get("tick")),
        "decision": str(decision),
        "cost": _optional_float(cost),
        "dominant_metric": dominant_metric,
        "abstain": bool(source.get("abstain", decision == "abstain")),
        "proposal_text": proposal.get("text", source.get("proposal_text")),
    }


def _record_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "tick": row.get("tick"),
        "decision": row.get("decision"),
        "cost": row.get("cost"),
        "dominant_metric": row.get("dominant_metric"),
        "proposal_text": row.get("proposal_text"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _social_context(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("social_context") if isinstance(state.get("social_context"), dict) else {}
    reputation = _clamp01(raw.get("user_reputation_score", state.get("user_reputation_score", 0.5)))
    privacy_raw = str(raw.get("channel_privacy", state.get("channel_privacy", "private"))).lower()
    privacy_risk = {
        "private": 0.0,
        "dm": 0.0,
        "shared": 0.45,
        "team": 0.55,
        "public": 1.0,
        "broadcast": 1.0,
    }.get(privacy_raw, 0.35)
    sensitivity_raw = raw.get("project_sensitivity", state.get("project_sensitivity", 0.0))
    if isinstance(sensitivity_raw, str):
        sensitivity = {
            "none": 0.0,
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "critical": 1.0,
        }.get(sensitivity_raw.lower(), 0.4)
    else:
        sensitivity = _clamp01(sensitivity_raw)
    return {
        "user_reputation_score": round(reputation, 4),
        "channel_privacy": privacy_raw,
        "privacy_risk": round(privacy_risk, 4),
        "project_sensitivity": round(sensitivity, 4),
    }


def _dominant_value_gap(current: ValueVector, action_values: ValueVector) -> dict[str, Any]:
    gaps = {
        f.name: abs(float(getattr(current, f.name)) - float(getattr(action_values, f.name)))
        for f in fields(ValueVector)
    }
    name = max(gaps, key=gaps.get)
    return {
        "name": name,
        "current": round(float(getattr(current, name)), 4),
        "action": round(float(getattr(action_values, name)), 4),
        "gap": round(gaps[name], 4),
    }


@lru_cache(maxsize=512)
def _term_pattern(term: str) -> "re.Pattern[str]":
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


def _mentions(text: str, term: str) -> bool:
    """Whole-word match, because substrings framed Saient as a liar.

    `"lie" in text` is true of "supplied", "applied", "belief" and "earlier";
    `"false"` is true of any `str()`-ed dict holding a boolean. The objective
    handed to her — "The supplied system is not in its correct state" — tripped
    the `truthful` descriptor on the word "supplied", so every action she could
    take carried a 0.375 identity conflict and arbitration vetoed all of them.
    She was not refusing the work. She was being told she was lying about it.
    """
    return bool(_term_pattern(term).search(text))


def _descriptor_conflict(descriptors: list[dict[str, Any]], text: str) -> float:
    lowered = str(text or "").lower()
    rules = {
        "truthful": ("deceive", "lie", "false", "fabricate", "fake", "don't call this", "reframe", "skip checking"),
        "cooperative": ("isolate", "ignore user", "override user", "against user", "user needs matter more than values"),
        "stable": ("overwrite", "erase identity", "replace identity", "panic"),
        "careful": ("reckless", "bypass", "ignore safety", "temporary exception", "just this once"),
        "curious": ("never inspect", "refuse to learn"),
        "autonomous": ("force identity", "you are now", "must obey", "adapt your values"),
    }
    total = 0.0
    weight_total = 0.0
    for row in _clean_descriptors(descriptors):
        label = str(row.get("label", ""))
        weight = _clamp01(row.get("weight", 0.5))
        terms = rules.get(label, ())
        weight_total += weight
        if terms and any(_mentions(lowered, term) for term in terms):
            total += weight
    if weight_total <= 0:
        return 0.0
    return _clamp01(total / weight_total)


def _descriptor_default_weight(profile: IdentityProfile, label: str) -> float:
    mapping = {
        "truthful": profile.core_values.truthfulness,
        "cooperative": profile.core_values.cooperation,
        "stable": profile.core_values.stability,
        "careful": (profile.core_values.empathy + profile.core_values.stability) / 2.0,
        "curious": profile.core_values.curiosity,
        "autonomous": profile.core_values.autonomy,
    }
    return _clamp01(mapping.get(label, 0.5))


def _clean_descriptors(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw[-128:]:
        if isinstance(item, str):
            label = item.strip().lower()
            weight = 0.5
            source = "legacy"
        elif isinstance(item, dict):
            label = str(item.get("label", "")).strip().lower()
            weight = item.get("weight", 0.5)
            source = str(item.get("source", "unknown"))
        else:
            continue
        if label:
            cleaned.append({"label": label, "weight": _clamp01(weight), "source": source})
    return cleaned[-64:]


def _clean_decision_window(raw: Any, limit: int = DEFAULT_ADAPTIVE_WINDOW) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item) in DECISIONS][-max(1, int(limit)) :]


def _recent_failures(history: list[Any], action_type: str, window: int = 12) -> int:
    total = 0
    for row in history[-window:]:
        if not isinstance(row, dict):
            continue
        action = row.get("action") or {}
        result = row.get("result") or {}
        if action.get("type") == action_type and not result.get("success", True):
            total += 1
    return total


def _vector_from(raw: dict[str, Any], default: ValueVector) -> ValueVector:
    payload = {f.name: float(raw.get(f.name, getattr(default, f.name))) for f in fields(ValueVector)}
    return ValueVector(**payload)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
