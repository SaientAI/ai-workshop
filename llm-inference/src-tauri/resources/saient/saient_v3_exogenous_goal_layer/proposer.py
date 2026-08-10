from __future__ import annotations

import re
from typing import Iterable

from .schema import CandidateGoal


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "unknown_anomaly"


def _next_goal_id(tick: int, idx: int, prefix: str = "g") -> str:
    return f"{prefix}-{tick}-{idx}"


def propose_candidates(state: dict) -> list[CandidateGoal]:
    tick = int(state.get("tick", 0))
    out: list[CandidateGoal] = []
    idx = 0

    mission = state.get("mission") or {}
    mission_target = mission.get("target")
    if mission_target:
        out.append(
            CandidateGoal(
                goal_id=_next_goal_id(tick, idx, "inherited"),
                objective=f"inherited:{mission_target}",
                proposal_origin="inherited_system",
                created_tick=tick,
                payload={"mission_id": mission.get("id"), "target": mission_target},
            )
        )
        idx += 1

    anomalies: Iterable[str] = state.get("anomalies", []) or []
    for a in anomalies:
        if not isinstance(a, str) or not a.strip():
            continue
        out.append(
            CandidateGoal(
                goal_id=_next_goal_id(tick, idx, "exo"),
                objective=f"investigate:{_slug(a)}",
                proposal_origin="exogenous_probe",
                created_tick=tick,
                payload={"anomaly": a},
            )
        )
        idx += 1

    if bool(state.get("state_conflict", False)):
        out.append(
            CandidateGoal(
                goal_id=_next_goal_id(tick, idx, "exo"),
                objective="resolve:state_conflict",
                proposal_origin="exogenous_probe",
                created_tick=tick,
                payload={"conflict": True},
            )
        )

    return out
