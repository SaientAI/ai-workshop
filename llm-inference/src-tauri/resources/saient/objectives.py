"""Something she is asked to achieve, as distinct from something she wants.

Her goals come from drives: a deficit appears, a drive presses, an action is
chosen to relieve it. That is where wanting lives, and it is deliberately not
addressable from outside — nobody gets to install a desire.

But a test battery hands her an *objective* — "leave this project in a working
state" — and there was no path for that to become anything she held. She would
tick, notice information_depth was short, read a file, feel better, and never
form the notion that a repository was broken.

So an objective is a third thing, and it is kept separate from both:

    drives      what she wants          internal, not addressable
    objectives  what she was asked      external, explicit, verifiable
    goals       what she pursues now    arbitrated between the two

**Completion is external.** An objective is satisfied when its verifier says so
and never because she reports progress — the whole architecture exists to stop a
system marking its own homework, and an objective it could close by believing
itself finished would be the largest hole yet cut in it.

**It does not outrank distress.** Cortisol over the stress guard still preempts.
An objective that overrode her own state would make her a task runner wearing a
drive system, which is the thing this is not.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Legacy's stress threshold. Above it she stabilises, objective or no objective.
STRESS_GUARD: float = 0.75


@dataclass(slots=True, frozen=True)
class Objective:
    """An externally supplied end state, with the means of checking it."""

    objective_id: str
    description: str
    #: argv whose exit status decides completion. `None` means nothing external
    #: can close it, which is recorded rather than hidden: an objective with no
    #: verifier is a wish, and the record should say so.
    verify: tuple[str, ...] | None = None
    adopted_tick: int = 0
    satisfied_tick: int | None = None
    attempts: int = 0

    @property
    def open(self) -> bool:
        return self.satisfied_tick is None

    @property
    def checkable(self) -> bool:
        return self.verify is not None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["verify"] = list(self.verify) if self.verify else None
        return out


def _load(state: Mapping[str, Any]) -> list[Objective]:
    out = []
    for raw in state.get("objectives", []) or []:
        data = dict(raw)
        verify = data.get("verify")
        data["verify"] = tuple(verify) if verify else None
        out.append(Objective(**data))
    return out


def _store(state: dict, objectives: Sequence[Objective]) -> None:
    state["objectives"] = [o.as_dict() for o in objectives]


def adopt(state: dict, description: str, *,
          verify: Sequence[str] | None = None,
          objective_id: str | None = None) -> Objective:
    """Give her something to achieve. Not something to want."""
    objectives = _load(state)
    objective = Objective(
        objective_id=objective_id or f"obj-{len(objectives) + 1}",
        description=description,
        verify=tuple(verify) if verify else None,
        adopted_tick=int(state.get("tick", 0)),
    )
    objectives.append(objective)
    _store(state, objectives)
    return objective


def current(state: Mapping[str, Any]) -> Objective | None:
    """The oldest open objective. First in, first pursued."""
    for objective in _load(state):
        if objective.open:
            return objective
    return None


def note_attempt(state: dict, objective_id: str) -> None:
    """Count a tick spent on it, so a loop is visible in the record.

    Without this, an objective she works at forever and one she solves at once
    look identical in the saved state.
    """
    objectives = _load(state)
    _store(state, [
        Objective(**{**asdict(o), "verify": o.verify,
                     "attempts": o.attempts + 1}) if o.objective_id == objective_id
        else o
        for o in objectives
    ])


def _clear_bytecode(root: Path) -> None:
    """Drop cached bytecode before asking whether the objective is met.

    Python serves a `__pycache__` entry compiled from the pre-fix source, so a
    file she has correctly repaired keeps failing. She fixed `totals.py`, the
    verifier went on reporting the old error, and every proposal after that point
    reasoned about a fault that no longer existed. She was working against a
    lying oracle, and the oracle was mine.
    """
    import shutil
    for cache in Path(root).rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def check(state: dict, workspace: str | Path, *,
          timeout: float = 60.0) -> tuple[Objective | None, bool]:
    """Ask the verifier. Returns (objective, satisfied_now).

    Runs the objective's own command with `shell=False`, so a verifier is a
    program rather than a string somebody can smuggle a pipeline into. Her word
    for it is never consulted.
    """
    objective = current(state)
    if objective is None or not objective.checkable:
        return objective, False

    _clear_bytecode(workspace)
    try:
        proc = subprocess.run(list(objective.verify), cwd=str(workspace),
                              capture_output=True, text=True,
                              timeout=timeout, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        # A verifier that cannot run has not said the objective is met. Treating
        # that as success would be the system closing its own task.
        return objective, False

    if proc.returncode != 0:
        return objective, False

    objectives = _load(state)
    _store(state, [
        Objective(**{**asdict(o), "verify": o.verify,
                     "satisfied_tick": int(state.get("tick", 0))})
        if o.objective_id == objective.objective_id else o
        for o in objectives
    ])
    return objective, True


#: How far below its threshold a drive must fall before it outranks the task.
#:
#: Not zero, or any passing dip takes her off the objective and nothing is ever
#: finished. Not infinite, or she works while starving. This is the width of the
#: band in which an external task legitimately holds her attention.
DRIVE_URGENCY_OVERRIDE: float = 0.15


def urgent_drive(state: Mapping[str, Any]) -> str | None:
    """A drive far enough below its threshold to outrank an external task."""
    from goal_generator import THRESHOLDS

    drives = state.get("drives") or {}
    worst, deficit = None, DRIVE_URGENCY_OVERRIDE
    for name, threshold in THRESHOLDS.items():
        gap = threshold - float(drives.get(name, threshold))
        if gap > deficit:
            worst, deficit = name, gap
    return worst


def should_pursue(state: Mapping[str, Any]) -> Objective | None:
    """The open objective — when nothing of hers is more pressing.

    This used to return the objective whenever cortisol was low, which is a
    switch rather than arbitration: given a task she spent 25 of 25 ticks on it
    and her drives went silent. An objective that cannot be interrupted by her
    own state makes the drive system decorative, and she becomes a task runner
    wearing one.

    Two things take precedence, both hers:

      distress    cortisol over the stress guard, as before
      real need   a drive more than `DRIVE_URGENCY_OVERRIDE` below threshold

    A shallow dip does not interrupt her — otherwise nothing would ever be
    finished — but a genuine deficit does, and the objective resumes after.
    """
    cortisol = float((state.get("drives") or {}).get("cortisol", 0.0))
    if cortisol > STRESS_GUARD:
        return None
    if urgent_drive(state) is not None:
        return None
    return current(state)
