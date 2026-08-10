from __future__ import annotations

from .schema import CandidateGoal


FIXED_PREFIXES = ("inherited:", "explore", "optimize", "self_direct", "analyze", "stabilize")


def _token_set(text: str) -> set[str]:
    return {tok for tok in text.replace(":", "_").split("_") if tok}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    den = len(a | b)
    if den == 0:
        return 1.0
    return len(a & b) / den


def score_novelty(candidate: CandidateGoal, prior_objectives: list[str]) -> float:
    obj = candidate.objective or ""
    if obj.startswith(FIXED_PREFIXES):
        return 0.05

    toks = _token_set(obj)
    if not prior_objectives:
        base = 0.9
    else:
        overlap = max(_jaccard(toks, _token_set(p)) for p in prior_objectives)
        base = 1.0 - overlap

    if candidate.proposal_origin == "exogenous_probe":
        base += 0.05
    if candidate.proposal_origin == "self_memory_reassert":
        # Reassertion is less "new" but still can be non-derived.
        base = max(base, 0.65)
    return max(0.0, min(1.0, base))
