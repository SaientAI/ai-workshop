def write_commitment(state: dict, action: dict) -> None:
    action_type = action.get("type", "idle")
    commitments = state.setdefault("commitments", [])
    if commitments and commitments[-1].get("action") == action_type:
        # Avoid stacking duplicate commitments on repeated same action.
        commitments[-1]["ttl"] = max(int(commitments[-1].get("ttl", 0)), 6)
        return
    commitments.append({
        "action": action_type,
        "ttl": 6,
    })


def apply_commitment_pressure(state: dict, action: dict) -> float:
    penalty = 0.0
    kept = []
    active = 0

    for commitment in state.get("commitments", []):
        ttl = int(commitment.get("ttl", 0))
        if ttl <= 0:
            continue

        active += 1
        if commitment.get("action") != action.get("type"):
            penalty += 0.08

        commitment["ttl"] = ttl - 1
        kept.append(commitment)

    # Drop expired commitments so pressure window stays bounded.
    state["commitments"] = [c for c in kept if c.get("ttl", 0) > 0]
    # Normalize + cap so cortisol does not saturate permanently.
    if active > 0:
        penalty = penalty / active
    return min(0.30, penalty)
