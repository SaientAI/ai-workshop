def reflect(state: dict, window: int = 20) -> dict | None:
    history = state.get("history", [])
    if len(history) < 10:
        return None

    recent = history[-window:]
    total = len(recent)
    success_rate = sum(1 for h in recent if h.get("result", {}).get("success")) / max(1, total)

    action_counts: dict[str, int] = {}
    for h in recent:
        action_type = h.get("action", {}).get("type", "idle")
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    dominant_action = max(action_counts, key=action_counts.get) if action_counts else "idle"

    # Detect repeated failure on same action to trigger stronger mutation.
    fail_streak = 0
    for h in reversed(recent):
        if h.get("result", {}).get("success"):
            break
        fail_streak += 1

    last = recent[-1]
    prev = recent[-5]
    improved = False
    core = ("information_depth", "efficiency", "autonomy")
    if "drives" in last and "drives" in prev:
        improved = (
            sum(float(last["drives"].get(k, 0.0)) for k in core)
            > sum(float(prev["drives"].get(k, 0.0)) for k in core)
        )

    return {
        "success_rate": success_rate,
        "dominant_action": dominant_action,
        "stagnating": success_rate < 0.6,
        "fail_streak": fail_streak,
        "sample_size": total,
        "anticipation_helped": improved,
    }
