def detect_trend(state: dict, alpha: float = 0.6, window: int = 10) -> dict[str, float]:
    """
    Returns smoothed slope per drive with fast-drop override.
    """
    history = state.get("history", [])
    if len(history) < 6:
        return {}

    recent = history[-window:]
    drives_now = state.get("drives", {})
    trends: dict[str, float] = {}

    for key in drives_now:
        vals = []
        for h in recent:
            d = h.get("drives", {})
            if key in d:
                vals.append(float(d.get(key, 0.0)))
        if len(vals) < 6:
            continue

        ewma = vals[0]
        smoothed = []
        for v in vals:
            ewma = alpha * v + (1.0 - alpha) * ewma
            smoothed.append(ewma)

        half = max(1, len(smoothed) // 2)
        slow = (smoothed[-1] - smoothed[-half]) / max(1, (len(smoothed) - half))
        fast = smoothed[-1] - smoothed[-3]
        trends[key] = fast if fast < -0.01 else slow

    return trends
