def mutate_strategy(state: dict, reflection: dict | None) -> dict:
    strategy = dict(state.get("strategy", {"mode": "balanced", "last_updated": 0}))
    params = state.setdefault("params", {"trend_gain": 1.0})
    if reflection is None:
        return strategy

    tick = int(state.get("tick", 0))
    mode = strategy.get("mode", "balanced")

    if reflection.get("fail_streak", 0) >= 3:
        # Hard pivot under consecutive failure.
        mode = "exploratory" if mode != "exploratory" else "aggressive"
    elif reflection.get("stagnating"):
        if mode == "balanced":
            mode = "exploratory"
        elif mode == "exploratory":
            mode = "aggressive"
        else:
            mode = "balanced"
    else:
        # Recover to stable behavior when outcomes are healthy.
        mode = "balanced"

    strategy["mode"] = mode
    strategy["last_updated"] = tick

    # Self-tune anticipation sensitivity from outcomes.
    tg = float(params.get("trend_gain", 1.0))
    if reflection:
        helped = reflection.get("anticipation_helped", None)
        if helped is True:
            inc = 1.08 if tg < 1.3 else 1.03
            tg *= inc
        elif helped is False:
            tg *= 0.94
    params["trend_gain"] = max(0.8, min(1.6, tg))

    return strategy
