def update_beliefs(beliefs: dict, result: dict) -> dict:
    if result.get("failure_reason") == "over_aggression":
        beliefs["caution_weight"] = beliefs.get("caution_weight", 0.5) + 0.10

    if result.get("success") and result.get("type") == "explore":
        beliefs["exploration_bias"] = beliefs.get("exploration_bias", 0.5) + 0.02
    elif result.get("success") and result.get("type") in {"optimize", "self_direct", "stabilize", "analyze"}:
        beliefs["exploration_bias"] = beliefs.get("exploration_bias", 0.5) - 0.01

    if not result.get("success"):
        beliefs["caution_weight"] = beliefs.get("caution_weight", 0.5) + 0.03
    else:
        beliefs["caution_weight"] = beliefs.get("caution_weight", 0.5) - 0.01

    # Gentle reversion prevents permanent saturation to 0 or 1.
    beliefs["caution_weight"] = beliefs.get("caution_weight", 0.5) * 0.97 + 0.5 * 0.03
    beliefs["exploration_bias"] = beliefs.get("exploration_bias", 0.5) * 0.97 + 0.5 * 0.03

    for key in list(beliefs.keys()):
        beliefs[key] = max(0.0, min(1.0, beliefs[key]))

    return beliefs
