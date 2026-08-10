def update_drives(state: dict, result: dict) -> None:
    drives = state["drives"]
    core = ("information_depth", "efficiency", "autonomy")

    # SUCCESS = diminishing returns (big recovery when low, small when high)
    if result.get("success"):
        for key in core:
            gain = 0.15 * (1.0 - float(drives.get(key, 0.5)))
            drives[key] += gain
        drives["cortisol"] = drives.get("cortisol", 0.2) - 0.03
    # FAILURE = softer penalty
    else:
        drives["information_depth"] -= 0.05
        drives["autonomy"] -= 0.05
        drives["cortisol"] = drives.get("cortisol", 0.2) + 0.06

    # SPECIAL: optimize is recovery action.
    if result.get("recovery"):
        strength = float(result.get("strength", 1.0))
        target = result.get("target")
        active = state.get("active_preempt")
        has_active_target = isinstance(active, dict) and active.get("drive") == target
        for key in core:
            if has_active_target:
                if key == target:
                    drives[key] += 0.2
                else:
                    drives[key] += 0.005
            else:
                if key == target:
                    delta = max(0.08, 0.18 * strength)
                    drives[key] += delta
                else:
                    drives[key] += 0.02
        drives["cortisol"] = drives.get("cortisol", 0.2) - 0.03

    # High-state boredom: perfect state creates its own instability.
    if all(float(drives.get(key, 0.0)) > 0.85 for key in core):
        drives["autonomy"] -= 0.05

    # BASELINE DECAY + REGEN + SOFT CAP DAMPING
    active = state.get("active_preempt")
    active_drive = active.get("drive") if isinstance(active, dict) else None
    active_stage = state.get("preempt_stage", {}).get(active_drive, {}).get("mode") if active_drive else None
    for key in core:
        if active_drive and active_stage == "commit":
            decay = 0.0 if key == active_drive else 0.015
            regen = 0.01 if key == active_drive else 0.002
        else:
            decay = 0.0 if key == active_drive else 0.015
            regen = 0.01
        drives[key] -= decay
        drives[key] += regen
        if drives[key] > 0.8:
            drives[key] -= (drives[key] - 0.8) * 0.5
        drives[key] = max(0.05, min(1.0, drives[key]))

    drives["cortisol"] = max(0.0, min(1.0, drives.get("cortisol", 0.2)))
