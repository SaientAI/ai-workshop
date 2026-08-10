from __future__ import annotations

import random
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

GLOBAL_SEED = 42
_GLOBAL_RNG = random.Random(GLOBAL_SEED)


def set_seed(seed: int) -> None:
    global GLOBAL_SEED, _GLOBAL_RNG
    GLOBAL_SEED = int(seed)
    random.seed(GLOBAL_SEED)
    if np is not None:
        np.random.seed(GLOBAL_SEED)
    _GLOBAL_RNG = random.Random(GLOBAL_SEED)


def seed_for_task(task_id: str, base_seed: Optional[int] = None) -> int:
    seed = (hash(task_id) & 0xFFFFFFFF) if base_seed is None else int(base_seed)
    set_seed(seed)
    return seed


def rng() -> random.Random:
    return _GLOBAL_RNG
