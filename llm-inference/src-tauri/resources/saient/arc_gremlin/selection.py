from __future__ import annotations

from .config import rng

OPS = ["identity", "flip_h", "flip_v", "rotate_90", "transpose", "color_shift"]


def generate_hypothesis() -> dict:
    r = rng()
    op = OPS[r.randrange(len(OPS))]
    return {"op": op, "param": r.randrange(10)}
