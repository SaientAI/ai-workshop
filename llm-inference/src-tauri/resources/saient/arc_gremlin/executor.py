from __future__ import annotations

from .dsl import TransformRule
from .ops import color_shift, flip_h, flip_v, rotate_90, transpose


def apply_transform(rule: TransformRule, grid: list[list[int]]) -> list[list[int]]:
    op = rule.op
    if op == "identity":
        return [row[:] for row in grid]
    if op == "flip_h":
        return flip_h(grid)
    if op == "flip_v":
        return flip_v(grid)
    if op == "rotate_90":
        return rotate_90(grid)
    if op == "transpose":
        return transpose(grid)
    if op.startswith("color_shift:"):
        return color_shift(grid, int(op.split(":", 1)[1]))
    return [row[:] for row in grid]
