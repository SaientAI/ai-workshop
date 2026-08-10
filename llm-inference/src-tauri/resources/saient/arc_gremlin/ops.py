from __future__ import annotations

from typing import Any


def to_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("Grid must be 2D")
    width = len(grid[0])
    out: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("Grid rows must have equal length")
        out.append([int(v) for v in row])
    return out


def flip_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in to_grid(grid)]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    g = to_grid(grid)
    return list(reversed([row[:] for row in g]))


def rotate_90(grid: list[list[int]]) -> list[list[int]]:
    g = to_grid(grid)
    h = len(g)
    w = len(g[0]) if g else 0
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def transpose(grid: list[list[int]]) -> list[list[int]]:
    g = to_grid(grid)
    h = len(g)
    w = len(g[0]) if g else 0
    return [[g[r][c] for r in range(h)] for c in range(w)]


def color_shift(grid: list[list[int]], param: int) -> list[list[int]]:
    p = int(param) % 10
    return [[(int(v) + p) % 10 for v in row] for row in to_grid(grid)]
