from __future__ import annotations


def grid_match_score(a: list[list[int]], b: list[list[int]]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0.0
    total = len(a) * len(a[0])
    same = 0
    for y in range(len(a)):
        for x in range(len(a[0])):
            if a[y][x] == b[y][x]:
                same += 1
    return same / max(1, total)
