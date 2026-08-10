from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

OPS = [
    "identity",
    "flip_h",
    "flip_v",
    "rotate_90",
    "transpose",
    "color_shift",
    "move_to_corner",
    "center_object",
    "duplicate_object",
    "remove_smallest",
    "crop_to_bbox",
    "paint_bbox",
    "object_recolor_by_rank",
    "extract_repeating_unit",
    "append_head_rows",
    "append_tail_rows",
    "recolor_nonzero_to",
    "self_tile_mask",
    "template_match",
]

_ROW_SIG_POS_CACHE: dict[tuple[int, ...], dict[str, Any]] = {}


@dataclass
class SubsetRule:
    mode: str = "row"
    k: int = 3
    weights: list[float] | None = None
    condition: dict[str, Any] | None = None
    order_sensitive: bool = True
    confidence: float = 0.0
    mean_score: float = 0.0
    var_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "row_linear_v2",
            "mode": self.mode,
            "k": int(self.k),
            "weights": list(self.weights or []),
            "condition": dict(self.condition) if self.condition else None,
            "order_sensitive": bool(self.order_sensitive),
            "confidence": float(self.confidence),
            "mean_score": float(self.mean_score),
            "var_score": float(self.var_score),
        }


@dataclass
class TransformRule:
    op: str

    def __repr__(self) -> str:
        return f"Transform({self.op})"


@dataclass
class ProgramRule:
    steps: list[Any]

    def __repr__(self) -> str:
        return " -> ".join(str(s) for s in self.steps)


def load_task(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        task = json.load(f)
    if "train" not in task or "test" not in task:
        raise ValueError("ARC task must contain 'train' and 'test'")
    return task


def _to_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("Grid must be 2D")
    width = len(grid[0])
    out: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("Grid rows must have equal length")
        out.append([int(v) for v in row])
    return out


def _shape(grid: list[list[int]]) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _flip_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def _flip_v(grid: list[list[int]]) -> list[list[int]]:
    return list(reversed([row[:] for row in grid]))


def _rot90(grid: list[list[int]]) -> list[list[int]]:
    h, w = _shape(grid)
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _transpose(grid: list[list[int]]) -> list[list[int]]:
    h, w = _shape(grid)
    return [[grid[r][c] for r in range(h)] for c in range(w)]


def _color_shift(grid: list[list[int]], param: int) -> list[list[int]]:
    p = int(param) % 10
    return [[(v + p) % 10 for v in row] for row in grid]


def extract_objects(grid: list[list[int]]) -> list[dict[str, Any]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    seen: set[tuple[int, int]] = set()
    objects: list[dict[str, Any]] = []

    for y in range(h):
        for x in range(w):
            color = g[y][x]
            if color == 0 or (y, x) in seen:
                continue
            stack = [(y, x)]
            seen.add((y, x))
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and g[ny][nx] == color:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            ys = [p[0] for p in coords]
            xs = [p[1] for p in coords]
            y0, x0, y1, x1 = min(ys), min(xs), max(ys), max(xs)
            shape = [[0 for _ in range((x1 - x0) + 1)] for _ in range((y1 - y0) + 1)]
            for cy, cx in coords:
                shape[cy - y0][cx - x0] = int(color)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "size": len(coords),
                    "bbox": (y0, x0, y1, x1),
                    "shape": shape,
                    "center": ((y0 + y1) // 2, (x0 + x1) // 2),
                }
            )
    return objects


def enrich_objects(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for o in objs:
        y0, x0, y1, x1 = o["bbox"]
        o["width"] = (x1 - x0) + 1
        o["height"] = (y1 - y0) + 1
        o["area"] = o["width"] * o["height"]
    return objs


def compute_relations(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    centers = [o["center"] for o in objs]
    for o in objs:
        cy, cx = o["center"]
        rank_left = sum(1 for _, x in centers if x < cx)
        rank_top = sum(1 for y, _ in centers if y < cy)
        o["rank_left"] = rank_left
        o["rank_top"] = rank_top
        o["is_leftmost"] = rank_left == 0
        o["is_topmost"] = rank_top == 0
    return objs


def _blank_like(grid: list[list[int]]) -> list[list[int]]:
    h, w = _shape(grid)
    return [[0 for _ in range(w)] for _ in range(h)]


def _stamp_object(dst: list[list[int]], src: list[list[int]], obj: dict[str, Any], top: int, left: int) -> None:
    h, w = _shape(dst)
    y0, x0, _, _ = obj["bbox"]
    for y, x in obj["coords"]:
        ny = top + (y - y0)
        nx = left + (x - x0)
        if 0 <= ny < h and 0 <= nx < w:
            dst[ny][nx] = src[y][x]


def _move_to_corner(grid: list[list[int]]) -> list[list[int]]:
    g = _to_grid(grid)
    objs = extract_objects(g)
    if not objs:
        return [row[:] for row in g]
    largest = max(objs, key=lambda o: int(o["size"]))
    out = _blank_like(g)
    _stamp_object(out, g, largest, 0, 0)
    return out


def _center_object(grid: list[list[int]]) -> list[list[int]]:
    g = _to_grid(grid)
    objs = extract_objects(g)
    if not objs:
        return [row[:] for row in g]
    largest = max(objs, key=lambda o: int(o["size"]))
    h, w = _shape(g)
    y0, x0, y1, x1 = largest["bbox"]
    oh = (y1 - y0) + 1
    ow = (x1 - x0) + 1
    top = max(0, (h - oh) // 2)
    left = max(0, (w - ow) // 2)
    out = _blank_like(g)
    _stamp_object(out, g, largest, top, left)
    return out


def _duplicate_object(grid: list[list[int]]) -> list[list[int]]:
    g = _to_grid(grid)
    objs = extract_objects(g)
    if not objs:
        return [row[:] for row in g]
    largest = max(objs, key=lambda o: int(o["size"]))
    out = [row[:] for row in g]
    y0, x0, y1, x1 = largest["bbox"]
    oh = (y1 - y0) + 1
    ow = (x1 - x0) + 1
    top = min(max(0, y0 + oh), max(0, len(g) - oh))
    left = min(max(0, x0 + ow), max(0, len(g[0]) - ow))
    _stamp_object(out, g, largest, top, left)
    return out


def _remove_smallest(grid: list[list[int]]) -> list[list[int]]:
    g = [row[:] for row in _to_grid(grid)]
    objs = extract_objects(g)
    if not objs:
        return g
    smallest = min(objs, key=lambda o: int(o["size"]))
    for y, x in smallest["coords"]:
        g[y][x] = 0
    return g


def _crop_to_bbox(grid: list[list[int]]) -> list[list[int]]:
    g = _to_grid(grid)
    objs = extract_objects(g)
    if not objs:
        return [row[:] for row in g]
    ys: list[int] = []
    xs: list[int] = []
    for obj in objs:
        y0, x0, y1, x1 = obj["bbox"]
        ys.extend([y0, y1])
        xs.extend([x0, x1])
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    return [row[min_x : max_x + 1] for row in g[min_y : max_y + 1]]


def _paint_bbox(grid: list[list[int]]) -> list[list[int]]:
    g = [row[:] for row in _to_grid(grid)]
    objs = extract_objects(g)
    if not objs:
        return g
    largest = max(objs, key=lambda o: int(o["size"]))
    y0, x0, y1, x1 = largest["bbox"]
    c = int(largest["color"])
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            g[y][x] = c
    return g


def _object_recolor_by_rank(grid: list[list[int]]) -> list[list[int]]:
    g = [row[:] for row in _to_grid(grid)]
    objs = sorted(extract_objects(g), key=lambda o: int(o["size"]), reverse=True)
    if not objs:
        return g
    for idx, obj in enumerate(objs):
        new_color = (9 - idx) % 10
        if new_color == 0:
            new_color = 1
        for y, x in obj["coords"]:
            g[y][x] = new_color
    return g


def _extract_repeating_unit(grid: list[list[int]], axis: str = "y") -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    axis = str(axis)
    if h == 0 or w == 0:
        return [row[:] for row in g]

    if axis == "x":
        # Find smallest width segment that best reconstructs the row-wise pattern.
        best_k = w
        best_score = -1.0
        for k in range(1, w + 1):
            same = 0
            total = h * w
            for y in range(h):
                for x in range(w):
                    if g[y][x] == g[y][x % k]:
                        same += 1
            score = same / max(1, total)
            if score > best_score:
                best_score = score
                best_k = k
            if score == 1.0:
                break
        return [row[:best_k] for row in g]

    # Default axis y: find smallest height prefix that best reconstructs vertical pattern.
    best_k = h
    best_score = -1.0
    for k in range(1, h + 1):
        same = 0
        total = h * w
        for y in range(h):
            for x in range(w):
                if g[y][x] == g[y % k][x]:
                    same += 1
        score = same / max(1, total)
        if score > best_score:
            best_score = score
            best_k = k
        if score == 1.0:
            break
    return [row[:] for row in g[:best_k]]


def _append_head_rows(grid: list[list[int]], rows: int = 0) -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0:
        return [row[:] for row in g]
    n = int(rows)
    if n <= 0:
        # Useful ARC default: append width-count rows.
        n = w
    n = max(1, min(h, n))
    out = [row[:] for row in g]
    out.extend([row[:] for row in g[:n]])
    return out


def _append_tail_rows(grid: list[list[int]], rows: int = 0) -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0:
        return [row[:] for row in g]
    n = int(rows)
    if n <= 0:
        n = w
    n = max(1, min(h, n))
    out = [row[:] for row in g]
    out.extend([row[:] for row in g[h - n : h]])
    return out


def _color_hist_score(grid: list[list[int]], window: list[list[int]]) -> float:
    g_hist = [0 for _ in range(10)]
    w_hist = [0 for _ in range(10)]
    for row in grid:
        for v in row:
            g_hist[int(v) % 10] += 1
    for row in window:
        for v in row:
            w_hist[int(v) % 10] += 1
    dot = 0.0
    g_norm = 0.0
    w_norm = 0.0
    for i in range(10):
        dot += g_hist[i] * w_hist[i]
        g_norm += g_hist[i] * g_hist[i]
        w_norm += w_hist[i] * w_hist[i]
    denom = (g_norm ** 0.5) * (w_norm ** 0.5)
    if denom <= 1e-8:
        return 0.0
    return dot / denom


def _slice_match_score(a: list[list[int]], b: list[list[int]]) -> float:
    if _shape(a) != _shape(b):
        return 0.0
    h, w = _shape(a)
    total = h * w
    if total <= 0:
        return 0.0
    same = 0
    for y in range(h):
        for x in range(w):
            if a[y][x] == b[y][x]:
                same += 1
    return same / total


def _periodicity_score(grid: list[list[int]], window: list[list[int]]) -> float:
    h, w = _shape(grid)
    k, wk = _shape(window)
    if k <= 0 or wk != w or h < k:
        return 0.0
    best = 0.0
    for s in range(0, h - k + 1):
        prev = [row[:] for row in grid[s : s + k]]
        best = max(best, _slice_match_score(prev, window))
    return best


def _select_best_window(grid: list[list[int]], rows: int, prefer: str = "head") -> tuple[int, list[list[int]], float]:
    g = _to_grid(grid)
    h, _ = _shape(g)
    k = max(1, min(h, int(rows)))
    best_s = 0
    best_w = [row[:] for row in g[:k]]
    best_score = -1.0

    for s in range(0, h - k + 1):
        window = [row[:] for row in g[s : s + k]]
        tail = [row[:] for row in g[h - k : h]]
        edge = _slice_match_score(tail, window)
        hist = _color_hist_score(g, window)
        per = _periodicity_score(g, window)
        score = (0.6 * edge) + (0.2 * hist) + (0.2 * per)
        if score > best_score:
            best_s = s
            best_w = window
            best_score = score
        elif abs(score - best_score) < 1e-12:
            if prefer == "head":
                if s < best_s:
                    best_s = s
                    best_w = window
            else:
                if s > best_s:
                    best_s = s
                    best_w = window
    return best_s, best_w, best_score


def _combo_complexity(ids: tuple[int, ...]) -> int:
    if len(ids) <= 1:
        return 0
    return sum(abs(ids[i] - ids[i - 1]) for i in range(1, len(ids)))


def _score_row_set(grid: list[list[int]], rows: list[list[int]]) -> float:
    g = _to_grid(grid)
    k, _ = _shape(rows)
    if k <= 0:
        return 0.0
    h, _ = _shape(g)
    tail = [row[:] for row in g[h - k : h]]
    edge = _slice_match_score(tail, rows)
    hist = _color_hist_score(g, rows)
    per = _periodicity_score(g, rows)
    return (0.6 * edge) + (0.2 * hist) + (0.2 * per)


def _stride_diversity_score(ids: tuple[int, ...]) -> float:
    if len(ids) < 2:
        return 0.0
    gaps = [ids[i] - ids[i - 1] for i in range(1, len(ids))]
    if not gaps:
        return 0.0
    unique = len(set(gaps))
    return unique / float(len(gaps))


def _cross_position_similarity(grid: list[list[int]], rows: list[list[int]]) -> float:
    g = _to_grid(grid)
    k, wk = _shape(rows)
    h, w = _shape(g)
    if k <= 0 or wk != w or h < k:
        return 0.0
    best = 0.0
    for s in range(0, h - k + 1):
        prev = [row[:] for row in g[s : s + k]]
        best = max(best, _slice_match_score(prev, rows))
    return best


def _tail_bias_penalty(ids: tuple[int, ...], h: int) -> float:
    k = len(ids)
    if ids == tuple(range(h - k, h)):
        return -0.3
    return 0.0


def _row_union_coverage(rows: list[list[int]]) -> float:
    if not rows:
        return 0.0
    w = len(rows[0]) if rows and rows[0] else 0
    if w <= 0:
        return 0.0
    union = [0 for _ in range(w)]
    redundancy = [0 for _ in range(w)]
    for row in rows:
        for x, v in enumerate(row):
            if int(v) != 0:
                union[x] = 1
                redundancy[x] += 1
    coverage = sum(union) / float(w)
    redundancy_penalty = sum(1 for c in redundancy if c > 1) / float(w)
    return coverage - (0.5 * redundancy_penalty)


def _contiguous_penalty(ids: tuple[int, ...]) -> float:
    if len(ids) < 2:
        return 0.0
    gaps = [ids[i] - ids[i - 1] for i in range(1, len(ids))]
    if all(g == 1 for g in gaps):
        return -0.2
    return 0.0


def _row_signature_pos(row: list[int]) -> dict[str, Any]:
    key = tuple(int(v) for v in row)
    cached = _ROW_SIG_POS_CACHE.get(key)
    if cached is not None:
        return cached
    hist = [0.0 for _ in range(10)]
    for v in row:
        hist[int(v) % 10] += 1.0
    total = sum(hist)
    if total > 1e-8:
        hist = [h / total for h in hist]

    w = len(row)
    if w <= 0:
        return {"hist": hist, "centroid": 0.0, "spread": 0.0, "trans": 0.0}

    mask = [1.0 if int(v) != 0 else 0.0 for v in row]
    mass = sum(mask)
    if mass > 0:
        numerator = 0.0
        for x, m in enumerate(mask):
            numerator += x * m
        centroid_raw = numerator / mass
        centroid = centroid_raw / max(1, w - 1)

        var_num = 0.0
        for x, m in enumerate(mask):
            d = x - centroid_raw
            var_num += (d * d) * m
        spread = (var_num / mass) / (float(w * w) + 1e-8)
    else:
        centroid = 0.0
        spread = 0.0

    trans = 0.0
    if w > 1:
        diffs = 0
        for i in range(1, w):
            if row[i] != row[i - 1]:
                diffs += 1
        trans = diffs / float(w - 1)

    sig = {"hist": hist, "centroid": centroid, "spread": spread, "trans": trans}
    _ROW_SIG_POS_CACHE[key] = sig
    return sig


def _sig_sim_pos(a: dict[str, Any], b: dict[str, Any]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a["hist"], b["hist"]))
    an = sum(float(x) * float(x) for x in a["hist"]) ** 0.5
    bn = sum(float(y) * float(y) for y in b["hist"]) ** 0.5
    cos = dot / max(1e-8, an * bn)
    pos = 1.0 - abs(float(a["centroid"]) - float(b["centroid"]))
    spr = 1.0 - abs(float(a["spread"]) - float(b["spread"]))
    trn = 1.0 - abs(float(a["trans"]) - float(b["trans"]))
    return (0.5 * cos) + (0.25 * pos) + (0.15 * spr) + (0.10 * trn)


def _build_target_row_sigs_pos(target_grid: list[list[int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _to_grid(target_grid):
        out.append(_row_signature_pos(row))
    return out


def _signature_match_score_pos(rows: list[list[int]], target_sigs_pos: list[dict[str, Any]] | None) -> float:
    if not target_sigs_pos:
        return 0.0
    cand = [_row_signature_pos(row) for row in rows]
    used: set[int] = set()
    score = 0.0
    for cs in cand:
        best_j = None
        best_s = -1.0
        for j, ts in enumerate(target_sigs_pos):
            if j in used:
                continue
            s = _sig_sim_pos(cs, ts)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None:
            used.add(best_j)
            score += best_s
    return score / max(1, len(cand))


def _sequence_match_score(rows: list[list[int]], target_rows: list[list[int]] | None) -> float:
    if not target_rows:
        return 0.0
    k = min(len(rows), len(target_rows))
    if k <= 0:
        return 0.0

    a = [_row_signature_pos(rows[i]) for i in range(k)]
    b = [_row_signature_pos(target_rows[i]) for i in range(k)]

    def _seq_sim(x: list[dict[str, Any]], y: list[dict[str, Any]]) -> float:
        s = 0.0
        for i in range(min(len(x), len(y))):
            s += _sig_sim_pos(x[i], y[i])
        return s / max(1, min(len(x), len(y)))

    return max(_seq_sim(a, b), _seq_sim(list(reversed(a)), b))


def _signature_match_score_pair(rows: list[list[int]], target_rows: list[list[int]] | None) -> float:
    if not target_rows:
        return 0.0
    k = min(len(rows), len(target_rows))
    if k <= 0:
        return 0.0
    a = [_row_signature_pos(rows[i]) for i in range(k)]
    b = [_row_signature_pos(target_rows[i]) for i in range(k)]
    used: set[int] = set()
    score = 0.0
    for x in a:
        best_j = None
        best_s = -1.0
        for j, y in enumerate(b):
            if j in used:
                continue
            s = _sig_sim_pos(x, y)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None:
            used.add(best_j)
            score += best_s
    return score / max(1, k)


def _target_mask_overlap(rows: list[list[int]], target_rows: list[list[int]] | None) -> float:
    if not target_rows or not rows:
        return 0.0
    w = len(rows[0]) if rows[0] else 0
    tw = len(target_rows[0]) if target_rows[0] else 0
    if w <= 0 or tw <= 0 or w != tw:
        return 0.0
    pred_union = [0 for _ in range(w)]
    tgt_union = [0 for _ in range(w)]
    for row in rows:
        for i, v in enumerate(row):
            if int(v) != 0:
                pred_union[i] = 1
    for row in target_rows:
        for i, v in enumerate(row):
            if int(v) != 0:
                tgt_union[i] = 1
    inter = sum(1 for p, t in zip(pred_union, tgt_union) if p == 1 and t == 1)
    union = sum(1 for p, t in zip(pred_union, tgt_union) if p == 1 or t == 1)
    return inter / max(1.0, float(union))


def _best_rowwise_alignment(rows: list[list[int]], target_rows: list[list[int]] | None) -> float:
    if not target_rows or not rows:
        return 0.0
    a = [_row_signature_pos(r) for r in rows]
    b = [_row_signature_pos(r) for r in target_rows]
    score = 0.0
    for x in a:
        best = 0.0
        for y in b:
            best = max(best, _sig_sim_pos(x, y))
        score += best
    return score / max(1, len(a))


def _select_best_rows(
    grid: list[list[int]],
    rows: int,
    prefer: str = "head",
    target_rows_sample: list[list[int]] | None = None,
    max_candidates: int = 5000,
) -> tuple[tuple[int, ...], list[list[int]], float]:
    g = _to_grid(grid)
    h, _ = _shape(g)
    k = max(1, min(h, int(rows)))
    if k > 4:
        start, win, sc = _select_best_window(g, k, prefer=prefer)
        ids = tuple(range(start, start + k))
        return ids, win, sc

    best_ids: tuple[int, ...] = tuple(range(0, k))
    best_rows = [row[:] for row in g[:k]]
    best_score = -1.0
    seen = 0

    for ids in combinations(range(h), k):
        seen += 1
        if seen > max_candidates:
            break
        rows_sel = [g[i][:] for i in ids]
        base = _score_row_set(g, rows_sel)
        cross = _cross_position_similarity(g, rows_sel)
        cont_pen = _contiguous_penalty(ids)
        sig_pair = _signature_match_score_pair(rows_sel, target_rows_sample)
        seq = _sequence_match_score(rows_sel, target_rows_sample)
        overlap = _target_mask_overlap(rows_sel, target_rows_sample)
        align = _best_rowwise_alignment(rows_sel, target_rows_sample)
        sc = (
            (0.08 * base)
            + (0.03 * cross)
            + (0.02 * cont_pen)
            + (0.24 * sig_pair)
            + (0.38 * seq)
            + (0.20 * overlap)
            + (0.05 * align)
        )
        if sc > best_score:
            best_ids = ids
            best_rows = rows_sel
            best_score = sc
            continue
        if abs(sc - best_score) < 1e-12:
            current_complexity = _combo_complexity(ids)
            best_complexity = _combo_complexity(best_ids)
            if current_complexity < best_complexity:
                best_ids = ids
                best_rows = rows_sel
                best_score = sc
                continue
            if current_complexity == best_complexity:
                if prefer == "head":
                    if sum(ids) < sum(best_ids):
                        best_ids = ids
                        best_rows = rows_sel
                        best_score = sc
                else:
                    tail_ids = sum((h - 1 - i) for i in ids)
                    tail_best = sum((h - 1 - i) for i in best_ids)
                    if tail_ids < tail_best:
                        best_ids = ids
                        best_rows = rows_sel
                        best_score = sc
    return best_ids, best_rows, best_score


def _select_best_rows_global(
    grid: list[list[int]],
    rows: int,
    prefer: str = "head",
    max_candidates: int = 5000,
) -> tuple[tuple[int, ...], list[list[int]], float]:
    g = _to_grid(grid)
    h, _ = _shape(g)
    k = max(1, min(h, int(rows)))
    if k > 4:
        start, win, sc = _select_best_window(g, k, prefer=prefer)
        ids = tuple(range(start, start + k))
        return ids, win, sc

    best_ids: tuple[int, ...] = tuple(range(0, k))
    best_rows = [row[:] for row in g[:k]]
    best_score = -1.0
    seen = 0
    for ids in combinations(range(h), k):
        seen += 1
        if seen > max_candidates:
            break
        rows_sel = [g[i][:] for i in ids]
        base = _score_row_set(g, rows_sel)
        cross = _cross_position_similarity(g, rows_sel)
        stride = _stride_diversity_score(ids)
        coverage = _row_union_coverage(rows_sel)
        tail_pen = _tail_bias_penalty(ids, h)
        cont_pen = _contiguous_penalty(ids)
        sc = (
            (0.35 * base)
            + (0.20 * cross)
            + (0.20 * stride)
            + (0.15 * coverage)
            + (0.05 * tail_pen)
            + (0.05 * cont_pen)
        )
        if sc > best_score:
            best_ids = ids
            best_rows = rows_sel
            best_score = sc
            continue
        if abs(sc - best_score) < 1e-12:
            if prefer == "head":
                if sum(ids) < sum(best_ids):
                    best_ids = ids
                    best_rows = rows_sel
                    best_score = sc
            else:
                tail_ids = sum((h - 1 - i) for i in ids)
                tail_best = sum((h - 1 - i) for i in best_ids)
                if tail_ids < tail_best:
                    best_ids = ids
                    best_rows = rows_sel
                    best_score = sc
    return best_ids, best_rows, best_score


def _subset_features(grid: list[list[int]], ids: tuple[int, ...]) -> list[float]:
    g = _to_grid(grid)
    h, _ = _shape(g)
    rows = [g[i][:] for i in ids]
    k = len(ids)
    mean_idx = (sum(ids) / float(max(1, k))) / float(max(1, h - 1))
    span = (max(ids) - min(ids)) / float(max(1, h - 1)) if ids else 0.0
    contiguous = 1.0 if all((ids[i] - ids[i - 1]) == 1 for i in range(1, len(ids))) else 0.0
    head = 1.0 if ids == tuple(range(0, k)) else 0.0
    tail = 1.0 if ids == tuple(range(max(0, h - k), h)) else 0.0
    return [
        1.0,  # bias
        mean_idx,
        span,
        contiguous,
        _stride_diversity_score(ids),
        _score_row_set(g, rows),
        _cross_position_similarity(g, rows),
        _row_union_coverage(rows),
        head,
        tail,
    ]


_RULE_FEATURE_INDEX = {
    "mean_idx": 1,
    "span": 2,
    "contiguous": 3,
    "stride_diversity": 4,
    "base": 5,
    "cross": 6,
    "coverage": 7,
    "head": 8,
    "tail": 9,
}


def _check_rule_condition(feats: list[float], condition: dict[str, Any] | None) -> bool:
    if not condition:
        return True
    fname = str(condition.get("feature", ""))
    op = str(condition.get("op", ""))
    value = float(condition.get("value", 0.0))
    idx = _RULE_FEATURE_INDEX.get(fname)
    if idx is None or idx >= len(feats):
        return True
    x = float(feats[idx])
    if op == "<":
        return x < value
    if op == "<=":
        return x <= value
    if op == ">":
        return x > value
    if op == ">=":
        return x >= value
    if op == "==":
        return abs(x - value) < 1e-9
    return True


def _rule_summary(rule: dict[str, Any] | None) -> str:
    if not rule:
        return "rule:none"
    if str(rule.get("type", "")) == "row_composite_v1":
        cond = rule.get("condition", {})
        ctxt = f"{cond.get('feature')} {cond.get('op')} {cond.get('value')}"
        t = rule.get("rule_true", {})
        f = rule.get("rule_false", {})
        conf = float(rule.get("confidence", 0.0))
        return (
            f"rule:row_composite_v1 conf={conf:.3f} IF {ctxt} "
            f"THEN {t.get('type','row_rule')}[{t.get('transform','identity')}] "
            f"ELSE {f.get('type','row_rule')}[{f.get('transform','identity')}]"
        )
    weights = [float(x) for x in rule.get("weights", [])]
    conf = float(rule.get("confidence", 0.0))
    cond = rule.get("condition")
    tr = str(rule.get("transform", "identity"))
    cond_txt = "none"
    if isinstance(cond, dict):
        cond_txt = f"{cond.get('feature')} {cond.get('op')} {cond.get('value')}"
    return f"rule:{rule.get('type','row_linear')} conf={conf:.3f} cond={cond_txt} tf={tr} w={weights}"


def _resolve_rule_leaf(rule: dict[str, Any], grid: list[list[int]]) -> dict[str, Any]:
    r = dict(rule)
    if str(r.get("type", "")) != "row_composite_v1":
        return r
    cond = r.get("condition", {})
    feature = str(cond.get("feature", "fg_density"))
    op = str(cond.get("op", "<"))
    threshold = float(cond.get("value", 0.5))
    val = _extract_global_feature(grid, feature)
    go_true = val < threshold if op == "<" else val >= threshold
    branch = r.get("rule_true") if go_true else r.get("rule_false")
    if isinstance(branch, dict):
        return _resolve_rule_leaf(branch, grid)
    return r


def _apply_transform_rows(rows: list[list[int]], transform: str) -> list[list[int]]:
    tr = str(transform or "identity")
    out = [row[:] for row in rows]
    if tr == "flip_h":
        return [list(reversed(row)) for row in out]
    if tr == "flip_v":
        return list(reversed(out))
    return out


def _execute_rule_program_window(
    grid: list[list[int]],
    n: int,
    prefer: str,
    rule_program: list[dict[str, Any]],
) -> tuple[list[list[int]], float]:
    current = [row[:] for row in _to_grid(grid)]
    score_acc = 0.0
    for step_rule in rule_program:
        leaf = _resolve_rule_leaf(step_rule, current)
        _, rows, sc = select_subset(current, n, target_grid=None, mode="row", rule=leaf, prefer=prefer)
        rows = _apply_transform_rows(rows, str(leaf.get("transform", "identity")))
        current = [row[:] for row in rows]
        score_acc += float(sc)
    return current, score_acc


def _program_to_hypothesis(program: ProgramRule) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    for step in program.steps:
        if isinstance(step, TransformRule):
            chain.append({"op": str(step.op), "param": 0})
        elif isinstance(step, dict):
            chain.extend(_normalize_hyp(step))
    return {"chain": chain}


def _generate_programs(base_rules: list[dict[str, Any]]) -> list[ProgramRule]:
    transform_ops = ["identity", "flip_h", "flip_v", "transpose", "rotate_90"]
    programs: list[ProgramRule] = []
    for t in transform_ops:
        programs.append(ProgramRule([TransformRule(t)]))
    for t1 in transform_ops:
        for t2 in transform_ops:
            programs.append(ProgramRule([TransformRule(t1), TransformRule(t2)]))
    for r in base_rules:
        for t in transform_ops:
            programs.append(ProgramRule([TransformRule(t), r]))
            programs.append(ProgramRule([r, TransformRule(t)]))
    return programs


def _select_best_rows_by_rule(
    grid: list[list[int]],
    rows: int,
    rule: dict[str, Any],
    prefer: str = "head",
    max_candidates: int = 5000,
) -> tuple[tuple[int, ...], list[list[int]], float]:
    g = _to_grid(grid)
    h, _ = _shape(g)
    k = max(1, min(h, int(rows)))
    if str(rule.get("type", "")) == "row_composite_v1":
        cond = rule.get("condition", {})
        feature = str(cond.get("feature", "fg_density"))
        op = str(cond.get("op", "<"))
        threshold = float(cond.get("value", 0.5))
        val = _extract_global_feature(g, feature)
        go_true = val < threshold if op == "<" else val >= threshold
        branch = rule.get("rule_true") if go_true else rule.get("rule_false")
        if isinstance(branch, dict):
            return _select_best_rows_by_rule(g, k, branch, prefer=prefer, max_candidates=max_candidates)
        return _select_best_rows_global(g, k, prefer=prefer, max_candidates=max_candidates)
    if k > 4:
        return _select_best_rows_global(g, k, prefer=prefer, max_candidates=max_candidates)

    w = [float(x) for x in rule.get("weights", [])]
    if not w:
        return _select_best_rows_global(g, k, prefer=prefer, max_candidates=max_candidates)
    condition = rule.get("condition") if isinstance(rule, dict) else None

    best_ids: tuple[int, ...] = tuple(range(0, k))
    best_rows = [row[:] for row in g[:k]]
    best_score = -1e18
    seen = 0
    accepted = 0
    for ids in combinations(range(h), k):
        seen += 1
        if seen > max_candidates:
            break
        feats = _subset_features(g, ids)
        if not _check_rule_condition(feats, condition):
            continue
        accepted += 1
        score = sum(wi * fi for wi, fi in zip(w, feats))
        if score > best_score:
            best_ids = ids
            best_rows = [g[i][:] for i in ids]
            best_score = score
            continue
        if abs(score - best_score) < 1e-12:
            if prefer == "head":
                if sum(ids) < sum(best_ids):
                    best_ids = ids
                    best_rows = [g[i][:] for i in ids]
                    best_score = score
            else:
                tail_ids = sum((h - 1 - i) for i in ids)
                tail_best = sum((h - 1 - i) for i in best_ids)
                if tail_ids < tail_best:
                    best_ids = ids
                    best_rows = [g[i][:] for i in ids]
                    best_score = score
    if accepted == 0:
        return _select_best_rows_global(g, k, prefer=prefer, max_candidates=max_candidates)
    return best_ids, best_rows, best_score


def _extract_global_feature(grid: list[list[int]], name: str) -> float:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h <= 0 or w <= 0:
        return 0.0
    if name == "fg_density":
        nonzero = sum(1 for row in g for v in row if int(v) != 0)
        return nonzero / float(h * w)
    if name == "height":
        return float(h)
    if name == "width":
        return float(w)
    if name == "centroid_y":
        ys: list[int] = []
        for y in range(h):
            for x in range(w):
                if int(g[y][x]) != 0:
                    ys.append(y)
        if not ys:
            return 0.0
        return (sum(ys) / float(len(ys))) / float(max(1, h - 1))
    if name == "centroid_x":
        xs: list[int] = []
        for y in range(h):
            for x in range(w):
                if int(g[y][x]) != 0:
                    xs.append(x)
        if not xs:
            return 0.0
        return (sum(xs) / float(len(xs))) / float(max(1, w - 1))
    return 0.0


def _infer_best_single_rule_for_group(
    examples: list[dict[str, Any]],
    chain: list[dict[str, Any]],
    pairs: list[tuple[list[list[int]], list[list[int]]]],
    k: int,
) -> dict[str, Any] | None:
    candidates = _build_candidate_row_rules(examples, k=k)
    if not candidates:
        return None
    scored: list[dict[str, Any]] = []
    for r in candidates:
        mean, var, conf = _rule_consistency(r, chain, pairs)
        rr = dict(r)
        rr["mean_score"] = mean
        rr["var_score"] = var
        rr["confidence"] = conf
        scored.append(rr)
    scored.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return scored[0] if scored else None


def _infer_partitioned_rule(
    chain: list[dict[str, Any]],
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
    examples_by_pair: list[dict[str, Any]],
    k: int,
) -> dict[str, Any] | None:
    if len(train_pairs) < 3 or len(examples_by_pair) < 3:
        return None
    feature_names = ["fg_density", "centroid_y", "centroid_x", "width", "height"]
    thresholds = [0.2, 0.4, 0.6, 0.8]
    best_rule: dict[str, Any] | None = None
    best_conf = -1e18

    for feat in feature_names:
        vals = [_extract_global_feature(p[0], feat) for p in train_pairs]
        local_thresholds = thresholds
        # width/height are absolute, so derive from data range too.
        if feat in {"width", "height"}:
            lo, hi = min(vals), max(vals)
            if hi > lo:
                local_thresholds = [lo + (hi - lo) * t for t in [0.25, 0.5, 0.75]]
            else:
                continue
        for thr in local_thresholds:
            idx_a: list[int] = []
            idx_b: list[int] = []
            for i, (inp, _) in enumerate(train_pairs):
                v = _extract_global_feature(inp, feat)
                if v < thr:
                    idx_a.append(i)
                else:
                    idx_b.append(i)
            if not idx_a or not idx_b:
                continue

            ex_a = [examples_by_pair[i] for i in idx_a if i < len(examples_by_pair)]
            ex_b = [examples_by_pair[i] for i in idx_b if i < len(examples_by_pair)]
            pairs_a = [train_pairs[i] for i in idx_a]
            pairs_b = [train_pairs[i] for i in idx_b]
            rule_a = _infer_best_single_rule_for_group(ex_a, chain, pairs_a, k)
            rule_b = _infer_best_single_rule_for_group(ex_b, chain, pairs_b, k)
            if rule_a is None or rule_b is None:
                continue

            composite = {
                "type": "row_composite_v1",
                "mode": "row",
                "k": int(k),
                "condition": {"feature": feat, "op": "<", "value": float(thr)},
                "rule_true": rule_a,
                "rule_false": rule_b,
            }
            mean, var, conf = _rule_consistency(composite, chain, train_pairs)
            composite["mean_score"] = mean
            composite["var_score"] = var
            composite["confidence"] = conf
            if conf > best_conf:
                best_conf = conf
                best_rule = composite
    return best_rule


def _build_candidate_row_rules(examples: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    learned = _infer_row_subset_rule_from_examples(examples)
    if learned:
        learned["mode"] = "row"
        learned["k"] = int(k)
        candidates.append(learned)

    # Small fixed candidate family to avoid overfitting one weight fit.
    templates = [
        # balanced structure
        ([0.0, 0.0, 0.2, -0.2, 0.4, 0.2, 0.2, 0.3, 0.0, 0.0], None),
        # head-biased
        ([0.0, -0.5, -0.2, 0.3, 0.1, 0.2, 0.2, 0.2, 0.6, -0.2], {"feature": "mean_idx", "op": "<=", "value": 0.6}),
        # tail-biased
        ([0.0, 0.5, -0.2, 0.3, 0.1, 0.2, 0.2, 0.2, -0.2, 0.6], {"feature": "mean_idx", "op": ">=", "value": 0.4}),
        # non-contiguous preference
        ([0.0, 0.0, 0.4, -0.4, 0.4, 0.15, 0.2, 0.25, 0.0, 0.0], {"feature": "contiguous", "op": "<=", "value": 0.5}),
    ]
    for t, cond in templates:
        candidates.append(
            {
                "type": "row_linear_v2",
                "mode": "row",
                "k": int(k),
                "weights": t,
                "condition": cond,
                "order_sensitive": True,
            }
        )
    return candidates


def _inject_rule_into_chain(chain: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in chain:
        s = dict(step)
        if s.get("op") in {"append_head_rows", "append_tail_rows"} and s.get("selector") == "best_rows":
            s["rule"] = rule
        out.append(s)
    return out


def _apply_chain(grid: list[list[int]], chain: list[dict[str, Any]], target_grid: list[list[int]] | None = None) -> list[list[int]]:
    out = [row[:] for row in _to_grid(grid)]
    for step in chain:
        out = _apply_single_op(out, step, target_grid=target_grid)
    return out


def _rule_consistency(
    rule: dict[str, Any],
    chain: list[dict[str, Any]],
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
) -> tuple[float, float, float]:
    rule_chain = _inject_rule_into_chain(chain, rule)
    scores: list[float] = []
    for inp, tgt in train_pairs:
        pred = _apply_chain(inp, rule_chain, target_grid=None)
        scores.append(_score_grid(pred, tgt))
    if not scores:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / float(len(scores))
    var = sum((s - mean) * (s - mean) for s in scores) / float(len(scores))
    conf = mean - var
    return mean, var, conf


def _generate_rule_programs(base_rules: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    transforms = ["identity", "flip_h", "flip_v"]
    expanded: list[dict[str, Any]] = []
    for r in base_rules:
        for tr in transforms:
            rr = dict(r)
            rr["transform"] = tr
            expanded.append(rr)
    programs: list[list[dict[str, Any]]] = []
    for r in expanded:
        programs.append([r])
    for i, r1 in enumerate(expanded):
        for j, r2 in enumerate(expanded):
            if i == j and r1.get("transform") == r2.get("transform"):
                continue
            programs.append([r1, r2])
    return programs


def _score_rule_program(
    chain: list[dict[str, Any]],
    program: list[dict[str, Any]],
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
) -> tuple[float, float]:
    scores: list[float] = []
    for inp, tgt in train_pairs:
        current = [row[:] for row in _to_grid(inp)]
        for step in chain:
            op = step.get("op")
            if op in {"append_head_rows", "append_tail_rows"} and step.get("selector") == "best_rows":
                n = int(step.get("rows", step.get("param", 0)) or 0)
                if n <= 0:
                    _, w = _shape(current)
                    n = w
                n = max(1, min(len(current), n))
                prefer = "tail" if op == "append_tail_rows" else "head"
                window, _ = _execute_rule_program_window(current, n, prefer=prefer, rule_program=program)
                current = [row[:] for row in current] + [row[:] for row in window]
            else:
                current = _apply_single_op(current, step, target_grid=None)
        scores.append(_score_grid(current, _to_grid(tgt)))
    if not scores:
        return 0.0, -1e18
    mean = sum(scores) / float(len(scores))
    penalized = mean - (0.05 * float(len(program)))
    return mean, penalized


def _infer_best_rule_program(
    chain: list[dict[str, Any]],
    base_rules: list[dict[str, Any]],
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
    baseline_mean: float,
) -> dict[str, Any] | None:
    if not base_rules:
        return None
    programs = _generate_rule_programs(base_rules[:3])
    best_prog = None
    best_pen = baseline_mean
    for p in programs:
        mean, pen = _score_rule_program(chain, p, train_pairs)
        if pen > best_pen:
            best_pen = pen
            best_prog = {
                "steps": p,
                "mean_score": mean,
                "penalized_score": pen,
                "num_steps": len(p),
            }
    return best_prog


def _infer_row_subset_rule_from_examples(
    examples: list[dict[str, Any]],
    max_candidates: int = 5000,
) -> dict[str, Any] | None:
    if not examples:
        return None
    dim = 10
    weights = [0.0 for _ in range(dim)]
    used = 0

    for ex in examples:
        g = _to_grid(ex["grid"])
        ids_pos = tuple(int(i) for i in ex["ids"])
        h, _ = _shape(g)
        k = len(ids_pos)
        if k <= 0 or k > 4:
            continue
        pos = _subset_features(g, ids_pos)

        neg_sum = [0.0 for _ in range(dim)]
        neg_count = 0
        seen = 0
        for ids in combinations(range(h), k):
            seen += 1
            if seen > max_candidates:
                break
            if ids == ids_pos:
                continue
            f = _subset_features(g, ids)
            for i in range(dim):
                neg_sum[i] += f[i]
            neg_count += 1
        if neg_count == 0:
            continue
        neg_avg = [v / float(neg_count) for v in neg_sum]
        for i in range(dim):
            weights[i] += pos[i] - neg_avg[i]
        used += 1

    if used == 0:
        return None
    weights = [w / float(used) for w in weights]
    return {"type": "row_linear_v1", "weights": weights}


def select_subset(
    grid: list[list[int]],
    k: int,
    target_grid: list[list[int]] | None = None,
    mode: str = "row",
    rule: dict[str, Any] | None = None,
    prefer: str = "head",
) -> tuple[tuple[int, ...], list[list[int]], float]:
    if mode != "row":
        raise ValueError(f"Unsupported subset mode: {mode}")
    if target_grid is None and rule:
        return _select_best_rows_by_rule(grid, k, rule=rule, prefer=prefer)
    if target_grid is None:
        return _select_best_rows_global(grid, k, prefer=prefer)
    target_rows_sample = _to_grid(target_grid)[-max(1, min(len(_to_grid(target_grid)), int(k))):]
    return _select_best_rows(grid, k, prefer=prefer, target_rows_sample=target_rows_sample)


def _append_head_rows_selected(
    grid: list[list[int]],
    rows: int = 0,
    selector: str | None = None,
    rule: dict[str, Any] | None = None,
    rule_ensemble: list[dict[str, Any]] | None = None,
    rule_program: list[dict[str, Any]] | None = None,
    target_grid: list[list[int]] | None = None,
) -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0:
        return [row[:] for row in g]
    n = int(rows)
    if n <= 0:
        n = w
    n = max(1, min(h, n))
    if target_grid is None and rule_program:
        window, _ = _execute_rule_program_window(g, n, prefer="head", rule_program=rule_program)
        return [row[:] for row in g] + window
    if target_grid is None and isinstance(rule, dict) and float(rule.get("confidence", 0.0)) < 0.6:
        _, window, _ = _select_best_rows_global(g, n, prefer="head")
        return [row[:] for row in g] + window
    if str(selector) == "best_window":
        _, window, _ = _select_best_window(g, n, prefer="head")
    elif str(selector) == "best_rows":
        if target_grid is None and rule_ensemble:
            best_score = -1e18
            best_window = None
            for r in rule_ensemble:
                _, wsel, sc = select_subset(g, n, target_grid=None, mode="row", rule=r, prefer="head")
                if sc > best_score:
                    best_score = sc
                    best_window = wsel
            window = best_window if best_window is not None else [row[:] for row in g[:n]]
        else:
            _, window, _ = select_subset(g, n, target_grid=target_grid, mode="row", rule=rule, prefer="head")
    else:
        window = [row[:] for row in g[:n]]
    return [row[:] for row in g] + window


def _append_tail_rows_selected(
    grid: list[list[int]],
    rows: int = 0,
    selector: str | None = None,
    rule: dict[str, Any] | None = None,
    rule_ensemble: list[dict[str, Any]] | None = None,
    rule_program: list[dict[str, Any]] | None = None,
    target_grid: list[list[int]] | None = None,
) -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0:
        return [row[:] for row in g]
    n = int(rows)
    if n <= 0:
        n = w
    n = max(1, min(h, n))
    if target_grid is None and rule_program:
        window, _ = _execute_rule_program_window(g, n, prefer="tail", rule_program=rule_program)
        return [row[:] for row in g] + window
    if target_grid is None and isinstance(rule, dict) and float(rule.get("confidence", 0.0)) < 0.6:
        _, window, _ = _select_best_rows_global(g, n, prefer="tail")
        return [row[:] for row in g] + window
    if str(selector) == "best_window":
        _, window, _ = _select_best_window(g, n, prefer="tail")
    elif str(selector) == "best_rows":
        if target_grid is None and rule_ensemble:
            best_score = -1e18
            best_window = None
            for r in rule_ensemble:
                _, wsel, sc = select_subset(g, n, target_grid=None, mode="row", rule=r, prefer="tail")
                if sc > best_score:
                    best_score = sc
                    best_window = wsel
            window = best_window if best_window is not None else [row[:] for row in g[h - n : h]]
        else:
            _, window, _ = select_subset(g, n, target_grid=target_grid, mode="row", rule=rule, prefer="tail")
    else:
        window = [row[:] for row in g[h - n : h]]
    return [row[:] for row in g] + window


def _recolor_nonzero_to(grid: list[list[int]], color: int = 2) -> list[list[int]]:
    g = _to_grid(grid)
    c = int(color) % 10
    if c == 0:
        c = 1
    return [[c if v != 0 else 0 for v in row] for row in g]


def _self_tile_mask(grid: list[list[int]]) -> list[list[int]]:
    return _self_tile_mask_param(grid, mode="mask_fill", axis="both", factor=0)


def _safe_factor(factor: int) -> int:
    if factor <= 0:
        return 1
    return min(9, int(factor))


def _self_tile_mask_param(grid: list[list[int]], mode: str, axis: str, factor: int) -> list[list[int]]:
    g = _to_grid(grid)
    h, w = _shape(g)
    mode = str(mode)
    axis = str(axis)
    f = _safe_factor(int(factor))

    if mode == "mask_fill":
        # Legacy behavior when factor is not set: use source grid dims.
        rep_y = h if factor <= 0 else f
        rep_x = w if factor <= 0 else f
        out_h, out_w = h * rep_y, w * rep_x
        out = [[0 for _ in range(out_w)] for _ in range(out_h)]
        for by in range(rep_y):
            for bx in range(rep_x):
                if g[by % h][bx % w] == 0:
                    continue
                off_y = by * h
                off_x = bx * w
                for y in range(h):
                    for x in range(w):
                        out[off_y + y][off_x + x] = g[y][x]
        return out

    rep_y = f if axis in ("y", "both") else 1
    rep_x = f if axis in ("x", "both") else 1
    out_h, out_w = h * rep_y, w * rep_x
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for by in range(rep_y):
        for bx in range(rep_x):
            tile = g
            if mode == "mirror":
                if bx % 2 == 1:
                    tile = _flip_h(tile)
                if by % 2 == 1:
                    tile = _flip_v(tile)
            elif mode != "repeat":
                tile = g
            off_y = by * h
            off_x = bx * w
            for y in range(h):
                for x in range(w):
                    out[off_y + y][off_x + x] = tile[y][x]
    return out


def _apply_single_op(
    grid: list[list[int]],
    step: dict[str, Any],
    target_grid: list[list[int]] | None = None,
) -> list[list[int]]:
    op = str(step.get("op", "identity"))
    param = int(step.get("param", 0))
    if op == "identity":
        return [row[:] for row in grid]
    if op == "flip_h":
        return _flip_h(grid)
    if op == "flip_v":
        return _flip_v(grid)
    if op == "rotate_90":
        return _rot90(grid)
    if op == "transpose":
        return _transpose(grid)
    if op == "color_shift":
        return _color_shift(grid, param)
    if op == "move_to_corner":
        return _move_to_corner(grid)
    if op == "center_object":
        return _center_object(grid)
    if op == "duplicate_object":
        return _duplicate_object(grid)
    if op == "remove_smallest":
        return _remove_smallest(grid)
    if op == "crop_to_bbox":
        return _crop_to_bbox(grid)
    if op == "paint_bbox":
        return _paint_bbox(grid)
    if op == "object_recolor_by_rank":
        return _object_recolor_by_rank(grid)
    if op == "extract_repeating_unit":
        axis = str(step.get("axis", "y"))
        return _extract_repeating_unit(grid, axis=axis)
    if op == "append_head_rows":
        rows = int(step.get("rows", step.get("param", 0)))
        selector = step.get("selector")
        rule = step.get("rule")
        rule_ensemble = step.get("rule_ensemble")
        rule_program = step.get("rule_program")
        if selector is None:
            return _append_head_rows(grid, rows=rows)
        return _append_head_rows_selected(
            grid,
            rows=rows,
            selector=str(selector),
            rule=rule if isinstance(rule, dict) else None,
            rule_ensemble=rule_ensemble if isinstance(rule_ensemble, list) else None,
            rule_program=rule_program if isinstance(rule_program, list) else None,
            target_grid=target_grid,
        )
    if op == "append_tail_rows":
        rows = int(step.get("rows", step.get("param", 0)))
        selector = step.get("selector")
        rule = step.get("rule")
        rule_ensemble = step.get("rule_ensemble")
        rule_program = step.get("rule_program")
        if selector is None:
            return _append_tail_rows(grid, rows=rows)
        return _append_tail_rows_selected(
            grid,
            rows=rows,
            selector=str(selector),
            rule=rule if isinstance(rule, dict) else None,
            rule_ensemble=rule_ensemble if isinstance(rule_ensemble, list) else None,
            rule_program=rule_program if isinstance(rule_program, list) else None,
            target_grid=target_grid,
        )
    if op == "recolor_nonzero_to":
        color = int(step.get("color", step.get("param", 2)))
        return _recolor_nonzero_to(grid, color=color)
    if op == "self_tile_mask":
        mode = str(step.get("mode", "mask_fill"))
        axis = str(step.get("axis", "both"))
        factor = int(step.get("factor", 0))
        return _self_tile_mask_param(grid, mode=mode, axis=axis, factor=factor)
    return [row[:] for row in grid]


def _shape_similarity(a: list[list[int]], b: list[list[int]]) -> float:
    if _shape(a) != _shape(b):
        return 0.0
    h, w = _shape(a)
    total = h * w
    if total == 0:
        return 0.0
    same = 0
    for y in range(h):
        for x in range(w):
            if a[y][x] == b[y][x]:
                same += 1
    return same / total


def _object_similarity(obj_a: dict[str, Any], obj_b: dict[str, Any]) -> float:
    score = 0.0
    score += _shape_similarity(obj_a["shape"], obj_b["shape"]) * 2.0
    if int(obj_a["color"]) == int(obj_b["color"]):
        score += 1.0
    sa = max(1, int(obj_a["size"]))
    sb = max(1, int(obj_b["size"]))
    score += min(sa, sb) / max(sa, sb)
    return score


def _match_objects(input_objs: list[dict[str, Any]], output_objs: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    used_out: set[int] = set()
    for i, obj_in in enumerate(input_objs):
        best_j = None
        best_score = -1.0
        for j, obj_out in enumerate(output_objs):
            if j in used_out:
                continue
            s = _object_similarity(obj_in, obj_out)
            if s > best_score:
                best_score = s
                best_j = j
        if best_j is not None:
            mapping[i] = best_j
            used_out.add(best_j)
    return mapping


def learn_selection(input_objs: list[dict[str, Any]], output_objs: list[dict[str, Any]]) -> set[int]:
    selected: set[int] = set()
    for i, obj_in in enumerate(input_objs):
        for obj_out in output_objs:
            if _object_similarity(obj_in, obj_out) > 2.5:
                selected.add(i)
                break
    return selected


def _learn_selection_rule(input_objs: list[dict[str, Any]], selected_ids: set[int]) -> dict[str, Any]:
    if not input_objs:
        return {"type": "all"}
    if not selected_ids:
        return {"type": "all"}
    if len(selected_ids) == len(input_objs):
        return {"type": "all"}

    sel = [input_objs[i] for i in sorted(selected_ids) if 0 <= i < len(input_objs)]
    if not sel:
        return {"type": "all"}

    # Simple compact rule space for first pass.
    if len(sel) == 1:
        target = sel[0]
        max_area = max(int(o.get("area", 0)) for o in input_objs)
        min_area = min(int(o.get("area", 0)) for o in input_objs)
        if int(target.get("area", 0)) == max_area:
            return {"type": "largest"}
        if int(target.get("area", 0)) == min_area:
            return {"type": "smallest"}

    colors = {int(o.get("color", 0)) for o in sel}
    if len(colors) == 1:
        return {"type": "color", "value": next(iter(colors))}

    if all(bool(o.get("is_leftmost", False)) for o in sel):
        return {"type": "leftmost"}
    if all(bool(o.get("is_topmost", False)) for o in sel):
        return {"type": "topmost"}

    return {"type": "all"}


def _apply_selection_rule(input_objs: list[dict[str, Any]], rule: dict[str, Any]) -> set[int]:
    rtype = str(rule.get("type", "all"))
    if rtype == "all":
        return set(range(len(input_objs)))
    if rtype == "color":
        c = int(rule.get("value", 0))
        return {i for i, o in enumerate(input_objs) if int(o.get("color", -1)) == c}
    if rtype == "leftmost":
        return {i for i, o in enumerate(input_objs) if bool(o.get("is_leftmost", False))}
    if rtype == "topmost":
        return {i for i, o in enumerate(input_objs) if bool(o.get("is_topmost", False))}
    if rtype == "largest":
        if not input_objs:
            return set()
        max_area = max(int(o.get("area", 0)) for o in input_objs)
        return {i for i, o in enumerate(input_objs) if int(o.get("area", 0)) == max_area}
    if rtype == "smallest":
        if not input_objs:
            return set()
        min_area = min(int(o.get("area", 0)) for o in input_objs)
        return {i for i, o in enumerate(input_objs) if int(o.get("area", 0)) == min_area}
    return set(range(len(input_objs)))


def _learn_transform(
    input_objs: list[dict[str, Any]],
    output_objs: list[dict[str, Any]],
    mapping: dict[int, int],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for i, j in mapping.items():
        obj_in = input_objs[i]
        obj_out = output_objs[j]
        dy = int(obj_out["center"][0]) - int(obj_in["center"][0])
        dx = int(obj_out["center"][1]) - int(obj_in["center"][1])
        rules.append(
            {
                "source_shape": obj_in["shape"],
                "source_color": int(obj_in["color"]),
                "translate": (dy, dx),
                "recolor": int(obj_out["color"]),
            }
        )
    return rules


def _score_grid(pred: list[list[int]], out: list[list[int]]) -> float:
    if _shape(pred) != _shape(out):
        return 0.0
    h, w = _shape(pred)
    total = h * w
    if total == 0:
        return 0.0
    same = 0
    for y in range(h):
        for x in range(w):
            if pred[y][x] == out[y][x]:
                same += 1
    return same / total


def apply_rules_selected(
    input_grid: Any,
    input_objs: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    selected_ids: set[int],
) -> list[list[int]]:
    src = _to_grid(input_grid)
    h, w = _shape(src)
    out = [[0 for _ in range(w)] for _ in range(h)]
    if not input_objs or not rules:
        return out

    for idx, obj in enumerate(input_objs):
        if idx not in selected_ids:
            continue
        best_rule = None
        best_score = -1.0
        for rule in rules:
            sim = 0.0
            sim += _shape_similarity(obj["shape"], rule["source_shape"]) * 2.0
            if int(obj["color"]) == int(rule["source_color"]):
                sim += 1.0
            if sim > best_score:
                best_score = sim
                best_rule = rule
        if best_rule is None:
            continue
        dy, dx = best_rule["translate"]
        recolor = int(best_rule["recolor"])
        for y, x in obj["coords"]:
            ny = y + int(dy)
            nx = x + int(dx)
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = recolor
    return out


def apply_rules(input_grid: Any, rules: list[dict[str, Any]], selection_rule: dict[str, Any] | None = None) -> list[list[int]]:
    src = _to_grid(input_grid)
    objs = compute_relations(enrich_objects(extract_objects(src)))
    selected = _apply_selection_rule(objs, selection_rule or {"type": "all"})
    return apply_rules_selected(src, objs, rules, selected)


def _mapping_rule_set(train_pairs: list[tuple[Any, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rules: list[dict[str, Any]] = []
    selection_rules: list[dict[str, Any]] = []
    for inp, out in train_pairs:
        in_objs = compute_relations(enrich_objects(extract_objects(_to_grid(inp))))
        out_objs = compute_relations(enrich_objects(extract_objects(_to_grid(out))))
        mapping = _match_objects(in_objs, out_objs)
        selected = learn_selection(in_objs, out_objs)
        rules = _learn_transform(in_objs, out_objs, mapping)
        selection_rules.append(_learn_selection_rule(in_objs, selected))
        all_rules.extend(rules)
    if selection_rules:
        # Majority rule by string key.
        counter: dict[str, int] = {}
        by_key: dict[str, dict[str, Any]] = {}
        for r in selection_rules:
            k = json.dumps(r, sort_keys=True)
            counter[k] = counter.get(k, 0) + 1
            by_key[k] = r
        best_key = max(counter, key=counter.get)
        return all_rules, by_key[best_key]
    return all_rules, {"type": "all"}


def _mapping_score(task: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    train = task.get("train", [])
    pairs: list[tuple[Any, Any]] = []
    for pair in train:
        if isinstance(pair, dict):
            pairs.append((pair.get("input"), pair.get("output")))
        else:
            pairs.append((pair[0], pair[1]))
    if not pairs:
        return 0.0, []

    # Leave-one-out for less optimistic scoring.
    scores: list[float] = []
    final_rules, final_selection_rule = _mapping_rule_set(pairs)
    for i, (inp, out) in enumerate(pairs):
        if len(pairs) > 1:
            other = pairs[:i] + pairs[i + 1 :]
            rules, selection_rule = _mapping_rule_set(other)
            if not rules:
                rules = final_rules
                selection_rule = final_selection_rule
        else:
            rules = final_rules
            selection_rule = final_selection_rule
        pred = apply_rules(inp, rules, selection_rule=selection_rule)
        scores.append(_score_grid(pred, _to_grid(out)))
    return (sum(scores) / len(scores)), [{"rules": final_rules, "selection_rule": final_selection_rule}]


def _normalize_hyp(hyp: dict[str, Any]) -> list[dict[str, Any]]:
    # Backward compatible with single-op hypothesis.
    if "chain" in hyp and isinstance(hyp["chain"], list):
        chain = []
        for step in hyp["chain"]:
            nstep = {"op": str(step.get("op", "identity")), "param": int(step.get("param", 0))}
            for key in ("mode", "axis", "selector"):
                if key in step:
                    nstep[key] = str(step.get(key))
            for key in ("factor", "rows", "color"):
                if key in step:
                    nstep[key] = int(step.get(key, 0))
            if "rule" in step and isinstance(step.get("rule"), dict):
                nstep["rule"] = step.get("rule")
            if "rule_ensemble" in step and isinstance(step.get("rule_ensemble"), list):
                nstep["rule_ensemble"] = step.get("rule_ensemble")
            if "rule_program" in step and isinstance(step.get("rule_program"), list):
                nstep["rule_program"] = step.get("rule_program")
            if "rule_summary" in step:
                nstep["rule_summary"] = str(step.get("rule_summary"))
            if "rule_type" in step:
                nstep["rule_type"] = str(step.get("rule_type"))
            if "program_summary" in step and isinstance(step.get("program_summary"), dict):
                nstep["program_summary"] = step.get("program_summary")
            chain.append(nstep)
        return chain
    return [{"op": str(hyp.get("op", "identity")), "param": int(hyp.get("param", 0))}]


def apply_hypothesis(grid: Any, hyp: dict[str, Any], target_grid: Any | None = None) -> list[list[int]]:
    if hyp.get("op") == "template_match":
        packed = hyp.get("rules", [])
        if packed and isinstance(packed[0], dict) and "rules" in packed[0]:
            rules = packed[0].get("rules", [])
            selection_rule = packed[0].get("selection_rule", {"type": "all"})
            return apply_rules(grid, rules, selection_rule=selection_rule)
        return apply_rules(grid, packed)
    g = _to_grid(grid)
    t = _to_grid(target_grid) if target_grid is not None else None
    out = [row[:] for row in g]
    for step in _normalize_hyp(hyp):
        out = _apply_single_op(out, step, target_grid=t)
    return out


def _baseline_steps_for_op(op: str) -> list[dict[str, Any]]:
    if op == "color_shift":
        return [{"op": op, "param": p} for p in [0, 1, 2, 3, 5, 7, 9]]
    if op == "extract_repeating_unit":
        return [
            {"op": op, "param": 0, "axis": "y"},
            {"op": op, "param": 0, "axis": "x"},
        ]
    if op == "append_head_rows":
        return [
            {"op": op, "param": 0, "rows": 0},
            {"op": op, "param": 1, "rows": 1},
            {"op": op, "param": 2, "rows": 2},
            {"op": op, "param": 3, "rows": 3},
            {"op": op, "param": 3, "rows": 3, "selector": "best_window"},
            {"op": op, "param": 3, "rows": 3, "selector": "best_rows"},
        ]
    if op == "append_tail_rows":
        return [
            {"op": op, "param": 0, "rows": 0},
            {"op": op, "param": 1, "rows": 1},
            {"op": op, "param": 2, "rows": 2},
            {"op": op, "param": 3, "rows": 3},
            {"op": op, "param": 3, "rows": 3, "selector": "best_window"},
            {"op": op, "param": 3, "rows": 3, "selector": "best_rows"},
        ]
    if op == "recolor_nonzero_to":
        return [{"op": op, "param": c, "color": c} for c in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    if op == "self_tile_mask":
        return [
            {"op": op, "param": 0, "mode": "mask_fill", "axis": "both", "factor": 0},
            {"op": op, "param": 0, "mode": "mask_fill", "axis": "both", "factor": 2},
            {"op": op, "param": 0, "mode": "mask_fill", "axis": "both", "factor": 3},
            {"op": op, "param": 0, "mode": "repeat", "axis": "x", "factor": 2},
            {"op": op, "param": 0, "mode": "repeat", "axis": "y", "factor": 2},
            {"op": op, "param": 0, "mode": "repeat", "axis": "y", "factor": 3},
            {"op": op, "param": 0, "mode": "repeat", "axis": "both", "factor": 2},
            {"op": op, "param": 0, "mode": "repeat", "axis": "both", "factor": 3},
            {"op": op, "param": 0, "mode": "mirror", "axis": "both", "factor": 2},
            {"op": op, "param": 0, "mode": "mirror", "axis": "x", "factor": 2},
            {"op": op, "param": 0, "mode": "mirror", "axis": "y", "factor": 2},
        ]
    return [{"op": op, "param": 0}]


def score_hypothesis(task: dict[str, Any], hyp: dict[str, Any]) -> float:
    train = task.get("train", [])
    if not train:
        return 0.0

    total_score = 0.0
    for pair in train:
        if isinstance(pair, dict):
            inp = pair.get("input")
            out = pair.get("output")
        else:
            inp, out = pair
        pred = apply_hypothesis(inp, hyp, target_grid=out)
        out_grid = _to_grid(out)
        if _shape(pred) != _shape(out_grid):
            continue
        matches = 0
        total = 0
        for r in range(len(pred)):
            for c in range(len(pred[0])):
                total += 1
                if pred[r][c] == out_grid[r][c]:
                    matches += 1
        total_score += matches / max(1.0, total)

    return total_score / float(len(train))


_OP_GROUP = {
    "identity": "neutral",
    "flip_h": "geom",
    "flip_v": "geom",
    "rotate_90": "geom",
    "transpose": "geom",
    "color_shift": "color",
    "move_to_corner": "object",
    "center_object": "object",
    "duplicate_object": "object",
    "remove_smallest": "object",
    "crop_to_bbox": "shape",
    "paint_bbox": "object",
    "object_recolor_by_rank": "color_object",
    "extract_repeating_unit": "shape_extract",
    "append_head_rows": "shape_expand",
    "append_tail_rows": "shape_expand",
    "recolor_nonzero_to": "color",
    "self_tile_mask": "shape_expand",
    "template_match": "mapping",
}


def _sample_weighted_op(
    state: dict[str, Any],
    best_ops: list[str],
    last_good_ops: list[str],
    rng: random.Random,
) -> str:
    r = rng or random
    op_stats = state.setdefault("arc_op_stats", {op: {"attempts": 0, "successes": 0} for op in OPS})
    weights: list[float] = []
    for op in OPS:
        st = op_stats.setdefault(op, {"attempts": 0, "successes": 0})
        attempts = int(st.get("attempts", 0))
        successes = int(st.get("successes", 0))
        success_rate = successes / max(1, attempts)
        novelty = 1.0 / (1.0 + attempts)
        w = 0.2 + (0.6 * success_rate) + (0.2 * novelty)
        if op in best_ops:
            w += 0.15
        if op in last_good_ops:
            w += 0.25
        if last_good_ops and any(_OP_GROUP.get(op) == _OP_GROUP.get(p) for p in last_good_ops):
            w += 0.10
        weights.append(max(0.01, w))
    return r.choices(OPS, weights=weights, k=1)[0]


def generate_hypothesis(rng: random.Random | None = None) -> dict[str, Any]:
    r = rng or random
    return {
        "chain": [{"op": r.choice(OPS), "param": r.randint(0, 9)}],
    }


def mutate_based_on_preferences(best: dict[str, Any] | None, state: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
    r = rng or random
    best_ops = [s["op"] for s in _normalize_hyp(best)] if best else []
    last_good_ops = [str(x) for x in state.get("arc_last_good_ops", [])]
    forced_op = str(state.get("arc_forced_op", "") or "")
    forced_prob = float(state.get("arc_forced_op_prob", 0.0) or 0.0)

    # Sequence-aware mutation: sample and lightly mutate previously successful chains.
    seq_stats = state.get("arc_sequence_stats", {})
    if isinstance(seq_stats, dict) and seq_stats and r.random() < 0.45:
        candidates: list[tuple[str, dict[str, Any], float]] = []
        for k, v in seq_stats.items():
            if not isinstance(v, dict):
                continue
            chain = v.get("chain")
            if not isinstance(chain, list) or not chain:
                continue
            attempts = int(v.get("attempts", 0))
            successes = int(v.get("successes", 0))
            score = float(v.get("best_score", 0.0))
            rate = successes / max(1, attempts)
            weight = max(0.01, 0.2 + (0.8 * rate) + (0.5 * score))
            candidates.append((k, v, weight))
        if candidates:
            _, picked, _ = r.choices(candidates, weights=[c[2] for c in candidates], k=1)[0]
            seq = [dict(step) for step in picked.get("chain", [])]
            # Small mutation to keep exploration alive.
            if seq and r.random() < 0.30:
                idx = r.randrange(len(seq))
                if r.random() < 0.5:
                    seq[idx]["param"] = r.randint(0, 9)
                else:
                    seq[idx]["op"] = _sample_weighted_op(state, best_ops, last_good_ops, r)
            # Forced missing-op inclusion as hard guidance.
            if forced_op and forced_op in OPS and r.random() < forced_prob:
                if seq:
                    seq[0]["op"] = forced_op
                else:
                    seq = [{"op": forced_op, "param": r.randint(0, 9)}]
            return {"chain": seq}

    chain_len = r.choices([1, 2, 3], weights=[0.55, 0.30, 0.15], k=1)[0]
    chain: list[dict[str, Any]] = []
    for i in range(chain_len):
        op = _sample_weighted_op(state, best_ops, last_good_ops, r)
        if i == 0 and chain_len > 1 and op == "identity":
            op = _sample_weighted_op(state, best_ops, last_good_ops, r)
        if best and best_ops and op == "color_shift" and best_ops[0] == "color_shift":
            best_chain = _normalize_hyp(best)
            base = int(best_chain[0].get("param", 0))
            param = (base + r.choice([-2, -1, 1, 2])) % 10
        else:
            param = r.randint(0, 9)
        chain.append({"op": op, "param": param})
    # Forced missing-op inclusion as hard guidance.
    if forced_op and forced_op in OPS and r.random() < forced_prob:
        if chain:
            chain[0]["op"] = forced_op
        else:
            chain.append({"op": forced_op, "param": r.randint(0, 9)})
    return {"chain": chain}


def solve_arc(
    task: dict[str, Any],
    iterations: int = 50,
    state: dict[str, Any] | None = None,
    seed: int | None = None,
    task_id: str = "unknown",
) -> tuple[dict[str, Any] | None, float, list[float]]:
    if iterations < 0:
        raise ValueError("iterations must be >= 0")

    s = state if state is not None else {}
    rng = random.Random(seed)

    mapping_score, mapping_rules = _mapping_score(task)
    best: dict[str, Any] | None = {"op": "template_match", "rules": mapping_rules}
    best_score = float(mapping_score)

    # Deterministic baselines before stochastic mutation.
    single_scored: list[tuple[float, dict[str, Any]]] = []
    for op in OPS:
        if op == "template_match":
            continue
        for step in _baseline_steps_for_op(op):
            hyp = {"chain": [step]}
            score = score_hypothesis(task, hyp)
            single_scored.append((float(score), hyp))
            if score > best_score:
                best = hyp
                best_score = float(score)

    # Small beam over deterministic 2-op chains.
    # If all singles tie at 0, widen beam so composition still gets a chance.
    single_scored.sort(key=lambda x: x[0], reverse=True)
    top_score = single_scored[0][0] if single_scored else 0.0
    if top_score <= 0.0:
        beam = [h for _, h in single_scored]
    else:
        beam = [h for _, h in single_scored[:3]]
    seen_chain: set[str] = set()
    for left in beam:
        for right in beam:
            chain = _normalize_hyp(left) + _normalize_hyp(right)
            hyp = {"chain": chain}
            key = json.dumps(hyp, sort_keys=True)
            if key in seen_chain:
                continue
            seen_chain.add(key)
            score = score_hypothesis(task, hyp)
            if score > best_score:
                best = hyp
                best_score = float(score)

    # Program candidates: transform-only and mixed transform/rule programs.
    program_hyps_seen: set[str] = set()
    for program in _generate_programs(beam):
        hyp = _program_to_hypothesis(program)
        key = json.dumps(hyp, sort_keys=True)
        if key in program_hyps_seen:
            continue
        program_hyps_seen.add(key)
        score = score_hypothesis(task, hyp)
        if score > best_score:
            best = hyp
            best_score = float(score)

    # Targeted deterministic compositions that frequently occur in ARC.
    deterministic_chain_pool: list[dict[str, Any]] = []
    for shift in [1, 2, 3]:
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_head_rows", "param": 0, "rows": 0},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "append_head_rows", "param": 0, "rows": 0},
                    {"op": "color_shift", "param": shift},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_tail_rows", "param": 0, "rows": 0},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "append_tail_rows", "param": 0, "rows": 0},
                    {"op": "color_shift", "param": shift},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_head_rows", "param": 3, "rows": 3, "selector": "best_window"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_tail_rows", "param": 3, "rows": 3, "selector": "best_window"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_head_rows", "param": 3, "rows": 3, "selector": "best_rows"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "color_shift", "param": shift},
                    {"op": "append_tail_rows", "param": 3, "rows": 3, "selector": "best_rows"},
                ]
            }
        )
    for c in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_head_rows", "param": 0, "rows": 0},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "append_head_rows", "param": 0, "rows": 0},
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_tail_rows", "param": 0, "rows": 0},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "append_tail_rows", "param": 0, "rows": 0},
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_head_rows", "param": 3, "rows": 3, "selector": "best_window"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_tail_rows", "param": 3, "rows": 3, "selector": "best_window"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_head_rows", "param": 3, "rows": 3, "selector": "best_rows"},
                ]
            }
        )
        deterministic_chain_pool.append(
            {
                "chain": [
                    {"op": "recolor_nonzero_to", "param": c, "color": c},
                    {"op": "append_tail_rows", "param": 3, "rows": 3, "selector": "best_rows"},
                ]
            }
        )
    for hyp in deterministic_chain_pool:
        score = score_hypothesis(task, hyp)
        if score > best_score:
            best = hyp
            best_score = float(score)
    progression: list[float] = []

    op_stats = s.setdefault("arc_op_stats", {op: {"attempts": 0, "successes": 0} for op in OPS})

    active_context = str(s.get("arc_current_context", "unknown"))

    for _ in range(iterations):
        hyp = mutate_based_on_preferences(best, s, rng)
        score = score_hypothesis(task, hyp)

        chain = _normalize_hyp(hyp)
        for step in chain:
            st = op_stats.setdefault(step["op"], {"attempts": 0, "successes": 0})
            st["attempts"] = int(st.get("attempts", 0)) + 1
        # Track attempts for sequences globally and per-task.
        seq_stats = s.setdefault("arc_sequence_stats", {})
        seq_by_task = s.setdefault("arc_sequence_stats_by_task", {})
        seq_task = seq_by_task.setdefault(task_id, {})
        sig = json.dumps(chain, sort_keys=True)
        cur = seq_stats.setdefault(sig, {"chain": chain, "attempts": 0, "successes": 0, "best_score": 0.0, "contexts": {}})
        cur["attempts"] = int(cur.get("attempts", 0)) + 1
        ctxs = cur.setdefault("contexts", {})
        ctxs[active_context] = int(ctxs.get(active_context, 0)) + 1
        cur_t = seq_task.setdefault(sig, {"chain": chain, "attempts": 0, "successes": 0, "best_score": 0.0, "contexts": {}})
        cur_t["attempts"] = int(cur_t.get("attempts", 0)) + 1
        ctxs_t = cur_t.setdefault("contexts", {})
        ctxs_t[active_context] = int(ctxs_t.get(active_context, 0)) + 1

        improved = score > best_score
        if score > best_score:
            best = hyp
            best_score = score
        if improved:
            for step in chain:
                st = op_stats.setdefault(step["op"], {"attempts": 0, "successes": 0})
                st["successes"] = int(st.get("successes", 0)) + 1
            s["arc_last_good_ops"] = [step["op"] for step in chain]
            # Reinforce successful rule sequences.
            cur = seq_stats.setdefault(sig, {"chain": chain, "attempts": 0, "successes": 0, "best_score": 0.0, "contexts": {}})
            cur["successes"] = int(cur.get("successes", 0)) + 1
            cur["best_score"] = max(float(cur.get("best_score", 0.0)), float(score))
            cur_t = seq_task.setdefault(sig, {"chain": chain, "attempts": 0, "successes": 0, "best_score": 0.0, "contexts": {}})
            cur_t["successes"] = int(cur_t.get("successes", 0)) + 1
            cur_t["best_score"] = max(float(cur_t.get("best_score", 0.0)), float(score))
            # Keep sequence memory bounded.
            if len(seq_stats) > 256:
                ranked = sorted(
                    seq_stats.items(),
                    key=lambda kv: (float(kv[1].get("best_score", 0.0)), int(kv[1].get("successes", 0))),
                    reverse=True,
                )[:256]
                s["arc_sequence_stats"] = {k: v for k, v in ranked}
        progression.append(best_score)

    return best, float(best_score), progression


def _attach_inferred_subset_rule(task: dict[str, Any], hyp: dict[str, Any]) -> dict[str, Any]:
    chain = _normalize_hyp(hyp)
    if not any(
        step.get("op") in {"append_head_rows", "append_tail_rows"} and step.get("selector") == "best_rows"
        for step in chain
    ):
        return hyp

    train = task.get("train", [])
    examples: list[dict[str, Any]] = []
    examples_by_pair: list[dict[str, Any]] = []
    train_pairs: list[tuple[list[list[int]], list[list[int]]]] = []
    for pair in train:
        if isinstance(pair, dict):
            inp = _to_grid(pair.get("input"))
            out = _to_grid(pair.get("output"))
        else:
            inp = _to_grid(pair[0])
            out = _to_grid(pair[1])
        train_pairs.append((inp, out))

        current = [row[:] for row in inp]
        pair_example_recorded = False
        for step in chain:
            op = step.get("op")
            if op in {"append_head_rows", "append_tail_rows"} and step.get("selector") == "best_rows":
                k = int(step.get("rows", step.get("param", 0)))
                if k <= 0:
                    _, w = _shape(current)
                    k = w
                k = max(1, min(len(current), k))
                prefer = "tail" if op == "append_tail_rows" else "head"
                target_rows_sample = out[-k:]
                ids, _, _ = _select_best_rows(current, k, prefer=prefer, target_rows_sample=target_rows_sample)
                examples.append({"grid": [row[:] for row in current], "ids": ids})
                if not pair_example_recorded:
                    examples_by_pair.append({"grid": [row[:] for row in current], "ids": ids})
                    pair_example_recorded = True
            current = _apply_single_op(current, step, target_grid=out)
        if not pair_example_recorded:
            examples_by_pair.append({"grid": [row[:] for row in inp], "ids": tuple()})

    k_values = [
        int(step.get("rows", step.get("param", 0)) or 0)
        for step in chain
        if step.get("op") in {"append_head_rows", "append_tail_rows"} and step.get("selector") == "best_rows"
    ]
    k = 3
    if k_values:
        k = max(1, k_values[0])

    candidates = _build_candidate_row_rules(examples, k=k)
    if not candidates:
        return hyp

    scored: list[dict[str, Any]] = []
    for r in candidates:
        mean, var, conf = _rule_consistency(r, chain, train_pairs)
        rr = dict(r)
        rr["mean_score"] = mean
        rr["var_score"] = var
        rr["confidence"] = conf
        scored.append(rr)
    scored.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    best_rule = scored[0]
    # If single-rule fit is unstable, try a depth-1 partitioned composite rule.
    if float(best_rule.get("var_score", 0.0)) > 0.01:
        composite = _infer_partitioned_rule(chain, train_pairs, examples_by_pair, k)
        if composite is not None and float(composite.get("confidence", -1e18)) > float(best_rule.get("confidence", -1e18)):
            best_rule = composite
            scored = [composite] + scored
    ensemble = scored[:3]
    baseline_mean, _, _ = _rule_consistency(best_rule, chain, train_pairs)
    program = _infer_best_rule_program(chain, ensemble, train_pairs, baseline_mean=baseline_mean)

    enriched_chain: list[dict[str, Any]] = []
    for step in chain:
        if step.get("op") in {"append_head_rows", "append_tail_rows"} and step.get("selector") == "best_rows":
            s = dict(step)
            s["rule"] = best_rule
            s["rule_ensemble"] = ensemble
            s["rule_summary"] = _rule_summary(best_rule)
            s["rule_type"] = "composite" if str(best_rule.get("type", "")) == "row_composite_v1" else "single"
            if isinstance(program, dict):
                s["rule_program"] = program.get("steps", [])
                s["program_summary"] = {
                    "num_steps": int(program.get("num_steps", 0)),
                    "mean_score": float(program.get("mean_score", 0.0)),
                    "penalized_score": float(program.get("penalized_score", 0.0)),
                }
            enriched_chain.append(s)
        else:
            enriched_chain.append(dict(step))
    return {"chain": enriched_chain}


def run_arc_reasoning(task_path: str, iterations: int = 80, seed: int | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    task = load_task(task_path)
    task_id = Path(task_path).name
    best, best_score, progression = solve_arc(task, iterations=iterations, state=state, seed=seed, task_id=task_id)
    if best is not None:
        best = _attach_inferred_subset_rule(task, best)

    test_pred: Any = None
    if best is not None:
        test_cases = task.get("test", [])
        if isinstance(test_cases, list):
            preds: list[list[list[int]]] = []
            for case in test_cases:
                if isinstance(case, dict):
                    inp = case.get("input")
                else:
                    inp = case
                preds.append(apply_hypothesis(inp, best))
            test_pred = preds
        else:
            test_pred = apply_hypothesis(test_cases, best)

    return {
        "type": "arc_attempt",
        "hypothesis": best,
        "score": best_score,
        "iterations": iterations,
        "score_progression": progression,
        "test_prediction": test_pred,
    }
