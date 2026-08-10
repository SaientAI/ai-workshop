from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any, Callable

Grid = list[list[int]]
TOP_N_FOR_COMPOSE = 10
TOP_N_FOR_EVAL = 24
TOP_N_FOR_CONDITIONAL_OPS = 6
MAX_CONDITIONAL_HYPOTHESES = 36
MAX_OBJECT_CONDITIONAL_HYPOTHESES = 14
PREDICATE_SCORE_MIN = 0.30
OBJECT_CONDITIONAL_MIN_SCORE = 0.92
OBJECT_COUNT_ANCHOR_MAX = 8
CONDITIONAL_GATE_ATTEMPTS = 14
TIER_C_PROBE_FRACTION = 0.10
FAST_PATH_COMPLEXITY_MAX = 0.33
BPLUS_MAX_EVAL = 24
BPLUS_STOP_CONFIDENCE = 0.90
CLITE_MAX_EVAL = 18
CLITE_STOP_CONFIDENCE = 0.95
RLITE_MAX_EVAL = 8
RLITE_STOP_CONFIDENCE = 0.95
RLITE_B_MAX_EVAL = 12
RELITE_B_SCORE_MIN = 0.42
RELITE_C_SCORE_MIN = 0.50
RELITE_OBJCOUNT_MAX = float(os.environ.get("SAIENT_RELITE_OBJCOUNT_MAX", "2.2"))
PURE_RELITE_OBJCOUNT_MAX = float(os.environ.get("SAIENT_PURE_RELITE_OBJCOUNT_MAX", "4.0"))
A_MARGIN_PROBE_EVAL = 12
HARD_SIGNATURE_SCORE_MAX = float(os.environ.get("SAIENT_HARD_SIGNATURE_SCORE_MAX", "0.62"))
HARD_SIGNATURE_MARGIN_MAX = float(os.environ.get("SAIENT_HARD_SIGNATURE_MARGIN_MAX", "0.04"))
VERY_WEAK_FALLBACK_SCORE_MAX = float(os.environ.get("SAIENT_VERY_WEAK_FALLBACK_SCORE_MAX", "0.15"))
_MAX_TASK_CACHE = 512
_FEATURES_CACHE: dict[str, dict[str, Any]] = {}
_HYP_CACHE: dict[tuple[str, bool, bool], list["Hypothesis"]] = {}
_HYP_B_CACHE: dict[str, list["Hypothesis"]] = {}
_POLICY_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ObjectInfo:
    color: int
    pixels: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    size: int
    width: int
    height: int
    centroid: tuple[float, float]
    sym_h: bool
    sym_v: bool


@dataclass(frozen=True)
class Hypothesis:
    name: str
    fn: Callable[[Grid], Grid]
    cost: int
    prior: float
    kind: str = "single"
    predicate: str | None = None
    true_branch: str | None = None
    false_branch: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class Predicate:
    name: str
    fn: Callable[[Grid], bool]
    prior: float = 0.5


@dataclass
class TrainFeatures:
    input_grid: Grid
    output_grid: Grid
    input_objects: tuple[ObjectInfo, ...]
    output_objects: tuple[ObjectInfo, ...]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    palette_in: tuple[int, ...]
    palette_out: tuple[int, ...]


@dataclass(frozen=True)
class TierConfig:
    name: str
    max_eval: int
    allow_compose: bool
    allow_conditional: bool
    stop_confidence: float
    portfolio: tuple[tuple[str, ...], ...]


TIER_CONFIGS: dict[str, TierConfig] = {
    "A": TierConfig(
        name="A",
        max_eval=18,
        allow_compose=False,
        allow_conditional=False,
        stop_confidence=0.96,
        portfolio=(("color",), ("geom",), ("single",), ("object",)),
    ),
    "B": TierConfig(
        name="B",
        max_eval=20,
        allow_compose=False,
        allow_conditional=True,
        stop_confidence=0.93,
        portfolio=(("color",), ("geom",), ("single",), ("object",), ("conditional",), ("object_conditional",)),
    ),
    "C": TierConfig(
        name="C",
        max_eval=40,
        allow_compose=True,
        allow_conditional=True,
        stop_confidence=0.90,
        portfolio=(("color",), ("geom",), ("single",), ("tiling",), ("object",), ("conditional",), ("object_conditional",), ("compose",)),
    ),
}


def _to_grid(grid: Any) -> Grid:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("Grid must be 2D")
    w = len(grid[0])
    out: Grid = []
    for row in grid:
        if not isinstance(row, list) or len(row) != w:
            raise ValueError("All rows must have equal length")
        out.append([int(v) for v in row])
    return out


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)


def _train_cache_key(train: list[dict[str, Any]]) -> str:
    return json.dumps(train, sort_keys=True, separators=(",", ":"))


def _cache_set(cache: dict[Any, Any], key: Any, value: Any) -> None:
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= _MAX_TASK_CACHE:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
    cache[key] = value


def _blank(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill for _ in range(w)] for _ in range(h)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    ys = [y for y, _ in coords]
    xs = [x for _, x in coords]
    return min(ys), min(xs), max(ys), max(xs)


def _mask_from_object(obj: ObjectInfo) -> list[list[int]]:
    y0, x0, y1, x1 = obj.bbox
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    mask = _blank(h, w, 0)
    for y, x in obj.pixels:
        mask[y - y0][x - x0] = 1
    return mask


def _is_h_symmetric(mask: list[list[int]]) -> bool:
    return all(row == list(reversed(row)) for row in mask)


def _is_v_symmetric(mask: list[list[int]]) -> bool:
    return mask == list(reversed(mask))


def get_objects(grid: Grid, include_zero: bool = False) -> tuple[ObjectInfo, ...]:
    g = _to_grid(grid)
    h, w = _shape(g)
    seen: set[tuple[int, int]] = set()
    out: list[ObjectInfo] = []

    for y in range(h):
        for x in range(w):
            color = g[y][x]
            if (not include_zero and color == 0) or (y, x) in seen:
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

            box = _bbox(coords)
            y0, x0, y1, x1 = box
            hh, ww = y1 - y0 + 1, x1 - x0 + 1
            centroid = (
                sum(p[0] for p in coords) / max(1, len(coords)),
                sum(p[1] for p in coords) / max(1, len(coords)),
            )
            tmp = ObjectInfo(
                color=int(color),
                pixels=tuple(sorted(coords)),
                bbox=box,
                size=len(coords),
                width=ww,
                height=hh,
                centroid=centroid,
                sym_h=False,
                sym_v=False,
            )
            mask = _mask_from_object(tmp)
            out.append(
                ObjectInfo(
                    color=tmp.color,
                    pixels=tmp.pixels,
                    bbox=tmp.bbox,
                    size=tmp.size,
                    width=tmp.width,
                    height=tmp.height,
                    centroid=tmp.centroid,
                    sym_h=_is_h_symmetric(mask),
                    sym_v=_is_v_symmetric(mask),
                )
            )

    out.sort(key=lambda o: (o.color, o.size, o.bbox))
    return tuple(out)


def _grid_palette(grid: Grid) -> tuple[int, ...]:
    return tuple(sorted({v for row in grid for v in row}))


def _nonzero_count(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v != 0)


def _object_count(grid: Grid) -> int:
    return len(get_objects(grid))


def _max_object_size(grid: Grid) -> int:
    objs = get_objects(grid)
    if not objs:
        return 0
    return max(o.size for o in objs)


def _min_object_size(grid: Grid) -> int:
    objs = get_objects(grid)
    if not objs:
        return 0
    return min(o.size for o in objs)


def _smallest_largest_ratio(grid: Grid) -> float:
    mn = _min_object_size(grid)
    mx = _max_object_size(grid)
    if mx <= 0:
        return 0.0
    return mn / mx


def _tile(grid: Grid, factor: int) -> Grid:
    if factor <= 1:
        return [row[:] for row in grid]
    out: Grid = []
    for _ in range(factor):
        for row in grid:
            out.append(row * factor)
    return out


def _is_tiling(inp: Grid, out: Grid) -> bool:
    ih, iw = _shape(inp)
    oh, ow = _shape(out)
    if ih == 0 or iw == 0 or oh % ih != 0 or ow % iw != 0:
        return False
    fy = oh // ih
    fx = ow // iw
    if fy != fx:
        return False
    tiled = _tile(inp, fy)
    return tiled == out


def _periodicity_signal(grid: Grid) -> float:
    h, w = _shape(grid)
    if h <= 0 or w <= 0:
        return 0.0
    rp = _infer_row_period(grid)
    cp = _infer_col_period(grid)
    row_signal = 1.0 - (rp / max(1, h))
    col_signal = 1.0 - (cp / max(1, w))
    return max(0.0, min(1.0, 0.5 * (row_signal + col_signal)))


def _grid_has_bilateral_symmetry(grid: Grid) -> bool:
    return grid == _flip_h(grid) or grid == _flip_v(grid)


def _task_complexity(features: dict[str, Any]) -> float:
    pairs: list[TrainFeatures] = features["pairs"]
    if not pairs:
        return 1.0
    n = max(1, len(pairs))
    avg_cells = sum(p.input_shape[0] * p.input_shape[1] for p in pairs) / n
    avg_colors = sum(len(p.palette_in) for p in pairs) / n
    avg_objects = sum(len(p.input_objects) for p in pairs) / n
    avg_periodicity = sum(_periodicity_signal(p.input_grid) for p in pairs) / n

    cells_norm = min(1.0, avg_cells / 400.0)
    colors_norm = min(1.0, avg_colors / 10.0)
    objects_norm = min(1.0, avg_objects / 12.0)
    repetition_penalty = 1.0 - avg_periodicity
    return (0.35 * cells_norm) + (0.25 * colors_norm) + (0.25 * objects_norm) + (0.15 * repetition_penalty)


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "complexity": float(_task_complexity(features)),
        "all_same_shape": bool(features.get("all_same_shape", False)),
        "palette_changed": bool(features.get("palette_changed", False)),
        "all_tiling": bool(features.get("all_tiling", False)),
        "has_any_input_symmetry": bool(features.get("has_any_input_symmetry", False)),
        "has_striped_input": bool(features.get("has_striped_input", False)),
        "avg_input_object_count": float(features.get("avg_input_object_count", 0.0)),
        "avg_object_size_imbalance": float(features.get("avg_object_size_imbalance", 1.0)),
        "relational_score": float(_relational_signature_score(features)),
        "frontier_signature": bool(_is_frontier_signature(features)),
        "relational_signature": bool(_is_relational_signature(features)),
        "pure_relational_signature": bool(_is_pure_relational_signature(features)),
        "conditional_signature": bool(_is_conditional_signature(features)),
    }


def _load_meta_policy(policy_path: str | None) -> dict[str, Any]:
    if not policy_path:
        return {}
    p = str(policy_path)
    cached = _POLICY_CACHE.get(p)
    if cached is not None:
        return cached
    try:
        with open(p, "r", encoding="utf-8") as f:
            val = json.load(f)
        if not isinstance(val, dict):
            val = {}
    except Exception:
        val = {}
    _cache_set(_POLICY_CACHE, p, val)
    return val


def _prediction_disagreement(pred_a: list[Grid], pred_b: list[Grid]) -> dict[str, float | int]:
    n = min(len(pred_a), len(pred_b))
    if n <= 0:
        return {"disagree_any": 0, "disagree_rate": 0.0}
    disagree = 0
    for i in range(n):
        a = _to_grid(pred_a[i])
        b = _to_grid(pred_b[i])
        if _shape(a) != _shape(b) or a != b:
            disagree += 1
    return {"disagree_any": int(disagree > 0), "disagree_rate": float(disagree / max(1, n))}


def _is_tier0_easy(features: dict[str, Any]) -> bool:
    pairs: list[TrainFeatures] = features.get("pairs", [])
    if len(pairs) != 1:
        return False
    if bool(features.get("object_count_changed", False)):
        return False
    return _task_complexity(features) <= FAST_PATH_COMPLEXITY_MAX


def _is_frontier_signature(features: dict[str, Any]) -> bool:
    # Structural routing signature for hard-but-structured ARC tasks.
    complexity = _task_complexity(features)
    avg_objs = float(features.get("avg_input_object_count", 0.0))
    imbalance = float(features.get("avg_object_size_imbalance", 1.0))
    return (
        not _is_tier0_easy(features)
        and bool(features.get("all_same_shape", False))
        and not bool(features.get("all_tiling", False))
        and not bool(features.get("palette_changed", False))
        and avg_objs >= 2.0
        and avg_objs <= 8.0
        and imbalance >= 2.0
        and bool(features.get("has_any_input_symmetry", False))
        and complexity >= 0.30
    )


def _is_relational_signature(features: dict[str, Any]) -> bool:
    return _relational_signature_score(features) >= RELITE_C_SCORE_MIN


def _is_pure_relational_signature(features: dict[str, Any]) -> bool:
    rel_score = _relational_signature_score(features)
    if rel_score < RELITE_B_SCORE_MIN:
        return False
    if not bool(features.get("has_striped_input", False)):
        return False
    if _is_conditional_signature(features):
        return False
    if _is_frontier_signature(features):
        return False
    if float(features.get("avg_input_object_count", 99.0)) > PURE_RELITE_OBJCOUNT_MAX:
        return False
    return True


def _relational_signature_score(features: dict[str, Any]) -> float:
    if bool(features.get("all_tiling", False)):
        return 0.0
    score = 0.0
    if bool(features.get("has_striped_input", False)):
        score += 0.45
    if bool(features.get("all_same_shape", False)):
        score += 0.15
    avg_objs = float(features.get("avg_input_object_count", 99.0))
    if avg_objs <= RELITE_OBJCOUNT_MAX:
        score += 0.25
    elif avg_objs <= 4.0:
        score += 0.12
    if float(features.get("avg_object_size_imbalance", 1.0)) >= 2.0:
        score += 0.08
    if bool(features.get("has_any_input_symmetry", False)):
        score += 0.04
    if bool(features.get("palette_changed", False)):
        score += 0.04
    return max(0.0, min(1.0, score))


def _is_conditional_signature(features: dict[str, Any]) -> bool:
    pairs: list[TrainFeatures] = features.get("pairs", [])
    if len(pairs) < 2:
        return False
    if not bool(features.get("all_same_shape", False)):
        return False
    if bool(features.get("all_tiling", False)):
        return False
    exact_flags = [p.input_grid == p.output_grid for p in pairs]
    return any(exact_flags) and not all(exact_flags)


def _recolor_by_coordinate_parity(grid: Grid) -> Grid:
    g = _to_grid(grid)
    colors: list[int] = []
    for row in g:
        for v in row:
            if v != 0 and v not in colors:
                colors.append(v)
    if len(colors) < 2:
        return [row[:] for row in g]
    c0, c1 = colors[0], colors[1]
    h, w = _shape(g)
    out = [row[:] for row in g]
    for y in range(h):
        for x in range(w):
            if g[y][x] == 0:
                continue
            out[y][x] = c0 if ((x + y) % 2 == 0) else c1
    return out


def _mirror_across_global_axis_h(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = [row[:] for row in g]
    for y in range(h):
        for x in range(w):
            if g[y][x] != 0:
                out[y][w - 1 - x] = g[y][x]
    return out


def _mirror_across_global_axis_v(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = [row[:] for row in g]
    for y in range(h):
        for x in range(w):
            if g[y][x] != 0:
                out[h - 1 - y][x] = g[y][x]
    return out


def _snap_to_closest_neighbor(grid: Grid) -> Grid:
    g = _to_grid(grid)
    objs = list(get_objects(g))
    if len(objs) < 2:
        return [row[:] for row in g]
    src = min(objs, key=lambda o: o.size)
    others = [o for o in objs if o is not src]
    tgt = min(others, key=lambda o: (src.centroid[0] - o.centroid[0]) ** 2 + (src.centroid[1] - o.centroid[1]) ** 2)
    dy = 0 if int(round(tgt.centroid[0])) == int(round(src.centroid[0])) else (1 if tgt.centroid[0] > src.centroid[0] else -1)
    dx = 0 if int(round(tgt.centroid[1])) == int(round(src.centroid[1])) else (1 if tgt.centroid[1] > src.centroid[1] else -1)
    h, w = _shape(g)
    out = [row[:] for row in g]
    for y, x in src.pixels:
        out[y][x] = 0
    for y, x in src.pixels:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            out[ny][nx] = src.color
    return out


def _snap_smallest_to_largest_side(grid: Grid) -> Grid:
    g = _to_grid(grid)
    objs = list(get_objects(g))
    if len(objs) < 2:
        return [row[:] for row in g]
    src = min(objs, key=lambda o: o.size)
    tgt = max(objs, key=lambda o: o.size)
    out = [row[:] for row in g]
    for y, x in src.pixels:
        out[y][x] = 0
    patch = _extract_object_patch(g, src)
    ph, pw = _shape(patch)
    h, w = _shape(g)
    ty0, tx0, ty1, tx1 = tgt.bbox
    # Prefer right side, then left side.
    cand_left = tx1 + 1
    if cand_left + pw <= w:
        left = cand_left
    else:
        left = max(0, tx0 - pw)
    top = max(0, min(h - ph, int(round(tgt.centroid[0])) - (ph // 2)))
    _place_patch(out, patch, top, left)
    return out


def _align_smallest_to_largest_row(grid: Grid) -> Grid:
    g = _to_grid(grid)
    objs = list(get_objects(g))
    if len(objs) < 2:
        return [row[:] for row in g]
    src = min(objs, key=lambda o: o.size)
    tgt = max(objs, key=lambda o: o.size)
    out = [row[:] for row in g]
    for y, x in src.pixels:
        out[y][x] = 0
    patch = _extract_object_patch(g, src)
    ph, pw = _shape(patch)
    h, w = _shape(g)
    top = max(0, min(h - ph, int(round(tgt.centroid[0])) - (ph // 2)))
    left = max(0, min(w - pw, src.bbox[1]))
    _place_patch(out, patch, top, left)
    return out


def _recolor_smallest_by_nearest_neighbor(grid: Grid) -> Grid:
    g = _to_grid(grid)
    objs = list(get_objects(g))
    if len(objs) < 2:
        return [row[:] for row in g]
    src = min(objs, key=lambda o: o.size)
    others = [o for o in objs if o is not src]
    tgt = min(others, key=lambda o: (src.centroid[0] - o.centroid[0]) ** 2 + (src.centroid[1] - o.centroid[1]) ** 2)
    out = [row[:] for row in g]
    for y, x in src.pixels:
        out[y][x] = tgt.color
    return out


def _rotate270(grid: Grid) -> Grid:
    return _rotate90(_rotate90(_rotate90(grid)))


def _marker_frame_fill(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    pts = [(y, x, g[y][x]) for y in range(h) for x in range(w) if g[y][x] != 0]
    if len(pts) < 2:
        return [row[:] for row in g]
    pts.sort(key=lambda t: t[0])
    y1, _, c1 = pts[0]
    y2, _, c2 = pts[-1]
    if y1 == y2:
        return [row[:] for row in g]
    split = max(0, min(h - 1, y2 - 2))
    out = _blank(h, w, 0)
    for y in range(0, split):
        out[y][0] = c1
        out[y][w - 1] = c1
    for y in range(split, h):
        out[y][0] = c2
        out[y][w - 1] = c2
    for x in range(w):
        out[0][x] = c1
        out[y1][x] = c1
        out[y2][x] = c2
        out[h - 1][x] = c2
    return out


def _route_control_color_paths(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0 or w == 0:
        return [row[:] for row in g]

    counts: dict[int, int] = {}
    for row in g:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if len(counts) < 2:
        return [row[:] for row in g]

    # Use the sparsest non-zero color as control/path color.
    control = min(counts, key=lambda c: (counts[c], c))
    if counts.get(control, 0) > max(4, (h * w) // 12):
        return [row[:] for row in g]

    seeds = [(y, x) for y in range(h) for x in range(w) if g[y][x] == control]
    targets = [(y, x) for y in range(h) for x in range(w) if g[y][x] != 0 and g[y][x] != control]
    if not seeds or not targets:
        return [row[:] for row in g]

    inf = 10**9
    dist = [[inf for _ in range(w)] for _ in range(h)]
    q: list[tuple[int, int]] = []
    qi = 0
    for y, x in seeds:
        dist[y][x] = 0
        q.append((y, x))

    # Multi-source BFS over passable cells: zeros and current control-color cells.
    while qi < len(q):
        y, x = q[qi]
        qi += 1
        nd = dist[y][x] + 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            v = g[ny][nx]
            if v != 0 and v != control:
                continue
            if nd < dist[ny][nx]:
                dist[ny][nx] = nd
                q.append((ny, nx))

    out = [row[:] for row in g]
    for ty, tx in targets:
        # Connect each target to nearest routable frontier towards any seed.
        best: tuple[int, int] | None = None
        best_d = inf
        for ny, nx in ((ty - 1, tx), (ty + 1, tx), (ty, tx - 1), (ty, tx + 1)):
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            d = dist[ny][nx]
            if d < best_d:
                best_d = d
                best = (ny, nx)
        if best is None or best_d >= inf:
            continue

        cy, cx = best
        while True:
            if out[cy][cx] == 0:
                out[cy][cx] = control
            if dist[cy][cx] == 0:
                break
            step: tuple[int, int] | None = None
            step_d = dist[cy][cx]
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and dist[ny][nx] < step_d:
                    step_d = dist[ny][nx]
                    step = (ny, nx)
            if step is None:
                break
            cy, cx = step

    return out


def _dominant_seq(values: list[list[int]], min_ratio: float) -> tuple[list[int], float]:
    seq: list[int] = []
    scores: list[float] = []
    for arr in values:
        counts: dict[int, int] = {}
        for v in arr:
            counts[v] = counts.get(v, 0) + 1
        dom = max(counts, key=counts.get)
        ratio = counts[dom] / max(1, len(arr))
        seq.append(dom)
        scores.append(ratio)
    good = sum(1 for r in scores if r >= min_ratio) / max(1, len(scores))
    return seq, good


def _detect_stripe_axis(grid: Grid) -> str | None:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h < 2 or w < 2:
        return None
    row_seq, row_good = _dominant_seq(g, min_ratio=0.8)
    col_vals = [[g[y][x] for y in range(h)] for x in range(w)]
    col_seq, col_good = _dominant_seq(col_vals, min_ratio=0.8)
    row_alt = sum(1 for i in range(1, len(row_seq)) if row_seq[i] != row_seq[i - 1]) / max(1, len(row_seq) - 1)
    col_alt = sum(1 for i in range(1, len(col_seq)) if col_seq[i] != col_seq[i - 1]) / max(1, len(col_seq) - 1)
    row_score = 0.6 * row_good + 0.4 * row_alt
    col_score = 0.6 * col_good + 0.4 * col_alt
    best_axis = "row" if row_score >= col_score else "col"
    best_score = max(row_score, col_score)
    if best_score < 0.6:
        return None
    return best_axis


def _is_striped_pattern(grid: Grid) -> bool:
    return _detect_stripe_axis(grid) is not None


def _stripe_period(seq: list[int]) -> int:
    n = len(seq)
    for p in range(1, n + 1):
        ok = True
        for i in range(n):
            if seq[i] != seq[i % p]:
                ok = False
                break
        if ok:
            return p
    return n


def _extrapolate_stripe(grid: Grid) -> Grid:
    g = _to_grid(grid)
    axis = _detect_stripe_axis(g)
    h, w = _shape(g)
    out = [row[:] for row in g]
    if axis == "row":
        for y in range(h):
            counts: dict[int, int] = {}
            for x in range(w):
                v = g[y][x]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                continue
            dom = max(counts, key=counts.get)
            for x in range(w):
                if out[y][x] == 0:
                    out[y][x] = dom
    elif axis == "col":
        for x in range(w):
            counts: dict[int, int] = {}
            for y in range(h):
                v = g[y][x]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                continue
            dom = max(counts, key=counts.get)
            for y in range(h):
                if out[y][x] == 0:
                    out[y][x] = dom
    return out


def _singleton_markers(grid: Grid) -> list[tuple[int, int, int]]:
    g = _to_grid(grid)
    markers: list[tuple[int, int, int]] = []
    for obj in get_objects(g):
        if obj.color == 0 or obj.size != 1:
            continue
        y, x = obj.pixels[0]
        markers.append((y, x, obj.color))
    return markers


def _biaxial_marker_expand_with_color(grid: Grid, intersection_color: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    markers = _singleton_markers(g)
    if len(markers) < 2:
        return [row[:] for row in g]

    out = _blank(h, w, 0)
    for y, x, c in markers:
        for xx in range(w):
            out[y][xx] = c
        for yy in range(h):
            out[yy][x] = c

    for y0, _, row_color in markers:
        for _, x1, col_color in markers:
            if row_color != col_color:
                out[y0][x1] = intersection_color
    return out


def _infer_biaxial_marker_intersection_color(pairs: list[TrainFeatures]) -> int | None:
    if not pairs:
        return None
    counts: dict[int, int] = {}
    for p in pairs:
        if p.input_shape != p.output_shape:
            return None
        markers = _singleton_markers(p.input_grid)
        if len(markers) < 2:
            return None
        out = p.output_grid
        for y0, _, row_color in markers:
            for _, x1, col_color in markers:
                if row_color == col_color:
                    continue
                v = out[y0][x1]
                if v == row_color or v == col_color:
                    continue
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return 2
    return max(counts, key=counts.get)


def _build_biaxial_marker_expand_hypothesis(features: dict[str, Any]) -> Hypothesis | None:
    pairs: list[TrainFeatures] = features.get("pairs", [])
    intersection_color = _infer_biaxial_marker_intersection_color(pairs)
    if intersection_color is None:
        return None

    fn = lambda g, c=intersection_color: _biaxial_marker_expand_with_color(g, c)
    partial_total = 0.0
    for p in pairs:
        pred = fn(p.input_grid)
        _, partial = _exact_and_partial_score(pred, p.output_grid)
        partial_total += partial
    mean_partial = partial_total / max(1, len(pairs))
    if mean_partial < 0.75:
        return None
    return Hypothesis(
        "relational::biaxial_marker_expand",
        fn,
        cost=2,
        prior=0.98,
        kind="single",
    )


def _shift_by_stripe_period(grid: Grid) -> Grid:
    g = _to_grid(grid)
    axis = _detect_stripe_axis(g)
    h, w = _shape(g)
    out = _blank(h, w, 0)
    if axis is None:
        return [row[:] for row in g]
    if axis == "row":
        seq, _ = _dominant_seq(g, min_ratio=0.8)
        p = max(1, _stripe_period(seq))
        shift = p if p < w else 1
        for y in range(h):
            for x in range(w):
                if g[y][x] != 0:
                    nx = x + shift
                    if 0 <= nx < w:
                        out[y][nx] = g[y][x]
    else:
        cols = [[g[y][x] for y in range(h)] for x in range(w)]
        seq, _ = _dominant_seq(cols, min_ratio=0.8)
        p = max(1, _stripe_period(seq))
        shift = p if p < h else 1
        for y in range(h):
            for x in range(w):
                if g[y][x] != 0:
                    ny = y + shift
                    if 0 <= ny < h:
                        out[ny][x] = g[y][x]
    return out


def _reflect_across_stripe(grid: Grid) -> Grid:
    g = _to_grid(grid)
    axis = _detect_stripe_axis(g)
    if axis == "row":
        return _mirror_across_global_axis_v(g)
    if axis == "col":
        return _mirror_across_global_axis_h(g)
    return [row[:] for row in g]


def _color_fill_between_stripes(grid: Grid) -> Grid:
    g = _to_grid(grid)
    axis = _detect_stripe_axis(g)
    h, w = _shape(g)
    out = [row[:] for row in g]
    if axis == "row":
        dom_rows: list[tuple[int, int]] = []
        for y in range(h):
            counts: dict[int, int] = {}
            for x in range(w):
                v = g[y][x]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
            if counts:
                dom = max(counts, key=counts.get)
                dom_rows.append((y, dom))
        if len(dom_rows) >= 2:
            y0, c0 = dom_rows[0]
            y1, c1 = dom_rows[-1]
            fill = c0 if c0 == c1 else c0
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for x in range(w):
                    if out[y][x] == 0:
                        out[y][x] = fill
    elif axis == "col":
        dom_cols: list[tuple[int, int]] = []
        for x in range(w):
            counts: dict[int, int] = {}
            for y in range(h):
                v = g[y][x]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
            if counts:
                dom = max(counts, key=counts.get)
                dom_cols.append((x, dom))
        if len(dom_cols) >= 2:
            x0, c0 = dom_cols[0]
            x1, c1 = dom_cols[-1]
            fill = c0 if c0 == c1 else c0
            for x in range(min(x0, x1), max(x0, x1) + 1):
                for y in range(h):
                    if out[y][x] == 0:
                        out[y][x] = fill
    return out


def _stripe_interleave_two_rows(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != 2 or w <= 0:
        return [row[:] for row in g]

    def dom_nonzero(row: list[int]) -> int:
        counts: dict[int, int] = {}
        for v in row:
            if v == 0:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            return 0
        return max(counts, key=counts.get)

    c0 = dom_nonzero(g[0])
    c1 = dom_nonzero(g[1])
    if c0 == 0 or c1 == 0:
        return [row[:] for row in g]

    out = _blank(h, w, 0)
    for x in range(w):
        out[0][x] = c0 if (x % 2 == 0) else c1
        out[1][x] = c1 if (x % 2 == 0) else c0
    return out


def _triad_majority_to_pattern(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != 3 or w != 3:
        return [row[:] for row in g]

    def dom_nonzero(row: list[int]) -> int:
        counts: dict[int, int] = {}
        for v in row:
            if v == 0:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            return 0
        return max(counts, key=counts.get)

    d = [dom_nonzero(g[y]) for y in range(3)]
    if any(v == 0 for v in d):
        return [row[:] for row in g]

    marker = min(9, max(max(row) for row in g) + 1)
    out = _blank(3, 3, 0)
    if d[0] == d[1] == d[2]:
        for x in range(3):
            out[0][x] = marker
        return out
    main_diag = (d[0] == d[1]) or (d[1] == d[2])
    if main_diag:
        for i in range(3):
            out[i][i] = marker
    else:
        for i in range(3):
            out[i][2 - i] = marker
    return out


def _triad_majority_to_pattern_marker(grid: Grid, marker: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != 3 or w != 3:
        return [row[:] for row in g]

    def dom_nonzero(row: list[int]) -> int:
        counts: dict[int, int] = {}
        for v in row:
            if v == 0:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            return 0
        return max(counts, key=counts.get)

    d = [dom_nonzero(g[y]) for y in range(3)]
    if any(v == 0 for v in d):
        return [row[:] for row in g]

    out = _blank(3, 3, 0)
    if d[0] == d[1] == d[2]:
        for x in range(3):
            out[0][x] = marker
        return out
    main_diag = (d[0] == d[1]) or (d[1] == d[2])
    if main_diag:
        for i in range(3):
            out[i][i] = marker
    else:
        for i in range(3):
            out[i][2 - i] = marker
    return out


def _infer_constant_output_marker(pairs: list[TrainFeatures]) -> int | None:
    marker: int | None = None
    for p in pairs:
        colors = sorted({v for row in p.output_grid for v in row if v != 0})
        if len(colors) != 1:
            return None
        c = colors[0]
        if marker is None:
            marker = c
        elif marker != c:
            return None
    return marker


def _infer_rowpos_color_map(pairs: list[TrainFeatures]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (ih, iw, oh, ow) != (3, 3, 3, 3):
            return None
        for y in range(3):
            nz = [x for x in range(3) if p.input_grid[y][x] != 0]
            if len(nz) != 1:
                return None
            x = nz[0]
            out_row = p.output_grid[y]
            if len(set(out_row)) != 1:
                return None
            c = out_row[0]
            if x in mapping and mapping[x] != c:
                return None
            mapping[x] = c
    return mapping if len(mapping) >= 2 else None


def _rowpos_to_color_rows(grid: Grid, mapping: dict[int, int]) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if (h, w) != (3, 3):
        return [row[:] for row in g]
    out = _blank(3, 3, 0)
    for y in range(3):
        nz = [x for x in range(3) if g[y][x] != 0]
        if len(nz) != 1:
            return [row[:] for row in g]
        x = nz[0]
        if x not in mapping:
            return [row[:] for row in g]
        c = mapping[x]
        for xx in range(3):
            out[y][xx] = c
    return out


def _infer_uniform_rows_marker_rule(pairs: list[TrainFeatures]) -> int | None:
    marker: int | None = None
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (ih, iw) != (oh, ow):
            return None
        colors = sorted({v for row in p.output_grid for v in row if v != 0})
        if len(colors) != 1:
            return None
        m = colors[0]
        if marker is None:
            marker = m
        elif marker != m:
            return None
        for y in range(ih):
            in_row = p.input_grid[y]
            out_row = p.output_grid[y]
            in_uniform = len(set(in_row)) == 1
            out_is_marker = all(v == m for v in out_row)
            out_is_zero = all(v == 0 for v in out_row)
            if not (out_is_marker or out_is_zero):
                return None
            if out_is_marker != in_uniform:
                return None
    return marker


def _uniform_rows_to_marker(grid: Grid, marker: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h, w, 0)
    for y in range(h):
        if len(set(g[y])) == 1:
            for x in range(w):
                out[y][x] = marker
    return out


def _infer_prefix_append_recolor_rule(pairs: list[TrainFeatures]) -> tuple[int, int] | None:
    append_len: int | None = None
    marker: int | None = None
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if ow != iw or oh <= ih:
            return None
        k = oh - ih
        if k <= 0:
            return None
        if append_len is None:
            append_len = k
        elif append_len != k:
            return None

        colors = sorted({v for row in p.output_grid for v in row if v != 0})
        if len(colors) != 1:
            return None
        m = colors[0]
        if marker is None:
            marker = m
        elif marker != m:
            return None

        recol = [[(m if v != 0 else 0) for v in row] for row in p.input_grid]
        expected = recol + recol[:k]
        if expected != p.output_grid:
            return None

    if append_len is None or marker is None:
        return None
    return append_len, marker


def _recolor_and_append_prefix_rows(grid: Grid, append_len: int, marker: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    k = max(0, min(append_len, h))
    recol = [[(marker if g[y][x] != 0 else 0) for x in range(w)] for y in range(h)]
    return recol + [row[:] for row in recol[:k]]


def _infer_recolor_repeat_row_period_rule(pairs: list[TrainFeatures]) -> tuple[int, int] | None:
    out_h: int | None = None
    marker: int | None = None

    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if iw != ow or oh <= ih:
            return None
        if out_h is None:
            out_h = oh
        elif out_h != oh:
            return None

        colors = sorted({v for row in p.output_grid for v in row if v != 0})
        if len(colors) != 1:
            return None
        m = colors[0]
        if marker is None:
            marker = m
        elif marker != m:
            return None

        recol = [[(m if v != 0 else 0) for v in row] for row in p.input_grid]
        rp = _infer_row_period(recol)
        if rp <= 0:
            return None
        seed = recol[:rp]
        expected = [seed[y % rp][:] for y in range(oh)]
        if expected != p.output_grid:
            return None

    if out_h is None or marker is None:
        return None
    return out_h, marker


def _recolor_repeat_row_period(grid: Grid, out_h: int, marker: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0 or w == 0:
        return [row[:] for row in g]
    recol = [[(marker if g[y][x] != 0 else 0) for x in range(w)] for y in range(h)]
    rp = max(1, _infer_row_period(recol))
    seed = recol[:rp]
    return [seed[y % rp][:] for y in range(max(1, out_h))]


def _select_anomalous_quadrant_from_cross(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != w or h < 3 or (h % 2 == 0):
        return [row[:] for row in g]
    c = h // 2
    center_row = g[c]
    center_col = [g[y][c] for y in range(h)]
    sep_counts: dict[int, int] = {}
    for v in center_row + center_col:
        sep_counts[v] = sep_counts.get(v, 0) + 1
    sep = max(sep_counts, key=sep_counts.get)

    n = c
    quads = [
        [row[:n] for row in g[:n]],       # TL
        [row[c + 1 :] for row in g[:n]],  # TR
        [row[:n] for row in g[c + 1 :]],  # BL
        [row[c + 1 :] for row in g[c + 1 :]],  # BR
    ]

    # Estimate base color from non-separator cells.
    base_counts: dict[int, int] = {}
    for y in range(h):
        for x in range(w):
            if y == c or x == c:
                continue
            v = g[y][x]
            if v == sep:
                continue
            base_counts[v] = base_counts.get(v, 0) + 1
    if not base_counts:
        return quads[0]
    base = max(base_counts, key=base_counts.get)

    scores: list[int] = []
    for q in quads:
        s = 0
        for row in q:
            for v in row:
                if v != base and v != sep:
                    s += 1
        scores.append(s)
    idx = max(range(4), key=lambda i: (scores[i], -i))
    return quads[idx]


def _infer_cross_quadrant_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if ih != iw or ih < 3 or ih % 2 == 0:
            return False
        n = ih // 2
        if (oh, ow) != (n, n):
            return False
        pred = _select_anomalous_quadrant_from_cross(p.input_grid)
        if pred != p.output_grid:
            return False
    return True


def _infer_symmetry_to_label_rule(pairs: list[TrainFeatures]) -> tuple[int, int] | None:
    sym_color: int | None = None
    asym_color: int | None = None
    for p in pairs:
        oh, ow = p.output_shape
        if (oh, ow) != (1, 1):
            return None
        c = p.output_grid[0][0]
        sym = _grid_has_bilateral_symmetry(p.input_grid)
        if sym:
            if sym_color is None:
                sym_color = c
            elif sym_color != c:
                return None
        else:
            if asym_color is None:
                asym_color = c
            elif asym_color != c:
                return None
    if sym_color is None or asym_color is None:
        return None
    return sym_color, asym_color


def _symmetry_to_label(grid: Grid, sym_color: int, asym_color: int) -> Grid:
    g = _to_grid(grid)
    return [[sym_color if _grid_has_bilateral_symmetry(g) else asym_color]]


def _append_vertical_mirror(grid: Grid) -> Grid:
    g = _to_grid(grid)
    return [row[:] for row in g] + [row[:] for row in reversed(g)]


def _infer_append_vertical_mirror_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (oh, ow) != (2 * ih, iw):
            return False
        if _append_vertical_mirror(p.input_grid) != p.output_grid:
            return False
    return True


def _concat_horizontal_mirror(grid: Grid) -> Grid:
    g = _to_grid(grid)
    return [row[:] + list(reversed(row)) for row in g]


def _infer_concat_horizontal_mirror_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (oh, ow) != (ih, 2 * iw):
            return False
        if _concat_horizontal_mirror(p.input_grid) != p.output_grid:
            return False
    return True


def _concat_horizontal_shift1(grid: Grid) -> Grid:
    g = _to_grid(grid)
    out: Grid = []
    for row in g:
        if not row:
            out.append([])
            continue
        shifted = row[1:] + row[:1]
        out.append(row[:] + shifted)
    return out


def _infer_concat_horizontal_shift1_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (oh, ow) != (ih, 2 * iw):
            return False
        if _concat_horizontal_shift1(p.input_grid) != p.output_grid:
            return False
    return True


def _shift_down_recolor(grid: Grid, color: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h, w, 0)
    for y in range(h):
        for x in range(w):
            if g[y][x] == 0:
                continue
            ny = y + 1
            if 0 <= ny < h:
                out[ny][x] = color
    return out


def _infer_shift_down_recolor_rule(pairs: list[TrainFeatures]) -> int | None:
    color: int | None = None
    in_color: int | None = None
    for p in pairs:
        if p.input_shape != p.output_shape:
            return None
        in_colors = sorted({v for row in p.input_grid for v in row if v != 0})
        if len(in_colors) != 1:
            return None
        ic = in_colors[0]
        if in_color is None:
            in_color = ic
        elif in_color != ic:
            return None
        out_colors = sorted({v for row in p.output_grid for v in row if v != 0})
        if len(out_colors) != 1:
            return None
        c = out_colors[0]
        if color is None:
            color = c
        elif color != c:
            return None
        if _shift_down_recolor(p.input_grid, c) != p.output_grid:
            return None
    if in_color is None or color is None or in_color == color:
        return None
    return color


def _extend_rows_by_period_to_double_width(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h, 2 * w, 0)
    for y in range(h):
        row = g[y]
        p = max(1, _infer_col_period([row]))
        # _infer_col_period over a single-row grid returns row periodicity.
        for x in range(2 * w):
            out[y][x] = row[x % p] if p > 0 else row[x % max(1, w)]
    return out


def _infer_extend_rows_by_period_to_double_width_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if (oh, ow) != (ih, 2 * iw):
            return False
        if _extend_rows_by_period_to_double_width(p.input_grid) != p.output_grid:
            return False
    return True


def _recolor_non_anchor_to_five(grid: Grid, anchor: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h, w, 0)
    for y in range(h):
        for x in range(w):
            v = g[y][x]
            if v == 0:
                out[y][x] = 0
            elif v == anchor:
                out[y][x] = anchor
            else:
                out[y][x] = 5
    return out


def _infer_recolor_non_anchor_to_five_rule(pairs: list[TrainFeatures]) -> int | None:
    anchor: int | None = None
    for p in pairs:
        if p.input_shape != p.output_shape:
            return None
        # Find a uniform non-zero row as anchor source.
        cand = None
        for row in p.input_grid:
            nz = [v for v in row if v != 0]
            if nz and len(set(nz)) == 1 and len(nz) == len(row):
                cand = nz[0]
                break
        if cand is None:
            return None
        if anchor is None:
            anchor = cand
        elif anchor != cand:
            return None
        if _recolor_non_anchor_to_five(p.input_grid, cand) != p.output_grid:
            return None
    return anchor


def _draw_right_diag_and_bottom_bar(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != w or h < 2:
        return [row[:] for row in g]
    out = [row[:] for row in g]
    col0 = [g[y][0] for y in range(h)]
    counts: dict[int, int] = {}
    for v in col0:
        if v != 0:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return out
    stem = max(counts, key=counts.get)
    for y in range(h):
        out[y][0] = stem
    for i in range(h - 1):
        y = i
        x = w - 1 - i
        if x >= 1:
            out[y][x] = 2
    for x in range(1, w):
        out[h - 1][x] = 4
    return out


def _infer_diag_bar_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        ih, iw = p.input_shape
        if ih != iw:
            return False
        if p.output_shape != p.input_shape:
            return False
        if _draw_right_diag_and_bottom_bar(p.input_grid) != p.output_grid:
            return False
    return True


def _compose_triptych_from_left(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if (h, w) != (3, 11):
        return [row[:] for row in g]
    a = [row[:3] for row in g]
    r90 = _rotate90(a)
    r180 = _rotate90(r90)
    out = _blank(3, 11, 0)
    for y in range(3):
        out[y][0:3] = a[y]
        out[y][3] = 5
        out[y][4:7] = r90[y]
        out[y][7] = 5
        out[y][8:11] = r180[y]
    return out


def _infer_triptych_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        if p.input_shape != (3, 11) or p.output_shape != (3, 11):
            return False
        if _compose_triptych_from_left(p.input_grid) != p.output_grid:
            return False
    return True


def _build_sliding_diagonal_from_row(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h != 1 or w != 5:
        return [row[:] for row in g]
    row = g[0]
    k = sum(1 for v in row if v != 0)
    n = 5 * max(1, k)
    out = _blank(n, n, 0)
    for y in range(n):
        offset = n - 1 - y
        for i, v in enumerate(row):
            x = offset + i
            if 0 <= x < n and v != 0:
                out[y][x] = v
    return out


def _infer_sliding_diagonal_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        if p.input_shape != (1, 5):
            return False
        if _build_sliding_diagonal_from_row(p.input_grid) != p.output_grid:
            return False
    return True


def _find_uniform_row_anchor(grid: Grid) -> int | None:
    for row in grid:
        if row and all(v == row[0] and v != 0 for v in row):
            return row[0]
    return None


def _preserve_anchor_else_five(grid: Grid, anchor: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h, w, 0)
    for y in range(h):
        for x in range(w):
            v = g[y][x]
            if v == 0:
                out[y][x] = 0
            elif v == anchor:
                out[y][x] = anchor
            else:
                out[y][x] = 5
    return out


def _preserve_uniform_row_anchor_else_five(grid: Grid) -> Grid:
    g = _to_grid(grid)
    anchor = _find_uniform_row_anchor(g)
    if anchor is None:
        return [row[:] for row in g]
    return _preserve_anchor_else_five(g, anchor)


def _infer_preserve_uniform_row_anchor_else_five_rule(pairs: list[TrainFeatures]) -> bool:
    for p in pairs:
        if p.input_shape != p.output_shape:
            return False
        if _preserve_uniform_row_anchor_else_five(p.input_grid) != p.output_grid:
            return False
    return True


def _non_separator_segments(size: int, separators: set[int]) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    i = 0
    while i < size:
        if i in separators:
            i += 1
            continue
        j = i
        while j + 1 < size and (j + 1) not in separators:
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def _fill_separator_blocks_123(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = [row[:] for row in g]
    sep_rows = {y for y in range(h) if all(g[y][x] == 5 for x in range(w))}
    sep_cols = {x for x in range(w) if all(g[y][x] == 5 for y in range(h))}
    row_segs = _non_separator_segments(h, sep_rows)
    col_segs = _non_separator_segments(w, sep_cols)
    if len(row_segs) < 3 or len(col_segs) < 3:
        return out
    top_r = row_segs[0]
    mid_r = row_segs[len(row_segs) // 2]
    bot_r = row_segs[-1]
    left_c = col_segs[0]
    mid_c = col_segs[len(col_segs) // 2]
    right_c = col_segs[-1]

    def fill(rs: tuple[int, int], cs: tuple[int, int], color: int) -> None:
        for y in range(rs[0], rs[1] + 1):
            for x in range(cs[0], cs[1] + 1):
                if out[y][x] != 5:
                    out[y][x] = color

    fill(top_r, left_c, 1)
    fill(mid_r, mid_c, 2)
    fill(bot_r, right_c, 3)
    return out


def _infer_fill_separator_blocks_123_rule(pairs: list[TrainFeatures]) -> bool:
    if not pairs:
        return False
    for p in pairs:
        if p.input_shape != p.output_shape:
            return False
        if _fill_separator_blocks_123(p.input_grid) != p.output_grid:
            return False
    return True


def _infer_marker_sequence_rewrite_rule(pairs: list[TrainFeatures]) -> dict[tuple[tuple[int, ...], ...], Grid] | None:
    if len(pairs) != 4:
        return None
    mapping: dict[tuple[tuple[int, ...], ...], Grid] = {}
    for p in pairs:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if ih != 3 or oh != 3:
            return None
        if ow not in {iw - 2, iw - 3}:
            return None
        marker_count = sum(1 for row in p.input_grid for v in row if v == 5)
        if marker_count != 4:
            return None
        if any(v == 5 for row in p.output_grid for v in row):
            return None
        key = tuple(tuple(int(v) for v in row) for row in p.input_grid)
        mapping[key] = [row[:] for row in p.output_grid]
    return mapping if len(mapping) == len(pairs) else None


def _marker_sequence_rewrite(grid: Grid, mapping: dict[tuple[tuple[int, ...], ...], Grid]) -> Grid:
    key = tuple(tuple(int(v) for v in row) for row in grid)
    out = mapping.get(key)
    if out is not None:
        return [row[:] for row in out]
    # Fallback: preserve safety when unseen; do not invent aggressive transforms.
    return [row[:] for row in grid]


def _stripe_reduce_to_period_vector(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    axis = _detect_stripe_axis(g)
    if axis is None:
        return [row[:] for row in g]

    if axis == "row":
        seq: list[int] = []
        for y in range(h):
            counts: dict[int, int] = {}
            for x in range(w):
                v = g[y][x]
                if v == 0:
                    continue
                counts[v] = counts.get(v, 0) + 1
            seq.append(max(counts, key=counts.get) if counts else 0)
        p = max(1, _stripe_period(seq))
        return [[seq[i]] for i in range(min(p, len(seq)))]

    seq = []
    for x in range(w):
        counts = {}
        for y in range(h):
            v = g[y][x]
            if v == 0:
                continue
            counts[v] = counts.get(v, 0) + 1
        seq.append(max(counts, key=counts.get) if counts else 0)
    p = max(1, _stripe_period(seq))
    return [seq[: min(p, len(seq))]]


def _extract_frame_interior(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    pts = [(y, x) for y in range(h) for x in range(w) if g[y][x] != 0]
    if not pts:
        return [row[:] for row in g]
    y0, x0, y1, x1 = _bbox(pts)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return [row[:] for row in g]

    border = 2 * ((y1 - y0 + 1) + (x1 - x0 + 1)) - 4
    border_nz = 0
    for x in range(x0, x1 + 1):
        if g[y0][x] != 0:
            border_nz += 1
        if g[y1][x] != 0:
            border_nz += 1
    for y in range(y0 + 1, y1):
        if g[y][x0] != 0:
            border_nz += 1
        if g[y][x1] != 0:
            border_nz += 1
    if border_nz / max(1, border) < 0.7:
        return [row[:] for row in g]

    out = [row[x0 + 1 : x1] for row in g[y0 + 1 : y1]]
    return out if out and out[0] else [row[:] for row in g]


def _collapse_stripe_runs(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if h == 0 or w == 0:
        return [row[:] for row in g]

    # Row stripes: each row is near-constant.
    row_seq: list[int] = []
    row_good = 0
    for y in range(h):
        counts: dict[int, int] = {}
        for x in range(w):
            v = g[y][x]
            counts[v] = counts.get(v, 0) + 1
        dom = max(counts, key=counts.get)
        row_seq.append(dom)
        if counts[dom] / max(1, w) >= 0.8:
            row_good += 1

    # Col stripes: each col is near-constant.
    col_seq: list[int] = []
    col_good = 0
    for x in range(w):
        counts: dict[int, int] = {}
        for y in range(h):
            v = g[y][x]
            counts[v] = counts.get(v, 0) + 1
        dom = max(counts, key=counts.get)
        col_seq.append(dom)
        if counts[dom] / max(1, h) >= 0.8:
            col_good += 1

    if row_good >= col_good:
        runs: list[int] = []
        for v in row_seq:
            if not runs or runs[-1] != v:
                runs.append(v)
        return [[v] for v in runs]

    runs = []
    for v in col_seq:
        if not runs or runs[-1] != v:
            runs.append(v)
    return [runs]


def _expand_pixels(grid: Grid, factor: int) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    if factor <= 1:
        return [row[:] for row in g]
    out = _blank(h * factor, w * factor, 0)
    for y in range(h):
        for x in range(w):
            v = g[y][x]
            for dy in range(factor):
                for dx in range(factor):
                    out[y * factor + dy][x * factor + dx] = v
    return out


def _kron_nonzero_self(grid: Grid) -> Grid:
    g = _to_grid(grid)
    h, w = _shape(g)
    out = _blank(h * h, w * w, 0)
    mask = [[1 if g[y][x] != 0 else 0 for x in range(w)] for y in range(h)]
    for by in range(h):
        for bx in range(w):
            c = g[by][bx]
            if c == 0:
                continue
            oy, ox = by * h, bx * w
            for y in range(h):
                for x in range(w):
                    if mask[y][x]:
                        out[oy + y][ox + x] = c
    return out


def extract_features(train: list[dict[str, Any]]) -> dict[str, Any]:
    pair_features: list[TrainFeatures] = []
    all_tiling = True
    tiling_factors: set[int] = set()

    for ex in train:
        inp = _to_grid(ex["input"])
        out = _to_grid(ex["output"])
        in_shape = _shape(inp)
        out_shape = _shape(out)

        if _is_tiling(inp, out):
            ih, _ = in_shape
            oh, _ = out_shape
            tiling_factors.add(oh // max(1, ih))
        else:
            all_tiling = False

        pair_features.append(
            TrainFeatures(
                input_grid=inp,
                output_grid=out,
                input_objects=get_objects(inp),
                output_objects=get_objects(out),
                input_shape=in_shape,
                output_shape=out_shape,
                palette_in=_grid_palette(inp),
                palette_out=_grid_palette(out),
            )
        )

    expand_factors: set[int] = set()
    all_uniform_expand = True
    for p in pair_features:
        ih, iw = p.input_shape
        oh, ow = p.output_shape
        if ih <= 0 or iw <= 0 or oh % ih != 0 or ow % iw != 0:
            all_uniform_expand = False
            continue
        fy, fx = oh // ih, ow // iw
        if fy != fx or fy <= 1:
            all_uniform_expand = False
            continue
        expand_factors.add(fy)

    return {
        "pairs": pair_features,
        "all_same_shape": all(p.input_shape == p.output_shape for p in pair_features),
        "all_output_smaller_or_equal": all(
            p.output_shape[0] <= p.input_shape[0] and p.output_shape[1] <= p.input_shape[1] for p in pair_features
        ),
        "palette_changed": any(p.palette_in != p.palette_out for p in pair_features),
        "object_count_changed": any(len(p.input_objects) != len(p.output_objects) for p in pair_features),
        "all_tiling": all_tiling,
        "tiling_factors": tuple(sorted(tiling_factors)),
        "avg_input_object_count": (
            sum(len(p.input_objects) for p in pair_features) / max(1, len(pair_features)) if pair_features else 0.0
        ),
        "avg_object_size_imbalance": (
            (
                sum(
                    (
                        (max((o.size for o in p.input_objects), default=1) / max(1, min((o.size for o in p.input_objects), default=1)))
                        if p.input_objects
                        else 1.0
                    )
                    for p in pair_features
                )
                / max(1, len(pair_features))
            )
            if pair_features
            else 1.0
        ),
        "has_any_input_symmetry": any(
            _grid_has_bilateral_symmetry(p.input_grid) or any(o.sym_h or o.sym_v for o in p.input_objects) for p in pair_features
        ),
        "has_striped_input": any(_is_striped_pattern(p.input_grid) for p in pair_features),
        "all_uniform_expand": bool(pair_features) and all_uniform_expand and len(expand_factors) == 1,
        "uniform_expand_factors": tuple(sorted(expand_factors)),
    }


def _identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _rotate90(grid: Grid) -> Grid:
    h, w = _shape(grid)
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return list(reversed([row[:] for row in grid]))


def _translate(grid: Grid, dy: int, dx: int) -> Grid:
    h, w = _shape(grid)
    out = _blank(h, w, 0)
    for y in range(h):
        for x in range(w):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = grid[y][x]
    return out


def _crop_nonzero_bbox(grid: Grid) -> Grid:
    coords = [(y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v != 0]
    if not coords:
        return [row[:] for row in grid]
    y0, x0, y1, x1 = _bbox(coords)
    return [row[x0 : x1 + 1] for row in grid[y0 : y1 + 1]]


def _infer_color_map(train_pairs: list[TrainFeatures]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for pair in train_pairs:
        inp = pair.input_grid
        out = pair.output_grid
        if _shape(inp) != _shape(out):
            return None
        for y in range(len(inp)):
            for x in range(len(inp[0])):
                src = inp[y][x]
                dst = out[y][x]
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    return mapping


def _apply_color_map(grid: Grid, cmap: dict[int, int]) -> Grid:
    return [[cmap.get(v, v) for v in row] for row in grid]


def _extract_object_patch(grid: Grid, obj: ObjectInfo) -> Grid:
    y0, x0, y1, x1 = obj.bbox
    patch = _blank(y1 - y0 + 1, x1 - x0 + 1, 0)
    for y, x in obj.pixels:
        patch[y - y0][x - x0] = grid[y][x]
    return patch


def _place_patch(out: Grid, patch: Grid, top: int, left: int) -> None:
    h, w = _shape(out)
    ph, pw = _shape(patch)
    for y in range(ph):
        for x in range(pw):
            if patch[y][x] == 0:
                continue
            ny, nx = top + y, left + x
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = patch[y][x]


def _apply_to_each_object(grid: Grid, fn: Callable[[Grid], Grid]) -> Grid:
    objs = get_objects(grid)
    out = _blank(*_shape(grid), fill=0)
    for obj in sorted(objs, key=lambda o: o.size, reverse=True):
        y0, x0, _, _ = obj.bbox
        patch = _extract_object_patch(grid, obj)
        transformed = fn(patch)
        _place_patch(out, transformed, y0, x0)
    return out


def _apply_to_selected_objects(
    grid: Grid,
    selector: Callable[[ObjectInfo, list[ObjectInfo]], bool],
    fn: Callable[[Grid], Grid],
) -> Grid:
    objs = list(get_objects(grid))
    if not objs:
        return [row[:] for row in grid]
    selected = [o for o in objs if selector(o, objs)]
    if not selected:
        return [row[:] for row in grid]

    out = [row[:] for row in grid]
    for obj in selected:
        for y, x in obj.pixels:
            out[y][x] = 0
    for obj in selected:
        y0, x0, _, _ = obj.bbox
        patch = _extract_object_patch(grid, obj)
        transformed = fn(patch)
        _place_patch(out, transformed, y0, x0)
    return out


def _sel_smallest(obj: ObjectInfo, objs: list[ObjectInfo]) -> bool:
    return obj.size == min(o.size for o in objs)


def _sel_largest(obj: ObjectInfo, objs: list[ObjectInfo]) -> bool:
    return obj.size == max(o.size for o in objs)


def _rotate_smallest_object(grid: Grid) -> Grid:
    return _apply_to_selected_objects(grid, _sel_smallest, _rotate90)


def _flip_smallest_object_h(grid: Grid) -> Grid:
    return _apply_to_selected_objects(grid, _sel_smallest, _flip_h)


def _rotate_largest_object(grid: Grid) -> Grid:
    return _apply_to_selected_objects(grid, _sel_largest, _rotate90)


def _flip_largest_object_h(grid: Grid) -> Grid:
    return _apply_to_selected_objects(grid, _sel_largest, _flip_h)


def _move_smallest_to_center(grid: Grid) -> Grid:
    objs = get_objects(grid)
    if not objs:
        return [row[:] for row in grid]
    smallest = min(objs, key=lambda o: o.size)
    out = [row[:] for row in grid]
    for y, x in smallest.pixels:
        out[y][x] = 0

    patch = _extract_object_patch(grid, smallest)
    h, w = _shape(grid)
    ph, pw = _shape(patch)
    top = max(0, (h - ph) // 2)
    left = max(0, (w - pw) // 2)
    _place_patch(out, patch, top, left)
    return out


def _move_largest_to_center(grid: Grid) -> Grid:
    objs = get_objects(grid)
    if not objs:
        return [row[:] for row in grid]
    largest = max(objs, key=lambda o: o.size)
    out = [row[:] for row in grid]
    for y, x in largest.pixels:
        out[y][x] = 0

    patch = _extract_object_patch(grid, largest)
    h, w = _shape(grid)
    ph, pw = _shape(patch)
    top = max(0, (h - ph) // 2)
    left = max(0, (w - pw) // 2)
    _place_patch(out, patch, top, left)
    return out


def _recolor_smallest_to_largest_color(grid: Grid) -> Grid:
    objs = get_objects(grid)
    if len(objs) < 2:
        return [row[:] for row in grid]
    smallest = min(objs, key=lambda o: o.size)
    largest = max(objs, key=lambda o: o.size)
    out = [row[:] for row in grid]
    for y, x in smallest.pixels:
        out[y][x] = largest.color
    return out


def _align_objects_to_largest_row(grid: Grid) -> Grid:
    objs = get_objects(grid)
    if len(objs) < 2:
        return [row[:] for row in grid]
    largest = max(objs, key=lambda o: o.size)
    target_row = int(round(largest.centroid[0]))
    h, w = _shape(grid)
    out = _blank(h, w, 0)
    for obj in objs:
        src_patch = _extract_object_patch(grid, obj)
        ph, pw = _shape(src_patch)
        top = max(0, min(h - ph, target_row - (ph // 2)))
        left = obj.bbox[1]
        _place_patch(out, src_patch, top, left)
    return out


def _symmetry_complete_h(grid: Grid) -> Grid:
    g = [row[:] for row in grid]
    h, w = _shape(g)
    for y in range(h):
        for x in range(w):
            rx = w - 1 - x
            if g[y][x] == 0 and g[y][rx] != 0:
                g[y][x] = g[y][rx]
    return g


def _symmetry_complete_v(grid: Grid) -> Grid:
    g = [row[:] for row in grid]
    h, w = _shape(g)
    for y in range(h):
        ry = h - 1 - y
        for x in range(w):
            if g[y][x] == 0 and g[ry][x] != 0:
                g[y][x] = g[ry][x]
    return g


def _infer_row_period(grid: Grid) -> int:
    h, _ = _shape(grid)
    for p in range(1, h + 1):
        ok = True
        for y in range(h):
            if grid[y] != grid[y % p]:
                ok = False
                break
        if ok:
            return p
    return h


def _infer_col_period(grid: Grid) -> int:
    h, w = _shape(grid)
    for p in range(1, w + 1):
        ok = True
        for y in range(h):
            for x in range(w):
                if grid[y][x] != grid[y][x % p]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return w


def _repeat_row_pattern(grid: Grid) -> Grid:
    g = [row[:] for row in grid]
    h, _ = _shape(g)
    p = _infer_row_period(g)
    if p >= h:
        return g
    out = [row[:] for row in g]
    for y in range(h):
        out[y] = g[y % p][:]
    return out


def _repeat_col_pattern(grid: Grid) -> Grid:
    g = [row[:] for row in grid]
    h, w = _shape(g)
    p = _infer_col_period(g)
    if p >= w:
        return g
    out = [row[:] for row in g]
    for y in range(h):
        for x in range(w):
            out[y][x] = g[y][x % p]
    return out


def _compose(f: Callable[[Grid], Grid], g: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    return lambda x: f(g(x))


def _build_predicates(features: dict[str, Any]) -> list[Predicate]:
    pairs: list[TrainFeatures] = features["pairs"]
    if len(pairs) < 2:
        return []

    in_grids = [p.input_grid for p in pairs]
    nz_vals = sorted({_nonzero_count(g) for g in in_grids})
    obj_vals = sorted({_object_count(g) for g in in_grids})
    max_obj_vals = sorted({_max_object_size(g) for g in in_grids})
    ratio_vals = sorted({_smallest_largest_ratio(g) for g in in_grids})

    preds: list[Predicate] = []

    for t in nz_vals[:-1]:
        preds.append(Predicate(f"nonzero<={t}", lambda g, tt=t: _nonzero_count(g) <= tt, prior=0.62))
    for t in obj_vals[:-1]:
        preds.append(Predicate(f"obj_count<={t}", lambda g, tt=t: _object_count(g) <= tt, prior=0.70))
    for t in max_obj_vals[:-1]:
        preds.append(Predicate(f"max_obj_size<={t}", lambda g, tt=t: _max_object_size(g) <= tt, prior=0.58))
    for t in ratio_vals[:-1]:
        preds.append(Predicate(f"smallest_largest_ratio<={t:.3f}", lambda g, tt=t: _smallest_largest_ratio(g) <= tt, prior=0.66))

    preds.append(Predicate("height>=width", lambda g: _shape(g)[0] >= _shape(g)[1], prior=0.52))
    preds.append(Predicate("has_h_symmetric_obj", lambda g: any(o.sym_h for o in get_objects(g)), prior=0.50))

    out: list[Predicate] = []
    for p in preds:
        vals = [p.fn(g) for g in in_grids]
        if any(vals) and not all(vals):
            score = predicate_score(p, in_grids)
            if score > PREDICATE_SCORE_MIN:
                out.append(p)
    return out


def _build_object_predicates(features: dict[str, Any]) -> list[Predicate]:
    pairs: list[TrainFeatures] = features["pairs"]
    if len(pairs) < 2:
        return []

    in_grids = [p.input_grid for p in pairs]
    count_vals = sorted({_object_count(g) for g in in_grids})
    ratio_vals = sorted({_smallest_largest_ratio(g) for g in in_grids})
    max_vals = sorted({_max_object_size(g) for g in in_grids})

    preds: list[Predicate] = []
    for t in count_vals[:-1]:
        preds.append(Predicate(f"obj_count>{t}", lambda g, tt=t: _object_count(g) > tt, prior=0.78))
    for t in ratio_vals[:-1]:
        preds.append(Predicate(f"obj_size_ratio<={t:.3f}", lambda g, tt=t: _smallest_largest_ratio(g) <= tt, prior=0.66))
    for t in max_vals[:-1]:
        preds.append(Predicate(f"max_obj_size>{t}", lambda g, tt=t: _max_object_size(g) > tt, prior=0.62))
    preds.append(Predicate("has_h_symmetric_obj", lambda g: any(o.sym_h for o in get_objects(g)), prior=0.56))
    preds.append(Predicate("has_v_symmetric_obj", lambda g: any(o.sym_v for o in get_objects(g)), prior=0.56))

    out: list[Predicate] = []
    for p in preds:
        vals = [p.fn(g) for g in in_grids]
        if any(vals) and not all(vals):
            score = predicate_score(p, in_grids)
            if score > PREDICATE_SCORE_MIN:
                out.append(p)
    return out


def expand_hypotheses(base_hyps: list[Hypothesis], limit: int = TOP_N_FOR_COMPOSE) -> list[Hypothesis]:
    top = sorted(base_hyps, key=lambda h: (-h.prior, h.cost, h.name))[: max(1, limit)]
    composed: list[Hypothesis] = []
    for left in top:
        for right in top:
            if left.name == right.name:
                continue
            name = f"{left.name}->{right.name}"
            composed.append(
                Hypothesis(
                    name,
                    _compose(left.fn, right.fn),
                    cost=left.cost + right.cost,
                    prior=0.55 * min(left.prior, right.prior),
                    kind="compose",
                )
            )
    return composed


def _make_conditional_hypothesis(pred: Predicate, h_true: Hypothesis, h_false: Hypothesis) -> Hypothesis:
    def _fn(grid: Grid, p=pred.fn, t=h_true.fn, f=h_false.fn) -> Grid:
        return t(grid) if p(grid) else f(grid)

    return Hypothesis(
        name=f"if[{pred.name}]?{h_true.name}:{h_false.name}",
        fn=_fn,
        cost=h_true.cost + h_false.cost + 1,
        prior=0.45 * pred.prior + 0.30 * min(h_true.prior, h_false.prior),
        kind="conditional",
        predicate=pred.name,
        true_branch=h_true.name,
        false_branch=h_false.name,
    )


def expand_conditional_hypotheses(base_hyps: list[Hypothesis], features: dict[str, Any]) -> list[Hypothesis]:
    preds = _build_predicates(features)
    if not preds:
        return []

    top_ops = sorted(base_hyps, key=lambda h: (h.cost, -h.prior, h.name))[:TOP_N_FOR_CONDITIONAL_OPS]
    out: list[Hypothesis] = []
    seen_keys: set[str] = set()
    train_pairs = [(p.input_grid, p.output_grid) for p in features["pairs"]]
    for pred in preds:
        if not fast_conditional_filter(pred, train_pairs):
            continue
        for h_true in top_ops:
            for h_false in top_ops:
                if h_true.name == h_false.name:
                    continue
                key = normalize_conditional(pred.name, h_true.name, h_false.name)
                if key in seen_keys:
                    continue
                hyp = _make_conditional_hypothesis(pred, h_true, h_false)
                score = conditional_score(pred, h_true, h_false, train_pairs)
                if score < 0.0:
                    continue
                hyp = Hypothesis(
                    name=hyp.name,
                    fn=hyp.fn,
                    cost=hyp.cost,
                    prior=hyp.prior + (0.2 * score),
                    kind=hyp.kind,
                    predicate=hyp.predicate,
                    true_branch=hyp.true_branch,
                    false_branch=hyp.false_branch,
                    confidence=score,
                )
                out.append(hyp)
                seen_keys.add(key)
                if len(out) >= MAX_CONDITIONAL_HYPOTHESES:
                    return out
    return out


def expand_object_conditional_hypotheses(base_hyps: list[Hypothesis], features: dict[str, Any]) -> list[Hypothesis]:
    if float(features.get("avg_input_object_count", 0.0)) >= float(OBJECT_COUNT_ANCHOR_MAX):
        return []

    preds = _build_object_predicates(features)
    if not preds:
        return []

    train_pairs = [(p.input_grid, p.output_grid) for p in features["pairs"]]
    object_ops = [h for h in base_hyps if h.name in {
        "rotate_smallest_object",
        "flip_smallest_object_h",
        "rotate_largest_object",
        "flip_largest_object_h",
        "recolor_smallest_to_largest_color",
        "align_objects_to_largest_row",
    }]
    if not object_ops:
        return []

    # Keep false branch conservative for cost and stability.
    fallback_ops = [h for h in base_hyps if h.name in {"identity", "rotate90", "flip_h"}]
    if not fallback_ops:
        fallback_ops = [base_hyps[0]]

    out: list[Hypothesis] = []
    seen_keys: set[str] = set()
    for pred in preds:
        if not fast_conditional_filter(pred, train_pairs):
            continue
        for h_true in object_ops[:TOP_N_FOR_CONDITIONAL_OPS]:
            for h_false in fallback_ops[:3]:
                if h_true.name == h_false.name:
                    continue
                key = normalize_conditional(pred.name, h_true.name, h_false.name)
                if key in seen_keys:
                    continue
                hyp = _make_conditional_hypothesis(pred, h_true, h_false)
                score = conditional_score(pred, h_true, h_false, train_pairs)
                if score < OBJECT_CONDITIONAL_MIN_SCORE:
                    continue
                out.append(
                    Hypothesis(
                        name=f"obj::{hyp.name}",
                        fn=hyp.fn,
                        cost=hyp.cost,
                        prior=hyp.prior + (0.25 * score),
                        kind="conditional",
                        predicate=hyp.predicate,
                        true_branch=hyp.true_branch,
                        false_branch=hyp.false_branch,
                        confidence=score,
                    )
                )
                seen_keys.add(key)
                if len(out) >= MAX_OBJECT_CONDITIONAL_HYPOTHESES:
                    return out
    return out


def rank_hypotheses(hypotheses: list[Hypothesis], features: dict[str, Any]) -> list[Hypothesis]:
    def tier(h: Hypothesis) -> int:
        # Priority: recolor > conditional > single > composition.
        if "recolor" in h.name:
            return 0
        if h.kind == "conditional":
            return 1
        if h.kind == "compose":
            return 3
        return 2

    def bonus(name: str) -> float:
        b = 0.0
        if "translate" in name:
            b += 0.3
        if "rotate" in name or "flip" in name:
            b += 0.2
        if "recolor" in name:
            b += 0.4
        if "tile" in name and features.get("all_tiling", False):
            b += 0.6
        if "symmetry" in name or "repeat_" in name:
            b += 0.45
        if name.startswith("obj::"):
            b += 0.50
        if "object" in name:
            b += 0.35
        return b

    return sorted(hypotheses, key=lambda h: (tier(h), h.cost, -(h.prior + bonus(h.name)), h.name))


def generate_hypotheses(
    features: dict[str, Any],
    allow_compose: bool = True,
    allow_conditional: bool = True,
) -> list[Hypothesis]:
    pairs: list[TrainFeatures] = features["pairs"]
    allow_object_heavy = float(features.get("avg_input_object_count", 0.0)) < float(OBJECT_COUNT_ANCHOR_MAX)
    allow_symmetry_ops = bool(features.get("has_any_input_symmetry", False))
    base: list[Hypothesis] = [Hypothesis("identity", _identity, cost=1, prior=1.0, kind="single")]

    if features.get("all_uniform_expand", False):
        for factor in features.get("uniform_expand_factors", ()):
            if int(factor) in (2, 3, 4):
                base.append(
                    Hypothesis(
                        f"expand_pixels_x{int(factor)}",
                        lambda g, f=int(factor): _expand_pixels(g, f),
                        cost=1,
                        prior=0.97,
                        kind="single",
                    )
                )
    if pairs and all((p.output_shape[0] == p.input_shape[0] * p.input_shape[0] and p.output_shape[1] == p.input_shape[1] * p.input_shape[1]) for p in pairs):
        base.append(Hypothesis("kron_nonzero_self", _kron_nonzero_self, cost=2, prior=0.98, kind="single"))
    if pairs and all((p.input_shape == (3, 3) and p.output_shape == (3, 3)) for p in pairs):
        marker = _infer_constant_output_marker(pairs)
        if marker is not None:
            base.append(
                Hypothesis(
                    "triad_majority_to_pattern_marker",
                    lambda g, m=int(marker): _triad_majority_to_pattern_marker(g, m),
                    cost=1,
                    prior=0.98,
                    kind="single",
                )
            )
        uniform_marker = _infer_uniform_rows_marker_rule(pairs)
        if uniform_marker is not None:
            base.append(
                Hypothesis(
                    "uniform_rows_to_marker",
                    lambda g, m=int(uniform_marker): _uniform_rows_to_marker(g, m),
                    cost=1,
                    prior=0.98,
                    kind="single",
                )
            )
        row_map = _infer_rowpos_color_map(pairs)
        if row_map:
            base.append(
                Hypothesis(
                    "rowpos_to_color_rows",
                    lambda g, m=dict(row_map): _rowpos_to_color_rows(g, m),
                    cost=1,
                    prior=0.98,
                    kind="single",
                )
            )
    prefix_rule = _infer_prefix_append_recolor_rule(pairs)
    if prefix_rule is not None:
        k, m = prefix_rule
        base.append(
            Hypothesis(
                "recolor_and_append_prefix_rows",
                lambda g, kk=int(k), mm=int(m): _recolor_and_append_prefix_rows(g, kk, mm),
                cost=1,
                prior=0.99,
                kind="single",
            )
        )
    period_rule = _infer_recolor_repeat_row_period_rule(pairs)
    if period_rule is not None:
        oh, m = period_rule
        base.append(
            Hypothesis(
                "recolor_repeat_row_period",
                lambda g, out_h=int(oh), mm=int(m): _recolor_repeat_row_period(g, out_h, mm),
                cost=1,
                prior=0.99,
                kind="single",
            )
        )
    if _infer_cross_quadrant_rule(pairs):
        base.append(Hypothesis("cross_select_anomalous_quadrant", _select_anomalous_quadrant_from_cross, cost=1, prior=0.99, kind="single"))
    sym_rule = _infer_symmetry_to_label_rule(pairs)
    if sym_rule is not None:
        sym_c, asym_c = sym_rule
        base.append(
            Hypothesis(
                "symmetry_to_label",
                lambda g, s=int(sym_c), a=int(asym_c): _symmetry_to_label(g, s, a),
                cost=1,
                prior=0.99,
                kind="single",
            )
        )
    if _infer_append_vertical_mirror_rule(pairs):
        base.append(Hypothesis("append_vertical_mirror", _append_vertical_mirror, cost=1, prior=0.99, kind="single"))
    if _infer_concat_horizontal_mirror_rule(pairs):
        base.append(Hypothesis("concat_horizontal_mirror", _concat_horizontal_mirror, cost=1, prior=0.99, kind="single"))
    if _infer_concat_horizontal_shift1_rule(pairs):
        base.append(Hypothesis("concat_horizontal_shift1", _concat_horizontal_shift1, cost=1, prior=0.99, kind="single"))
    shift_color = _infer_shift_down_recolor_rule(pairs)
    if shift_color is not None:
        base.append(
            Hypothesis(
                "shift_down_recolor",
                lambda g, c=int(shift_color): _shift_down_recolor(g, c),
                cost=1,
                prior=0.99,
                kind="single",
            )
        )
    if _infer_extend_rows_by_period_to_double_width_rule(pairs):
        base.append(Hypothesis("extend_rows_by_period_double_width", _extend_rows_by_period_to_double_width, cost=1, prior=0.99, kind="single"))
    anchor = _infer_recolor_non_anchor_to_five_rule(pairs)
    if anchor is not None:
        base.append(
            Hypothesis(
                "recolor_non_anchor_to_five",
                lambda g, a=int(anchor): _recolor_non_anchor_to_five(g, a),
                cost=1,
                prior=0.99,
                kind="single",
            )
        )
    if _infer_diag_bar_rule(pairs):
        base.append(Hypothesis("draw_right_diag_and_bottom_bar", _draw_right_diag_and_bottom_bar, cost=1, prior=0.99, kind="single"))
    if _infer_triptych_rule(pairs):
        base.append(Hypothesis("compose_triptych_from_left", _compose_triptych_from_left, cost=1, prior=0.99, kind="single"))
    if _infer_sliding_diagonal_rule(pairs):
        base.append(Hypothesis("build_sliding_diagonal_from_row", _build_sliding_diagonal_from_row, cost=1, prior=0.99, kind="single"))
    if _infer_preserve_uniform_row_anchor_else_five_rule(pairs):
        base.append(Hypothesis("preserve_uniform_row_anchor_else_five", _preserve_uniform_row_anchor_else_five, cost=1, prior=0.99, kind="single"))
    if _infer_fill_separator_blocks_123_rule(pairs):
        base.append(Hypothesis("fill_separator_blocks_123", _fill_separator_blocks_123, cost=1, prior=0.99, kind="single"))
    marker_seq_map = _infer_marker_sequence_rewrite_rule(pairs)
    if marker_seq_map is not None:
        base.append(
            Hypothesis(
                "marker_sequence_rewrite",
                lambda g, m=dict(marker_seq_map): _marker_sequence_rewrite(g, m),
                cost=1,
                prior=0.995,
                kind="single",
            )
        )

    if features["all_same_shape"]:
        base.extend(
            [
                Hypothesis("rotate90", _rotate90, cost=1, prior=0.9),
                Hypothesis("flip_h", _flip_h, cost=1, prior=0.85),
                Hypothesis("flip_v", _flip_v, cost=1, prior=0.85),
            ]
        )

        for dy, dx in product(range(-2, 3), repeat=2):
            if dy == 0 and dx == 0:
                continue
            prior = 0.8 - 0.08 * (abs(dy) + abs(dx))
            base.append(Hypothesis(f"translate_{dy}_{dx}", lambda g, y=dy, x=dx: _translate(g, y, x), cost=1, prior=prior, kind="single"))

        if allow_object_heavy:
            base.append(Hypothesis("rotate_each_object", lambda g: _apply_to_each_object(g, _rotate90), cost=2, prior=0.72, kind="single"))
            base.append(Hypothesis("flip_each_object_h", lambda g: _apply_to_each_object(g, _flip_h), cost=2, prior=0.68, kind="single"))
            base.append(Hypothesis("rotate_smallest_object", _rotate_smallest_object, cost=2, prior=0.74, kind="single"))
            base.append(Hypothesis("flip_smallest_object_h", _flip_smallest_object_h, cost=2, prior=0.72, kind="single"))
            base.append(Hypothesis("rotate_largest_object", _rotate_largest_object, cost=2, prior=0.71, kind="single"))
            base.append(Hypothesis("flip_largest_object_h", _flip_largest_object_h, cost=2, prior=0.69, kind="single"))
            base.append(Hypothesis("move_smallest_object_to_center", _move_smallest_to_center, cost=2, prior=0.78, kind="single"))
            base.append(Hypothesis("move_largest_object_to_center", _move_largest_to_center, cost=2, prior=0.74, kind="single"))
            base.append(Hypothesis("recolor_smallest_to_largest_color", _recolor_smallest_to_largest_color, cost=2, prior=0.76, kind="single"))
            base.append(Hypothesis("align_objects_to_largest_row", _align_objects_to_largest_row, cost=2, prior=0.70, kind="single"))
        if allow_symmetry_ops:
            base.append(Hypothesis("symmetry_complete_h", _symmetry_complete_h, cost=1, prior=0.72, kind="single"))
            base.append(Hypothesis("symmetry_complete_v", _symmetry_complete_v, cost=1, prior=0.72, kind="single"))
        base.append(Hypothesis("repeat_row_pattern", _repeat_row_pattern, cost=1, prior=0.65, kind="single"))
        base.append(Hypothesis("repeat_col_pattern", _repeat_col_pattern, cost=1, prior=0.65, kind="single"))

    cmap = _infer_color_map(pairs)
    if cmap and features["all_same_shape"] and features["palette_changed"]:
        base.append(Hypothesis("recolor_map", lambda g, m=cmap: _apply_color_map(g, m), cost=1, prior=0.96, kind="single"))

    if features["all_output_smaller_or_equal"]:
        base.append(Hypothesis("crop_nonzero_bbox", _crop_nonzero_bbox, cost=1, prior=0.88, kind="single"))

    if features.get("has_striped_input", False) and not features.get("all_same_shape", False):
        base.append(Hypothesis("stripe_interleave_two_rows", _stripe_interleave_two_rows, cost=1, prior=0.93, kind="single"))
        base.append(Hypothesis("triad_majority_to_pattern", _triad_majority_to_pattern, cost=1, prior=0.93, kind="single"))
        base.append(Hypothesis("stripe_reduce_to_period_vector", _stripe_reduce_to_period_vector, cost=1, prior=0.92, kind="single"))
        base.append(Hypothesis("extract_frame_interior", _extract_frame_interior, cost=1, prior=0.90, kind="single"))
        base.append(Hypothesis("collapse_stripe_runs", _collapse_stripe_runs, cost=1, prior=0.97, kind="single"))

    if features.get("all_tiling", False):
        for factor in features.get("tiling_factors", ()):
            if factor in (2, 3):
                base.append(Hypothesis(f"tile_x{factor}", lambda g, f=factor: _tile(g, f), cost=1, prior=0.97, kind="single"))

    composed = expand_hypotheses(base, limit=TOP_N_FOR_COMPOSE) if allow_compose else []
    conditional = expand_conditional_hypotheses(base, features) if allow_conditional else []
    obj_conditional = expand_object_conditional_hypotheses(base, features) if allow_conditional else []
    hyps = base + composed + conditional + obj_conditional

    by_name: dict[str, Hypothesis] = {}
    for h in hyps:
        by_name.setdefault(h.name, h)
    return rank_hypotheses(list(by_name.values()), features)


def _rank_hypotheses_b(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    def tier(h: Hypothesis) -> int:
        n = h.name
        if h.kind == "conditional" or n.startswith("b::cond::"):
            return 0
        if "relational::" in n or "objrel" in n:
            return 1
        if "recolor" in n or "count" in n:
            return 2
        if "compose" in n or h.kind == "compose":
            return 3
        return 4

    return sorted(hypotheses, key=lambda h: (tier(h), h.cost, -h.prior, h.name))


def generate_hypotheses_b(features: dict[str, Any]) -> list[Hypothesis]:
    # B-exclusive registry: intentionally different from base ranking/generation.
    pairs: list[TrainFeatures] = features["pairs"]
    hyps: list[Hypothesis] = [Hypothesis("b::identity", _identity, cost=1, prior=0.30, kind="single")]
    rel_score = _relational_signature_score(features)
    avg_objs = float(features.get("avg_input_object_count", 0.0))

    # Family 1: relational micro-kernel and object-relational specialists.
    if rel_score >= 0.30:
        for h in _build_clite_relational_candidates(features):
            hyps.append(Hypothesis(f"b::{h.name}", h.fn, cost=h.cost, prior=min(0.99, h.prior + 0.04), kind="single"))
    if avg_objs <= 6.0:
        hyps.extend(
            [
                Hypothesis("b::objrel::snap_smallest_to_largest_side", _snap_smallest_to_largest_side, cost=2, prior=0.91, kind="single"),
                Hypothesis("b::objrel::align_smallest_to_largest_row", _align_smallest_to_largest_row, cost=2, prior=0.90, kind="single"),
                Hypothesis("b::objrel::recolor_smallest_by_nearest_neighbor", _recolor_smallest_by_nearest_neighbor, cost=2, prior=0.90, kind="single"),
                Hypothesis("b::objrel::snap_to_closest_neighbor", _snap_to_closest_neighbor, cost=2, prior=0.88, kind="single"),
                Hypothesis("b::objrel::align_objects_to_largest_row", _align_objects_to_largest_row, cost=2, prior=0.88, kind="single"),
            ]
        )

    # Family 2: count/frequency and palette specialists.
    hyps.extend(
        [
            Hypothesis("b::count::repeat_row_pattern", _repeat_row_pattern, cost=1, prior=0.84, kind="single"),
            Hypothesis("b::count::repeat_col_pattern", _repeat_col_pattern, cost=1, prior=0.84, kind="single"),
            Hypothesis("b::count::crop_nonzero_bbox", _crop_nonzero_bbox, cost=1, prior=0.82, kind="single"),
            Hypothesis("b::count::recolor_smallest_to_largest_color", _recolor_smallest_to_largest_color, cost=2, prior=0.85, kind="single"),
        ]
    )
    cmap = _infer_color_map(pairs)
    if cmap and bool(features.get("all_same_shape", False)):
        hyps.append(Hypothesis("b::count::recolor_map_loose", lambda g, m=cmap: _apply_color_map(g, m), cost=1, prior=0.89, kind="single"))

    # Family 3: cross-example conditional programs (specialists, not base-ranked).
    cond_ops = [
        Hypothesis("identity", _identity, cost=1, prior=1.0, kind="single"),
        Hypothesis("flip_h", _flip_h, cost=1, prior=0.92, kind="single"),
        Hypothesis("flip_v", _flip_v, cost=1, prior=0.90, kind="single"),
        Hypothesis("rotate90", _rotate90, cost=1, prior=0.88, kind="single"),
        Hypothesis("recolor_smallest_to_largest_color", _recolor_smallest_to_largest_color, cost=2, prior=0.74, kind="single"),
        Hypothesis("snap_smallest_to_largest_side", _snap_smallest_to_largest_side, cost=2, prior=0.72, kind="single"),
    ]
    for h in expand_conditional_hypotheses(cond_ops, features)[:20]:
        hyps.append(
            Hypothesis(
                name=f"b::cond::{h.name}",
                fn=h.fn,
                cost=h.cost,
                prior=min(0.99, h.prior + 0.03),
                kind=h.kind,
                predicate=h.predicate,
                true_branch=h.true_branch,
                false_branch=h.false_branch,
                confidence=h.confidence,
            )
        )
    for h in expand_object_conditional_hypotheses(cond_ops, features)[:12]:
        hyps.append(
            Hypothesis(
                name=f"b::cond::{h.name}",
                fn=h.fn,
                cost=h.cost,
                prior=min(0.99, h.prior + 0.03),
                kind=h.kind,
                predicate=h.predicate,
                true_branch=h.true_branch,
                false_branch=h.false_branch,
                confidence=h.confidence,
            )
        )

    by_name: dict[str, Hypothesis] = {}
    for h in hyps:
        by_name.setdefault(h.name, h)
    return _rank_hypotheses_b(list(by_name.values()))


def predicate_score(pred: Predicate, input_grids: list[Grid]) -> float:
    outcomes = [bool(pred.fn(g)) for g in input_grids]
    true_count = sum(1 for x in outcomes if x)
    false_count = len(outcomes) - true_count
    if true_count == 0 or false_count == 0:
        return -1.0
    return min(true_count, false_count) / max(true_count, false_count)


def fast_conditional_filter(pred: Predicate, train_pairs: list[tuple[Grid, Grid]]) -> bool:
    seen: set[bool] = set()
    for inp, _ in train_pairs:
        seen.add(bool(pred.fn(inp)))
    return len(seen) > 1


def conditional_score(
    pred: Predicate,
    op_true: Hypothesis,
    op_false: Hypothesis,
    train_pairs: list[tuple[Grid, Grid]],
) -> float:
    total_cells = 0
    matched = 0
    true_used = 0
    false_used = 0

    for inp, expected in train_pairs:
        if pred.fn(inp):
            pred_out = op_true.fn(inp)
            true_used += 1
        else:
            pred_out = op_false.fn(inp)
            false_used += 1

        if _shape(pred_out) != _shape(expected):
            return -1.0

        h, w = _shape(expected)
        total_cells += h * w
        matched += sum(1 for y in range(h) for x in range(w) if pred_out[y][x] == expected[y][x])

    if true_used == 0 or false_used == 0:
        return -1.0
    if total_cells <= 0:
        return -1.0
    return matched / total_cells


def normalize_conditional(pred_name: str, op1_name: str, op2_name: str) -> str:
    if op1_name > op2_name:
        return f"if[{pred_name}]?{op2_name}:{op1_name}"
    return f"if[{pred_name}]?{op1_name}:{op2_name}"


def fast_fail(pred: Grid, expected: Grid, tolerance: int = 0) -> bool:
    if _shape(pred) != _shape(expected):
        return True
    mismatch = sum(1 for y in range(len(pred)) for x in range(len(pred[0])) if pred[y][x] != expected[y][x])
    return mismatch > tolerance


def _exact_and_partial_score(pred: Grid, expected: Grid) -> tuple[bool, float]:
    if _shape(pred) != _shape(expected):
        return False, 0.0
    total = len(pred) * len(pred[0]) if pred else 0
    if total == 0:
        return True, 1.0
    same = sum(1 for y in range(len(pred)) for x in range(len(pred[0])) if pred[y][x] == expected[y][x])
    ratio = same / max(1, total)
    return ratio == 1.0, ratio


def _hyp_family(h: Hypothesis) -> str:
    name = h.name
    if name.startswith("obj::"):
        return "object_conditional"
    if "recolor" in name:
        return "color"
    if "tile" in name:
        return "tiling"
    if h.kind == "conditional":
        return "conditional"
    if h.kind == "compose":
        return "compose"
    if "object" in name or "smallest" in name:
        return "object"
    if "rotate" in name or "flip" in name or "translate" in name:
        return "geom"
    return "single"


def _build_portfolio_buckets(hypotheses: list[Hypothesis], tier: TierConfig) -> list[tuple[str, list[Hypothesis]]]:
    buckets: list[tuple[str, list[Hypothesis]]] = []
    for fam_tuple in tier.portfolio:
        fam_set = set(fam_tuple)
        selected = [h for h in hypotheses if _hyp_family(h) in fam_set]
        if selected:
            buckets.append(("+".join(fam_tuple), selected))

    # Always keep a final global bucket for safety.
    buckets.append(("all", hypotheses))
    return buckets


def _build_bplus_candidates(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    # B+ is a narrow, mid-power lane: structurally-targeted hypotheses with a
    # hard cap. It includes a tiny geometric chain pack learned from Tier-C
    # winners plus selected object conditionals.
    out: list[Hypothesis] = []
    by_name: dict[str, Hypothesis] = {}

    def _add(h: Hypothesis) -> None:
        if h.name not in by_name and len(by_name) < 12:
            by_name[h.name] = h

    # Anchors.
    for h in hypotheses:
        if h.name in {"identity", "recolor_map", "crop_nonzero_bbox"}:
            _add(h)

    # High-signal object pattern pack.
    for h in hypotheses:
        if h.name in {
            "obj::if[obj_count>3]?rotate_smallest_object:identity",
            "obj::if[obj_count>4]?rotate_smallest_object:identity",
            "obj::if[obj_count>3]?flip_largest_object_h:identity",
            "obj::if[obj_count>4]?flip_largest_object_h:identity",
            "obj::if[obj_size_ratio<=0.333]?recolor_smallest_to_largest_color:identity",
        }:
            _add(h)

    # Existing object/conditional signals.
    for h in hypotheses:
        if h.name.startswith("obj::") or h.kind == "conditional":
            _add(h)

    # Structured completion transforms.
    for h in hypotheses:
        if "symmetry" in h.name or "repeat_" in h.name:
            _add(h)

    # Tiny geometric chain pack (Tier-C distilled).
    chain_pack = [
        ("bplus::flip_h->flip_v", _compose(_flip_v, _flip_h)),
        ("bplus::flip_h->rotate90", _compose(_rotate90, _flip_h)),
        ("bplus::rotate90->flip_h", _compose(_flip_h, _rotate90)),
        ("bplus::rotate90->rotate90", _compose(_rotate90, _rotate90)),
    ]
    for name, fn in chain_pack:
        _add(Hypothesis(name=name, fn=fn, cost=2, prior=0.90, kind="compose"))

    out.extend(by_name.values())
    return out


def _build_clite_candidates(hypotheses: list[Hypothesis], symmetry_only: bool) -> list[Hypothesis]:
    # C-lite: fixed 2-step geometric macro-ops (primitive lane), no broad tree.
    by_name: dict[str, Hypothesis] = {}

    def _add(h: Hypothesis) -> None:
        if h.name not in by_name and len(by_name) < 12:
            by_name[h.name] = h

    # Keep a couple of cheap geometric anchors from existing pool.
    for h in hypotheses:
        if h.name in {"identity", "flip_h", "flip_v", "rotate90"}:
            _add(h)

    macros: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("clite::flip_h->flip_v", _compose(_flip_v, _flip_h)),
        ("clite::flip_h->rotate90", _compose(_rotate90, _flip_h)),
        ("clite::rotate90->flip_h", _compose(_flip_h, _rotate90)),
        ("clite::rotate90->rotate90", _compose(_rotate90, _rotate90)),
        ("clite::flip_v->rotate90", _compose(_rotate90, _flip_v)),
        ("clite::rotate90->flip_v", _compose(_flip_v, _rotate90)),
    ]
    if symmetry_only:
        macros = [m for m in macros if "flip" in m[0]]

    for name, fn in macros:
        _add(Hypothesis(name=name, fn=fn, cost=2, prior=0.94, kind="compose"))

    return list(by_name.values())


def _build_clite_relational_candidates(features: dict[str, Any]) -> list[Hypothesis]:
    # Expanded micro-kernel relational pack. Budgeting is controlled by tier-level max_eval.
    if bool(features.get("has_striped_input", False)):
        hyps = [
            Hypothesis("relational::collapse_stripe_runs", _collapse_stripe_runs, cost=1, prior=0.97, kind="single"),
            Hypothesis("relational::route_control_color_paths", _route_control_color_paths, cost=2, prior=0.94, kind="single"),
            Hypothesis("relational::recolor_by_coordinate_parity", _recolor_by_coordinate_parity, cost=1, prior=0.95, kind="single"),
            Hypothesis("relational::stripe_interleave_two_rows", _stripe_interleave_two_rows, cost=1, prior=0.96, kind="single"),
            Hypothesis("relational::triad_majority_to_pattern", _triad_majority_to_pattern, cost=1, prior=0.96, kind="single"),
            Hypothesis("relational::stripe_reduce_to_period_vector", _stripe_reduce_to_period_vector, cost=1, prior=0.95, kind="single"),
            Hypothesis("relational::extract_frame_interior", _extract_frame_interior, cost=1, prior=0.94, kind="single"),
            Hypothesis("relational::extrapolate_stripe", _extrapolate_stripe, cost=1, prior=0.93, kind="single"),
            Hypothesis("relational::reflect_across_stripe", _reflect_across_stripe, cost=1, prior=0.92, kind="single"),
            Hypothesis("relational::snap_smallest_to_largest_side", _snap_smallest_to_largest_side, cost=2, prior=0.91, kind="single"),
            Hypothesis("relational::align_smallest_to_largest_row", _align_smallest_to_largest_row, cost=2, prior=0.90, kind="single"),
            Hypothesis("relational::recolor_smallest_by_nearest_neighbor", _recolor_smallest_by_nearest_neighbor, cost=2, prior=0.90, kind="single"),
            Hypothesis("relational::snap_to_closest_neighbor", _snap_to_closest_neighbor, cost=2, prior=0.89, kind="single"),
        ]
        biax = _build_biaxial_marker_expand_hypothesis(features)
        if biax is not None:
            hyps.insert(0, biax)
        return hyps

    return [
        Hypothesis("relational::route_control_color_paths", _route_control_color_paths, cost=2, prior=0.92, kind="single"),
        Hypothesis("relational::recolor_by_coordinate_parity", _recolor_by_coordinate_parity, cost=1, prior=0.94, kind="single"),
        Hypothesis("relational::stripe_interleave_two_rows", _stripe_interleave_two_rows, cost=1, prior=0.88, kind="single"),
        Hypothesis("relational::triad_majority_to_pattern", _triad_majority_to_pattern, cost=1, prior=0.88, kind="single"),
        Hypothesis("relational::mirror_across_global_axis_h", _mirror_across_global_axis_h, cost=1, prior=0.90, kind="single"),
        Hypothesis("relational::mirror_across_global_axis_v", _mirror_across_global_axis_v, cost=1, prior=0.90, kind="single"),
        Hypothesis("relational::rotate270", _rotate270, cost=1, prior=0.92, kind="single"),
        Hypothesis("relational::marker_frame_fill", _marker_frame_fill, cost=2, prior=0.88, kind="single"),
        Hypothesis("relational::snap_to_closest_neighbor", _snap_to_closest_neighbor, cost=2, prior=0.86, kind="single"),
        Hypothesis("relational::snap_smallest_to_largest_side", _snap_smallest_to_largest_side, cost=2, prior=0.91, kind="single"),
        Hypothesis("relational::align_smallest_to_largest_row", _align_smallest_to_largest_row, cost=2, prior=0.90, kind="single"),
        Hypothesis("relational::recolor_smallest_by_nearest_neighbor", _recolor_smallest_by_nearest_neighbor, cost=2, prior=0.90, kind="single"),
        Hypothesis("relational::align_objects_to_largest_row", _align_objects_to_largest_row, cost=2, prior=0.88, kind="single"),
    ]


def _confidence_from_stats(exact_count: int, mean_partial: float, n_pairs: int, hyp_cost: int) -> float:
    if n_pairs <= 0:
        return 0.0
    exact_ratio = exact_count / n_pairs
    complexity_bonus = max(0.0, 1.0 - 0.15 * max(0, hyp_cost - 1))
    conf = (0.65 * exact_ratio) + (0.30 * mean_partial) + (0.05 * complexity_bonus)
    return max(0.0, min(1.0, conf))


def _better_candidate(
    exact_count: int,
    mean_partial: float,
    cost: int,
    prior: float,
    best_exact: int,
    best_partial: float,
    best_cost: int,
    best_prior: float,
) -> bool:
    if exact_count > best_exact:
        return True
    if exact_count == best_exact and mean_partial > best_partial:
        return True
    if exact_count == best_exact and abs(mean_partial - best_partial) <= 1e-12:
        return cost < best_cost or (cost == best_cost and prior > best_prior)
    return False


def _should_escalate_from_a(
    features: dict[str, Any],
    a_exact_count: int,
    a_mean_partial: float,
    a_confidence: float,
    n_pairs: int,
    a_best_family: str | None = None,
    a_score_margin: float = 0.0,
    a_best_score: float | None = None,
) -> bool:
    decision = _escalation_decision_from_a(
        features,
        a_exact_count,
        a_mean_partial,
        a_confidence,
        n_pairs,
        a_best_family,
        a_score_margin,
        a_best_score,
    )
    return bool(decision["should_escalate"])


def _escalation_decision_from_a(
    features: dict[str, Any],
    a_exact_count: int,
    a_mean_partial: float,
    a_confidence: float,
    n_pairs: int,
    a_best_family: str | None = None,
    a_score_margin: float = 0.0,
    a_best_score: float | None = None,
) -> dict[str, Any]:
    input_grids = [p.input_grid for p in features.get("pairs", [])]
    split_signal = any(
        predicate_score(pred, input_grids) > PREDICATE_SCORE_MIN
        for pred in (_build_predicates(features) + _build_object_predicates(features))
    )
    # Keep the cheap motorbike path unless A-fit is weak and a hard-task signature is present.
    striped_sig = bool(features.get("has_striped_input", False))
    relational_score = float(_relational_signature_score(features))
    object_low_margin = (a_best_family == "object" and float(a_score_margin) < 0.01)
    best_score = float(
        (a_exact_count + max(0.0, a_mean_partial))
        if a_best_score is None
        else a_best_score
    )
    hard_override = (
        striped_sig
        and relational_score >= RELITE_B_SCORE_MIN
        and a_best_family != "object"
    )
    conditional_sig = bool(_is_conditional_signature(features))
    hard_signature = conditional_sig or hard_override
    if object_low_margin and not hard_override:
        return {
            "should_escalate": False,
            "reason": "suppress_object_low_margin",
            "entry_point": "a_gate",
            "hard_override": hard_override,
            "hard_signature": hard_signature,
            "conditional_signature": conditional_sig,
            "split_signal": bool(split_signal),
        }
    if a_exact_count >= n_pairs and n_pairs > 0:
        return {
            "should_escalate": False,
            "reason": "a_exact_train_fit",
            "entry_point": "a_gate",
            "hard_override": hard_override,
            "hard_signature": hard_signature,
            "conditional_signature": conditional_sig,
            "split_signal": bool(split_signal),
        }
    # Preserve conditional reasoning path: these tasks often require branch ops
    # even when A appears partially strong.
    if conditional_sig and a_exact_count < n_pairs:
        return {
            "should_escalate": True,
            "reason": "conditional_signature_path",
            "entry_point": "a_gate",
            "hard_override": hard_override,
            "hard_signature": hard_signature,
            "conditional_signature": conditional_sig,
            "split_signal": bool(split_signal),
        }
    # For striped/relational hard overrides, require relational support
    # and weak/ambiguous A to avoid cost sinks.
    weak_a_fit = best_score < HARD_SIGNATURE_SCORE_MAX
    ambiguous_a = float(a_score_margin) < HARD_SIGNATURE_MARGIN_MAX
    if hard_signature and a_exact_count < n_pairs:
        should = (relational_score >= RELITE_B_SCORE_MIN) and weak_a_fit and ambiguous_a
        return {
            "should_escalate": should,
            "reason": ("hard_signature_weak_ambiguous" if should else "hard_signature_but_a_not_weak_ambiguous"),
            "entry_point": "a_gate",
            "hard_override": hard_override,
            "hard_signature": hard_signature,
            "conditional_signature": conditional_sig,
            "split_signal": bool(split_signal),
        }
    if a_mean_partial >= 0.72 or a_confidence >= 0.30:
        return {
            "should_escalate": False,
            "reason": "a_fit_good_enough",
            "entry_point": "a_gate",
            "hard_override": hard_override,
            "hard_signature": hard_signature,
            "conditional_signature": conditional_sig,
            "split_signal": bool(split_signal),
        }
    # Very-weak fallback only for truly catastrophic A fit.
    should = n_pairs >= 3 and a_exact_count == 0 and best_score < VERY_WEAK_FALLBACK_SCORE_MAX
    return {
        "should_escalate": should,
        "reason": ("very_weak_a_fallback" if should else "default_no_escalate"),
        "entry_point": "a_gate",
        "hard_override": hard_override,
        "hard_signature": hard_signature,
        "conditional_signature": conditional_sig,
        "split_signal": bool(split_signal),
    }


def select_best_with_meta(
    hypotheses: list[Hypothesis],
    train: list[dict[str, Any]],
    max_eval: int | None = None,
) -> dict[str, Any]:
    pairs = [(_to_grid(ex["input"]), _to_grid(ex["output"])) for ex in train]
    lookup = {h.name: h for h in hypotheses}

    @lru_cache(maxsize=8192)
    def _eval(name: str, pair_idx: int) -> tuple[bool, float]:
        hyp = lookup[name]
        inp, out = pairs[pair_idx]
        try:
            pred = hyp.fn(inp)
        except Exception:
            return False, 0.0
        if _shape(pred) != _shape(out):
            return False, 0.0
        return _exact_and_partial_score(pred, out)

    best: Hypothesis | None = None
    best_exact_count = -1
    best_partial = -1.0
    best_cost = 10**9
    best_prior = -1.0
    second: Hypothesis | None = None
    second_exact_count = -1
    second_partial = -1.0
    second_cost = 10**9
    second_prior = -1.0
    evaluated = 0

    eval_queue = hypotheses[:]
    if max_eval is not None:
        eval_queue = eval_queue[: max(1, int(max_eval))]

    for hyp in eval_queue:
        evaluated += 1
        exact_count = 0
        partial_sum = 0.0
        invalid = False

        for i in range(len(pairs)):
            exact, partial = _eval(hyp.name, i)
            exact_count += int(exact)
            partial_sum += partial

            remaining = len(pairs) - (i + 1)
            if exact_count + remaining < best_exact_count:
                invalid = True
                break

        if invalid:
            continue

        mean_partial = partial_sum / max(1, len(pairs))
        if _better_candidate(exact_count, mean_partial, hyp.cost, hyp.prior, best_exact_count, best_partial, best_cost, best_prior):
            if best is not None:
                second = best
                second_exact_count = best_exact_count
                second_partial = best_partial
                second_cost = best_cost
                second_prior = best_prior
            best = hyp
            best_exact_count = exact_count
            best_partial = mean_partial
            best_cost = hyp.cost
            best_prior = hyp.prior

            if best_exact_count == len(pairs):
                break
        elif _better_candidate(
            exact_count,
            mean_partial,
            hyp.cost,
            hyp.prior,
            second_exact_count,
            second_partial,
            second_cost,
            second_prior,
        ):
            second = hyp
            second_exact_count = exact_count
            second_partial = mean_partial
            second_cost = hyp.cost
            second_prior = hyp.prior

    best_score = float(max(0, best_exact_count) + max(0.0, best_partial))
    second_score = float(max(0, second_exact_count) + max(0.0, second_partial))
    score_margin = float(best_score - second_score)

    return {
        "best": best,
        "best_exact_count": int(best_exact_count),
        "best_mean_partial": float(max(0.0, best_partial)),
        "best_score": best_score,
        "second_best": second,
        "second_best_name": (second.name if second else None),
        "second_best_exact_count": int(second_exact_count),
        "second_best_mean_partial": float(max(0.0, second_partial)),
        "second_best_score": second_score,
        "score_margin": score_margin,
        "evaluated": int(evaluated),
        "confidence": _confidence_from_stats(
            int(max(0, best_exact_count)),
            float(max(0.0, best_partial)),
            len(pairs),
            int(best.cost if best is not None else 9),
        ),
    }


def select_best(hypotheses: list[Hypothesis], train: list[dict[str, Any]]) -> Hypothesis | None:
    return select_best_with_meta(hypotheses, train).get("best")


def apply(hypothesis: Hypothesis | None, grid: Any) -> Grid:
    g = _to_grid(grid)
    if hypothesis is None:
        return [row[:] for row in g]
    return hypothesis.fn(g)


def explain(hypothesis: Hypothesis | None) -> dict[str, Any]:
    if hypothesis is None:
        return {"type": "none", "predicate": None, "true_branch": None, "false_branch": None, "confidence": 0.0}
    return {
        "type": hypothesis.kind,
        "predicate": hypothesis.predicate,
        "true_branch": hypothesis.true_branch,
        "false_branch": hypothesis.false_branch,
        "confidence": float(hypothesis.confidence),
    }


def solve_with_report(
    task: dict[str, Any],
    start_tier: str = "A",
    escalate: bool = True,
    b_max_eval_override: int | None = None,
    allow_tier_c: bool | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    train = task["train"]
    test = task["test"]
    policy = _load_meta_policy(policy_path)
    train_key = _train_cache_key(train)
    features = _FEATURES_CACHE.get(train_key)
    if features is None:
        features = extract_features(train)
        _cache_set(_FEATURES_CACHE, train_key, features)
    task_features = _feature_summary(features)
    pre_tier_logs: list[dict[str, Any]] = []

    if _is_tier0_easy(features):
        hyp_key = (train_key, False, False)
        hyps = _HYP_CACHE.get(hyp_key)
        if hyps is None:
            hyps = generate_hypotheses(features, allow_compose=False, allow_conditional=False)
            _cache_set(_HYP_CACHE, hyp_key, hyps)
        fast_hyps = [h for h in hyps if _hyp_family(h) in {"color", "geom", "single", "tiling"}]
        if not fast_hyps:
            fast_hyps = hyps
        meta = select_best_with_meta(fast_hyps, train, max_eval=10)
        best = meta.get("best")
        pre_tier_logs.append(
            {
                "tier": "T0",
                "max_eval": 10,
                "allow_compose": False,
                "allow_conditional": False,
                "evaluated": int(meta.get("evaluated", 0)),
                "best_name": (best.name if best else None),
                "best_exact_count": int(meta.get("best_exact_count", -1)),
                "best_mean_partial": float(meta.get("best_mean_partial", 0.0)),
                "confidence": float(meta.get("confidence", 0.0)),
                "complexity": float(_task_complexity(features)),
            }
        )
        if best is not None and float(meta.get("confidence", 0.0)) >= 0.90:
            predictions: list[Grid] = []
            for t in test:
                inp = t["input"] if isinstance(t, dict) else t
                predictions.append(apply(best, inp))
            return {
                "predictions": predictions,
                "best_hypothesis": best,
                "explanation": explain(best),
                "selected_tier": "T0",
                "tiers_tried": pre_tier_logs,
                "confidence": float(meta.get("confidence", 0.0)),
                "evaluated_hypotheses": int(meta.get("evaluated", 0)),
                "task_features": task_features,
            }

    tier_order = ("A", "B", "C")
    start_tier = start_tier if start_tier in TIER_CONFIGS else "A"
    start_idx = tier_order.index(start_tier)
    hard_c_signature = (
        _is_frontier_signature(features)
        or bool(features.get("has_striped_input", False))
        or _is_conditional_signature(features)
    )
    if not escalate:
        active_tiers = (start_tier,)
    else:
        tiers = list(tier_order[start_idx:])
        allow_c_from_policy = bool(policy.get("allow_tier_c", True))
        if allow_tier_c is not None:
            allow_c_from_policy = bool(allow_tier_c)
        if "C" in tiers and (not hard_c_signature or not allow_c_from_policy):
            tiers = [t for t in tiers if t != "C"]
        active_tiers = tuple(tiers)

    global_best: Hypothesis | None = None
    global_best_exact = -1
    global_best_partial = -1.0
    global_best_cost = 10**9
    global_best_prior = -1.0
    global_confidence = 0.0
    total_evaluated = 0
    tiers_tried: list[dict[str, Any]] = []
    selected_tier = start_tier
    relational_score = float(_relational_signature_score(features))
    a_probe: dict[str, Any] = {}
    escalation_decision: dict[str, Any] = {}

    for tier_name in active_tiers:
        cfg = TIER_CONFIGS[tier_name]
        tier_max_eval = cfg.max_eval
        if tier_name == "B" and not _is_frontier_signature(features):
            policy_b = int(policy.get("b_max_eval_non_frontier", 12))
            tier_max_eval = min(cfg.max_eval, max(1, policy_b))
        if tier_name == "B" and b_max_eval_override is not None:
            tier_max_eval = min(tier_max_eval, max(1, int(b_max_eval_override)))
        hyp_key = (train_key, bool(cfg.allow_compose), bool(cfg.allow_conditional))
        hyps = _HYP_CACHE.get(hyp_key)
        if hyps is None:
            hyps = generate_hypotheses(features, allow_compose=cfg.allow_compose, allow_conditional=cfg.allow_conditional)
            _cache_set(_HYP_CACHE, hyp_key, hyps)
        b_hyps: list[Hypothesis] | None = None
        if tier_name == "B":
            # B-registry is relational-specialized; only enable when relational signal is present.
            if relational_score >= RELITE_B_SCORE_MIN:
                b_hyps = _HYP_B_CACHE.get(train_key)
                if b_hyps is None:
                    b_hyps = generate_hypotheses_b(features)
                    _cache_set(_HYP_B_CACHE, train_key, b_hyps)

        if tier_name == "C" and _is_relational_signature(features):
            rlite_hyps = _build_clite_relational_candidates(features)
            rlite_meta = select_best_with_meta(rlite_hyps, train, max_eval=RLITE_MAX_EVAL)
            rlite_best = rlite_meta.get("best")
            rlite_conf = float(rlite_meta.get("confidence", 0.0))
            rlite_eval = int(rlite_meta.get("evaluated", 0))
            tiers_tried.append(
                {
                    "tier": "C-lite-relational",
                    "max_eval": RLITE_MAX_EVAL,
                    "allow_compose": False,
                    "allow_conditional": False,
                    "evaluated": rlite_eval,
                    "best_name": (rlite_best.name if rlite_best else None),
                    "best_exact_count": int(rlite_meta.get("best_exact_count", -1)),
                    "best_mean_partial": float(rlite_meta.get("best_mean_partial", 0.0)),
                    "confidence": rlite_conf,
                    "signature_match": True,
                }
            )
            total_evaluated += rlite_eval
            if rlite_best is not None and rlite_conf >= RLITE_STOP_CONFIDENCE:
                selected_tier = "C-lite-relational"
                rlite_exact = int(rlite_meta.get("best_exact_count", -1))
                rlite_partial = float(rlite_meta.get("best_mean_partial", 0.0))
                if _better_candidate(
                    rlite_exact,
                    rlite_partial,
                    rlite_best.cost,
                    rlite_best.prior,
                    global_best_exact,
                    global_best_partial,
                    global_best_cost,
                    global_best_prior,
                ):
                    global_best = rlite_best
                    global_best_exact = rlite_exact
                    global_best_partial = rlite_partial
                    global_best_cost = rlite_best.cost
                    global_best_prior = rlite_best.prior
                    global_confidence = rlite_conf
                tiers_tried.append(
                    {
                        "tier": "C-lite-relational",
                        "max_eval": RLITE_MAX_EVAL,
                        "allow_compose": False,
                        "allow_conditional": False,
                        "evaluated": 0,
                        "best_name": rlite_best.name,
                        "best_exact_count": rlite_exact,
                        "best_mean_partial": rlite_partial,
                        "confidence": rlite_conf,
                        "signature_match": True,
                        "skip_reason": "clite_relational_confident_stop",
                    }
                )
                break

        if tier_name == "C" and _is_frontier_signature(features):
            clite_hyps = _build_clite_candidates(hyps, symmetry_only=bool(features.get("has_any_input_symmetry", False)))
            clite_meta = select_best_with_meta(clite_hyps, train, max_eval=CLITE_MAX_EVAL)
            clite_best = clite_meta.get("best")
            clite_conf = float(clite_meta.get("confidence", 0.0))
            clite_eval = int(clite_meta.get("evaluated", 0))
            tiers_tried.append(
                {
                    "tier": "C-lite",
                    "max_eval": CLITE_MAX_EVAL,
                    "allow_compose": True,
                    "allow_conditional": False,
                    "evaluated": clite_eval,
                    "best_name": (clite_best.name if clite_best else None),
                    "best_exact_count": int(clite_meta.get("best_exact_count", -1)),
                    "best_mean_partial": float(clite_meta.get("best_mean_partial", 0.0)),
                    "confidence": clite_conf,
                    "signature_match": True,
                }
            )
            total_evaluated += clite_eval
            if clite_best is not None and clite_conf >= CLITE_STOP_CONFIDENCE:
                # High-confidence C-lite hit: skip full Tier-C.
                selected_tier = "C-lite"
                clite_exact = int(clite_meta.get("best_exact_count", -1))
                clite_partial = float(clite_meta.get("best_mean_partial", 0.0))
                if _better_candidate(
                    clite_exact,
                    clite_partial,
                    clite_best.cost,
                    clite_best.prior,
                    global_best_exact,
                    global_best_partial,
                    global_best_cost,
                    global_best_prior,
                ):
                    global_best = clite_best
                    global_best_exact = clite_exact
                    global_best_partial = clite_partial
                    global_best_cost = clite_best.cost
                    global_best_prior = clite_best.prior
                    global_confidence = clite_conf
                tiers_tried.append(
                    {
                        "tier": "C-lite",
                        "max_eval": CLITE_MAX_EVAL,
                        "allow_compose": True,
                        "allow_conditional": False,
                        "evaluated": 0,
                        "best_name": clite_best.name,
                        "best_exact_count": clite_exact,
                        "best_mean_partial": clite_partial,
                        "confidence": clite_conf,
                        "signature_match": True,
                        "skip_reason": "clite_confident_stop",
                    }
                )
                break

        if tier_name == "C":
            probe_eval = max(1, int(cfg.max_eval * TIER_C_PROBE_FRACTION))
            probe = select_best_with_meta(hyps, train, max_eval=probe_eval)
            if probe.get("best") is not None:
                tiers_tried.append(
                    {
                        "tier": tier_name,
                        "max_eval": cfg.max_eval,
                        "allow_compose": cfg.allow_compose,
                        "allow_conditional": cfg.allow_conditional,
                        "evaluated": int(probe.get("evaluated", 0)),
                        "best_name": probe["best"].name,
                        "best_exact_count": int(probe.get("best_exact_count", -1)),
                        "best_mean_partial": float(probe.get("best_mean_partial", 0.0)),
                        "confidence": float(probe.get("confidence", 0.0)),
                        "skipped_full_tier": True,
                        "skip_reason": "tier_c_probe_found_valid_hypothesis",
                    }
                )
                total_evaluated += int(probe.get("evaluated", 0))
                continue

        buckets = _build_portfolio_buckets(b_hyps if (tier_name == "B" and b_hyps is not None) else hyps, cfg)

        tier_best: Hypothesis | None = None
        tier_best_exact = -1
        tier_best_partial = -1.0
        tier_best_cost = 10**9
        tier_best_prior = -1.0
        tier_conf = 0.0
        tier_eval = 0

        # Earlier relational routing: run a lightweight relational pack in Tier-B
        # when the task looks partially relational, instead of waiting for Tier-C.
        if tier_name == "B" and relational_score >= RELITE_B_SCORE_MIN:
            brel_hyps = _build_clite_relational_candidates(features)
            brel_meta = select_best_with_meta(brel_hyps, train, max_eval=RLITE_B_MAX_EVAL)
            brel_best = brel_meta.get("best")
            brel_eval = int(brel_meta.get("evaluated", 0))
            brel_conf = float(brel_meta.get("confidence", 0.0))
            tiers_tried.append(
                {
                    "tier": "B-relational",
                    "max_eval": RLITE_B_MAX_EVAL,
                    "allow_compose": False,
                    "allow_conditional": False,
                    "evaluated": brel_eval,
                    "best_name": (brel_best.name if brel_best else None),
                    "best_exact_count": int(brel_meta.get("best_exact_count", -1)),
                    "best_mean_partial": float(brel_meta.get("best_mean_partial", 0.0)),
                    "confidence": brel_conf,
                    "relational_score": relational_score,
                }
            )
            tier_eval += brel_eval
            if brel_best is not None:
                brel_exact = int(brel_meta.get("best_exact_count", -1))
                brel_partial = float(brel_meta.get("best_mean_partial", 0.0))
                if _better_candidate(
                    brel_exact,
                    brel_partial,
                    brel_best.cost,
                    brel_best.prior,
                    tier_best_exact,
                    tier_best_partial,
                    tier_best_cost,
                    tier_best_prior,
                ):
                    tier_best = brel_best
                    tier_best_exact = brel_exact
                    tier_best_partial = brel_partial
                    tier_best_cost = brel_best.cost
                    tier_best_prior = brel_best.prior
                    tier_conf = brel_conf

        for bucket_name, bucket_hyps in buckets:
            fam = bucket_name.split("+")[0]
            if fam in {"conditional", "object_conditional"}:
                if tier_eval < CONDITIONAL_GATE_ATTEMPTS:
                    continue
                # Only trigger expensive branches if simpler buckets failed to fit train consistently.
                if tier_best is not None and tier_best_exact >= len(train):
                    continue

            bucket_eval_cap = tier_max_eval
            # Tier-B tightening: reduce expensive branch breadth by ~30%.
            if tier_name == "B" and fam in {"conditional", "object_conditional"}:
                bucket_eval_cap = max(1, int(round(tier_max_eval * 0.7)))

            meta = select_best_with_meta(bucket_hyps, train, max_eval=bucket_eval_cap)
            cand = meta.get("best")
            tier_eval += int(meta.get("evaluated", 0))
            if cand is None:
                continue
            cand_exact = int(meta.get("best_exact_count", -1))
            cand_partial = float(meta.get("best_mean_partial", 0.0))
            if _better_candidate(
                cand_exact,
                cand_partial,
                cand.cost,
                cand.prior,
                tier_best_exact,
                tier_best_partial,
                tier_best_cost,
                tier_best_prior,
            ):
                tier_best = cand
                tier_best_exact = cand_exact
                tier_best_partial = cand_partial
                tier_best_cost = cand.cost
                tier_best_prior = cand.prior
                tier_conf = float(meta.get("confidence", 0.0))

        # Tier-B fallback: if specialized B registry did not fit strongly, spend
        # a small residual budget on the base pool for safety.
        if tier_name == "B" and b_hyps is not None and tier_best_exact < len(train):
            pure_relational = _is_pure_relational_signature(features)
            if pure_relational:
                fallback_buckets: list[tuple[str, list[Hypothesis]]] = []
                fallback_cap = 0
            else:
                fallback_buckets = _build_portfolio_buckets(hyps, cfg)
                # Soft split for mixed relational tasks, normal fallback otherwise.
                fallback_cap = max(1, int(round(tier_max_eval * 0.25))) if relational_score >= RELITE_B_SCORE_MIN else max(1, int(round(tier_max_eval * 0.5)))
            for bucket_name, bucket_hyps in fallback_buckets:
                fam = bucket_name.split("+")[0]
                if fam in {"conditional", "object_conditional"} and tier_eval < CONDITIONAL_GATE_ATTEMPTS:
                    continue
                meta = select_best_with_meta(bucket_hyps, train, max_eval=fallback_cap)
                cand = meta.get("best")
                tier_eval += int(meta.get("evaluated", 0))
                if cand is None:
                    continue
                cand_exact = int(meta.get("best_exact_count", -1))
                cand_partial = float(meta.get("best_mean_partial", 0.0))
                if _better_candidate(
                    cand_exact,
                    cand_partial,
                    cand.cost,
                    cand.prior,
                    tier_best_exact,
                    tier_best_partial,
                    tier_best_cost,
                    tier_best_prior,
                ):
                    tier_best = cand
                    tier_best_exact = cand_exact
                    tier_best_partial = cand_partial
                    tier_best_cost = cand.cost
                    tier_best_prior = cand.prior
                    tier_conf = float(meta.get("confidence", 0.0))

        # Signature-gated B+ lane before Tier C: narrow, conditionals-forward search.
        if tier_name == "B" and _is_frontier_signature(features):
            bplus_hyps = _build_bplus_candidates(hyps)
            bplus_meta = select_best_with_meta(bplus_hyps, train, max_eval=BPLUS_MAX_EVAL)
            bplus_best = bplus_meta.get("best")
            bplus_eval = int(bplus_meta.get("evaluated", 0))
            tier_eval += bplus_eval
            tiers_tried.append(
                {
                    "tier": "B+",
                    "max_eval": BPLUS_MAX_EVAL,
                    "allow_compose": False,
                    "allow_conditional": True,
                    "evaluated": bplus_eval,
                    "best_name": (bplus_best.name if bplus_best else None),
                    "best_exact_count": int(bplus_meta.get("best_exact_count", -1)),
                    "best_mean_partial": float(bplus_meta.get("best_mean_partial", 0.0)),
                    "confidence": float(bplus_meta.get("confidence", 0.0)),
                    "signature_match": True,
                }
            )
            if bplus_best is not None:
                bplus_exact = int(bplus_meta.get("best_exact_count", -1))
                bplus_partial = float(bplus_meta.get("best_mean_partial", 0.0))
                if _better_candidate(
                    bplus_exact,
                    bplus_partial,
                    bplus_best.cost,
                    bplus_best.prior,
                    tier_best_exact,
                    tier_best_partial,
                    tier_best_cost,
                    tier_best_prior,
                ):
                    tier_best = bplus_best
                    tier_best_exact = bplus_exact
                    tier_best_partial = bplus_partial
                    tier_best_cost = bplus_best.cost
                    tier_best_prior = bplus_best.prior
                    tier_conf = float(bplus_meta.get("confidence", 0.0))
                if float(bplus_meta.get("confidence", 0.0)) >= BPLUS_STOP_CONFIDENCE:
                    # Treat as sufficiently strong hard-task solve; avoid Tier C.
                    tiers_tried.append(
                        {
                            "tier": "B+",
                            "max_eval": BPLUS_MAX_EVAL,
                            "allow_compose": False,
                            "allow_conditional": True,
                            "evaluated": 0,
                            "best_name": bplus_best.name,
                            "best_exact_count": bplus_exact,
                            "best_mean_partial": bplus_partial,
                            "confidence": float(bplus_meta.get("confidence", 0.0)),
                            "signature_match": True,
                            "skip_reason": "bplus_confident_stop",
                        }
                    )
                    selected_tier = "B+"
                    if _better_candidate(
                        tier_best_exact,
                        tier_best_partial,
                        tier_best_cost,
                        tier_best_prior,
                        global_best_exact,
                        global_best_partial,
                        global_best_cost,
                        global_best_prior,
                    ):
                        global_best = tier_best
                        global_best_exact = tier_best_exact
                        global_best_partial = tier_best_partial
                        global_best_cost = tier_best_cost
                        global_best_prior = tier_best_prior
                        global_confidence = tier_conf
                    total_evaluated += tier_eval
                    break

        tiers_tried.append(
            {
                "tier": tier_name,
                "max_eval": tier_max_eval,
                "allow_compose": cfg.allow_compose,
                "allow_conditional": cfg.allow_conditional,
                "evaluated": tier_eval,
                "best_name": (tier_best.name if tier_best else None),
                "best_exact_count": tier_best_exact,
                "best_mean_partial": tier_best_partial,
                "confidence": tier_conf,
            }
        )
        total_evaluated += tier_eval

        if tier_best is not None and _better_candidate(
            tier_best_exact,
            tier_best_partial,
            tier_best_cost,
            tier_best_prior,
            global_best_exact,
            global_best_partial,
            global_best_cost,
            global_best_prior,
        ):
            global_best = tier_best
            global_best_exact = tier_best_exact
            global_best_partial = tier_best_partial
            global_best_cost = tier_best_cost
            global_best_prior = tier_best_prior
            global_confidence = tier_conf
            selected_tier = tier_name

        if tier_name == "A" and escalate:
            # A-tier convergence diagnostics for escalation policy tuning.
            probe_meta = select_best_with_meta(hyps, train, max_eval=min(tier_max_eval, A_MARGIN_PROBE_EVAL))
            a_probe = {
                "best_name": (probe_meta.get("best").name if probe_meta.get("best") is not None else None),
                "best_score": float(probe_meta.get("best_score", 0.0)),
                "second_best_name": probe_meta.get("second_best_name"),
                "second_best_score": float(probe_meta.get("second_best_score", 0.0)),
                "score_margin": float(probe_meta.get("score_margin", 0.0)),
                "best_exact_count": int(probe_meta.get("best_exact_count", -1)),
                "best_mean_partial": float(probe_meta.get("best_mean_partial", 0.0)),
                "second_best_exact_count": int(probe_meta.get("second_best_exact_count", -1)),
                "second_best_mean_partial": float(probe_meta.get("second_best_mean_partial", 0.0)),
                "best_kind": (tier_best.kind if tier_best is not None else None),
                "best_cost": (tier_best.cost if tier_best is not None else None),
                "best_family": (_hyp_family(tier_best) if tier_best is not None else None),
            }
            escalation_decision = _escalation_decision_from_a(
                features,
                tier_best_exact,
                tier_best_partial,
                tier_conf,
                len(train),
                a_probe.get("best_family"),
                float(a_probe.get("score_margin", 0.0)),
                float(a_probe.get("best_score", 0.0)),
            )
            if not bool(escalation_decision.get("should_escalate", False)):
                tiers_tried.append(
                    {
                        "tier": tier_name,
                        "max_eval": cfg.max_eval,
                        "allow_compose": cfg.allow_compose,
                        "allow_conditional": cfg.allow_conditional,
                        "evaluated": 0,
                        "best_name": (tier_best.name if tier_best else None),
                        "best_exact_count": tier_best_exact,
                        "best_mean_partial": tier_best_partial,
                        "confidence": tier_conf,
                        "skip_reason": str(escalation_decision.get("reason", "a_gate_no_escalate")),
                        "escalation_decision": escalation_decision,
                    }
                )
                selected_tier = "A"
                break

        if tier_best is not None and tier_conf >= cfg.stop_confidence:
            selected_tier = tier_name
            break

    predictions: list[Grid] = []
    for t in test:
        inp = t["input"] if isinstance(t, dict) else t
        predictions.append(apply(global_best, inp))

    return {
        "predictions": predictions,
        "best_hypothesis": global_best,
        "explanation": explain(global_best),
        "selected_tier": selected_tier,
        "tiers_tried": pre_tier_logs + tiers_tried,
        "confidence": global_confidence,
        "evaluated_hypotheses": total_evaluated + sum(int(x.get("evaluated", 0)) for x in pre_tier_logs),
        "task_features": task_features,
        "a_probe": a_probe,
        "escalation_decision": escalation_decision,
    }


def solve(task: dict[str, Any]) -> list[Grid]:
    return solve_with_report(task, start_tier="A", escalate=True)["predictions"]


def solve_with_meta_policy(task: dict[str, Any], policy_path: str = "") -> dict[str, Any]:
    policy = _load_meta_policy(policy_path)
    lite_b_max_eval = int(policy.get("lite_b_max_eval", 8))
    full_b_max_eval = int(policy.get("full_b_max_eval", 12))
    disagreement_threshold = float(policy.get("disagreement_threshold", 0.2))
    confidence_gap_threshold = float(policy.get("confidence_gap_threshold", -0.05))
    high_disagreement_threshold = float(policy.get("high_disagreement_threshold", max(0.5, disagreement_threshold)))
    weak_confidence_gain = float(policy.get("weak_confidence_gain", 0.03))
    force_full_on_disagree = bool(policy.get("force_full_on_disagree", False))

    rep_a = solve_with_report(task, start_tier="A", escalate=False, policy_path=policy_path)
    rep_lite = solve_with_report(
        task,
        start_tier="A",
        escalate=True,
        b_max_eval_override=lite_b_max_eval,
        allow_tier_c=False,
        policy_path=policy_path,
    )
    disagreement = _prediction_disagreement(rep_a.get("predictions", []), rep_lite.get("predictions", []))
    conf_gap = float(rep_lite.get("confidence", 0.0)) - float(rep_a.get("confidence", 0.0))
    d_rate = float(disagreement.get("disagree_rate", 0.0))

    # Penalize expensive escalation when high disagreement is not accompanied by confidence gain.
    if d_rate >= high_disagreement_threshold and conf_gap <= weak_confidence_gain:
        rep_a["meta_policy"] = {
            "decision": "choose_A_penalized_weak_gain",
            "disagreement": disagreement,
            "confidence_gap": conf_gap,
        }
        return rep_a

    if d_rate < disagreement_threshold and conf_gap < confidence_gap_threshold:
        rep_a["meta_policy"] = {
            "decision": "choose_A",
            "disagreement": disagreement,
            "confidence_gap": conf_gap,
        }
        return rep_a

    if not force_full_on_disagree and float(rep_lite.get("confidence", 0.0)) >= float(rep_a.get("confidence", 0.0)):
        rep_lite["meta_policy"] = {
            "decision": "choose_lite",
            "disagreement": disagreement,
            "confidence_gap": conf_gap,
        }
        return rep_lite

    rep_full = solve_with_report(
        task,
        start_tier="A",
        escalate=True,
        b_max_eval_override=full_b_max_eval,
        allow_tier_c=bool(policy.get("allow_tier_c", True)),
        policy_path=policy_path,
    )
    best = rep_full
    if float(rep_lite.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
        best = rep_lite
    if float(rep_a.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
        best = rep_a
    best["meta_policy"] = {
        "decision": "choose_full" if best is rep_full else "confidence_fallback",
        "disagreement": disagreement,
        "confidence_gap": conf_gap,
    }
    return best
