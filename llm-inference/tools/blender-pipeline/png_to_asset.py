#!/usr/bin/env python3
"""Batch PNG-to-GLB wrapper for the Blender asset pipeline.

Drop transparent PNGs into assets/source-png, then run:

  npm run asset:build

This wrapper is intentionally normal Python. It validates inputs, finds Blender,
and calls the Blender-side script once per PNG. Use --dry-run or --self-test when
Blender is not installed yet; those paths must exit quickly and never hang.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:  # pragma: no cover - exercised by users without Pillow
    Image = None
    ImageDraw = None
    ImageFilter = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by users without numpy
    np = None


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_SCRIPT = SCRIPT_DIR / "blender_png_to_glb.py"
DEFAULT_INPUT = ROOT / "assets" / "source-png"
DEFAULT_OUTPUT = ROOT / "assets" / "game-assets"


@dataclass(frozen=True)
class AssetJob:
    source: Path
    output: Path
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert transparent PNGs into GLB game assets through Blender.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="PNG file or directory of PNGs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for .glb assets.")
    parser.add_argument("--pattern", default="*.png", help="Glob used when --input is a directory.")
    parser.add_argument("--blender", default=os.environ.get("BLENDER_BIN", "blender"), help="Blender executable path.")
    parser.add_argument("--width", type=float, default=1.0, help="World width of the generated billboard mesh.")
    parser.add_argument("--height", type=float, default=1.0, help="World height before aspect correction.")
    parser.add_argument("--thickness", type=float, default=0.16, help="Maximum depth of generated mesh volume.")
    parser.add_argument("--mesh-mode", choices=("relief", "silhouette", "billboard"), default="relief", help="Output geometry style.")
    parser.add_argument("--relief-grid", type=int, default=96, help="Longest-side grid resolution for relief meshes.")
    parser.add_argument("--cutout", choices=("auto", "always", "never"), default="auto", help="Create a transparent cutout for opaque PNGs before export.")
    parser.add_argument("--crop-padding", type=int, default=18, help="Transparent cutout crop padding in pixels.")
    parser.add_argument("--flood-tolerance", type=float, default=52.0, help="Background flood-fill color tolerance for opaque cutouts.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned outputs without running Blender.")
    parser.add_argument("--self-test", action="store_true", help="Run a fast smoke test that ends even without Blender.")
    return parser.parse_args()


def require_pillow() -> None:
    if Image is None:
        raise SystemExit("Pillow is required for validation/self-test. Install with: python3 -m pip install pillow")


def require_numpy() -> None:
    if np is None:
        raise SystemExit("NumPy is required for opaque PNG cutouts. Install with: python3 -m pip install numpy")


def find_blender(binary: str) -> str | None:
    candidate = Path(binary).expanduser()
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which(binary)
    return found


def discover_jobs(input_path: Path, output_dir: Path, pattern: str) -> list[AssetJob]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_path.is_file():
        sources = [input_path]
    elif input_path.is_dir():
        sources = sorted(p for p in input_path.glob(pattern) if p.is_file())
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")

    jobs: list[AssetJob] = []
    for source in sources:
        if source.suffix.lower() != ".png":
            continue
        jobs.append(AssetJob(source=source, output=output_dir / f"{source.stem}.glb", name=source.stem))
    return jobs


def validate_png(path: Path) -> tuple[int, int, bool]:
    require_pillow()
    with Image.open(path) as img:
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        return img.width, img.height, has_alpha


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgba = image.convert("RGBA")
    return rgba.getchannel("A").getbbox()


def crop_to_alpha(image: Image.Image, padding: int) -> Image.Image:
    rgba = image.convert("RGBA")
    bbox = alpha_bbox(rgba)
    if not bbox:
        return rgba
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgba.width, right + padding)
    bottom = min(rgba.height, bottom + padding)
    return rgba.crop((left, top, right, bottom))


def image_has_useful_alpha(image: Image.Image) -> bool:
    if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    extrema = alpha.getextrema()
    return extrema[0] < 250


def flood_background(rgb: "np.ndarray", tolerance: float) -> "np.ndarray":
    height, width, _ = rgb.shape
    background = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(width):
        q.append((x, 0))
        q.append((x, height - 1))
    for y in range(1, height - 1):
        q.append((0, y))
        q.append((width - 1, y))

    while q:
        x, y = q.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True
        background[y, x] = True
        here = rgb[y, x]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx]:
                continue
            diff = rgb[ny, nx] - here
            if float(np.sqrt(np.dot(diff, diff))) <= tolerance:
                q.append((nx, ny))
    return background


def make_opaque_cutout(source: Path, output: Path, padding: int, tolerance: float) -> bool:
    require_pillow()
    require_numpy()
    image = Image.open(source).convert("RGBA")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    background = flood_background(rgb, tolerance)
    subject = ~background

    # Drop tiny isolated specks from noisy generated backgrounds, then soften
    # the cut edge enough that GLB texture filtering does not show stair-steps.
    alpha = Image.fromarray((subject.astype("uint8") * 255), mode="L")
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.6))
    cutout = image.copy()
    cutout.putalpha(alpha)
    cropped = crop_to_alpha(cutout, padding)
    bbox = alpha_bbox(cropped)
    if not bbox:
        return False
    visible_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1, image.width * image.height)
    if visible_area < 0.02:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output)
    return True


def make_rembg_cutout(source: Path, output: Path, padding: int) -> bool:
    require_pillow()
    try:
        from rembg import remove
    except Exception:
        return False

    image = Image.open(source).convert("RGBA")
    result = remove(image)
    if not isinstance(result, Image.Image):
        result = Image.open(result).convert("RGBA")
    result = crop_to_alpha(result, padding)
    bbox = alpha_bbox(result)
    if not bbox:
        return False
    visible_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1, image.width * image.height)
    if visible_area < 0.02:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return True


def prepare_source_png(job: AssetJob, work_dir: Path, cutout: str, padding: int, tolerance: float) -> Path:
    require_pillow()
    image = Image.open(job.source)
    should_cutout = cutout == "always" or (cutout == "auto" and not image_has_useful_alpha(image))
    prepared = work_dir / f"{job.name}_cutout.png"

    if should_cutout:
        if make_rembg_cutout(job.source, prepared, padding):
            print(f"  cutout {job.source.name}: background removed with rembg", flush=True)
            return prepared
        if make_opaque_cutout(job.source, prepared, padding, tolerance):
            print(f"  cutout {job.source.name}: opaque background removed with fallback mask", flush=True)
            return prepared
        print(f"  warning {job.source.name}: cutout failed; using original PNG", flush=True)
        return job.source

    cropped = crop_to_alpha(image, padding)
    if cropped.size != image.size:
        cropped.save(prepared)
        print(f"  crop {job.source.name}: transparent bounds tightened", flush=True)
        return prepared
    return job.source


def _largest_alpha_contour(mask: "np.ndarray") -> list[tuple[float, float]]:
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        epsilon = max(2.0, peri * 0.006)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > 220:
            epsilon *= 1.35
            approx = cv2.approxPolyDP(contour, epsilon, True)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    try:
        from skimage import measure
    except Exception:
        return []

    contours = measure.find_contours(mask, 127)
    if not contours:
        return []
    contour = max(contours, key=len)
    stride = max(1, len(contour) // 180)
    return [(float(x), float(y)) for y, x in contour[::stride]]


def write_silhouette_mesh_json(image_path: Path, json_path: Path, width: float, height: float, depth: float) -> bool:
    require_pillow()
    require_numpy()
    image = Image.open(image_path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    mask = (alpha > 24).astype("uint8") * 255
    points = _largest_alpha_contour(mask)
    if len(points) < 3:
        return False

    img_w, img_h = image.size
    min_x = min(px for px, _ in points)
    max_x = max(px for px, _ in points)
    min_y = min(py for _, py in points)
    max_y = max(py for _, py in points)
    contour_w = max(max_x - min_x, 1.0)
    contour_h = max(max_y - min_y, 1.0)
    real_height = height * (contour_h / contour_w)
    verts = []
    uvs = []
    for px, py in points:
        x = ((px - min_x) / contour_w - 0.5) * width
        z = (1.0 - (py - min_y) / contour_h) * real_height
        verts.append([x, z])
        uvs.append([px / max(img_w - 1, 1), 1.0 - py / max(img_h - 1, 1)])

    if polygon_area(verts) < 0:
        verts.reverse()
        uvs.reverse()

    payload = {
        "width": width,
        "height": real_height,
        "depth": max(0.0, depth),
        "vertices": verts,
        "uvs": uvs,
    }
    json_path.write_text(json.dumps(payload))
    return True


def write_relief_mesh_json(image_path: Path, json_path: Path, width: float, height: float, depth: float, resolution: int) -> bool:
    require_pillow()
    require_numpy()
    try:
        import cv2
    except Exception:
        return False

    image = Image.open(image_path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    mask = alpha > 24
    bbox = Image.fromarray((mask.astype("uint8") * 255), mode="L").getbbox()
    if not bbox:
        return False

    left, top, right, bottom = bbox
    subject_w = max(right - left, 1)
    subject_h = max(bottom - top, 1)
    aspect = subject_h / subject_w
    real_height = height * aspect

    longest = max(24, min(int(resolution), 160))
    if aspect >= 1.0:
        rows = longest
        cols = max(24, int(round(longest / aspect)))
    else:
        cols = longest
        rows = max(24, int(round(longest * aspect)))

    roi_mask = mask[top:bottom, left:right].astype("uint8")
    distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    positive = distance[distance > 0]
    max_distance = float(np.percentile(positive, 95)) if positive.size else 1.0
    max_depth = max(float(depth), 0.02)
    edge_depth = min(max_depth * 0.16, 0.022)

    def source_pixel(i: int, j: int) -> tuple[int, int]:
        px = left + int(round((i / max(cols - 1, 1)) * (subject_w - 1)))
        py = top + int(round((j / max(rows - 1, 1)) * (subject_h - 1)))
        return min(max(px, 0), image.width - 1), min(max(py, 0), image.height - 1)

    def is_cell_inside(i: int, j: int) -> bool:
        px = left + int(round(((i + 0.5) / max(cols - 1, 1)) * (subject_w - 1)))
        py = top + int(round(((j + 0.5) / max(rows - 1, 1)) * (subject_h - 1)))
        px = min(max(px, 0), image.width - 1)
        py = min(max(py, 0), image.height - 1)
        return bool(mask[py, px])

    cell_inside = [[is_cell_inside(i, j) for i in range(cols - 1)] for j in range(rows - 1)]
    used: set[tuple[int, int]] = set()
    for j, row in enumerate(cell_inside):
        for i, inside in enumerate(row):
            if inside:
                used.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))

    if len(used) < 4:
        return False

    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    front_index: dict[tuple[int, int], int] = {}
    back_index: dict[tuple[int, int], int] = {}

    def depth_at(px: int, py: int) -> float:
        dx = min(max(px - left, 0), subject_w - 1)
        dy = min(max(py - top, 0), subject_h - 1)
        if not mask[py, px]:
            return edge_depth
        normalized = min(float(distance[dy, dx]) / max(max_distance, 1e-6), 1.0)
        return edge_depth + (max_depth - edge_depth) * (normalized ** 0.65)

    for i, j in sorted(used, key=lambda p: (p[1], p[0])):
        px, py = source_pixel(i, j)
        x = (i / max(cols - 1, 1) - 0.5) * width
        z = (1.0 - j / max(rows - 1, 1)) * real_height
        d = depth_at(px, py)
        uv = [px / max(image.width - 1, 1), 1.0 - py / max(image.height - 1, 1)]

        front_index[(i, j)] = len(vertices)
        vertices.append([x, -d * 0.5, z])
        uvs.append(uv)
        back_index[(i, j)] = len(vertices)
        vertices.append([x, d * 0.5, z])
        uvs.append(uv)

    faces: list[list[int]] = []
    material_indices: list[int] = []

    def add_face(face: list[int], material_index: int) -> None:
        faces.append(face)
        material_indices.append(material_index)

    for j, row in enumerate(cell_inside):
        for i, inside in enumerate(row):
            if not inside:
                continue
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            front = [front_index[p] for p in corners]
            back = [back_index[p] for p in reversed(corners)]
            add_face([front[0], front[1], front[2]], 0)
            add_face([front[0], front[2], front[3]], 0)
            add_face([back[0], back[1], back[2]], 0)
            add_face([back[0], back[2], back[3]], 0)

            edge_specs = [
                (((i, j), (i + 1, j)), j == 0 or not cell_inside[j - 1][i]),
                (((i + 1, j), (i + 1, j + 1)), i == cols - 2 or not cell_inside[j][i + 1]),
                (((i + 1, j + 1), (i, j + 1)), j == rows - 2 or not cell_inside[j + 1][i]),
                (((i, j + 1), (i, j)), i == 0 or not cell_inside[j][i - 1]),
            ]
            for (a, b), exposed in edge_specs:
                if exposed:
                    add_face([front_index[a], front_index[b], back_index[b], back_index[a]], 1)

    payload = {
        "mesh_kind": "relief",
        "width": width,
        "height": real_height,
        "depth": max_depth,
        "vertices": vertices,
        "uvs": uvs,
        "faces": faces,
        "material_indices": material_indices,
    }
    json_path.write_text(json.dumps(payload, separators=(",", ":")))
    return True


def polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for i, (x1, z1) in enumerate(points):
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area * 0.5


def print_plan(jobs: list[AssetJob]) -> None:
    if not jobs:
        print("No PNG inputs found.")
        return
    print(f"Planned assets: {len(jobs)}")
    for job in jobs:
        width, height, has_alpha = validate_png(job.source)
        alpha = "alpha" if has_alpha else "opaque"
        rel_src = job.source.relative_to(ROOT) if job.source.is_relative_to(ROOT) else job.source
        rel_out = job.output.relative_to(ROOT) if job.output.is_relative_to(ROOT) else job.output
        print(f"  {rel_src} ({width}x{height}, {alpha}) -> {rel_out}")


def run_job(
    blender: str,
    job: AssetJob,
    input_png: Path,
    work_dir: Path,
    width: float,
    height: float,
    thickness: float,
    mesh_mode: str,
    relief_grid: int,
) -> None:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_json = work_dir / f"{job.name}_mesh.json"
    mesh_arg: list[str] = []
    if mesh_mode == "relief" and write_relief_mesh_json(input_png, mesh_json, width, height, thickness, relief_grid):
        mesh_arg = ["--mesh-json", str(mesh_json)]
        print(f"  mesh {job.source.name}: built inflated relief mesh with max thickness {thickness:g}", flush=True)
    elif mesh_mode == "relief":
        print(f"  warning {job.source.name}: relief mesh failed; trying silhouette mesh", flush=True)
        if write_silhouette_mesh_json(input_png, mesh_json, width, height, thickness):
            mesh_arg = ["--mesh-json", str(mesh_json)]
            print(f"  mesh {job.source.name}: traced silhouette mesh with thickness {thickness:g}", flush=True)
    elif mesh_mode == "silhouette" and write_silhouette_mesh_json(input_png, mesh_json, width, height, thickness):
        mesh_arg = ["--mesh-json", str(mesh_json)]
        print(f"  mesh {job.source.name}: traced silhouette mesh with thickness {thickness:g}", flush=True)
    elif mesh_mode == "silhouette":
        print(f"  warning {job.source.name}: silhouette tracing failed; using billboard geometry", flush=True)

    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--input",
        str(input_png),
        "--output",
        str(job.output),
        "--name",
        job.name,
        "--width",
        str(width),
        "--height",
        str(height),
        "--thickness",
        str(thickness),
    ]
    cmd.extend(mesh_arg)
    print(f"Building {job.output.name} ...", flush=True)
    subprocess.run(cmd, check=True)


def create_test_png(path: Path) -> None:
    require_pillow()
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((24, 16, 72, 78), radius=8, fill=(156, 91, 42, 255), outline=(48, 32, 24, 255), width=3)
    draw.polygon([(18, 42), (48, 8), (78, 42)], fill=(58, 138, 192, 255), outline=(26, 57, 84, 255))
    draw.rectangle((42, 52, 54, 78), fill=(48, 29, 18, 255))
    img.save(path)


def self_test(args: argparse.Namespace) -> int:
    print("Running Blender pipeline self-test...")
    with tempfile.TemporaryDirectory(prefix="blender-pipeline-") as td:
        tmp = Path(td)
        source_dir = tmp / "source"
        out_dir = tmp / "out"
        source_dir.mkdir()
        sample = source_dir / "sample_building.png"
        create_test_png(sample)
        jobs = discover_jobs(source_dir, out_dir, "*.png")
        if len(jobs) != 1:
            raise SystemExit(f"Expected one self-test job, got {len(jobs)}")
        print_plan(jobs)

        blender = find_blender(args.blender)
        if blender is None:
            print("Blender not found on PATH. Dry-run self-test passed; real conversion is skipped.")
            print("Set BLENDER_BIN=/path/to/blender or install Blender to enable npm run asset:build.")
            return 0

        with tempfile.TemporaryDirectory(prefix="blender-pipeline-cutout-") as prep:
            input_png = prepare_source_png(jobs[0], Path(prep), args.cutout, args.crop_padding, args.flood_tolerance)
            run_job(blender, jobs[0], input_png, Path(prep), args.width, args.height, args.thickness, args.mesh_mode, args.relief_grid)
        if not jobs[0].output.exists() or jobs[0].output.stat().st_size == 0:
            raise SystemExit("Self-test did not produce a GLB output.")
        print(f"Self-test GLB created: {jobs[0].output}")
        return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test(args)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    jobs = discover_jobs(input_path, output_dir, args.pattern)
    print_plan(jobs)
    if args.dry_run:
        return 0
    if not jobs:
        return 0

    blender = find_blender(args.blender)
    if blender is None:
        print("\nBlender not found. Nothing was converted.")
        print("Install Blender or set BLENDER_BIN=/path/to/blender, then rerun: npm run asset:build")
        return 2

    with tempfile.TemporaryDirectory(prefix="blender-pipeline-cutout-") as prep:
        prep_dir = Path(prep)
        for job in jobs:
            input_png = prepare_source_png(job, prep_dir, args.cutout, args.crop_padding, args.flood_tolerance)
            run_job(blender, job, input_png, prep_dir, args.width, args.height, args.thickness, args.mesh_mode, args.relief_grid)
    print("Asset build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
