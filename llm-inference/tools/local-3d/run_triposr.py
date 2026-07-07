#!/usr/bin/env python3
"""Run local TripoSR image-to-3D and copy GLBs into assets/game-assets."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "assets" / "source-png"
DEFAULT_OUTPUT = ROOT / "assets" / "game-assets"
VENDOR_DIR = ROOT / "tools" / "local-3d" / "vendor" / "TripoSR"
VENV_PY = ROOT / ".venvs" / "triposr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@dataclass(frozen=True)
class Job:
    source: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real local 3D GLBs from images with TripoSR.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Image file or directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for GLBs.")
    parser.add_argument("--pattern", default="*.png", help="Glob for directory input.")
    parser.add_argument("--device", default="cuda:0", help="TripoSR device, e.g. cuda:0 or cpu.")
    parser.add_argument("--mc-resolution", type=int, default=256, help="Marching cubes resolution.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="TripoSR chunk size.")
    parser.add_argument("--bake-texture", action="store_true", help="Bake a texture atlas instead of vertex colors.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without running TripoSR.")
    return parser.parse_args()


def discover_jobs(input_path: Path, output_dir: Path, pattern: str) -> list[Job]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_path.is_file():
        sources = [input_path]
    elif input_path.is_dir():
        sources = sorted(p for p in input_path.glob(pattern) if p.is_file())
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")

    jobs = []
    for source in sources:
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        jobs.append(Job(source=source, output=output_dir / f"{source.stem}.glb"))
    return jobs


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_plan(jobs: list[Job]) -> None:
    if not jobs:
        print("No image inputs found.")
        return
    print(f"Planned local 3D assets: {len(jobs)}")
    for job in jobs:
        print(f"  {rel(job.source)} -> {rel(job.output)}")


def ensure_ready() -> None:
    if not VENDOR_DIR.exists():
        raise SystemExit("TripoSR is not installed. Run: npm run local3d:setup")
    if not VENV_PY.exists():
        raise SystemExit("TripoSR venv is missing. Run: npm run local3d:setup")
    if not (VENDOR_DIR / "run.py").exists():
        raise SystemExit(f"TripoSR run.py not found in {VENDOR_DIR}")


def run_job(job: Job, args: argparse.Namespace) -> None:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="saient-triposr-") as td:
        tmp = Path(td)
        cmd = [
            str(VENV_PY),
            str(VENDOR_DIR / "run.py"),
            str(job.source),
            "--output-dir",
            str(tmp),
            "--model-save-format",
            "glb",
            "--device",
            args.device,
            "--mc-resolution",
            str(args.mc_resolution),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.bake_texture:
            cmd.append("--bake-texture")

        print(f"Building local 3D {job.output.name} ...", flush=True)
        subprocess.run(cmd, cwd=VENDOR_DIR, check=True)
        generated = tmp / "0" / "mesh.glb"
        if not generated.exists():
            raise SystemExit(f"TripoSR did not write expected output: {generated}")
        shutil.copy2(generated, job.output)
        print(f"wrote {rel(job.output)}", flush=True)


def main() -> int:
    args = parse_args()
    jobs = discover_jobs(Path(args.input), Path(args.output), args.pattern)
    print_plan(jobs)
    if args.dry_run:
        return 0
    if not jobs:
        return 0
    ensure_ready()
    for job in jobs:
        run_job(job, args)
    print("Local 3D build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
