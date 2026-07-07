#!/usr/bin/env python3
"""Set up a local TripoSR image-to-3D environment.

This script keeps TripoSR isolated from Saient's existing Python environments:

  .venvs/triposr/                 Python environment
  tools/local-3d/vendor/TripoSR/  Official upstream checkout

It does not run any paid/cloud generation. Model weights are downloaded by the
TripoSR/Hugging Face stack on first inference and cached locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
VENDOR_DIR = LOCAL_DIR / "vendor" / "TripoSR"
TORCHMCUBES_DIR = LOCAL_DIR / "vendor" / "torchmcubes-cpu"
VENV_DIR = ROOT / ".venvs" / "triposr"
REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
REPO_REF = "107cefdc244c39106fa830359024f6a2f1c78871"
TORCHMCUBES_URL = "https://github.com/tatsy/torchmcubes.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install local TripoSR image-to-3D support.")
    parser.add_argument("--venv", default=str(VENV_DIR), help="Virtualenv path.")
    parser.add_argument("--repo", default=str(VENDOR_DIR), help="TripoSR checkout path.")
    parser.add_argument("--torchmcubes-repo", default=str(TORCHMCUBES_DIR), help="Patched CPU torchmcubes checkout path.")
    parser.add_argument(
        "--torch-index",
        default=os.environ.get("TRIPOSR_TORCH_INDEX", "https://download.pytorch.org/whl/cu128"),
        help="PyTorch wheel index. Use cpu or a PyTorch index URL.",
    )
    parser.add_argument("--skip-torch", action="store_true", help="Do not install torch/torchvision.")
    parser.add_argument("--with-gradio", action="store_true", help="Install TripoSR's optional Gradio demo dependencies.")
    parser.add_argument("--force-clone", action="store_true", help="Replace the existing TripoSR checkout.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def python_bin(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def clone_repo(repo_dir: Path, force: bool) -> None:
    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)
    if repo_dir.exists():
        print(f"TripoSR checkout already exists: {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    # Pinning makes the local pipeline reproducible while still using upstream.
    try:
        run(["git", "fetch", "--depth", "1", "origin", REPO_REF], cwd=repo_dir)
        run(["git", "checkout", REPO_REF], cwd=repo_dir)
    except subprocess.CalledProcessError:
        print("Warning: could not pin TripoSR checkout; using cloned default branch.")


def create_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        print(f"Creating venv: {venv_dir}")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    return python_bin(venv_dir)


def filtered_requirements(repo_dir: Path, include_gradio: bool) -> Path:
    lines = []
    for raw in (repo_dir / "requirements.txt").read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = stripped.split("==", 1)[0].split("[", 1)[0].lower()
        if package == "gradio" and not include_gradio:
            continue
        if "torchmcubes" in stripped.lower():
            continue
        lines.append(stripped)

    tmp = tempfile.NamedTemporaryFile("w", prefix="triposr-cli-", suffix=".txt", delete=False)
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    return Path(tmp.name)


def patch_torchmcubes_cpu(repo_dir: Path) -> None:
    cmake = repo_dir / "CMakeLists.txt"
    text = cmake.read_text()
    start = text.find("# CUDA settings")
    end = text.find("set(CMAKE_CXX_STANDARD", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate torchmcubes CUDA block to patch.")
    replacement = """# CUDA settings\nmessage(WARNING \"SAIENT: installing torchmcubes CPU extension only.\")\n\n"""
    cmake.write_text(text[:start] + replacement + text[end:])

    cxx_cmake = repo_dir / "cxx" / "CMakeLists.txt"
    cxx_text = cxx_cmake.read_text()
    start = cxx_text.find("\nif (CMAKE_CUDA_COMPILER)")
    end = cxx_text.find("\ntarget_link_libraries", start)
    if start != -1 and end != -1:
        cxx_cmake.write_text(cxx_text[:start] + "\n" + cxx_text[end:])


def install_torchmcubes_cpu(py: Path, repo_dir: Path) -> None:
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", TORCHMCUBES_URL, str(repo_dir)])
    patch_torchmcubes_cpu(repo_dir)
    shutil.rmtree(repo_dir / "build", ignore_errors=True)
    run([str(py), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(repo_dir)])


def install_dependencies(
    py: Path,
    repo_dir: Path,
    torchmcubes_dir: Path,
    torch_index: str,
    skip_torch: bool,
    include_gradio: bool,
) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82"])
    if not skip_torch:
        if torch_index == "cpu":
            run([str(py), "-m", "pip", "install", "torch", "torchvision"])
        else:
            run([str(py), "-m", "pip", "install", "torch", "torchvision", "--index-url", torch_index])
    req = filtered_requirements(repo_dir, include_gradio)
    try:
        run([str(py), "-m", "pip", "install", "-r", str(req)])
    finally:
        req.unlink(missing_ok=True)
    # TripoSR pins an older trimesh release whose GLB exporter still calls
    # ndarray.ptp, which NumPy 2.x removed. Keep the modern rembg/scipy/opencv
    # stack, stay below NumPy 2.5 for numba, and upgrade only the exporter
    # dependency that needs NumPy 2 support.
    run([str(py), "-m", "pip", "install", "numpy<2.5", "onnxruntime", "trimesh>=4.9.0"])
    install_torchmcubes_cpu(py, torchmcubes_dir)


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo).expanduser().resolve()
    torchmcubes_dir = Path(args.torchmcubes_repo).expanduser().resolve()
    venv_dir = Path(args.venv).expanduser().resolve()

    clone_repo(repo_dir, args.force_clone)
    py = create_venv(venv_dir)
    install_dependencies(py, repo_dir, torchmcubes_dir, args.torch_index, args.skip_torch, args.with_gradio)

    print("\nLocal TripoSR setup complete.")
    print(f"Python: {py}")
    print(f"Repo:   {repo_dir}")
    print("Run:    npm run local3d:run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
