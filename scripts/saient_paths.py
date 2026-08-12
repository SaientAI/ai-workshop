"""Shared Saient runtime paths for helper scripts."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("SAIENT_DATA_DIR", repo_root() / "data")).expanduser()


def config_dir() -> Path:
    return Path(os.environ.get("SAIENT_CONFIG_DIR", data_dir() / "config" / "saient")).expanduser()


def models_dir() -> Path:
    return Path(os.environ.get("SAIENT_MODELS_DIR", data_dir() / "models")).expanduser()


def model_scan_dirs() -> list[Path]:
    root = models_dir()
    return [
        root,
        root / "image",
        root / "saient",
        root / "home-models",
        root / "projects-models",
        root / "llm-runtime",
    ]


def cache_dir(name: str) -> Path:
    new = config_dir() / name
    legacy = Path.home() / ".config" / "ai-workshop" / name
    if not new.exists() and legacy.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(new)
        except Exception:
            return legacy
    return new


def shared_cache_dir(name: str) -> Path:
    """Return an app-owned cache path shared by debug and release builds."""
    root = Path(os.environ.get("SAIENT_CACHE_DIR", data_dir() / "cache")).expanduser()
    return root / name


def configure_hf_cache(*, offline: bool = True) -> None:
    """Contain library state and make inference local-only by default.

    Bundled runtime scripts must never turn the general Internet switch into
    Hugging Face access. Explicit downloader utilities may pass offline=False.
    The temporary directory is cleared by the desktop app on its next start.
    """
    hf_home = data_dir() / "runtime-tmp" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def runtime_assets_dir() -> Path:
    return Path(
        os.environ.get("SAIENT_RUNTIME_ASSETS_DIR", data_dir() / "runtime-assets")
    ).expanduser()
