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


def configure_hf_cache() -> None:
    hf_home = data_dir() / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
