from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import fcntl

from .state import SystemState, migrate_state


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


def atomic_write(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(p.parent), delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, default=_json_default)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, p)


def locked_write(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    lock_path = p.with_suffix(p.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        atomic_write(p, data)


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(default or {})


def load_system_state(path: str | Path) -> SystemState:
    raw = load_json(path, default={"schema_version": "0.0.0"})
    migrated = migrate_state(raw)
    try:
        return SystemState(**migrated)
    except Exception:
        return SystemState(**migrate_state({"schema_version": "0.0.0"}))


def save_system_state(path: str | Path, state: SystemState) -> None:
    payload = state.model_dump() if hasattr(state, "model_dump") else dict(state.__dict__)
    payload = migrate_state(payload)
    locked_write(path, payload)
