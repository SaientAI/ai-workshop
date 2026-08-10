import json
import os
import shutil
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from datetime import datetime, timezone

from arc_gremlin.persistence import load_json, locked_write

BASE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SAIENT_STATE_DIR", str(BASE / "data"))).expanduser().resolve()
STATE_PATH = DATA_DIR / "state.json"
BELIEF_PATH = DATA_DIR / "belief_state.json"
EVENT_LOG_PATH = DATA_DIR / "tick_events.jsonl"
CORE_DRIVES = ("information_depth", "efficiency", "autonomy")


@contextmanager
def _tick_lock():
    """Serialize the complete load-to-save tick across desktop processes."""
    _ensure_data_dir()
    tick_lock_path = DATA_DIR / "tick.lock"
    with tick_lock_path.open("a+b") as lock:
        if tick_lock_path.stat().st_size == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def serialized(func):
    """Run one state transaction under the cross-process tick lock."""
    @wraps(func)
    def locked(*args, **kwargs):
        if kwargs.get("persist", True) is False:
            return func(*args, **kwargs)
        with _tick_lock():
            return func(*args, **kwargs)
    return locked


def _default_coupling_matrix() -> dict:
    return {
        k: {j: (1.0 if k == j else 0.05) for j in CORE_DRIVES}
        for k in CORE_DRIVES
    }


def _default_gains() -> dict:
    return {k: 0.18 for k in CORE_DRIVES}


def _default_preferences() -> dict:
    return {
        k: {
            "score": 0.0,
            "attempts": 0,
            "successes": 0,
        }
        for k in CORE_DRIVES
    }


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_beliefs() -> dict:
    _ensure_data_dir()
    default = {"caution_weight": 0.5, "exploration_bias": 0.5}
    payload = load_json(BELIEF_PATH, default=default)
    if not isinstance(payload, dict):
        return dict(default)
    out = dict(default)
    out.update(payload)
    return out


def save_beliefs(beliefs: dict) -> None:
    _ensure_data_dir()
    locked_write(BELIEF_PATH, beliefs)


def load_state() -> dict:
    _ensure_data_dir()
    if not STATE_PATH.exists():
        state = {
            "drives": {
                "information_depth": 0.7,
                "efficiency": 0.7,
                "autonomy": 0.5,
                "cortisol": 0.2,
            },
            "strategy": {
                "mode": "balanced",
                "last_updated": 0,
            },
            "params": {
                "trend_gain": 1.0,
                "coupling_matrix": _default_coupling_matrix(),
                "effect_history": {k: [] for k in CORE_DRIVES},
                "gains": _default_gains(),
                "decay": 0.015,
                "spill": 0.02,
            },
            "cooldown": {
                "drive": None,
                "ttl": 0,
            },
            "preempt_stage": {},
            "last_target": None,
            "preempt_refractory": 0,
            "active_preempt": None,
            "post_success_lock": None,
            "pending_preempts": [],
            "metrics": {
                "ticks": 0,
                "goal_switches": 0,
                "switch_rate_20": 0.0,
                "preempt_total": 0,
                "preempt_evaluated": 0,
                "preempt_improved": 0,
                "preempt_success_rate": 0.0,
                "overshoot_count": 0,
                "prediction_error": [],
                "wasted_commits": 0,
            },
            "commitments": [],
            "beliefs": load_beliefs(),
            "history": [],
            "history_total": 0,
            "tick": 0,
            "mission": None,
            "subgoals": [],
            "mission_history": [],
            "continuity": {"target": None, "ttl": 0, "streak": 0},
            "preferences": _default_preferences(),
            "identity": {"dominant": None, "stability": 0},
        }
        return state
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            state = json.load(f)
            state.setdefault("mission", None)
            state.setdefault("subgoals", [])
            state.setdefault("mission_history", [])
            state.setdefault("continuity", {"target": None, "ttl": 0, "streak": 0})
            state.setdefault("preferences", _default_preferences())
            state.setdefault("identity", {"dominant": None, "stability": 0})
            state.setdefault("history", [])
            state.setdefault("history_total", int(state.get("tick", 0)))
            state.setdefault("tick", 0)
            state.setdefault("strategy", {"mode": "balanced", "last_updated": 0})
            state.setdefault("params", {"trend_gain": 1.0})
            state["params"].setdefault("coupling_matrix", _default_coupling_matrix())
            state["params"].setdefault("effect_history", {k: [] for k in CORE_DRIVES})
            state["params"].setdefault("gains", _default_gains())
            state["params"].setdefault("decay", 0.015)
            state["params"].setdefault("spill", 0.02)
            state.setdefault("cooldown", {"drive": None, "ttl": 0})
            state.setdefault("preempt_stage", {})
            state.setdefault("last_target", None)
            state.setdefault("preempt_refractory", 0)
            state.setdefault("active_preempt", None)
            state.setdefault("post_success_lock", None)
            state.setdefault("pending_preempts", [])
            state.setdefault(
                "metrics",
                {
                    "ticks": 0,
                    "goal_switches": 0,
                    "switch_rate_20": 0.0,
                    "preempt_total": 0,
                    "preempt_evaluated": 0,
                    "preempt_improved": 0,
                    "preempt_success_rate": 0.0,
                    "overshoot_count": 0,
                    "prediction_error": [],
                    "wasted_commits": 0,
                },
            )
            state.setdefault("commitments", [])
            state.setdefault("beliefs", load_beliefs())
            state.setdefault(
                "drives",
                {
                    "information_depth": 0.7,
                    "efficiency": 0.7,
                    "autonomy": 0.5,
                    "cortisol": 0.2,
                },
            )
            for key, v in _default_preferences().items():
                pref = state["preferences"].setdefault(key, v)
                pref.setdefault("score", 0.0)
                pref.setdefault("attempts", 0)
                pref.setdefault("successes", 0)
            ident = state.setdefault("identity", {"dominant": None, "stability": 0})
            ident.setdefault("dominant", None)
            ident["stability"] = int(ident.get("stability", 0))
            return state
    except Exception:
        # Backup broken state and reinitialize from defaults.
        try:
            bad = STATE_PATH.with_suffix(".json.bad")
            shutil.copy2(STATE_PATH, bad)
        except Exception:
            pass
        state = {
            "drives": {
                "information_depth": 0.7,
                "efficiency": 0.7,
                "autonomy": 0.5,
                "cortisol": 0.2,
            },
            "strategy": {
                "mode": "balanced",
                "last_updated": 0,
            },
            "params": {
                "trend_gain": 1.0,
                "coupling_matrix": _default_coupling_matrix(),
                "effect_history": {k: [] for k in CORE_DRIVES},
                "gains": _default_gains(),
                "decay": 0.015,
                "spill": 0.02,
            },
            "cooldown": {
                "drive": None,
                "ttl": 0,
            },
            "preempt_stage": {},
            "last_target": None,
            "preempt_refractory": 0,
            "active_preempt": None,
            "post_success_lock": None,
            "pending_preempts": [],
            "metrics": {
                "ticks": 0,
                "goal_switches": 0,
                "switch_rate_20": 0.0,
                "preempt_total": 0,
                "preempt_evaluated": 0,
                "preempt_improved": 0,
                "preempt_success_rate": 0.0,
                "overshoot_count": 0,
                "prediction_error": [],
                "wasted_commits": 0,
            },
            "commitments": [],
            "beliefs": load_beliefs(),
            "history": [],
            "history_total": 0,
            "tick": 0,
            "mission": None,
            "subgoals": [],
            "mission_history": [],
            "continuity": {"target": None, "ttl": 0, "streak": 0},
            "preferences": _default_preferences(),
            "identity": {"dominant": None, "stability": 0},
        }
        return state


def save_state(state: dict) -> None:
    _ensure_data_dir()
    tmp = STATE_PATH.with_name(f"{STATE_PATH.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)


def append_event(event: dict) -> None:
    _ensure_data_dir()
    payload = dict(event)
    payload.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")))
        f.write("\n")
