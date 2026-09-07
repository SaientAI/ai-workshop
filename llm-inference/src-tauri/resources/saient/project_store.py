"""Durable, single-runner state for explicitly authorized long projects.

SQLite commits the JSON state and its event together. Model-call reservations and
tool intentions are committed *before* execution. An interrupted action is an
unknown outcome, never a replay instruction. No credentials or token-price
estimates belong in this store; ``steps_used`` counts reserved model calls.
"""
from __future__ import annotations

import errno
from contextlib import closing
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import posixpath
import sqlite3
import stat
import tempfile
import time
import uuid


STATUSES = frozenset({"RUNNING", "WAITING", "PAUSED", "BLOCKED",
                      "BUDGET_EXHAUSTED", "COMPLETED", "STOPPED"})
DEFAULT_POLICY = {"max_steps": 10000, "max_seconds": 86400,
                  "max_no_progress": 8, "tool_timeout": 600,
                  "allow_shell": False, "allow_write": False, "verify_command": ""}
MAX_EVENTS = 1000
MAX_EVENT_BYTES = 32768
MAX_RECENT_EVENTS = 100
SCHEMA_VERSION = 1


class ProjectStoreError(RuntimeError):
    """A project cannot safely proceed."""


class ProjectLocked(ProjectStoreError):
    """Another cooperating process already owns this workspace's runner lock."""


class CorruptProject(ProjectStoreError):
    """Persistent state is invalid. Never silently replace it with a new goal."""


class InvalidTransition(ProjectStoreError):
    """The requested state change would violate project execution invariants."""


class BudgetExhausted(ProjectStoreError):
    """The project's persisted call or wall-clock allowance is exhausted."""


def validate_policy(policy=None):
    """Return a validated policy; omitted permissions default to denied."""
    if policy is None:
        policy = {}
    if not isinstance(policy, dict) or set(policy) - set(DEFAULT_POLICY):
        raise ValueError("Unknown long-project policy keys")
    result = dict(DEFAULT_POLICY, **policy)
    for key, upper in (("max_steps", 10000000), ("max_no_progress", 100000)):
        value = result[key]
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be an integer from 1 to {upper}")
    for key, upper in (("max_seconds", 366 * 86400), ("tool_timeout", 7 * 86400)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= upper:
            raise ValueError(f"{key} must be finite and from 1 to {upper} seconds")
    for key in ("allow_shell", "allow_write"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(result["verify_command"], str) or len(result["verify_command"]) > 8000:
        raise ValueError("verify_command must be a string of at most 8000 characters")
    return result


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _path_identity(path, *, windows=None):
    """Normalize an already-canonical path, including Rust's Windows prefixes."""
    windows = os.name == "nt" if windows is None else windows
    module = ntpath if windows else posixpath
    identity = module.normcase(os.fspath(path))
    if windows:
        if identity.startswith("\\\\?\\unc\\"):
            identity = "\\\\" + identity[8:]
        elif identity.startswith("\\\\?\\"):
            identity = identity[4:]
    return identity


def _path_within(path, root, *, windows=None):
    windows = os.name == "nt" if windows is None else windows
    module = ntpath if windows else posixpath
    candidate = _path_identity(path, windows=windows)
    boundary = _path_identity(root, windows=windows)
    try:
        return module.commonpath([candidate, boundary]) == boundary
    except ValueError:  # Different Windows drives or UNC shares.
        return False


def _bounded(value):
    encoded = _json(value)
    if len(encoded.encode("utf-8")) <= MAX_EVENT_BYTES:
        return value
    return {"truncated": True, "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "preview": encoded[:MAX_EVENT_BYTES // 8]}


def _regular(path):
    """Reject links, junctions and non-regular files without following them."""
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400 or info.st_nlink != 1):
        raise ProjectStoreError(f"Unsafe project state file: {path}")


def _directory(path, create=False):
    # Inspect every component, including ancestors of the supplied state root.
    for component in reversed((path, *path.parents)):
        if not component.exists() and not component.is_symlink() and create:
            component.mkdir(mode=0o700, exist_ok=True)
        info = component.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ProjectStoreError(f"Unsafe project state directory: {component}")


def _fsync_directory(path):
    # Windows does not expose directory fsync through the Python stdlib.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ProjectStore:
    """A context-managed, nonblocking runner lock and transactional journal.

    The workspace must exist. State must be outside it, under a user-controlled
    state directory. The workspace key is a SHA-256 of its canonical native path
    (case-normalized on Windows), so aliases share one lock and one allowance.
    The directory is not a sandbox against malicious same-user processes.
    """

    def __init__(self, state_dir, workspace):
        canonical = Path(workspace).resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError("Project workspace must be an existing directory")
        self.workspace = _path_identity(canonical)
        self.project_id = hashlib.sha256(self.workspace.encode("utf-8")).hexdigest()
        self.state_dir = Path(os.path.abspath(os.fspath(state_dir)))
        if _path_within(self.state_dir.resolve(), canonical):
            raise ValueError("Long-project state must be outside the project workspace")
        self.project_dir = self.state_dir / "projects" / self.project_id
        self.directory = self.project_dir
        self.db_path = self.project_dir / "project.sqlite3"
        self.lock_path = self.project_dir / "runner.lock"
        self._lock = None
        self._db = None

    def __enter__(self):
        if self._lock is not None:
            raise ProjectStoreError("ProjectStore is already open")
        _directory(self.project_dir, create=True)
        if self.lock_path.exists() or self.lock_path.is_symlink():
            _regular(self.lock_path)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._lock = os.fdopen(descriptor, "r+b")
        try:
            self._acquire_lock()
            self._check_paths()
            existed = self.db_path.exists()
            self._db = sqlite3.connect(str(self.db_path), timeout=0, isolation_level=None)
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA busy_timeout=0")
            if not existed:
                with self._transaction():
                    self._db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL)")
                    self._db.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL)")
                    self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                _fsync_directory(self.project_dir)
            if self._db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise CorruptProject("SQLite integrity check failed")
            if self._db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise CorruptProject("Unsupported project database schema")
            # Check both tables even when a project has not yet been created.
            self._db.execute("SELECT seq, at, kind, data FROM events LIMIT 1")
            self.load()
            return self
        except sqlite3.Error as error:
            self.close()
            raise CorruptProject(f"Cannot safely open project database: {error}") from error
        except BaseException:
            self.close()
            raise

    def _acquire_lock(self):
        try:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ProjectLocked("Another runner already owns this project") from error
            raise

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            # Closing releases native locks, including on exceptional paths.
            self._lock.close()
            self._lock = None

    def _check_paths(self):
        _directory(self.project_dir)
        for path in (self.lock_path, self.db_path, Path(str(self.db_path) + "-journal"),
                     Path(str(self.db_path) + "-wal"), Path(str(self.db_path) + "-shm")):
            if path.exists() or path.is_symlink():
                _regular(path)

    def _require_open(self):
        if self._db is None:
            raise ProjectStoreError("Use ProjectStore as a context manager")
        self._check_paths()

    def _transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def transaction():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return transaction()

    def load(self):
        self._require_open()
        try:
            rows = self._db.execute("SELECT id,state FROM metadata").fetchall()
            if not rows:
                if self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
                    raise CorruptProject("Project metadata is missing but events remain")
                return None
            if len(rows) != 1 or rows[0][0] != 1:
                raise CorruptProject("Invalid metadata row count")
            state = json.loads(rows[0][1])
            self._validate_state(state)
            return state
        except (sqlite3.Error, ValueError, TypeError, KeyError) as error:
            raise CorruptProject(f"Invalid persistent project state: {error}") from error

    snapshot = load

    def read_snapshot(self):
        """Read committed status without taking the runner lock or creating files.

        A transient SQLite writer contention is reported, not mistaken for an
        absent project. This method never replays journal recovery or changes
        budgets. The runner exclusively owns execution and writable state.
        """
        if self._db is not None:
            return self.load()
        if not self.project_dir.exists() and not self.project_dir.is_symlink():
            return None
        self._check_paths()
        if not self.db_path.exists():
            return None
        try:
            with closing(sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
                if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    raise CorruptProject("Unsupported project database schema")
                rows = connection.execute("SELECT id,state FROM metadata").fetchall()
                if not rows:
                    if connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
                        raise CorruptProject("Project metadata is missing but events remain")
                    return None
                if len(rows) != 1 or rows[0][0] != 1:
                    raise CorruptProject("Invalid metadata row count")
                state = json.loads(rows[0][1])
                self._validate_state(state)
                return state
        except (sqlite3.Error, ValueError, TypeError, KeyError) as error:
            raise CorruptProject(f"Cannot read committed project status: {error}") from error

    def _validate_state(self, state):
        required = {"schema_version", "project_id", "workspace", "goal", "policy",
                    "created_at", "updated_at", "deadline_at", "status", "reason",
                    "steps_used", "no_progress_count", "inflight", "recovery_required",
                    "last_action", "revision"}
        if (not isinstance(state, dict) or set(state) != required
                or type(state.get("schema_version")) is not int
                or state.get("schema_version") != SCHEMA_VERSION):
            raise CorruptProject("Unsupported state schema")
        if state["project_id"] != self.project_id or state["workspace"] != self.workspace:
            raise CorruptProject("Project identity does not match the workspace")
        if state["status"] not in STATUSES:
            raise CorruptProject("Unknown project status")
        if validate_policy(state["policy"]) != state["policy"]:
            raise CorruptProject("Incomplete project policy")
        if not isinstance(state["goal"], str) or not 1 <= len(state["goal"]) <= 16384:
            raise CorruptProject("Invalid project goal")
        for key in ("created_at", "updated_at", "deadline_at"):
            if type(state[key]) not in (int, float) or not math.isfinite(state[key]):
                raise CorruptProject("Invalid project timestamp")
        if state["deadline_at"] != state["created_at"] + state["policy"]["max_seconds"]:
            raise CorruptProject("Project deadline does not match the authorized allowance")
        for key in ("steps_used", "no_progress_count", "revision"):
            if type(state[key]) is not int or state[key] < 0:
                raise CorruptProject("Invalid usage counter")
        if state["steps_used"] > state["policy"]["max_steps"]:
            raise CorruptProject("Project model-call usage exceeds its allowance")
        if (type(state["recovery_required"]) is not bool
                or not isinstance(state["reason"], str) or len(state["reason"]) > 8192):
            raise CorruptProject("Invalid recovery state")
        action = state["inflight"]
        if action is not None:
            if (not isinstance(action, dict) or set(action) != {"id", "tool", "started_at"}
                    or not isinstance(action["id"], str)
                    or not isinstance(action["tool"], (str, dict))):
                raise CorruptProject("Invalid in-flight action")
            uuid.UUID(action["id"])
            if type(action["started_at"]) not in (int, float) or not math.isfinite(action["started_at"]):
                raise CorruptProject("Invalid action timestamp")
        if state["recovery_required"] and action is None:
            raise CorruptProject("Unknown outcome is missing its action identity")
        previous = state["last_action"]
        if previous is not None:
            if (not isinstance(previous, dict) or set(previous) != {"id", "tool", "outcome"}
                    or not isinstance(previous["id"], str)
                    or not isinstance(previous["tool"], (str, dict))
                    or not isinstance(previous["outcome"], dict)):
                raise CorruptProject("Invalid recorded action outcome")
            uuid.UUID(previous["id"])

    def _save(self, state, kind, data=None):
        self._require_open()
        state["updated_at"] = time.time()
        state["revision"] += 1
        self._validate_state(state)
        encoded = _json(state)
        event = _json(_bounded({} if data is None else data))
        with self._transaction():
            self._db.execute("INSERT OR REPLACE INTO metadata(id,state) VALUES(1,?)", (encoded,))
            self._db.execute("INSERT INTO events(at,kind,data) VALUES(?,?,?)", (state["updated_at"], kind, event))
            self._db.execute("DELETE FROM events WHERE seq <= (SELECT COALESCE(MAX(seq),0)-? FROM events)", (MAX_EVENTS,))
        return state

    def _state(self):
        state = self.load()
        if state is None:
            raise ProjectStoreError("No project has been created for this workspace")
        return state

    def create(self, goal, policy=None):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 16384:
            raise ValueError("Goal must contain 1 to 16384 characters")
        policy = validate_policy(policy)
        if self.load() is not None:
            raise InvalidTransition("A project already exists; it cannot be overwritten or reset")
        now = time.time()
        state = {"schema_version": SCHEMA_VERSION, "project_id": self.project_id,
                 "workspace": self.workspace, "goal": goal.strip(), "policy": policy,
                 "created_at": now, "updated_at": now, "deadline_at": now + policy["max_seconds"],
                 "status": "RUNNING", "reason": "User started project", "steps_used": 0,
                 "no_progress_count": 0, "inflight": None, "recovery_required": False,
                 "last_action": None, "revision": 0}
        return self._save(state, "created", {"goal": state["goal"], "policy": policy})

    def _check_budget(self, state, for_step=False):
        reason = None
        if time.time() >= state["deadline_at"]:
            reason = "Authorized wall-clock deadline reached (including paused time)"
        elif state["steps_used"] >= state["policy"]["max_steps"] and for_step:
            reason = "Authorized model-call allowance exhausted"
        if reason:
            state["status"], state["reason"] = "BUDGET_EXHAUSTED", reason
            self._save(state, "budget_exhausted", {"reason": reason})
            raise BudgetExhausted(reason)

    def _running(self):
        state = self._state()
        if state["status"] != "RUNNING" or state["recovery_required"]:
            raise InvalidTransition(f"Project is {state['status']}; explicit resume may be required")
        self._check_budget(state)
        return state

    def transition(self, status, reason=""):
        if status not in STATUSES or status == "RUNNING":
            raise InvalidTransition("Use resume() to enter RUNNING")
        state = self._state()
        if state["status"] in {"COMPLETED", "STOPPED", "BUDGET_EXHAUSTED"}:
            raise InvalidTransition(f"Project is terminal: {state['status']}")
        if status == "COMPLETED" and (state["inflight"] is not None or state["recovery_required"]):
            raise InvalidTransition("An unknown action cannot be declared completed")
        if not isinstance(reason, str) or len(reason) > 8192:
            raise ValueError("Transition reason must be a string of at most 8192 characters")
        state["status"], state["reason"] = status, reason
        return self._save(state, "transition", {"status": status, "reason": reason})

    def reserve_step(self):
        state = self._running()
        if state["inflight"] is not None:
            raise InvalidTransition("Resolve the current action before another model call")
        self._check_budget(state, for_step=True)
        state["steps_used"] += 1
        return self._save(state, "model_call_reserved", {"steps_used": state["steps_used"]})

    def begin_action(self, tool):
        if not isinstance(tool, (str, dict)) or not tool:
            raise ValueError("Tool intention must be a non-empty string or object")
        state = self._running()
        if state["inflight"] is not None:
            raise InvalidTransition("Only one action can be in flight")
        if state["steps_used"] < 1:
            raise InvalidTransition("Reserve a model call before starting its action")
        action = {"id": str(uuid.uuid4()), "tool": _bounded(tool), "started_at": time.time()}
        state["inflight"] = action
        self._save(state, "action_started", action)
        return action["id"]

    def finish_action(self, action_id, row):
        if not isinstance(row, dict):
            raise ValueError("Action outcome must be an object")
        state = self._state()
        action = state["inflight"]
        if action is None or action["id"] != action_id:
            raise InvalidTransition("Action result does not match the persisted in-flight identity")
        if state["recovery_required"]:
            raise InvalidTransition("A recovered unknown outcome requires explicit acknowledgment")
        outcome = {"id": action_id, "tool": action["tool"], "outcome": _bounded(row)}
        state["last_action"] = outcome
        state["inflight"] = None
        return self._save(state, "action_finished", outcome)

    def recover(self):
        state = self._state()
        if state["inflight"] is not None:
            if state["recovery_required"]:
                return state
            state["recovery_required"] = True
            if state["status"] not in {"STOPPED", "BUDGET_EXHAUSTED"}:
                state["status"] = "BLOCKED"
            state["reason"] = "Previous action has an unknown outcome; inspect its effects before acknowledging recovery"
            return self._save(state, "unknown_action", state["inflight"])
        if state["status"] == "RUNNING":
            state["status"] = "PAUSED"
            state["reason"] = "Runner reopened; user resume required (reserved calls remain charged)"
            return self._save(state, "recovered_paused")
        return state

    def resume(self, acknowledge_unknown=False):
        if type(acknowledge_unknown) is not bool:
            raise ValueError("Recovery acknowledgment must be an explicit boolean")
        state = self._state()
        if state["status"] not in {"PAUSED", "WAITING", "BLOCKED"}:
            raise InvalidTransition(f"Cannot resume a {state['status']} project")
        if state["inflight"] is not None or state["recovery_required"]:
            if not acknowledge_unknown:
                raise InvalidTransition("Inspect the unknown action, then explicitly acknowledge recovery")
            state = self.acknowledge_unknown()
        self._check_budget(state, for_step=True)
        state["status"], state["reason"] = "RUNNING", "User resumed project"
        # Resuming is an explicit decision to try again, not permission to reset budgets.
        state["no_progress_count"] = 0
        return self._save(state, "resumed")

    def wake_from_wait(self):
        """Continue after availability returns, without impersonating user resume.

        Waiting does not reset the no-progress breaker or any allowance. Only
        an idle, non-ambiguous WAITING project can automatically continue.
        """
        state = self._state()
        if state["status"] != "WAITING" or state["inflight"] is not None or state["recovery_required"]:
            raise InvalidTransition("Automatic availability recovery requires an idle WAITING project")
        self._check_budget(state, for_step=True)
        state["status"] = "RUNNING"
        state["reason"] = "Model availability recovered; continuing within the existing allowance"
        return self._save(state, "availability_recovered")

    def acknowledge_unknown(self):
        """Record explicit user review, without resuming or resetting allowances.

        The controller must invoke this only for a user's explicit recovery
        acknowledgment, never on a model's request or as automatic crash repair.
        It also permits a stopped project's reviewed unknown to be archived.
        """
        state = self._state()
        if state["status"] == "RUNNING":
            raise InvalidTransition("A running action cannot be acknowledged as a recovered unknown")
        unknown = state["inflight"]
        if unknown is None:
            raise InvalidTransition("No unknown action requires acknowledgment")
        state["inflight"] = None
        state["recovery_required"] = False
        state["last_action"] = {"id": unknown["id"], "tool": unknown["tool"],
                                "outcome": {"status": "unknown", "user_acknowledged": True}}
        state["reason"] = "User acknowledged unknown action outcome; no action was replayed"
        return self._save(state, "recovery_acknowledged", state["last_action"])

    def record_progress(self, made_progress, reason=""):
        if type(made_progress) is not bool or not isinstance(reason, str) or len(reason) > 8192:
            raise ValueError("Progress requires a boolean and a bounded reason")
        state = self._state()
        if state["status"] != "RUNNING" or state["inflight"] is not None:
            raise InvalidTransition("Progress is recorded only after a running action is resolved")
        state["no_progress_count"] = 0 if made_progress else state["no_progress_count"] + 1
        if state["no_progress_count"] >= state["policy"]["max_no_progress"]:
            state["status"] = "BLOCKED"
            state["reason"] = "No-progress limit reached; user review required"
        return self._save(state, "progress", {"made_progress": made_progress, "reason": reason})

    def usage(self):
        state = self._state()
        return {"steps_used": state["steps_used"],
                "steps_remaining": max(0, state["policy"]["max_steps"] - state["steps_used"]),
                "seconds_remaining": max(0, state["deadline_at"] - time.time()),
                "deadline_at": state["deadline_at"], "no_progress_count": state["no_progress_count"]}

    def extend_budget(self, add_steps=0, add_seconds=0):
        """Add explicitly user-approved allowances without resetting history.

        This is not callable as a model tool. The controller must require a
        user's explicit command. Exhausted projects become PAUSED, not RUNNING;
        permissions, goal, prior reservations and no-progress evidence remain.
        """
        if type(add_steps) is not int or add_steps < 0:
            raise ValueError("Additional steps must be a nonnegative integer")
        if (type(add_seconds) not in (int, float) or not math.isfinite(add_seconds)
                or add_seconds < 0 or (add_steps == 0 and add_seconds == 0)):
            raise ValueError("Add a positive, finite number of steps or seconds")
        state = self._state()
        if state["status"] not in {"PAUSED", "BLOCKED", "BUDGET_EXHAUSTED"}:
            raise InvalidTransition("Pause the project before extending its authorized budget")
        if state["inflight"] is not None or state["recovery_required"]:
            raise InvalidTransition("Review unknown outcomes before extending the project")
        policy = dict(state["policy"])
        policy["max_steps"] += add_steps
        policy["max_seconds"] += add_seconds
        state["policy"] = validate_policy(policy)
        state["deadline_at"] = state["created_at"] + policy["max_seconds"]
        if state["status"] == "BUDGET_EXHAUSTED":
            state["status"] = "PAUSED"
        state["reason"] = "User extended the budget; explicit resume is still required"
        return self._save(state, "budget_extended", {"add_steps": add_steps, "add_seconds": add_seconds,
                                                    "max_steps": policy["max_steps"],
                                                    "deadline_at": state["deadline_at"]})

    def archive(self):
        """Explicitly retain the old database before allowing another goal.

        A crash before the clearing transaction leaves the original project
        intact, possibly with an extra safe backup. A crash after it leaves a
        complete backup and an empty current project. Never replace the lock
        file, reset an active project, or discard an unknown action outcome.
        """
        state = self._state()
        if state["status"] in {"RUNNING", "WAITING"}:
            raise InvalidTransition("Pause or stop the project before explicitly archiving it")
        if state["inflight"] is not None or state["recovery_required"]:
            raise InvalidTransition("Unknown action outcomes must be reviewed before archival")
        archives = self.project_dir / "archives"
        _directory(archives, create=True)
        destination = archives / f"project-{uuid.uuid4().hex}.sqlite3"
        descriptor, name = tempfile.mkstemp(prefix=".archive-", suffix=".sqlite3", dir=archives)
        temporary = Path(name)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(str(temporary))) as backup:
                self._db.backup(backup)
                if backup.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise CorruptProject("Archive integrity check failed; current project retained")
            with temporary.open("r+b") as file:
                os.fsync(file.fileno())
            # link is atomic and refuses existing names on both supported OSes.
            os.link(temporary, destination)
            temporary.unlink()
            _fsync_directory(archives)
            self._check_paths()
            with self._transaction():
                self._db.execute("DELETE FROM metadata")
                self._db.execute("DELETE FROM events")
            return destination
        finally:
            if temporary.exists():
                temporary.unlink()

    def recent_events(self, limit=20):
        self._require_open()
        if type(limit) is not int or not 1 <= limit <= MAX_RECENT_EVENTS:
            raise ValueError(f"Recent event limit must be between 1 and {MAX_RECENT_EVENTS}")
        try:
            rows = self._db.execute("SELECT seq,at,kind,data FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
            return [{"seq": seq, "at": at, "kind": kind, "data": json.loads(data)}
                    for seq, at, kind, data in reversed(rows)]
        except (sqlite3.Error, ValueError) as error:
            raise CorruptProject(f"Invalid event journal: {error}") from error
