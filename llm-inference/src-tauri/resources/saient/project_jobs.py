"""Owned, bounded background commands for opt-in long projects (stdlib only).

A detached supervisor owns the reader, command and deadlines, including after
the calling UI/terminal exits. Each job has a private directory containing
bounded output and atomic JSON evidence. ``inspect_record`` marks nonterminal
records ``recover_unknown``; it never reattaches, signals a saved PID, or retries
a possibly effectful command. The project controller must reconcile before
resuming those actions. A crash of the supervisor itself is still ambiguous.

Only explicitly supplied ``popen_options`` can enable a shell. Commands must
remain in their process group/tree; deliberately daemonized or detached children
are outside this supervisor's guarantee. Native Windows uses taskkill /T /F;
its errors are surfaced, not treated as successful cancellation.
"""

from __future__ import annotations

import json
import errno
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable
import uuid


TERMINAL_STATUSES = frozenset({"completed", "failed", "timed_out", "cancelled"})
MAX_TIMEOUT_SECONDS = 365 * 24 * 60 * 60
MAX_LOG_BYTES = 8 * 1024 * 1024
TAIL_BYTES = 64 * 1024
_WINDOWS_JOB = None


def _runner_lock(handle, acquire):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _supervisor_active(directory):
    try:
        with (directory / "runner.lock").open("r+b") as handle:
            try:
                _runner_lock(handle, True)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    return True
                raise
            _runner_lock(handle, False)
            return False
    except FileNotFoundError:
        return False
    except OSError:
        return None  # An unreadable lock is not evidence that its owner exited.


class _WindowsContainment:
    """Put this detached worker and its descendants in a kill-on-close job.

    The supervisor joins BEFORE Popen, avoiding a child-spawn/assignment race.
    The unnamed, non-inheritable handle deliberately remains open until process
    shutdown. Windows closes it even if the worker crashes, terminating all its
    ordinary CreateProcess descendants. No breakaway permissions are enabled.
    https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
    """

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        self.wintypes = wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.kernel.IsProcessInJob.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not self.kernel.AssignProcessToJobObject(self.handle, self.kernel.GetCurrentProcess())):
            error = ctypes.WinError(ctypes.get_last_error())
            self.kernel.CloseHandle(self.handle)
            raise error  # Fail before launching an uncontained command.

    def remaining_children(self):
        # Job accounting can lag process exit. Inspect current native handles
        # instead of treating a stale count as a surviving command tree. These
        # IDs come from this live Job Object, never persisted PID metadata.
        ctypes, wintypes = self.ctypes, self.wintypes
        capacity = 64
        while capacity <= 65536:
            class ProcessList(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("ids", ctypes.c_size_t * capacity)]
            info = ProcessList()
            ok = self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None)
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            if not ok or info.count < info.assigned:
                capacity = max(capacity * 2, info.assigned)
                continue
            for pid in info.ids[:info.count]:
                if pid == os.getpid():
                    continue  # The supervisor itself remains associated.
                handle = self.kernel.OpenProcess(0x00101000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
                if not handle:
                    if ctypes.get_last_error() == 87:  # Process already vanished.
                        continue
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    member = wintypes.BOOL()
                    if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    status = self.kernel.WaitForSingleObject(handle, 0)
                    if status not in (0, 258):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if member.value and status == 258:
                        return True
                finally:
                    self.kernel.CloseHandle(handle)
            return False
        raise RuntimeError("Windows job process list exceeded its inspection bound")


def _duration(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= MAX_TIMEOUT_SECONDS:
        raise ValueError(name + " must be between 0 and 365 days")
    return value


def _validate_start(argv, cwd, timeout_seconds, idle_timeout_seconds, popen_options, max_log_bytes):
    timeout_seconds = _duration(timeout_seconds, "timeout_seconds")
    if idle_timeout_seconds is not None:
        idle_timeout_seconds = _duration(idle_timeout_seconds, "idle_timeout_seconds")
    if (isinstance(max_log_bytes, bool) or not isinstance(max_log_bytes, int)
            or not 1024 <= max_log_bytes <= 64 * 1024 * 1024):
        raise ValueError("max_log_bytes must be an integer between 1024 and 64 MiB")
    options = dict(popen_options or {})
    allowed = {"shell", "executable", "start_new_session", "creationflags"}
    if set(options) - allowed:
        raise ValueError("unsupported popen_options: " + ", ".join(sorted(set(options) - allowed)))
    if isinstance(argv, str):
        if not options.get("shell"):
            raise ValueError("string commands require an explicitly enabled shell")
    elif not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("argv must be a nonempty sequence of strings")
    if not argv or not argv[0]:
        raise ValueError("empty command")
    if os.name == "nt":
        # These are noninteractive jobs with pipes, not a new terminal. A
        # detached supervisor otherwise causes Windows to create a console
        # host, which can outlive a successful command inside our Job Object.
        options["creationflags"] = (options.get("creationflags", 0) |
                                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    else:
        if options.get("start_new_session") is False:
            raise ValueError("owned process groups require start_new_session=True")
        options["start_new_session"] = True
    workspace = Path(cwd).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("cwd must be a directory")
    return str(workspace), timeout_seconds, idle_timeout_seconds, options


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".job-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Native Windows readers/virus scanners can briefly deny replacement.
        # Retry sharing/lock violations only, for at most 310 ms; permanent
        # access errors still fail closed and are recorded by the supervisor.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (32, 33) or attempt == 5:
                    raise
                time.sleep(0.01 * 2 ** attempt)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


class _OwnedJob:
    """Private worker-side Popen ownership; never reconstructed from metadata."""

    @classmethod
    def start(cls, argv, *, cwd, env, state_dir, timeout_seconds,
              idle_timeout_seconds=None, popen_options=None,
              max_log_bytes=MAX_LOG_BYTES, _job_id=None):
        workspace, timeout_seconds, idle_timeout_seconds, options = _validate_start(
            argv, cwd, timeout_seconds, idle_timeout_seconds, popen_options, max_log_bytes)
        self = cls()
        self.job_id = _job_id or uuid.uuid4().hex
        root = Path(state_dir).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = root / self.job_id
        self.directory.mkdir(mode=0o700, exist_ok=_job_id is not None)
        self.log_path = self.directory / "output.log"
        self.metadata_path = self.directory / "job.json"
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._reader_done = threading.Event()
        self._cancel_requested = False
        self._status = "starting"
        self._returncode = None
        self._tail = b""
        self._output_bytes = 0
        self._log_truncated = False
        self._error = None
        self._cleanup_error = None
        self._timeout_reason = None
        self._reader_error = None
        self._started = time.monotonic()
        self._last_output = self._started
        self._finished = None
        self._created_at = time.time()
        self._timeout = timeout_seconds
        self._idle_timeout = idle_timeout_seconds
        self._max_log_bytes = max_log_bytes
        self._cwd = str(workspace)
        self._process = None
        # No environment or command text is persisted here: they can contain secrets.
        self._persist()
        try:
            fd = os.open(self.log_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            self._log = os.fdopen(fd, "r+b", buffering=0)
            self._process = subprocess.Popen(
                argv, cwd=str(workspace), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options)
            self._status = "running"
            self._persist()
        except Exception as exc:
            self._error = "launch failed: %s" % exc
            if self._process is not None:
                self._terminate_tree()
                self._process.stdout.close()
            if hasattr(self, "_log"):
                self._log.close()
            self._reader_done.set()
            self._finish("failed")
            return self
        self._reader = threading.Thread(target=self._read_output, name="saient-job-output-" + self.job_id, daemon=True)
        self._watchdog = threading.Thread(target=self._supervise, name="saient-job-watch-" + self.job_id, daemon=True)
        try:
            self._reader.start()
            self._watchdog.start()
        except Exception as exc:
            self._error = "supervisor threads could not start: %s" % exc
            self._terminate_tree()
            if self._reader.ident is None:
                self._process.stdout.close()
                self._log.close()
                self._reader_done.set()
            elif not self._reader_done.wait(2):
                self._cleanup_error = self._cleanup_error or "output pipe remained open after thread startup failure"
            self._finish("failed")
        return self

    def poll(self):
        """Return evidence without waiting for output or child completion."""
        with self._lock:
            return {
                "job_id": self.job_id, "status": self._status,
                "returncode": self._returncode,
                "tail": self._tail.decode("utf-8", errors="replace"),
                "elapsed": (self._finished or time.monotonic()) - self._started,
                "log_path": str(self.log_path), "metadata_path": str(self.metadata_path),
                "output_bytes": self._output_bytes, "log_truncated": self._log_truncated,
                "cancel_requested": self._cancel_requested,
                "timeout_reason": self._timeout_reason, "error": self._error,
                "cleanup_error": self._cleanup_error,
            }

    def cancel(self):
        """Request tree cancellation; wait() or poll() observes its final result."""
        with self._lock:
            if self._status not in TERMINAL_STATUSES:
                self._cancel_requested = True
        self._wake.set()
        return self.poll()

    def wait(self, poll_interval=0.1, on_progress: Callable | None = None,
             should_stop: Callable | None = None):
        """Wait with responsive cancellation, including silent or no-newline jobs.

        Progress callbacks must be nonblocking. Callback exceptions (including
        KeyboardInterrupt) request cancellation before propagating to the caller.
        """
        interval = min(_duration(poll_interval, "poll_interval"), 0.1)
        try:
            while True:
                snapshot = self.poll()
                if on_progress is not None:
                    on_progress(snapshot)
                if snapshot["status"] in TERMINAL_STATUSES:
                    return snapshot
                if should_stop is not None and should_stop():
                    self.cancel()
                self._wake.wait(interval)
                # Only the watchdog clears this signal. Avoid spinning while it
                # handles a potentially slower native process-tree termination.
                if self._wake.is_set():
                    time.sleep(interval)
        except BaseException:
            self.cancel()
            raise

    @staticmethod
    def inspect_record(metadata_path):
        """Read history, never restore process ownership from a persisted PID."""
        path = Path(metadata_path)
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != 1:
            raise ValueError("unsupported job metadata schema")
        record["original_status"] = record.get("status")
        record["terminal"] = record.get("status") in TERMINAL_STATUSES and not record.get("cleanup_error")
        record["supervisor_active"] = _supervisor_active(path.parent)
        if not record["terminal"]:
            record["recorded_status"] = record.get("status")
            record["status"] = "recover_unknown"
            record["recovery_note"] = "Prior owner execution is not attached; reconcile effects and processes before retry."
        return record

    def _persist(self):
        snapshot = self.poll()
        snapshot.pop("tail")  # The bounded log owns output, not repeated JSON snapshots.
        snapshot.update(schema_version=1, owner_pid=os.getpid(),
                        pid=self._process.pid if self._process else None,
                        cwd=self._cwd, created_at=self._created_at,
                        updated_at=time.time(), timeout_seconds=self._timeout,
                        idle_timeout_seconds=self._idle_timeout)
        _atomic_json(self.metadata_path, snapshot)

    def _read_output(self):
        try:
            while True:
                chunk = self._process.stdout.read1(8192)
                if not chunk:
                    break
                with self._lock:
                    self._last_output = time.monotonic()
                    self._output_bytes += len(chunk)
                    self._tail = (self._tail + chunk)[-TAIL_BYTES:]
                    size = self._log.seek(0, os.SEEK_END)
                    if size + len(chunk) > self._max_log_bytes:
                        # Compact by half a cap to avoid rewriting 8 MiB for
                        # every subsequent 8 KiB output chunk.
                        retain = min(size, max(0, self._max_log_bytes // 2 - len(chunk)))
                        self._log.seek(size - retain)
                        chunk = self._log.read(retain) + chunk
                        chunk = chunk[-self._max_log_bytes:]
                        self._log.seek(0)
                        self._log.truncate()
                        self._log_truncated = True
                    self._log.write(chunk)
        except Exception as exc:
            with self._lock:
                self._reader_error = "output capture failed: %s" % exc
            self._wake.set()
        finally:
            try:
                self._process.stdout.close()
                self._log.flush()
                os.fsync(self._log.fileno())
            except Exception as exc:
                with self._lock:
                    self._reader_error = "output could not be finalized: %s" % exc
            finally:
                self._log.close()
            self._reader_done.set()
            self._wake.set()

    def _terminate_tree(self):
        """Only act on the live Popen owned by this instance, never a disk PID."""
        proc = self._process
        if proc is None:
            return
        try:
            if os.name == "nt":
                taskkill = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "taskkill.exe")
                result = subprocess.run(
                    [taskkill, "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode != 0:
                    raise RuntimeError("taskkill exited %s: %s" % (result.returncode, (result.stderr or result.stdout).strip()))
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # This owned process group has already ceased to exist.
            proc.wait(timeout=2)
        except Exception as exc:
            self._cleanup_error = "process-tree cleanup failed: %s" % exc
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception as fallback:
                    self._cleanup_error += "; direct-child fallback failed: %s" % fallback

    def _finish(self, status):
        with self._lock:
            self._status = "failed" if self._cleanup_error or self._reader_error else status
            self._returncode = self._process.returncode if self._process else None
            self._finished = time.monotonic()
            if self._reader_error:
                self._error = self._reader_error
            try:
                self._persist()
            except Exception as exc:
                self._status = "failed"
                self._error = "final evidence could not be saved: %s" % exc
        self._wake.set()

    def _supervise(self):
        next_save = time.monotonic() + 1
        ended_at = None
        try:
            while True:
                now = time.monotonic()
                with self._lock:
                    cancelled, last_output = self._cancel_requested, self._last_output
                rc = self._process.poll()
                if rc is not None and ended_at is None:
                    ended_at = now
                if rc is not None and self._reader_done.is_set():
                    if os.name == "nt":
                        residual = _WINDOWS_JOB is not None and _WINDOWS_JOB.remaining_children()
                    else:
                        try:
                            os.killpg(self._process.pid, 0)
                            residual = True
                        except ProcessLookupError:
                            residual = False
                    if residual:
                        # Windows may still be shutting down a console/helper
                        # after the command handle signals. Observe actual job
                        # liveness for a bounded grace period; do not declare
                        # success while a descendant is still active.
                        if (os.name == "nt" and now - ended_at < 1 and not cancelled
                                and now - self._started < self._timeout):
                            self._wake.wait(0.05)
                            self._wake.clear()
                            continue
                        self._error = "command exited while descendants were still running; owned tree terminated"
                        self._terminate_tree()
                        self._finish("failed")
                    else:
                        self._finish("completed" if rc == 0 else "failed")
                    return
                reason = None
                if cancelled:
                    reason = "cancelled"
                elif self._reader_error:
                    reason = "failed"
                elif now - self._started >= self._timeout:
                    self._timeout_reason, reason = "wall", "timed_out"
                elif self._idle_timeout is not None and now - last_output >= self._idle_timeout:
                    self._timeout_reason, reason = "idle", "timed_out"
                elif ended_at is not None and now - ended_at >= 1:
                    # An exited root whose descendants retain output is not a
                    # completed command. Reap its owned tree rather than hang.
                    self._error = "command exited but descendants kept its output open"
                    reason = "failed"
                if reason:
                    self._terminate_tree()
                    if not self._reader_done.wait(2):
                        self._cleanup_error = self._cleanup_error or "output pipe remained open after tree cleanup"
                    self._finish(reason)
                    return
                if now >= next_save:
                    self._persist()
                    next_save = now + 1
                self._wake.wait(0.05)
                self._wake.clear()
        except BaseException as exc:
            self._error = "job supervisor failed: %s" % exc
            self._terminate_tree()
            if not self._reader_done.wait(2):
                self._cleanup_error = self._cleanup_error or "output pipe remained open after supervisor failure"
            self._finish("failed")


class ManagedJob:
    """A durable background job supervised separately from its caller.

    start() returns promptly; poll() reads atomic evidence and a bounded log tail.
    cancel() requests cancellation without waiting. wait() observes the final
    state while calling optional short progress/stop callbacks. Retained job
    directories are user evidence; this module never automatically deletes them.
    """

    @classmethod
    def start(cls, argv, *, cwd, env, state_dir, timeout_seconds,
              idle_timeout_seconds=None, popen_options=None, max_log_bytes=MAX_LOG_BYTES):
        workspace, timeout_seconds, idle_timeout_seconds, options = _validate_start(
            argv, cwd, timeout_seconds, idle_timeout_seconds, popen_options, max_log_bytes)
        self = cls()
        self.job_id = uuid.uuid4().hex
        root = Path(state_dir).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = root / self.job_id
        self.directory.mkdir(mode=0o700)
        self.log_path = self.directory / "output.log"
        self.metadata_path = self.directory / "job.json"
        self.worker_log_path = self.directory / "supervisor.log"
        self._worker = None
        self._cancel_requested = False
        self._initial = {
            "schema_version": 1, "job_id": self.job_id, "status": "starting",
            "returncode": None, "elapsed": 0.0, "tail": "",
            "log_path": str(self.log_path), "metadata_path": str(self.metadata_path),
            "output_bytes": 0, "log_truncated": False, "cancel_requested": False,
            "timeout_reason": None, "error": None, "cleanup_error": None,
            "created_at": time.time(), "updated_at": time.time(), "pid": None,
            "owner_pid": os.getpid(), "cwd": workspace, "timeout_seconds": timeout_seconds,
            "idle_timeout_seconds": idle_timeout_seconds,
        }
        _atomic_json(self.metadata_path, self._initial)
        launch_path = self.directory / "launch.json"
        payload = dict(argv=argv, cwd=workspace, env=env, state_dir=str(root),
                       timeout_seconds=timeout_seconds, idle_timeout_seconds=idle_timeout_seconds,
                       popen_options=options, max_log_bytes=max_log_bytes, _job_id=self.job_id)
        worker_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                          if os.name == "nt" else {"start_new_session": True})
        try:
            # Credentials can occur in env/argv: private, transient transfer only.
            _atomic_json(launch_path, payload)
            fd = os.open(self.worker_log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as diagnostics:
                self._worker = subprocess.Popen(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(launch_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=diagnostics,
                    **worker_options)
        except Exception as exc:
            self._initial.update(status="failed", error="supervisor launch failed: %s" % exc)
            _atomic_json(self.metadata_path, self._initial)
            launch_path.unlink(missing_ok=True)
        return self

    @staticmethod
    def inspect_record(metadata_path):
        return _OwnedJob.inspect_record(metadata_path)

    @staticmethod
    def request_cancel(metadata_path):
        """Request a retained job stop without adopting or signalling any PID."""
        path = Path(metadata_path)
        if path.name != "job.json" or path.is_symlink() or path.parent.is_symlink():
            raise ValueError("cancellation requires the original nonsymlink job.json")
        record = ManagedJob.inspect_record(path)
        job_id = record.get("job_id", "")
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id) or path.parent.name != job_id:
            raise ValueError("job identity does not match its directory")
        if not record["terminal"]:
            target = path.parent / "cancel.request"
            try:
                fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            except FileExistsError:
                if target.is_symlink() or not target.is_file():
                    raise ValueError("invalid cancellation request file")
            record["cancel_requested"] = True
        return record

    def poll(self):
        try:
            with self.metadata_path.open(encoding="utf-8") as handle:
                snapshot = json.load(handle)
            if self._worker is not None and self._worker.poll() is not None:
                # Read after observing worker exit: its final replace may have
                # raced the first read. Never infer completion from worker rc.
                with self.metadata_path.open(encoding="utf-8") as handle:
                    snapshot = json.load(handle)
                if snapshot.get("status") not in TERMINAL_STATUSES:
                    snapshot["recorded_status"] = snapshot.get("status")
                    snapshot["status"] = "recover_unknown"
                    snapshot["error"] = "supervisor exited without final execution evidence; reconcile before retry"
            if snapshot.get("cleanup_error"):
                snapshot["recorded_status"] = snapshot.get("status")
                snapshot["status"] = "recover_unknown"
            if snapshot.get("status") not in TERMINAL_STATUSES:
                snapshot["elapsed"] = max(0, time.time() - snapshot["created_at"])
            snapshot["tail"] = ""
            if self.log_path.exists():
                with self.log_path.open("rb") as handle:
                    handle.seek(max(0, self.log_path.stat().st_size - TAIL_BYTES))
                    snapshot["tail"] = handle.read(TAIL_BYTES).decode("utf-8", errors="replace")
        except (OSError, ValueError, KeyError) as exc:
            snapshot = dict(self._initial, status="recover_unknown", error="job evidence unavailable: %s" % exc)
        snapshot["worker_log_path"] = str(self.worker_log_path)
        snapshot["cancel_requested"] = self._cancel_requested or snapshot.get("cancel_requested", False)
        return snapshot

    def cancel(self):
        snapshot = self.poll()
        if snapshot["status"] not in TERMINAL_STATUSES:
            self.request_cancel(self.metadata_path)
            self._cancel_requested = True
        return self.poll()

    def wait(self, poll_interval=0.1, on_progress: Callable | None = None,
             should_stop: Callable | None = None):
        interval = min(_duration(poll_interval, "poll_interval"), 0.1)
        try:
            while True:
                snapshot = self.poll()
                if on_progress is not None:
                    on_progress(snapshot)
                if snapshot["status"] in TERMINAL_STATUSES or snapshot["status"] == "recover_unknown":
                    if self._worker is not None and (snapshot["status"] in TERMINAL_STATUSES or
                                                     snapshot.get("recorded_status") in TERMINAL_STATUSES):
                        # The worker has persisted final state and is exiting;
                        # reap it without concealing an anomalous stuck worker.
                        try:
                            self._worker.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            snapshot["error"] = "final command evidence exists but supervisor shutdown is pending"
                            snapshot["status"] = "recover_unknown"
                    return snapshot
                if should_stop is not None and should_stop():
                    self.cancel()
                time.sleep(interval)
        except BaseException:
            self.cancel()
            raise


def _worker_main(launch_path):
    global _WINDOWS_JOB
    path = Path(launch_path)
    # Exclusive claim is retained even after exit: invoking a worker twice must
    # never replay its command. The held native lock separately proves liveness.
    fd = os.open(path.parent / "runner.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    with os.fdopen(fd, "r+b") as runner:
        _runner_lock(runner, True)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        path.unlink()  # Do not retain arguments or environment after delivery.
        if os.name == "nt":
            _WINDOWS_JOB = _WindowsContainment()
        job = _OwnedJob.start(**payload)
        cancellation = job.directory / "cancel.request"
        try:
            job.wait(should_stop=cancellation.exists)
        except BaseException:
            job.cancel()
            job.wait()
            raise


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is launched through ManagedJob.start().")
    _worker_main(sys.argv[2])
