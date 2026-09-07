"""Opt-in durable project controller for the existing, formally bound terminal.

No model/provider SDK, background spending, auto-start on boot, or new authority.
The model proposes actions; existing Saient conscience/execution verifies them.
Project completion requires a user-supplied acceptance command, not model prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import socket
import stat
import tempfile
import threading
import time

from project_store import ProjectStore, ProjectStoreError, BudgetExhausted
from project_jobs import ManagedJob


HELP = """Long Project mode (local model; time/step budgets, not a token billing meter)
  /project start --hours 24 --steps 10000 --tool-timeout 3600 --allow-writes --verify 'npm test' -- build the project
  /project start {"goal":"Build the project","hours":24,"allow_write":true,"verify_command":"npm test"}
  /project status
  /project pause             (also Ctrl-C; unknown tool outcomes require review)
  /project resume            (same saved goal, permissions and remaining budget)
  /project resume --acknowledge-unknown
  /project jobs              (inspect retained command status/log paths)
  /project cancel-jobs       (request cancellation; never signal saved PIDs)
  /project acknowledge       (reviewed unknown outcome; does not resume)
  /project extend --hours 24 --steps 10000  (explicit extra allowance; no auto-run)
  /project stop              (stops this project; does not mark it finished)
  /project archive           (preserve a settled run before starting another)
Optional --allow-shell permits unattended host shell commands. This is NOT an OS
sandbox: shell commands run with your user account's access. File tools remain
workspace-bound. No command authorizes unrelated deletion, spending or publishing.
Without --verify, finishing pauses for review instead of claiming completion.
Keep the desktop/model running. After restart use /project resume explicitly.
"""

PROJECT_INSTRUCTIONS = """LONG PROJECT CONTROLLER:
Work only toward the saved user goal within its permissions. Calls automatically
continue: do not ask the user to type continue. The controller supplies a bounded
recent evidence journal; it is not the entire project history. Inspect real files
when older detail is needed. Do not recreate a file just because its history is
absent. Return ONE ordinary fenced JSON tool proposal at a time.
When ready for acceptance testing, return {"name":"project_finish"}. Only the
controller can record COMPLETED after the configured acceptance command passes.
If genuinely blocked, return {"name":"project_blocked","reason":"specific missing requirement"}.
You cannot change permissions, budgets, control commands or acceptance criteria.
Unattended shell permission does not authorize unrelated deletes, purchases,
publication, system changes or edits outside the project. Do not modify the
acceptance test merely to manufacture a pass. Runtime metadata is not a project
file and must not be edited. Read-only inspection is not proof that work is done.
"""


class CommandError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CommandError(message)


def parse_command(text):
    """JSON form avoids shell-quoting ambiguity, including native Windows paths."""
    parts = text.strip().split(None, 2)
    if not parts or parts[0] != "/project":
        raise CommandError("expected /project")
    operation = parts[1] if len(parts) > 1 else "help"
    rest = parts[2] if len(parts) > 2 else ""
    if operation == "start":
        if rest.lstrip().startswith("{"):
            values = json.loads(rest)
            if not isinstance(values, dict):
                raise CommandError("project options must be a JSON object")
            allowed = {"goal", "hours", "steps", "tool_timeout", "allow_write", "allow_shell", "verify_command", "max_no_progress"}
            if set(values) - allowed:
                raise CommandError("unknown project option")
        else:
            parser = Parser(add_help=False, allow_abbrev=False)
            parser.add_argument("--hours", type=float, default=24)
            parser.add_argument("--steps", type=int, default=10000)
            parser.add_argument("--tool-timeout", type=float, default=600)
            parser.add_argument("--allow-writes", action="store_true", dest="allow_write")
            parser.add_argument("--allow-shell", action="store_true")
            parser.add_argument("--verify", default="", dest="verify_command")
            parser.add_argument("--max-no-progress", type=int, default=8)
            parser.add_argument("goal", nargs="+")
            values = vars(parser.parse_args(shlex.split(rest)))
            values["goal"] = " ".join(values["goal"])
        goal = values.get("goal", "")
        hours = values.get("hours", 24)
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 16000:
            raise CommandError("goal must contain 1–16000 characters")
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
            raise CommandError("hours must be a finite positive number")
        return operation, {"goal": goal.strip(), "policy": {
            "max_seconds": hours * 3600,
            "max_steps": values.get("steps", 10000),
            "tool_timeout": values.get("tool_timeout", 600),
            "allow_write": values.get("allow_write", False),
            "allow_shell": values.get("allow_shell", False),
            "verify_command": values.get("verify_command", ""),
            "max_no_progress": values.get("max_no_progress", 8),
        }}
    if operation == "resume":
        if rest not in ("", "--acknowledge-unknown"):
            raise CommandError("resume accepts only --acknowledge-unknown")
        return operation, {"acknowledge_unknown": bool(rest)}
    if operation == "extend":
        parser = Parser(add_help=False, allow_abbrev=False)
        parser.add_argument("--hours", type=float, default=0)
        parser.add_argument("--steps", type=int, default=0)
        values = parser.parse_args(shlex.split(rest))
        return operation, {"add_seconds": values.hours * 3600, "add_steps": values.steps}
    if operation not in {"help", "status", "pause", "stop", "archive", "jobs", "cancel-jobs", "acknowledge"} or rest:
        raise CommandError("unknown command; use /project help")
    return operation, {}


def action_rows(events):
    return [event["data"]["outcome"] for event in events
            if event["kind"] == "action_finished" and isinstance(event["data"].get("outcome"), dict)
            and "proposed" in event["data"]["outcome"]]


def project_messages(system, snapshot, events, feedback=""):
    """Bound prompt growth independently of durable on-disk execution history."""
    recent = []
    for row in action_rows(events)[-20:]:
        recent.append({key: str(row.get(key, ""))[-600:] if key in ("path", "command", "result") else row.get(key)
                       for key in ("proposed", "path", "command", "executed", "success", "verified", "result")})
    policy = snapshot["policy"]
    context = {"goal": snapshot["goal"], "workspace": snapshot["workspace"],
               "steps_used": snapshot["steps_used"], "permissions": {
                   "write": policy["allow_write"], "shell": policy["allow_shell"]},
               "acceptance_command": policy["verify_command"], "recent_evidence": recent,
               "controller_feedback": feedback[-2000:],
               "reviewed_unknown_actions": [
                   {"tool": str(event["data"].get("tool", ""))[:1200],
                    "note": "Outcome was unknown and user acknowledged inspection; do not blindly repeat it."}
                   for event in events[-30:] if event["kind"] == "recovery_acknowledged"]}
    return [{"role": "system", "content": system + "\n" + PROJECT_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]


class InferenceControl:
    """Cancels only this HTTP response; does not kill a shared model server."""
    def __init__(self):
        self.cancelled = threading.Event()
        self.response = None

    def attach(self, response):
        self.response = response
        if self.cancelled.is_set():
            self.cancel()

    def cancel(self):
        self.cancelled.set()
        response = self.response
        if response is not None:
            # urllib's buffered reader can hold a read lock; shutting down its
            # own socket first unblocks that read before closing the response.
            stream_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if stream_socket is not None:
                try:
                    stream_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # An already-closed connection is already cancelled.


class ProjectRunner:
    def __init__(self, cli, store):
        self.cli, self.store = cli, store
        self.stop_status = None
        self.deferred_input = []
        self.acceptance_active = False
        self.original_tool = cli["run_tool"]
        self.last_job = None

    def status(self):
        state = self.store.snapshot()
        if state is None:
            print("No saved project in this workspace.")
            return
        usage = self.store.usage()
        print("Project %s · %s · %d steps used · %d steps / %.0fs remaining\n%s\n%s" % (
            state["project_id"], state["status"], state["steps_used"],
            usage["steps_remaining"], usage["seconds_remaining"], state["goal"], state["reason"]))

    def controls(self):
        while True:
            try:
                line = self.cli["INPUT_QUEUE"].get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.stop_status = self.stop_status or "PAUSED"
            elif line.strip() == "/project status":
                self.status()
            elif line.strip() in ("/project stop", "/exit", "/quit", "/q"):
                self.stop_status = "STOPPED"
            elif line.strip() == "/project pause":
                self.stop_status = self.stop_status or "PAUSED"
            elif line.strip():
                # Never reinterpret newly typed instructions as pre-approved
                # unattended actions. Pause and preserve them for the prompt.
                self.deferred_input.append(line)
                self.stop_status = self.stop_status or "PAUSED"
        if self.store.usage()["seconds_remaining"] <= 0:
            self.stop_status = self.stop_status or "BUDGET_EXHAUSTED"
        return self.stop_status is not None

    def inference(self, port, messages):
        self.store.reserve_step()
        result = queue.Queue(maxsize=1)
        control = InferenceControl()
        def request():
            try:
                result.put((True, self.cli["stream"](port, messages, control=control)))
            except BaseException as exc:
                result.put((False, exc))
        worker = threading.Thread(target=request, daemon=True, name="saient-project-inference")
        worker.start()
        try:
            while worker.is_alive():
                if self.controls():
                    raise KeyboardInterrupt()
                worker.join(0.1)
        except KeyboardInterrupt:
            control.cancel()
            # No model result from this request can initiate an action after
            # cancellation, even if connection setup has not yet returned.
            worker.join(0.5)
            raise
        ok, value = result.get()
        if not ok:
            raise value
        if self.controls():
            raise KeyboardInterrupt()
        return value

    def tool(self, request, ignored_yolo):
        policy = self.store.snapshot()["policy"]
        name = request.get("name")
        if name == "tempdir":
            raise CommandError("Long Project mode uses its saved workspace; temporary roots cannot be resumed")
        if name in ("write", "edit") and not policy["allow_write"]:
            raise CommandError("project write permission is off; no file was changed")
        if name != "bash":
            if "path" in request:
                path = Path(self.cli["safe_path"](request["path"]))
                if not self.cli["path_within"](path, self.cli["WORKSPACE"]):
                    raise CommandError("Long Project file tools must stay in the saved workspace, not an old temporary root")
                if path.exists():
                    info = path.stat()
                    if name in ("read", "write", "edit") and not stat.S_ISREG(info.st_mode):
                        raise CommandError("Long Project file tools reject pipes, devices and other non-regular targets")
                    if name in ("read", "edit") and info.st_size > 8 * 1024 * 1024:
                        raise CommandError("File exceeds the 8 MiB synchronous inspection limit; use an explicitly approved bounded command")
                    if name == "ls" and not stat.S_ISDIR(info.st_mode):
                        raise CommandError("ls requires a directory")
            return self.original_tool(request, name in ("write", "edit") and policy["allow_write"])
        if not policy["allow_shell"] and not (self.acceptance_active and request.get("command") == policy["verify_command"]):
            raise CommandError("project shell permission is off; command was not executed")
        if self.controls():
            raise KeyboardInterrupt()
        command, options = self.cli["tool_process_command"](request["command"])
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        timeout = min(policy["tool_timeout"], self.store.usage()["seconds_remaining"])
        if timeout <= 0:
            raise KeyboardInterrupt()
        job_root = self.store.directory / "jobs"
        # Bound retained supervisor output across jobs as well as per command.
        # Do not silently erase old evidence to make room for another command.
        retained = sum(path.stat().st_size for path in job_root.rglob("*") if path.is_file()) if job_root.exists() else 0
        if retained + 9 * 1024 * 1024 > 256 * 1024 * 1024:
            raise CommandError("Retained command evidence reached the 256 MiB workspace allowance; review/archive logs before continuing")
        job = ManagedJob.start(command, cwd=self.cli["WORKSPACE"], env=environment,
                               state_dir=job_root, timeout_seconds=timeout,
                               popen_options=options)
        self.last_job = job
        print("Project command %s · log %s" % (job.poll()["job_id"], job.poll()["log_path"]))
        try:
            result = job.wait(should_stop=self.controls)
        except KeyboardInterrupt:
            self.stop_status = self.stop_status or "PAUSED"
            job.cancel()
            result = job.wait()
        if result["status"] == "recover_unknown" or result.get("cleanup_error"):
            print("Command outcome/cleanup unknown; inspect retained job evidence before recovery.")
            raise self.cli["ToolExecutionInterrupted"](True)
        if self.stop_status or result["status"] == "cancelled":
            raise self.cli["ToolExecutionInterrupted"](True)
        ok = result["status"] == "completed" and result["returncode"] == 0 and not result.get("cleanup_error")
        suffix = "exit 0" if ok else "%s; exit %s; %s" % (result["status"], result["returncode"], result.get("cleanup_error") or result.get("error") or "")
        output = self.cli["trunc"](result["tail"]) + "\n[%s]" % suffix
        print(output)
        return "bash " + request["command"], output, ok, True

    def execute(self, tool):
        action_id = self.store.begin_action(tool)
        try:
            reply, executor = self.cli["run_bound_tool"](tool, False, self.store.snapshot()["goal"])
            row = {"tick": reply.tick, "proposed": str(tool["name"]), "selected": str(reply.action),
                   "conscience": str(reply.conscience), "redirected": bool(reply.redirected),
                   "success": bool(reply.success), "verified": bool(reply.verified),
                   "executed": bool(reply.detail.get("requested_tool_executed", executor.requested_executed)),
                   "result": str(reply.detail.get("tool_result", executor.result)),
                   "path": str(tool.get("path", "")), "command": str(tool.get("command", "")),
                   "source": "project_controller" if self.acceptance_active else "host",
                   "implemented_redirect": str(reply.detail.get("implemented_redirect", "")),
                   "artifacts": executor.artifacts, "interrupted": bool(reply.detail.get("interrupted"))}
            if row["interrupted"] and row["executed"]:
                self.store.recover()
                return None  # Retain unknown in-flight mutation; never replay.
            self.store.finish_action(action_id, row)
            return row
        except CommandError as exc:
            self.store.finish_action(action_id, {"proposed": tool["name"], "executed": False,
                "success": False, "verified": False, "result": str(exc), "path": tool.get("path", ""),
                "command": tool.get("command", "")})
            self.store.transition("BLOCKED", str(exc))
            return None
        except BaseException:
            # If the process failed between execution and persistence, its
            # side effects are unknown regardless of the exception's wording.
            self.store.recover()
            raise

    def finish(self):
        if self.controls():
            raise KeyboardInterrupt()
        state = self.store.snapshot()
        command = state["policy"]["verify_command"]
        if not command:
            self.store.transition("PAUSED", "Review required: no user-supplied acceptance command; completion is not verified")
            return ""
        self.acceptance_active = True
        try:
            row = self.execute({"name": "bash", "command": command})
        finally:
            self.acceptance_active = False
        if row and row["executed"] and row["success"] and row["verified"]:
            journal = action_rows(self.store.recent_events(100))
            missing = self.cli["completion_error"](state["goal"], journal)
            # Verification can read files and take time. A queued stop or an
            # expired deadline wins over its result at the commit boundary.
            if self.controls():
                raise KeyboardInterrupt()
            if missing:
                return "Acceptance command passed but required evidence is missing: " + missing
            self.store.transition("COMPLETED", "Configured acceptance command passed; recorded scope is this check, not proof of every possible behaviour")
            return ""
        if row and not row["executed"]:
            return ("Acceptance command was NOT executed; this is not a failing test or evidence of incorrect files. "
                    "Saient selected %s (conscience=%s). Recorded result: %s. "
                    "Do not invent a test failure. A later project_finish can request the same acceptance check, "
                    "still subject to conscience; never bypass its decision." %
                    (row["selected"], row["conscience"], row["result"][-1200:]))
        return ("Acceptance command did not pass. Recorded result: %s. "
                "Inspect the evidence; do not invent a cause or weaken the acceptance test." %
                (row["result"][-1200:] if row else "No verified command outcome"))

    def run(self):
        feedback = ""
        errors = 0
        self.cli["run_tool"] = self.tool
        print("Long Project active. /project status · /project pause · /project stop; Ctrl-C pauses. No automatic publishing permission.")
        try:
            while self.store.snapshot()["status"] in ("RUNNING", "WAITING"):
                if self.controls():
                    break
                port, model = self.cli["find_server"]()
                if not port:
                    if self.store.snapshot()["status"] != "WAITING":
                        self.store.transition("WAITING", "Local model unavailable; waiting with the original deadline")
                    until = time.monotonic() + 5
                    while time.monotonic() < until and not self.controls():
                        time.sleep(0.1)
                    continue
                if not self.cli["ensure_formal_binding"](port):
                    self.store.transition("BLOCKED", "Model binding unavailable; use /bind or /rebind explicitly")
                    break
                if self.store.snapshot()["status"] == "WAITING":
                    self.store.wake_from_wait()
                snapshot = self.store.snapshot()
                events = self.store.recent_events(100)
                messages = project_messages(self.cli["SYSTEM"].replace("__WS__", self.cli["WORKSPACE"])
                    .replace("__OS__", self.cli["platform"].system())
                    .replace("__SHELL__", "PowerShell" if os.name == "nt" else "Bash/sh")
                    .replace("__SHELL_EXAMPLE__", "")
                    .replace("__HOME__", str(Path.home()))
                    .replace("__DESKTOP__", str(self.cli["desktop_dir"]() or "(none)")), snapshot, events, feedback)
                try:
                    text = self.inference(port, messages)
                    errors = 0
                except (OSError, ValueError) as exc:
                    errors += 1
                    self.store.record_progress(False, "Inference failure: " + str(exc)[:300])
                    if errors >= snapshot["policy"]["max_no_progress"]:
                        self.store.transition("BLOCKED", "Repeated inference failures; model connection needs review")
                        break
                    until = time.monotonic() + min(30, 2 ** errors)
                    while time.monotonic() < until and not self.controls():
                        time.sleep(0.1)
                    continue
                tool = self.cli["extract_tool"](text)
                if tool and tool.get("name") == "project_blocked":
                    self.store.transition("BLOCKED", "Model-reported blocker (unverified): " +
                                          str(tool.get("reason", "No reason supplied"))[:2000])
                    break
                if tool and tool.get("name") == "project_finish":
                    feedback = self.finish()
                    if self.store.snapshot()["status"] != "RUNNING":
                        break
                    self.store.record_progress(False, feedback)
                    continue
                problem = self.cli["tool_argument_error"](tool) if tool else "Prose is not execution or verified completion. Return a tool, project_finish or project_blocked."
                if not problem and len(json.dumps(tool)) > 131072:
                    problem = "Tool proposal exceeds the 128 KiB durable action limit"
                if problem:
                    feedback = problem
                    self.store.record_progress(False, problem)
                    continue
                row = self.execute(tool)
                if row is None:
                    break
                fingerprint = lambda value: hashlib.sha256(json.dumps({k: value.get(k) for k in
                    ("proposed", "path", "command", "result", "success", "verified", "artifacts")}, sort_keys=True).encode()).hexdigest()
                previous = {fingerprint(item) for item in action_rows(events)}
                progress = bool(row["executed"] and row["success"] and row["verified"] and fingerprint(row) not in previous)
                self.store.record_progress(progress, "New verified observation/action" if progress else "Repeated, rejected or unsuccessful action")
                feedback = row["result"][-2000:]
        except BudgetExhausted:
            pass  # Store persists BUDGET_EXHAUSTED before raising.
        except KeyboardInterrupt:
            self.stop_status = self.stop_status or "PAUSED"
        except Exception as exc:
            state = self.store.snapshot()
            if state and state["inflight"]:
                self.store.recover()
            elif state and state["status"] in ("RUNNING", "WAITING"):
                self.store.transition("BLOCKED", "Controller error: %s: %s" % (type(exc).__name__, exc))
            print("Project error: %s: %s" % (type(exc).__name__, exc))
        finally:
            self.cli["run_tool"] = self.original_tool
            state = self.store.snapshot()
            if self.stop_status and state:
                if self.stop_status == "STOPPED" and state["status"] in ("RUNNING", "WAITING", "BLOCKED", "PAUSED"):
                    self.store.transition("STOPPED", "User stopped this project; any unknown tool outcome remains recorded for review")
                elif state["status"] in ("RUNNING", "WAITING"):
                    self.store.transition(self.stop_status, "User paused or project deadline reached")
            for line in self.deferred_input:
                self.cli["INPUT_QUEUE"].put(line)
            self.status()


def handle_command(text, cli):
    try:
        operation, arguments = parse_command(text)
        if operation == "help":
            print(HELP)
            return
        root = os.environ.get("SAIENT_STATE_DIR")
        if not root:
            raise CommandError("SAIENT_STATE_DIR is required; refusing unbound durable state")
        store = ProjectStore(root, cli["WORKSPACE"])
        if operation in ("jobs", "cancel-jobs"):
            for path in sorted((store.directory / "jobs").glob("*/job.json")):
                record = ManagedJob.inspect_record(path)
                if operation == "cancel-jobs" and record["status"] == "recover_unknown":
                    ManagedJob.request_cancel(path)
                    record = ManagedJob.inspect_record(path)
                    record["cancellation_requested_by_user"] = True
                print(json.dumps(record, ensure_ascii=False))
            return
        if operation == "status":
            state = store.read_snapshot()
            print(json.dumps(state, indent=2, ensure_ascii=False) if state else "No saved project in this workspace.")
            return
        with store:
            if operation == "start":
                policy = arguments["policy"]
                print("Workspace: %s\nUnattended writes: %s; host shell: %s\nAcceptance command: %s\nBudget: %s steps / %.1f hours" %
                      (cli["WORKSPACE"], policy["allow_write"], policy["allow_shell"],
                       policy["verify_command"] or "none (finish requires review)", policy["max_steps"], policy["max_seconds"] / 3600))
                if policy["allow_shell"]:
                    print("WARNING: host shell access is not OS-sandboxed and can reach outside this folder.")
                if cli["read_line"]("Start this saved project with these permissions? [y/N] ").strip().lower() not in ("y", "yes"):
                    print("Project not started.")
                    return
                store.create(arguments["goal"], policy)
            elif operation == "resume":
                store.recover()
                ensure_jobs_settled(store, acknowledge=arguments["acknowledge_unknown"])
                store.resume(**arguments)
            elif operation in ("pause", "stop"):
                store.recover()
                store.transition("PAUSED" if operation == "pause" else "STOPPED", "Explicit user command")
                print(operation + " saved; no completion claimed")
                return
            elif operation == "archive":
                ensure_jobs_settled(store)
                print("Project archived to " + str(store.archive()))
                return
            elif operation == "acknowledge":
                store.recover()
                ensure_jobs_settled(store, acknowledge=True)
                store.acknowledge_unknown()
                print("Unknown outcome acknowledged after user review; project has not resumed.")
                return
            elif operation == "extend":
                store.extend_budget(**arguments)
                print("Additional allowance saved; usage is unchanged. Use /project resume to continue.")
                return
            ProjectRunner(cli, store).run()
    except (CommandError, ProjectStoreError, ValueError, OSError) as exc:
        print("Project command rejected: %s" % exc)
    except KeyboardInterrupt:
        print("Project command cancelled; no automatic resume.")


def ensure_jobs_settled(store, acknowledge=False):
    for path in (store.directory / "jobs").glob("*/job.json"):
        record = ManagedJob.inspect_record(path)
        if record["status"] == "recover_unknown":
            review = path.parent / "user-recovery-review.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if record.get("supervisor_active") is not False:
                raise CommandError("A prior command supervisor is still active or its ownership is uncertain. Inspect /project jobs and use /project cancel-jobs before recovery.")
            if review.exists():
                if review.is_symlink():
                    raise CommandError("Unsafe job review record")
                value = json.loads(review.read_text(encoding="utf-8"))
                if value.get("metadata_sha256") == digest and value.get("user_acknowledged") is True:
                    continue
            if not acknowledge:
                raise CommandError("A prior command has an unknown outcome and no active supervisor. Inspect its processes/files, then explicitly use /project acknowledge; no completion is assumed.")
            # Explicit user acknowledgment of inspected uncertainty, not proof
            # that commands succeeded. Preserve the original job evidence.
            value = {"metadata_sha256": digest, "user_acknowledged": True,
                     "at": time.time(), "outcome": "unknown; user reviewed; never automatically replay"}
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                json.dump(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, review)
