#!/usr/bin/env python3
"""Real store/file/process integration with deterministic host proposals.

The model/binding boundary is a fixture; the actual embedded terminal executor,
project controller, durable SQLite store and background job worker are exercised.
This does not substitute for a live-model or multi-day target-platform soak.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src-tauri/resources/saient"
sys.path.insert(0, str(RUNTIME))
from project_controller import ProjectRunner, handle_command, parse_command, project_messages
from project_store import ProjectStore

SOURCE = (ROOT / "src-tauri/src/pty.rs").read_text().split('const SAIENT_CLI_PY: &str = r####"', 1)[1].split('"####;', 1)[0]


class ProjectControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saient-long-project-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.state_dir = self.root / "state"
        env = patch.dict(os.environ, {"SAIENT_WORKSPACE": str(self.workspace),
            "SAIENT_RUNTIME_DIR": str(RUNTIME), "SAIENT_STATE_DIR": str(self.state_dir)})
        env.start()
        self.addCleanup(env.stop)
        self.cli = {"__name__": "project_terminal_under_test"}
        exec(compile(SOURCE, "embedded_saient_cli.py", "exec"), self.cli)
        self.cli.update(find_server=lambda: (1, "fixture"), ensure_formal_binding=lambda *_: True,
                        read_line=lambda *_: "y")
        self.host_calls = 0
        self.output = io.StringIO()
        self.original_run_tool = self.cli["run_tool"]

        def bound(tool, yolo, user):
            executor = self.cli["TerminalToolExecutor"](tool, yolo, user)
            result = executor.execute({"type": tool["name"]}, {})
            reply = SimpleNamespace(tick=self.host_calls, action=tool["name"], conscience="fixture_allow",
                redirected=False, success=result.success, verified=result.verified, detail=result.detail)
            return reply, executor
        self.cli["run_bound_tool"] = bound

    def host(self, proposals):
        iterator = iter(proposals)
        def stream(port, messages, control=None):
            self.host_calls += 1
            proposal = next(iterator)
            if callable(proposal):
                proposal = proposal()
            return json.dumps(proposal) if isinstance(proposal, dict) else proposal
        self.cli["stream"] = stream

    def run_project(self, proposals, policy=None, goal="Build notes.txt"):
        self.host(proposals)
        with ProjectStore(self.state_dir, self.workspace) as store:
            store.create(goal, dict({"allow_write": True}, **(policy or {})))
            with contextlib.redirect_stdout(self.output):
                ProjectRunner(self.cli, store).run()
            self.assertIs(self.cli["run_tool"], self.original_run_tool)
            return store.snapshot(), store.recent_events(100)

    @staticmethod
    def write(text="hello"):
        return {"name": "write", "path": "notes.txt", "content": text}

    def test_json_options_preserve_windows_backslashes_and_unicode(self):
        goal = 'Build C:\\Users\\me\\Desktop\\工程 with "quoted" labels'
        op, args = parse_command('/project start ' + json.dumps({"goal": goal, "hours": 72}))
        self.assertEqual(op, "start")
        self.assertEqual(args["goal"], goal)
        self.assertEqual(args["policy"]["max_seconds"], 72 * 3600)
        self.assertFalse(args["policy"]["allow_shell"])

    def test_nonfinite_budget_and_unknown_settings_rejected(self):
        for options in ({"goal": "x", "hours": float("inf")}, {"goal": "x", "yolo": True}, {"goal": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                parse_command('/project start ' + json.dumps(options))

    def test_continues_beyond_old_25_step_limit_and_pauses_without_acceptance(self):
        proposals = [self.write(str(i)) for i in range(30)] + [{"name": "project_finish"}]
        state, events = self.run_project(proposals)
        self.assertEqual(state["steps_used"], 31)
        self.assertEqual(state["status"], "PAUSED")
        self.assertIn("no user-supplied acceptance", state["reason"])
        self.assertEqual((self.workspace / "notes.txt").read_text(), "29")

    def test_user_acceptance_command_runs_without_general_shell_permission(self):
        command = 'python -c "from pathlib import Path; assert Path(\'notes.txt\').read_text() == \'hello\'"'
        state, events = self.run_project([self.write(), {"name": "project_finish"}],
                                       {"verify_command": command})
        self.assertEqual(state["status"], "COMPLETED", self.output.getvalue())
        self.assertEqual(state["steps_used"], 2)
        self.assertFalse(state["policy"]["allow_shell"])
        self.assertTrue(any(e["kind"] == "action_finished" and e["data"]["outcome"].get("command") == command for e in events))

    def test_failed_acceptance_cannot_complete(self):
        state, _ = self.run_project([self.write(), {"name": "project_finish"},
            {"name": "project_blocked", "reason": "Acceptance failed"}],
            {"verify_command": 'python -c "raise SystemExit(7)"'})
        self.assertEqual(state["status"], "BLOCKED")

    def test_denied_shell_never_starts_process(self):
        state, _ = self.run_project([{"name": "bash", "command": 'python -c "open(\'escape\',\'w\').write(\'bad\')"'}])
        self.assertEqual(state["status"], "BLOCKED")
        self.assertFalse((self.workspace / "escape").exists())
        self.assertIsNone(state["inflight"])

    def test_model_call_budget_exhaustion_is_not_completion(self):
        state, _ = self.run_project([self.write(str(i)) for i in range(3)], {"max_steps": 3})
        self.assertEqual(state["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(self.host_calls, 3)

    def test_repeated_identical_observations_eventually_block(self):
        state, _ = self.run_project([{"name": "ls", "path": "."}] * 5, {"max_no_progress": 3})
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(self.host_calls, 4)

    def test_stop_during_inference_discards_later_tool_response(self):
        entered = threading.Event()
        def stream(port, messages, control=None):
            entered.set()
            control.cancelled.wait(2)
            return json.dumps(self.write("must not run"))
        self.cli["stream"] = stream
        def stop():
            entered.wait(2)
            self.cli["INPUT_QUEUE"].put("/project stop")
        stopper = threading.Thread(target=stop)
        stopper.start()
        with ProjectStore(self.state_dir, self.workspace) as store:
            store.create("Build notes.txt", {"allow_write": True})
            with contextlib.redirect_stdout(self.output):
                ProjectRunner(self.cli, store).run()
            self.assertEqual(store.snapshot()["status"], "STOPPED")
        stopper.join(2)
        self.assertFalse((self.workspace / "notes.txt").exists())

    def test_stop_during_command_retains_unknown_outcome_for_review(self):
        command = 'python -c "from pathlib import Path; import time; Path(\'started\').write_text(\'yes\'); time.sleep(30); Path(\'late\').write_text(\'bad\')"'
        def stop():
            deadline = time.monotonic() + 8
            while not (self.workspace / "started").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.cli["INPUT_QUEUE"].put("/project pause")
        stopper = threading.Thread(target=stop)
        stopper.start()
        state, _ = self.run_project([{"name": "bash", "command": command}], {"allow_shell": True})
        stopper.join(9)
        self.assertTrue((self.workspace / "started").exists(), self.output.getvalue())
        self.assertFalse((self.workspace / "late").exists())
        self.assertEqual(state["status"], "BLOCKED")
        self.assertTrue(state["recovery_required"])
        self.assertIsNotNone(state["inflight"])

    def test_resume_keeps_goal_usage_and_does_not_replay_old_write(self):
        self.run_project([self.write(), {"name": "project_finish"}])
        self.host([{"name": "project_blocked", "reason": "Need review"}])
        with contextlib.redirect_stdout(self.output):
            handle_command("/project resume", self.cli)
        state = ProjectStore(self.state_dir, self.workspace).read_snapshot()
        self.assertEqual(state["steps_used"], 3)
        self.assertEqual(state["goal"], "Build notes.txt")
        self.assertEqual((self.workspace / "notes.txt").read_text(), "hello")
        self.assertEqual(state["status"], "BLOCKED")

    def test_context_is_bounded_independently_of_history(self):
        events = [{"kind": "action_finished", "data": {"outcome": {"proposed": "read", "result": "x" * 100000}}}] * 1000
        with ProjectStore(self.state_dir, self.workspace) as store:
            state = store.create("Review project")
            messages = project_messages("system", state, events)
        self.assertLess(len(json.dumps(messages)), 20000)

    def test_start_requires_explicit_user_confirmation(self):
        self.cli["read_line"] = lambda *_: "n"
        with contextlib.redirect_stdout(self.output):
            handle_command('/project start {"goal":"Build notes.txt","allow_write":true}', self.cli)
        self.assertIsNone(ProjectStore(self.state_dir, self.workspace).read_snapshot())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special-file fixture")
    def test_fifo_is_rejected_without_blocking_controller(self):
        os.mkfifo(self.workspace / "pipe")
        state, _ = self.run_project([{"name": "read", "path": "pipe"}])
        self.assertEqual(state["status"], "BLOCKED")
        self.assertIn("non-regular", state["reason"])

    def test_oversized_regular_read_is_rejected(self):
        with (self.workspace / "large.bin").open("wb") as handle:
            handle.truncate(8 * 1024 * 1024 + 1)
        state, _ = self.run_project([{"name": "read", "path": "large.bin"}])
        self.assertEqual(state["status"], "BLOCKED")
        self.assertIn("8 MiB", state["reason"])

    def test_identical_file_content_is_not_new_progress(self):
        state, _ = self.run_project([self.write("unchanged")] * 5, {"max_no_progress": 3})
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(self.host_calls, 4)

    def test_stop_during_final_evidence_check_prevents_completed_state(self):
        real_check = self.cli["completion_error"]
        checked = []

        def stop_after_real_check(goal, journal):
            result = real_check(goal, journal)
            checked.append(result)
            self.cli["INPUT_QUEUE"].put("/project stop")
            return result

        self.cli["completion_error"] = stop_after_real_check
        command = 'python -c "from pathlib import Path; assert Path(\'notes.txt\').read_text() == \'hello\'"'
        state, events = self.run_project([self.write(), {"name": "project_finish"}],
                                        {"verify_command": command})
        self.assertEqual(checked, [None], self.output.getvalue())
        self.assertEqual(state["status"], "STOPPED", self.output.getvalue())
        self.assertEqual(state["steps_used"], 2)
        self.assertIsNone(state["inflight"])
        self.assertFalse(state["recovery_required"])
        self.assertTrue(self.cli["INPUT_QUEUE"].empty())
        self.assertFalse(any(event["kind"] == "transition" and
                             event["data"].get("status") == "COMPLETED" for event in events))

    def test_deadline_expiring_during_final_evidence_check_prevents_completed_state(self):
        real_check = self.cli["completion_error"]
        checked = []
        clock_patch = []

        def expire_after_real_check(goal, journal):
            result = real_check(goal, journal)
            checked.append(result)
            state = ProjectStore(self.state_dir, self.workspace).read_snapshot()
            # The file mutation, command, process shutdown and evidence check
            # above are real. Advance only the clock at this final boundary so
            # CI need not sleep through a long authorized project allowance.
            patched = patch("project_store.time.time", return_value=state["deadline_at"] + 1)
            patched.start()
            clock_patch.append(patched)
            return result

        self.cli["completion_error"] = expire_after_real_check
        command = 'python -c "from pathlib import Path; assert Path(\'notes.txt\').read_text() == \'hello\'"'
        try:
            state, events = self.run_project([self.write(), {"name": "project_finish"}],
                                            {"verify_command": command})
        finally:
            for patched in reversed(clock_patch):
                patched.stop()
        self.assertEqual(checked, [None], self.output.getvalue())
        self.assertEqual(state["status"], "BUDGET_EXHAUSTED", self.output.getvalue())
        self.assertEqual(state["steps_used"], 2)
        self.assertIsNone(state["inflight"])
        self.assertFalse(state["recovery_required"])
        self.assertFalse(any(event["kind"] == "transition" and
                             event["data"].get("status") == "COMPLETED" for event in events))
if __name__ == "__main__":
    unittest.main()
