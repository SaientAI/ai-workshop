#!/usr/bin/env python3
"""Cancellation regressions for the actual embedded terminal, without a model."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src-tauri/src/pty.rs").read_text().split(
    'const SAIENT_CLI_PY: &str = r####"', 1)[1].split('"####;', 1)[0]


class TerminalInterruptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="saient-interrupt-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(os.environ, {
            "SAIENT_WORKSPACE": str(self.root),
            "SAIENT_RUNTIME_DIR": str(ROOT / "src-tauri/resources/saient"),
            "SAIENT_STATE_DIR": str(self.root / "state"),
        })
        env.start()
        self.addCleanup(env.stop)
        self.cli = {"__name__": "terminal_under_test"}
        exec(compile(SOURCE, "embedded_saient_cli.py", "exec"), self.cli)
        self.assertIsNotNone(self.cli["SAIENT_RUNTIME"], self.cli.get("BINDING_ERROR"))

    @staticmethod
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    def loop(self, prompts, **overrides):
        incoming = iter(prompts)
        self.cli.update(
            header=lambda *args: None,
            find_server=lambda: (12345, "fixture-model"),
            read_user_prompt=lambda: next(incoming),
            ensure_formal_binding=lambda *args: True,
            BINDING_MANIFEST=None,
            remember_turn=lambda *args: None,
            express_final=lambda *args: SimpleNamespace(text="fixture report"),
        )
        self.cli.update(overrides)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                self.cli["main"]()
        except KeyboardInterrupt:
            self.fail("Ctrl-C escaped the terminal instead of returning to its prompt")
        return output.getvalue()

    def test_interrupting_a_direct_history_reply_returns_to_prompt(self):
        output = self.loop(
            ["What did I ask previously?", "/exit"],
            direct_session_reply=lambda *args: "prior task",
            express_final=self.interrupted,
        )
        self.assertIn("interrupted", output)
        self.assertNotIn("fixture report", output)

    def test_interrupting_the_replacement_server_stream_returns_to_prompt(self):
        calls = []

        def host(*args):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("old server disconnected")
            raise KeyboardInterrupt()

        output = self.loop(["Build index.html", "/exit"], stream=host)
        self.assertEqual(len(calls), 2)
        self.assertIn("interrupted", output)
        self.assertIn("Pending task retained", output)

    def test_interrupting_a_tool_bridge_returns_to_prompt_and_retains_the_task(self):
        output = self.loop(
            ["Build index.html", "/exit"],
            stream=lambda *args: json.dumps({"name": "write", "path": "index.html", "content": "fixture"}),
            run_bound_tool=self.interrupted,
        )
        self.assertIn("interrupted", output)
        self.assertIn("Pending task retained", output)
        self.assertNotIn("fixture report", output)

    def test_interrupted_executor_result_stops_the_loop_before_another_host_call(self):
        calls = []
        tool = {"name": "write", "path": "index.html", "content": "fixture"}

        def host(*args):
            calls.append(1)
            self.assertEqual(len(calls), 1, "Ctrl-C must stop this turn")
            return json.dumps(tool)

        def execute(*args):
            detail = {"tool_result": "interrupted by user; outcome unverified",
                      "requested_tool_executed": True, "interrupted": True}
            reply = SimpleNamespace(
                tick=1, action="write", conscience="allow", redirected=False,
                success=False, verified=False, detail=detail)
            executor = SimpleNamespace(
                result=detail["tool_result"], requested_executed=True,
                shown=False, label="write", artifacts={})
            return reply, executor

        output = self.loop(["Build index.html", "/exit"], stream=host, run_bound_tool=execute)
        self.assertEqual(len(calls), 1)
        self.assertIn("Pending task retained", output)

    def test_interrupting_confirmation_never_starts_the_write(self):
        self.cli["read_line"] = self.interrupted
        executor = self.cli["TerminalToolExecutor"](
            {"name": "write", "path": "index.html", "content": "fixture"}, False)
        result = executor.execute({"type": "write"}, {})
        self.assertFalse(executor.requested_executed)
        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertTrue(result.detail["interrupted"])
        self.assertFalse((self.root / "index.html").exists())

    @unittest.skipIf(os.name == "nt", "POSIX process-group signal test")
    def test_shell_interrupt_kills_child_promptly_and_preserves_unverified_partial_work(self):
        request = {"name": "bash", "command": "touch partial.txt; sleep 10"}
        executor = self.cli["TerminalToolExecutor"](request, True)
        self.cli["BASH_TIMEOUT"] = 2
        proc_module = self.cli["subprocess"]
        spawn = proc_module.Popen
        children = []
        timers = []

        def start(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            timer = threading.Timer(0.15, lambda: os.kill(os.getpid(), signal.SIGINT))
            timers.append(timer)
            timer.start()
            return child

        started = time.monotonic()
        try:
            with patch.object(proc_module, "Popen", side_effect=start), contextlib.redirect_stdout(io.StringIO()):
                try:
                    result = executor.execute({"type": "bash"}, {})
                except KeyboardInterrupt:
                    self.fail("the interrupted action needs a recorded failure outcome")
        finally:
            for timer in timers:
                timer.cancel()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        self.assertLess(time.monotonic() - started, 1.5, "Ctrl-C waited for the watchdog")
        self.assertTrue((self.root / "partial.txt").exists())
        self.assertTrue(executor.requested_executed)
        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertTrue(result.detail["interrupted"])
        self.assertEqual(executor.artifacts, {})
        self.assertTrue(all(child.poll() is not None for child in children))


if __name__ == "__main__":
    unittest.main()
