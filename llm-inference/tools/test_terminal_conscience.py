#!/usr/bin/env python3
"""Offline conscience-to-terminal checks using real files and the bundled policy.

No model is consulted. SAIENT_TERMINAL_TEST_SOURCE optionally selects an older
generated CLI for a before/after run of the same executor regressions.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = os.environ.get("SAIENT_TERMINAL_TEST_SOURCE")
SOURCE = Path(BASELINE).read_text() if BASELINE else (
    ROOT / "src-tauri/src/pty.rs").read_text().split(
        'const SAIENT_CLI_PY: &str = r####"', 1)[1].split('"####;', 1)[0]


class TerminalConscienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="saient-terminal-conscience-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.env = patch.dict(os.environ, {
            "SAIENT_WORKSPACE": str(self.workspace),
            "SAIENT_RUNTIME_DIR": str(ROOT / "src-tauri/resources/saient"),
            "SAIENT_STATE_DIR": str(self.root / "state"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cli = {"__name__": "terminal_under_test"}
        exec(compile(SOURCE, "embedded_saient_cli.py", "exec"), self.cli)
        self.assertIsNotNone(self.cli["SAIENT_RUNTIME"], self.cli.get("BINDING_ERROR"))
        import main
        self.kernel = main
        self.state = {
            "conscience_layer": "enforce",
            "conscience_profile_path": str(self.root / "profile.json"),
            "conscience_log_dir": str(self.root / "logs"),
            "drives": {"cortisol": 0.3146},
            "history": [],
        }
        self.request = {"name": "write", "path": "index.html", "content": "<h1>calculator</h1>"}

    def arbitrate_and_execute(self, request=None, *, yolo=True):
        request = request or self.request
        action = {"type": request["name"], **{k: v for k, v in request.items() if k != "name"}}
        selected, meta = self.kernel._apply_conscience(
            self.state, {"type": request["name"]}, action)
        executor = self.cli["TerminalToolExecutor"](request, yolo)
        result = executor.execute(selected, self.state)
        return meta, result, executor

    def test_stressed_write_probes_twice_then_runs_through_the_existing_policy(self):
        for attempt in range(3):
            meta, result, executor = self.arbitrate_and_execute()
            if attempt < 2:
                self.assertEqual(meta["decision"], "clarify")
                self.assertEqual(result.action_type, "analyze")
                self.assertTrue(result.success)
                self.assertTrue(result.verified)
                self.assertFalse(executor.requested_executed)
                self.assertFalse((self.workspace / "index.html").exists())
            else:
                self.assertEqual(meta["decision"], "allow_with_uncertainty")
                self.assertFalse(meta["redirected"])
                self.assertTrue(result.success)
                self.assertTrue(result.verified)
                self.assertTrue(executor.requested_executed)
                self.assertEqual((self.workspace / "index.html").read_text(), self.request["content"])

    def test_reported_conscience_cost_is_reconciled_without_lowering_thresholds(self):
        meta, _, _ = self.arbitrate_and_execute()
        evaluation = meta["evaluation"]
        self.assertEqual(evaluation["threshold"], 0.75)
        self.assertEqual(evaluation["cost"], 0.5998)
        self.assertEqual(evaluation["contributions"], {
            "harm": 0.2701, "social": 0.2281, "value": 0.1016, "identity": 0.0,
        })
        self.assertAlmostEqual(sum(evaluation["contributions"].values()), evaluation["cost"])

    def test_failed_command_does_not_poison_a_different_repair_command(self):
        failed = "python -c 'assert True'"
        self.state["history"] = [{"action": {"type": "bash", "command": failed},
                                  "result": {"success": False}}] * 2
        request = {"name": "bash", "command": "python3 -c 'assert True'"}
        action = {"type": "bash", "command": request["command"]}
        _, meta = self.kernel._apply_conscience(self.state, {"type": "bash"}, action)
        self.assertEqual(meta["evaluation"]["proposal"]["aux"]["recent_failures"], 0)
        self.assertNotEqual(meta["decision"], "veto")

    def test_repeating_the_failed_command_keeps_its_failure_penalty(self):
        command = "python -c 'assert True'"
        self.state["history"] = [{"action": {"type": "bash", "command": command},
                                  "result": {"success": False}}] * 2
        _, meta = self.kernel._apply_conscience(
            self.state, {"type": "bash"}, {"type": "bash", "command": command})
        self.assertEqual(meta["evaluation"]["proposal"]["aux"]["recent_failures"], 2)
        self.assertEqual(meta["decision"], "veto")

    def test_abstain_never_executes_a_write_even_in_yolo_mode(self):
        self.state["conscience_estimator_advice"] = {"predicted_harm": 0.99}
        meta, result, executor = self.arbitrate_and_execute()
        self.assertEqual(meta["decision"], "abstain")
        self.assertEqual(result.action_type, "stabilize")
        self.assertFalse(executor.requested_executed)
        self.assertFalse(result.verified)
        self.assertFalse((self.workspace / "index.html").exists())

    def test_veto_never_executes_a_command_even_in_yolo_mode(self):
        self.state["conscience_threshold"] = 0.55
        request = {"name": "bash", "command": "touch veto-must-not-create.txt"}
        meta, result, executor = self.arbitrate_and_execute(request)
        self.assertEqual(meta["decision"], "veto")
        self.assertEqual(result.action_type, "stabilize")
        self.assertFalse(executor.requested_executed)
        self.assertFalse(result.verified)
        self.assertFalse((self.workspace / "veto-must-not-create.txt").exists())

    def test_allowed_with_uncertainty_still_honors_user_denial(self):
        self.arbitrate_and_execute()
        self.arbitrate_and_execute()
        self.cli["read_line"] = lambda prompt: "n"
        meta, result, executor = self.arbitrate_and_execute(yolo=False)
        self.assertEqual(meta["decision"], "allow_with_uncertainty")
        self.assertFalse(result.success)
        self.assertFalse(executor.requested_executed)
        self.assertFalse((self.workspace / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
