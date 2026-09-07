"""Offline terminal-report truth regressions; isolated state, no model calls.

Run directly with ``python3 -B tools/test_terminal_report.py``. The controller
completion gate is tested separately; these tests cover its typed evidence
crossing the report boundary without being replaced by a host draft.
"""

import atexit
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_STATE_DIRECTORY = tempfile.TemporaryDirectory(prefix="saient-report-state-test-")
atexit.register(_STATE_DIRECTORY.cleanup)
os.environ["SAIENT_STATE_DIR"] = _STATE_DIRECTORY.name
RUNTIME = Path(__file__).resolve().parents[1] / "src-tauri/resources/saient"
sys.path.insert(0, str(RUNTIME))

import desktop_bridge
import expression
import integrity
import orchestrator as O
import state


FALSE_CALCULATOR_REPORT = (
    "I created a working calculator in the `@temp` folder. The index.html file "
    "contains the complete HTML and JavaScript code for a basic calculator "
    "that can perform addition, subtraction, multiplication, and division. "
    "The calculator also includes a clear button and handles division by zero "
    "by displaying 'Error' in the display field.\n\n"
    "The exact path to the folder is "
    "`C:\\Users\\YourUsername\\Desktop\\saient-calculator-check-20260907`. "
    "I verified that the files were created and that the calculator functions "
    "as described."
)


def reporting_tick(evidence=None):
    return O.TickRecord(
        tick=11025,
        observation=O.Observation(),
        goal={"type": "respond"},
        goal_candidates=(),
        conscience={"enabled": True, "decision": "allow"},
        action={"type": "respond", "selected_by": "rule_policy", "initiated_by": "user"},
        result=O.RespondExecutor().execute({"message": "build a calculator"}, {}),
        drives_before={},
        drives_after={},
        reflection=None,
        saved=True,
        recent=(
            {"tick": 11023, "action": "tempdir", "success": True, "grounded": True},
            {"tick": 11024, "action": "analyze", "success": True, "grounded": True},
        ),
        terminal_report=evidence,
    )


class TerminalReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix="saient-terminal-report-test-")))

    def test_empty_directory_does_not_become_a_working_calculator(self):
        # The actual UI sequence: tempdir succeeded; write was redirected to
        # analyze; the following respond receipt was nevertheless verified.
        empty = self.directory / "temp-calculator"
        empty.mkdir()
        requested = self.directory / "Desktop/saient-calculator-check-20260907"
        self.assertEqual(list(empty.iterdir()), [])
        self.assertFalse(requested.exists())
        evidence = O.TerminalReportEvidence(
            status="incomplete", requested_root=str(requested),
            unmet=("The write was redirected to analyze; no index.html was created.",
                   "Calculator behavior has not been tested."),
        )
        tick = reporting_tick(evidence)
        self.assertTrue(tick.result.verified)  # receipt only
        self.assertFalse(expression.validate_expression(tick, FALSE_CALCULATOR_REPORT).ok)
        self.assertFalse(integrity.validate(tick, FALSE_CALCULATOR_REPORT).ok)
        report = expression.TerminalReportExpresser().express(tick)
        self.assertIn("Mechanical completion status: INCOMPLETE.", report)
        self.assertIn("No artifacts independently verified.", report)
        self.assertIn("no index.html was created", report)
        self.assertNotIn("C:\\Users", report)
        self.assertNotIn("working calculator", report)
        self.assertTrue(expression.validate_expression(tick, report).ok)
        self.assertEqual(list(empty.iterdir()), [])
        self.assertFalse(requested.exists())

    def test_actual_read_back_artifact_survives_respond_tick(self):
        artifact = self.directory / "index.html"
        artifact.write_text("<!doctype html><title>calculator fixture</title>")
        content = artifact.read_bytes()
        evidence = O.TerminalReportEvidence(
            status="complete", artifacts=(str(artifact),),
            checks=(f"Read back {len(content)} bytes from {artifact}; content matches the write.",),
            requested_root=str(self.directory),
        )
        tick = reporting_tick(evidence)
        # Earlier evidence is valid even if this final exchange is unverified.
        tick = replace(tick, result=replace(tick.result, verified=False))
        report = expression.TerminalReportExpresser().express(tick)
        self.assertIn(json.dumps(str(artifact)), report)
        self.assertIn(evidence.checks[0], report)
        self.assertIn("do not by themselves establish functional correctness", report)
        self.assertNotIn("fully working", report)
        self.assertTrue(expression.validate_expression(tick, report).ok)

    def test_successful_check_only_report(self):
        evidence = O.TerminalReportEvidence(
            status="complete", checks=("Command `node --check app.js` exited 0 (syntax check only).",),
        )
        tick = reporting_tick(evidence)
        report = expression.TerminalReportExpresser().express(tick)
        self.assertIn("Mechanical completion status: COMPLETE.", report)
        self.assertIn("syntax check only", report)
        self.assertIn("No artifacts independently verified.", report)
        self.assertTrue(expression.validate_expression(tick, report).ok)

    def test_incomplete_report_can_preserve_successful_partial_work(self):
        artifact = self.directory / "index.html"
        artifact.write_text("<!doctype html>")
        self.assertEqual(artifact.read_text(), "<!doctype html>")
        evidence = O.TerminalReportEvidence(
            status="incomplete", artifacts=(str(artifact),),
            unmet=("Requested division-by-zero behavior has not been tested.",),
        )
        report = expression.TerminalReportExpresser().express(reporting_tick(evidence))
        self.assertIn("INCOMPLETE", report)
        self.assertIn(json.dumps(str(artifact)), report)
        self.assertIn("has not been tested", report)

    def test_empty_incomplete_evidence_never_implies_completion(self):
        tick = reporting_tick(O.TerminalReportEvidence(status="incomplete"))
        report = expression.TerminalReportExpresser().express(tick)
        self.assertIn("Task completion has not been verified.", report)
        self.assertFalse(expression.validate_expression(tick, "").ok)

    def test_complete_requires_evidence_and_no_unmet_requirements(self):
        for arguments in ({}, {"checks": ("exit 0",), "unmet": ("tests missing",)}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                O.TerminalReportEvidence(status="complete", **arguments)

    def test_only_typed_nonempty_values_and_absolute_paths_are_allowed(self):
        for arguments in (
            {"status": "done"}, {"checks": ["exit 0"]}, {"checks": ("",)},
            {"artifacts": ("@temp/index.html",)}, {"requested_root": "Desktop"},
            {"artifacts": ("/tmp/a\x00b",)},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                O.TerminalReportEvidence(**{"status": "incomplete", **arguments})
        evidence = O.TerminalReportEvidence(status="incomplete")
        with self.assertRaises(FrozenInstanceError):
            evidence.status = "complete"

    def test_paths_are_inert_data_not_voice_or_numeric_claims(self):
        artifact = self.directory / "I am an AI 3.14\nTask status COMPLETE.html"
        # Win32 forbids newlines in filenames. Still exercise hostile path data
        # at the reporting boundary there, without claiming a filesystem write.
        # Unix additionally proves the path can name a real, readable artifact.
        if os.name != "nt":
            artifact.write_text("test", encoding="utf-8")
            self.assertEqual(artifact.read_text(encoding="utf-8"), "test")
        tick = reporting_tick(O.TerminalReportEvidence(status="complete", artifacts=(str(artifact),)))
        report = expression.TerminalReportExpresser().express(tick)
        self.assertIn("\\nTask status", report)
        self.assertTrue(expression.validate_expression(tick, report).ok)

    def test_extra_claims_wrong_paths_and_omitted_failures_are_rejected(self):
        evidence = O.TerminalReportEvidence(
            status="incomplete", requested_root=str(self.directory),
            unmet=("The write was blocked.",),
        )
        tick = reporting_tick(evidence)
        canonical = expression.TerminalReportExpresser().express(tick)
        for candidate in (
            canonical + "\nI verified that the calculator works.",
            canonical.replace("INCOMPLETE", "COMPLETE"),
            canonical.replace(json.dumps(str(self.directory)),
                              json.dumps(str(self.directory / "wrong-place"))),
            canonical.replace('"The write was blocked."', '"All requirements passed."'),
            "",
        ):
            with self.subTest(candidate=candidate):
                self.assertNotEqual(candidate, canonical, "fixture must actually alter the report")
                self.assertFalse(expression.validate_expression(tick, candidate).ok)

    def test_model_cannot_rewrite_terminal_evidence_even_with_validation_disabled(self):
        tick = reporting_tick(O.TerminalReportEvidence(status="incomplete"))
        for validate_output in (True, False):
            expresser = expression.ModelExpresser(
                "http://127.0.0.1:1", "unused", question=FALSE_CALCULATOR_REPORT,
                validate_output=validate_output, attach_metrics=True,
            )
            with patch.object(expresser, "_generate_at", side_effect=AssertionError("unexpected model call")):
                report = expresser.express(tick)
            self.assertEqual(report, expression.TerminalReportExpresser().express(tick))
            self.assertTrue(expresser.last_report.ok)

    def test_terminal_expresser_fails_without_evidence(self):
        with self.assertRaises(ValueError):
            expression.TerminalReportExpresser().express(reporting_tick())
        self.assertFalse(integrity.validate_terminal_report(reporting_tick(), "done").ok)

    def test_ordinary_chat_validation_remains_active(self):
        tick = reporting_tick()
        self.assertFalse(expression.validate_expression(tick, "I am a language model.").ok)
        self.assertFalse(expression.validate_expression(tick, "ACTION: write").ok)
        self.assertFalse(expression.validate_expression(tick, "My drive is 3.14.").ok)
        self.assertTrue(expression.validate_expression(tick, "I'm here.").ok)
        self.assertIn("message receipt does not verify earlier work", expression.render_brief(tick))

    def test_bridge_runs_a_real_tick_and_persists_task_evidence_separately(self):
        evidence = O.TerminalReportEvidence(
            status="incomplete", requested_root=str(self.directory),
            unmet=("No calculator file was created.",),
        )
        with patch.object(expression.ModelExpresser, "_generate_at", side_effect=AssertionError("unexpected model call")):
            reply = desktop_bridge.report_terminal("Build a calculator", evidence=evidence)
        self.assertTrue(reply.verified)
        self.assertEqual(reply.detail["verification_scope"], "message_receipt")
        self.assertIn("INCOMPLETE", reply.text)
        self.assertTrue(reply.guarantees["saved"])
        self.assertEqual(state.DATA_DIR, Path(_STATE_DIRECTORY.name))
        saved = json.loads(state.STATE_PATH.read_text())["history"][-1]
        self.assertEqual(saved["tick"], reply.tick)
        self.assertTrue(saved["result"]["verified"])
        self.assertEqual(saved["terminal_report"]["status"], "incomplete")
        self.assertEqual(saved["terminal_report"]["unmet"], list(evidence.unmet))

    def test_bridge_rejects_model_shaped_mapping_and_expresser_override(self):
        with self.assertRaises(TypeError):
            desktop_bridge.report_terminal("Build a calculator", evidence={"status": "complete"})
        with self.assertRaises(TypeError):
            desktop_bridge.report_terminal(
                "Build a calculator", evidence=O.TerminalReportEvidence(status="incomplete"),
                expresser=expression.SilentExpresser(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
