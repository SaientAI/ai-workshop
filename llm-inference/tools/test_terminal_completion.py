#!/usr/bin/env python3
"""Offline regressions against the actual embedded desktop terminal controller."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src-tauri/src/pty.rs").read_text().split(
    'const SAIENT_CLI_PY: &str = r####"', 1)[1].split('"####;', 1)[0]


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="saient-controller-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {
            "SAIENT_WORKSPACE": str(self.root),
            "SAIENT_RUNTIME_DIR": str(ROOT / "src-tauri/resources/saient"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cli = {"__name__": "terminal_under_test"}
        exec(compile(SOURCE, "embedded_saient_cli.py", "exec"), self.cli)

    def test_build_request_cannot_finish_without_files(self):
        self.assertTrue(self.cli["completion_error"]("Build me a working calculator", []))

    def test_polite_action_is_not_mistaken_for_question(self):
        self.assertTrue(self.cli["is_action_request"]("Can you build a calculator?"))
        self.assertTrue(self.cli["is_action_request"]("Please edit index.html to fix the button"))
        self.assertFalse(self.cli["is_action_request"]("How do I build a calculator?"))
        self.assertFalse(self.cli["is_action_request"]("write a haiku about rust"))

    def test_coding_intentions_are_actions_but_creative_text_is_not(self):
        for prompt in ("Write me a program that adds two numbers",
                       "I want you to build an app", "Please make a spreadsheet"):
            with self.subTest(prompt=prompt):
                self.assertTrue(self.cli["is_action_request"](prompt))
        for prompt in ("Create a poem about spring", "Please make a joke about rust"):
            with self.subTest(prompt=prompt):
                self.assertFalse(self.cli["is_action_request"](prompt))

    def test_ordinary_chat_does_not_access_undefined_created_names(self):
        self.assertIsNone(self.cli["direct_session_reply"]("hello", []))
        self.assertIsNone(self.cli["direct_session_reply"]("Can you explain recursion?", []))

    def test_unrelated_existing_read_and_true_do_not_complete_a_build(self):
        path = self.root / "README.md"
        path.write_text("unrelated existing documentation")
        journal = [
            {"proposed": "read", "executed": True, "success": True,
             "verified": True, "path": "README.md", "result": path.read_text()},
            {"proposed": "bash", "executed": True, "success": True,
             "verified": True, "command": "true", "result": "[exit 0]"},
        ]
        self.assertFalse((self.root / "index.html").exists())
        self.assertTrue(self.cli["completion_error"](
            "Build a website in index.html and test it", journal))
        self.assertEqual(self.cli["terminal_report_evidence"](
            "Build a website in index.html and test it", journal).status, "incomplete")

    def test_all_explicit_deliverable_filenames_must_exist(self):
        path = self.root / "index.html"
        path.write_text("<!doctype html>")
        journal = [{"proposed": "write", "executed": True, "success": True,
                    "verified": True, "path": "index.html", "result": "written"}]
        self.assertTrue(self.cli["completion_error"](
            "Build a website with index.html and README.md", journal))

    def test_echoing_requested_folder_name_does_not_verify_that_location(self):
        path = self.root / "other/index.html"
        path.parent.mkdir()
        path.write_text("<!doctype html>")
        journal = [
            {"proposed": "write", "executed": True, "success": True,
             "verified": True, "path": "other/index.html", "result": "written"},
            {"proposed": "bash", "executed": True, "success": True,
             "verified": True, "command": "echo 'wanted'", "result": "wanted\n[exit 0]"},
        ]
        self.assertFalse((self.root / "wanted").exists())
        self.assertTrue(self.cli["completion_error"](
            "Build an app in a folder named wanted", journal))

    def test_unexecuted_mkdir_proposal_is_not_folder_evidence(self):
        journal = [{"proposed": "bash", "executed": False, "success": True,
                    "verified": True, "command": "mkdir wanted",
                    "result": "original bash not executed; conscience selected analyze"}]
        self.assertFalse(self.cli["journal_used_requested_name"](journal, "wanted"))

    def test_desktop_and_called_folder_are_preserved(self):
        user = "Build me a calculator on my Desktop in a new folder called saient-calculator-check-20260907. Use a single index.html."
        self.assertTrue(self.cli["requested_desktop"](user))
        self.assertEqual(self.cli["requested_folder_name"](user), "saient-calculator-check-20260907")

    def test_command_text_does_not_prove_files_exist(self):
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": "echo 'mkdir imaginary'", "result": "[exit 0]"}]
        self.assertEqual(self.cli["created_paths"](journal), [])

    def test_removed_file_cannot_be_reported_as_verified(self):
        path = self.root / "index.html"
        path.write_text("hello")
        journal = [{"proposed": "write", "executed": True, "success": True,
                    "verified": True, "path": "index.html", "result": "wrote 5 bytes"}]
        path.unlink()
        self.assertEqual(self.cli["created_paths"](journal), [])
        self.assertTrue(self.cli["completion_error"]("Build a calculator in index.html", journal))

    def test_exit_zero_alone_does_not_satisfy_build(self):
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": "true", "result": "[exit 0]"}]
        self.assertTrue(self.cli["completion_error"]("Build a calculator", journal))

    def test_shell_file_creation_is_not_its_own_verification_command(self):
        command = ("Set-Content -LiteralPath index.html -Value calculator" if os.name == "nt"
                   else "printf calculator > index.html")
        tool = {"name": "bash", "command": command}
        executor = self.cli["TerminalToolExecutor"](tool, True)
        result = executor.execute({"type": "bash"}, {})
        self.assertTrue(result.verified)
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": tool["command"],
                    "artifacts": executor.artifacts, "result": executor.result}]
        error = self.cli["completion_error"]("Build a calculator in index.html and test it", journal)
        self.assertTrue(error)
        self.assertIn("verification command", error)

    def test_assertions_are_verification_requirements_not_optional_prose(self):
        (self.root / "index.html").write_text("calculator")
        journal = [{"proposed": "write", "executed": True, "success": True,
                    "verified": True, "path": "index.html", "result": "wrote 10 bytes"}]
        self.assertTrue(self.cli["completion_error"](
            "Fix index.html. Then run Node assertions against its JavaScript.", journal))

    def test_unrelated_commands_cannot_verify_created_file(self):
        (self.root / "index.html").write_text("calculator")
        write = {"proposed": "write", "executed": True, "success": True,
                 "verified": True, "path": "index.html", "result": "written"}
        for command in ("true", "cat index.html", "echo 'assert index.html'",
                        "python -c 'assert open(\"other.html\").read()'"):
            with self.subTest(command=command):
                journal = [write, {"proposed": "bash", "executed": True,
                                  "success": True, "verified": True,
                                  "command": command, "result": "[exit 0]"}]
                self.assertTrue(self.cli["completion_error"](
                    "Build a calculator in index.html and test its arithmetic", journal))

    def test_assertion_about_created_file_satisfies_mechanical_check_gate(self):
        (self.root / "index.html").write_text("calculator")
        journal = [
            {"proposed": "write", "executed": True, "success": True,
             "verified": True, "path": "index.html", "result": "written"},
            {"proposed": "bash", "executed": True, "success": True,
             "verified": True, "command": "python -c 'assert open(\"index.html\").read()'",
             "result": "[exit 0]"},
        ]
        self.assertIsNone(self.cli["completion_error"](
            "Build index.html and check it", journal))

    def test_test_runner_cache_is_not_a_new_deliverable_mutation(self):
        (self.root / "index.html").write_text("calculator")
        tool = {"name": "bash", "command":
                "python -c 'from pathlib import Path; "
                "Path(\".pytest_cache\").mkdir(); "
                "Path(\".pytest_cache/nodeids\").write_text(\"[]\"); "
                "assert Path(\"index.html\").read_text() == \"calculator\"'"}
        executor = self.cli["TerminalToolExecutor"](tool, True)
        result = executor.execute({"type": "bash"}, {})
        self.assertTrue(result.verified)
        journal = [
            {"proposed": "write", "executed": True, "success": True,
             "verified": True, "path": "index.html", "result": "written"},
            {"proposed": "bash", "executed": True, "success": True,
             "verified": True, "command": tool["command"],
             "artifacts": executor.artifacts, "result": executor.result},
        ]
        self.assertIsNone(self.cli["completion_error"](
            "Build index.html and test it", journal))

    def test_testing_an_existing_file_does_not_require_modifying_it(self):
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": "node tests.js index.html",
                    "artifacts": {}, "result": "6 assertions passed\n[exit 0]"}]
        self.assertIsNone(self.cli["completion_error"]("Run tests on index.html", journal))

    def test_bash_python_uses_running_interpreter_when_alias_is_missing(self):
        original_which = self.cli["shutil"].which
        with patch.object(self.cli["shutil"], "which", side_effect=lambda name:
                          None if name == "python" else original_which(name)):
            _, result, ok, _ = self.cli["run_tool"](
                {"name": "bash", "command": "python -c 'import sys; print(sys.executable)'"}, True)
        self.assertTrue(ok, result)
        self.assertIn(self.cli["sys"].executable, result)

    def test_bash_keeps_existing_python_on_path(self):
        binary = self.root / "bin" / ("python.cmd" if os.name == "nt" else "python")
        binary.parent.mkdir()
        binary.write_text("@echo custom-python\n" if os.name == "nt" else "#!/bin/sh\nprintf custom-python")
        binary.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}):
            _, result, ok, _ = self.cli["run_tool"]({"name": "bash", "command": "python"}, True)
        self.assertTrue(ok, result)
        self.assertIn("custom-python", result)

    def test_user_turn_only_requires_existing_binding(self):
        class Binding:
            def require_binding(inner, *args):
                return {"binding_status": "bound"}, None
            def ensure_binding(inner, *args):
                self.fail("a user task must not initiate profiling")
        self.cli.update(SAIENT_BINDING=Binding(), BINDING_ERROR=None, BINDING_MANIFEST=None)
        self.assertTrue(self.cli["ensure_formal_binding"](12345))

    def loop_harness(self, prompts, **overrides):
        """Exercise main's real loop with canned host/bridge boundaries only."""
        prompt_iter = iter(prompts)
        self.cli.update(
            header=lambda *args: None,
            find_server=lambda: (12345, "fixture-model"),
            read_user_prompt=lambda: next(prompt_iter),
            ensure_formal_binding=lambda port: True,
            BINDING_MANIFEST=None,
            remember_turn=lambda *args: None,
            express_final=lambda *args: SimpleNamespace(text="fixture report"),
        )
        self.cli.update(overrides)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.cli["main"]()
        return stdout.getvalue()

    def loop_tool_result(self, tool, tick, *, decision="allow"):
        redirected = decision in ("clarify", "veto")
        selected = "analyze" if decision == "clarify" else "stabilize" if redirected else tool["name"]
        detail = {
            "tool_result": "fixture observation",
            "requested_tool_executed": not redirected,
            "implemented_redirect": "read-only listing" if decision == "clarify" else "",
        }
        reply = SimpleNamespace(
            tick=tick, action=selected, conscience=decision, redirected=redirected,
            success=True, verified=decision != "veto", detail=detail,
        )
        executor = SimpleNamespace(
            result=detail["tool_result"], requested_executed=not redirected,
            shown=False, label=tool["name"], artifacts={},
        )
        return reply, executor

    def test_interrupting_final_expression_returns_to_prompt(self):
        def interrupted_expression(*args):
            raise KeyboardInterrupt()

        try:
            output = self.loop_harness(
                ["hello", "/exit"], stream=lambda *args: "Hello.",
                express_final=interrupted_expression,
            )
        except KeyboardInterrupt:
            self.fail("cancelling expression must not terminate the terminal agent")
        self.assertIn("expression interrupted", output)
        self.assertNotIn("fixture report", output)

    def test_bounded_clarify_retries_same_proposal_before_asking_host_again(self):
        tool = {"name": "write", "path": "index.html", "content": "fixture"}
        host_calls, executions = [], []

        def host(port, messages):
            host_calls.append(port)
            return json.dumps(tool) if len(host_calls) == 1 else "Created the requested file."

        def execute(proposal, yolo, user=""):
            executions.append(dict(proposal))
            decision = "clarify" if len(executions) < 3 else "allow_with_uncertainty"
            if len(executions) == 3:
                (self.root / "index.html").write_text("fixture")
            return self.loop_tool_result(proposal, len(executions), decision=decision)

        output = self.loop_harness(
            ["Build a website in index.html", "/exit"], stream=host, run_bound_tool=execute,
        )
        self.assertEqual(executions, [tool, tool, tool])
        self.assertEqual(len(host_calls), 2)
        self.assertNotIn("repeat blocked", output)

    def test_conscience_veto_does_not_enter_clarify_retry(self):
        tool = {"name": "write", "path": "index.html", "content": "fixture"}
        responses = iter([json.dumps(tool), KeyboardInterrupt()])
        executions = []

        def host(port, messages):
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return response

        def execute(proposal, yolo, user=""):
            executions.append(proposal)
            return self.loop_tool_result(proposal, 1, decision="veto")

        self.loop_harness(
            ["Build a website in index.html", "/exit"], stream=host, run_bound_tool=execute,
        )
        self.assertEqual(executions, [tool])
        self.assertFalse((self.root / "index.html").exists())

    def test_continue_retains_original_request_and_prior_verified_journal(self):
        request = "Build a website in index.html and test it"
        messages_seen, executions = [], []
        tools = iter([
            {"name": "write", "path": "index.html", "content": "fixture"},
            {"name": "bash", "command": "fixture-syntax-check index.html"},
        ])

        def host(port, messages):
            messages_seen.append(json.loads(json.dumps(messages)))
            return json.dumps(next(tools))

        def execute(proposal, yolo, user=""):
            executions.append(proposal)
            if proposal["name"] == "write":
                (self.root / "index.html").write_text("fixture")
            return self.loop_tool_result(proposal, len(executions))

        output = self.loop_harness(
            [request, "continue", "/exit"], MAX_STEPS=1,
            stream=host, run_bound_tool=execute,
        )
        self.assertEqual([item["name"] for item in executions], ["write", "bash"])
        self.assertIn("continuing pending request: " + request, output)
        second_prompt = "\n".join(message["content"] for message in messages_seen[1])
        self.assertIn(request, second_prompt)
        self.assertIn("PRIOR VERIFIED TASK JOURNAL", second_prompt)
        self.assertIn('"executed": true', second_prompt)
        self.assertIn('"path": "index.html"', second_prompt)

    def test_server_replacement_requires_binding_before_retrying_inference(self):
        servers = iter([(12345, "bound-model"), (12345, "bound-model"), (23456, "unbound-model")])
        binding_checks, host_calls = [], []

        def binding(port):
            binding_checks.append(port)
            return port == 12345

        def host(port, messages):
            host_calls.append(port)
            raise ConnectionError("fixture server stopped")

        def execute(proposal, yolo, user=""):
            self.fail("unbound replacement must not execute tools")

        output = self.loop_harness(
            ["Build a website in index.html", "/exit"],
            find_server=lambda: next(servers), ensure_formal_binding=binding,
            stream=host, run_bound_tool=execute,
        )
        self.assertEqual(binding_checks, [12345, 23456])
        self.assertEqual(host_calls, [12345])
        self.assertIn("replacement server is not bound", output)


if __name__ == "__main__":
    unittest.main()
