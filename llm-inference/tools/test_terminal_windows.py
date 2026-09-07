#!/usr/bin/env python3
"""Windows branch contracts on every OS, plus native Windows execution checks."""
import base64
import contextlib
import io
import ntpath
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src-tauri/src/pty.rs").read_text(encoding="utf-8").split(
    'const SAIENT_CLI_PY: &str = r####"', 1)[1].split('"####;', 1)[0]


class WindowsTerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="saient-windows-contract-")
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

    def windows(self):
        windows = SimpleNamespace(**vars(os))
        windows.name, windows.path, windows.sep = "nt", SimpleNamespace(**vars(ntpath)), "\\"
        # These are synthetic branch-contract paths, not real Windows files.
        # Never send the UNC fixtures through native filesystem/network lookup.
        windows.path.realpath = ntpath.normpath
        self.cli["os"] = windows
        self.cli["WORKSPACE"] = r"C:\Workspace"
        return windows

    def test_case_insensitive_workspace_paths_are_accepted(self):
        self.windows()
        actual = self.cli["safe_path"](r"c:\workspace\file.txt")
        self.assertEqual(ntpath.normcase(actual), r"c:\workspace\file.txt")

    def test_drive_root_children_are_inside_the_workspace(self):
        self.windows()
        self.cli["WORKSPACE"] = "C:\\"
        self.assertEqual(self.cli["safe_path"]("file.txt"), r"C:\file.txt")

    def test_extended_windows_root_matches_the_same_normal_path(self):
        self.windows()
        self.cli["WORKSPACE"] = r"\\?\C:\Workspace"
        self.assertTrue(self.cli["path_within"](r"C:\Workspace\file.txt", self.cli["WORKSPACE"]))
        self.assertTrue(self.cli["path_within"](r"\\server\share\folder\file.txt", r"\\?\UNC\server\share\folder"))

    def test_synthetic_windows_paths_never_resolve_a_remote_filesystem(self):
        with patch.object(ntpath, "realpath", side_effect=AssertionError("native filesystem lookup")):
            self.windows()
            self.assertTrue(self.cli["path_within"](
                r"\\server\share\folder\file.txt", r"\\?\UNC\server\share\folder"))

    def test_backslash_temp_handle_uses_the_managed_directory(self):
        self.windows()
        self.cli["TEMP_ROOTS"] = [r"C:\Temp\task"]
        self.assertEqual(self.cli["safe_path"](r"@temp\notes.txt"), r"C:\Temp\task\notes.txt")
        self.assertTrue(self.cli["in_temp"](r"c:\temp\TASK\notes.txt"))

    def test_other_drives_siblings_and_traversal_stay_outside(self):
        self.windows()
        for path in (r"D:\file.txt", r"C:\Workspace-other\file.txt", r"..\outside.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.cli["safe_path"](path)

    def test_windows_mkdir_evidence_preserves_backslashes(self):
        windows = self.windows()
        windows.path.exists = lambda path: ntpath.normcase(path) == r"c:\workspace\folder"
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": r"mkdir C:\Workspace\folder"}]
        self.assertEqual(self.cli["created_paths"](journal), [r"C:\Workspace\folder"])

    def test_powershell_new_item_provides_observed_folder_evidence(self):
        windows = self.windows()
        windows.path.exists = lambda path: ntpath.normcase(path) == r"c:\workspace\new folder"
        journal = [{"proposed": "bash", "executed": True, "success": True,
                    "verified": True, "command": "New-Item -ItemType Directory -Path 'C:\\Workspace\\new folder'"}]
        self.assertEqual(self.cli["created_paths"](journal), [r"C:\Workspace\new folder"])

    def test_windows_native_test_executable_is_recognized(self):
        self.windows()
        self.assertTrue(self.cli["verification_command_targets_task"](
            r"C:\Python312\python.exe -m pytest -q", "Build an app and test it", {}))
        self.assertTrue(self.cli["verification_command_targets_task"](
            "npm.cmd test", "Build an app and test it", {}))

    def test_windows_shell_matches_the_prompt_and_preserves_the_script(self):
        self.windows()
        script = "Get-ChildItem -LiteralPath 'C:\\Workspace\\José''s files'"
        with patch.object(self.cli["shutil"], "which", side_effect=lambda name:
                          r"C:\Windows\powershell.exe" if name == "powershell.exe" else None):
            command, kwargs = self.cli["tool_process_command"](script)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(command[0], r"C:\Windows\powershell.exe")
        self.assertIn("-NonInteractive", command)
        self.assertIn("-NoProfile", command)
        decoded = base64.b64decode(command[-1]).decode("utf-16le")
        self.assertIn(script, decoded)
        self.assertIn("$LASTEXITCODE", decoded)
        self.assertNotIn("-ExecutionPolicy", command)

    def test_missing_python_alias_uses_the_running_interpreter_without_requoting(self):
        self.windows()
        source = 'from pathlib import Path; assert Path("index.html").exists()'
        with patch.object(self.cli["shutil"], "which", return_value=None):
            command, kwargs = self.cli["tool_process_command"]("python3 -c '" + source + "'")
        self.assertEqual(command, [sys.executable, "-c", source])
        self.assertFalse(kwargs["shell"])

    def test_existing_python_alias_is_preserved(self):
        self.windows()
        with patch.object(self.cli["shutil"], "which", return_value=r"D:\venv\python.exe"):
            command, _ = self.cli["tool_process_command"]("python -c 'print(123)'")
        self.assertEqual(command[0], r"D:\venv\python.exe")

    def test_windows_cancellation_targets_the_child_tree_by_pid(self):
        self.windows()
        child = SimpleNamespace(pid=43123, poll=lambda: None)
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stderr="")
        with patch.object(self.cli["subprocess"], "run", side_effect=run):
            self.cli["terminate_tool_process"](child)
        args, kwargs = calls[0]
        self.assertEqual(ntpath.basename(args[0]).lower(), "taskkill.exe")
        self.assertEqual(args[1:], ["/PID", "43123", "/T", "/F"])
        self.assertFalse(kwargs.get("shell", False))
        self.assertLessEqual(kwargs["timeout"], 10)

    def test_windows_tree_termination_failure_is_not_reported_as_success(self):
        self.windows()
        child = SimpleNamespace(pid=43123, poll=lambda: None)
        with patch.object(self.cli["subprocess"], "run", return_value=SimpleNamespace(
                returncode=5, stderr="Access is denied")):
            with self.assertRaisesRegex(RuntimeError, "Access is denied"):
                self.cli["terminate_tool_process"](child)

    def test_unicode_file_tools_round_trip_in_utf8(self):
        text = "Saient — café 日本語 ✓\n"
        request = {"name": "write", "path": "café.txt", "content": text}
        _, result, ok, _ = self.cli["run_tool"](request, True)
        self.assertTrue(ok, result)
        self.assertEqual((self.root / "café.txt").read_bytes(), text.encode("utf-8"))
        self.assertTrue(self.cli["verify_tool"](request, result, ok)[0])

    def test_edit_does_not_lossily_rewrite_undecodable_files(self):
        original = b"caf\xe9 and original text\r\n"
        target = self.root / "legacy.txt"
        target.write_bytes(original)
        _, result, ok, _ = self.cli["run_tool"]({"name": "edit", "path": "legacy.txt",
            "old": "original", "new": "changed"}, True)
        self.assertFalse(ok, result)
        self.assertEqual(target.read_bytes(), original)

    def test_host_resolved_desktop_is_used_in_tools_and_completion(self):
        desktop = self.root / "OneDrive" / "Desktop"
        desktop.mkdir(parents=True)
        with patch.dict(os.environ, {"SAIENT_DESKTOP": str(desktop)}):
            self.assertEqual(self.cli["desktop_dir"](), desktop)
            _, facts, ok, _ = self.cli["run_tool"]({"name": "env"}, True)
            self.assertTrue(ok)
            self.assertIn("desktop=" + str(desktop), facts)

    @unittest.skipUnless(os.name == "nt", "native Windows shell required")
    def test_native_powershell_cmdlets_and_unicode(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, result, ok, _ = self.cli["run_tool"]({"name": "bash", "command":
                "Set-Content -LiteralPath 'native.txt' -Value 'café 日本語' -Encoding UTF8; Get-Content -LiteralPath 'native.txt'"}, True)
        self.assertTrue(ok, result)
        self.assertIn("café 日本語", result)

    @unittest.skipUnless(os.name == "nt", "native Windows shell required")
    def test_native_python_assertion_source_and_exit_code_survive(self):
        (self.root / "index.html").write_text("calculator", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            _, result, ok, _ = self.cli["run_tool"]({"name": "bash", "command":
                "python3 -c 'from pathlib import Path; assert Path(\"index.html\").read_text() == \"calculator\"'"}, True)
            self.assertTrue(ok, result)
            _, result, ok, _ = self.cli["run_tool"]({"name": "bash", "command":
                "python3 -c 'raise SystemExit(7)'"}, True)
        self.assertFalse(ok)
        self.assertIn("[exit 7]", result)

    @unittest.skipUnless(os.name == "nt", "native Windows shell required")
    def test_native_shell_propagates_nonzero_and_utf8_output(self):
        # Force the PowerShell path, rather than the single-Python fast path.
        with contextlib.redirect_stdout(io.StringIO()):
            _, result, ok, _ = self.cli["run_tool"]({"name": "bash", "command":
                "Write-Output 'café 日本語'; python3 -c 'raise SystemExit(7)'"}, True)
        self.assertFalse(ok)
        self.assertIn("café 日本語", result)
        self.assertIn("[exit 7]", result)

    @unittest.skipUnless(os.name == "nt", "native Windows process tree required")
    def test_native_watchdog_terminates_the_descendant(self):
        # The timed process starts another Python; both must be terminated.
        child_code = "import time; time.sleep(30)"
        parent_code = (
            "import pathlib,subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,\"-c\",\"" + child_code + "\"]); "
            "pathlib.Path(\"child.pid\").write_text(str(p.pid)); time.sleep(30)")
        self.cli["BASH_TIMEOUT"] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            _, result, ok, _ = self.cli["run_tool"](
                {"name": "bash", "command": "python3 -c '" + parent_code + "'"}, True)
        self.assertFalse(ok)
        self.assertIn("exceeded 1s", result)
        pid = int((self.root / "child.pid").read_text(encoding="utf-8"))
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                code = ctypes.c_ulong()
                self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                self.assertNotEqual(code.value, 259, "descendant remains STILL_ACTIVE")
            finally:
                kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
