"""Native process tests; run with python3 -B -m unittest discover -s tools -p test_project_jobs.py."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "src-tauri/resources/saient/project_jobs.py"
SPEC = importlib.util.spec_from_file_location("project_jobs", MODULE)
jobs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jobs)


def executing(pid):
    """A Linux zombie has stopped executing, even before its adopter reaps it."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
                return False
            raise OSError(ctypes.get_last_error(), "OpenProcess failed during process verification")
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    stat = Path("/proc") / str(pid) / "stat"
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class ProjectJobsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="saient-project-jobs-test-")
        self.root = Path(self.temporary.name)
        self.jobs = []

    def tearDown(self):
        for job in self.jobs:
            job.cancel()
            job.wait()
        self.temporary.cleanup()

    def start(self, source, **options):
        config = dict(cwd=self.root, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                      state_dir=self.root / "jobs", timeout_seconds=10)
        config.update(options)
        job = jobs.ManagedJob.start([sys.executable, "-u", "-c", source], **config)
        self.jobs.append(job)
        return job

    def until(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.025)
        self.fail("condition did not become true within %.1fs" % timeout)

    def assert_final(self, job, status, returncode=None):
        result = job.wait()
        self.assertEqual(result["status"], status, result)
        if returncode is not None:
            self.assertEqual(result["returncode"], returncode)
        record = jobs.ManagedJob.inspect_record(job.metadata_path)
        self.assertEqual(record["status"], status, record)
        self.assertIsNone(result["cleanup_error"], result)
        self.assertFalse((job.directory / "launch.json").exists())
        self.assertEqual(job.worker_log_path.read_text(), "")
        return result

    def test_success_unicode_working_directory_and_environment(self):
        env = dict(os.environ, PROJECT_JOB_TEST="héllo ∑", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        job = self.start("import os; print(os.getcwd()); print(os.environ['PROJECT_JOB_TEST'])", env=env)
        result = self.assert_final(job, "completed", 0)
        self.assertTrue(os.path.samefile(result["tail"].splitlines()[0], self.root))
        self.assertIn("héllo ∑", result["tail"])
        self.assertEqual(result["tail"], job.log_path.read_bytes().decode("utf-8"))
        record = job.metadata_path.read_text()
        self.assertNotIn("PROJECT_JOB_TEST", record)
        self.assertNotIn("héllo", record)
        if os.name != "nt":
            self.assertEqual(job.directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(job.metadata_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(job.log_path.stat().st_mode & 0o777, 0o600)

    def test_atomic_metadata_retries_only_transient_windows_sharing_errors(self):
        target = self.root / "metadata.json"
        replace = os.replace
        failure = PermissionError("injected Windows sharing violation")
        failure.winerror = 32
        with patch.object(jobs.os, "replace", side_effect=[failure, None]) as mocked, patch.object(jobs.time, "sleep"):
            jobs._atomic_json(target, {"status": "running"})
            self.assertEqual(mocked.call_count, 2)
            # Complete the mocked successful rename for an actual-byte check.
            replace(*mocked.call_args.args)
        self.assertEqual(json.loads(target.read_text())["status"], "running")

    def test_atomic_metadata_does_not_hide_other_errors_or_retry_forever(self):
        for code, expected_attempts in ((5, 1), (33, 6)):
            failure = PermissionError("injected metadata error")
            failure.winerror = code
            with self.subTest(code=code), patch.object(jobs.os, "replace", side_effect=failure) as mocked, patch.object(jobs.time, "sleep"):
                with self.assertRaises(PermissionError):
                    jobs._atomic_json(self.root / "metadata.json", {})
                self.assertEqual(mocked.call_count, expected_attempts)
            self.assertFalse(list(self.root.glob(".job-*.tmp")))

    def test_failure_exitcode_and_stderr_preserved(self):
        job = self.start("import sys; print('actual failure evidence', file=sys.stderr); sys.exit(17)")
        result = self.assert_final(job, "failed", 17)
        self.assertIn("actual failure evidence", result["tail"])

    def test_launch_failure_is_durable(self):
        job = jobs.ManagedJob.start([str(self.root / "missing-executable")], cwd=self.root,
                                    env=os.environ.copy(), state_dir=self.root / "jobs", timeout_seconds=10)
        self.jobs.append(job)
        result = self.assert_final(job, "failed")
        self.assertIn("launch failed", result["error"])
        self.assertIsNone(result["returncode"])

    def test_silent_wall_timeout_without_caller_polling(self):
        job = self.start("import time; time.sleep(30)", timeout_seconds=0.35)
        time.sleep(0.8)
        result = self.assert_final(job, "timed_out")
        self.assertEqual(result["timeout_reason"], "wall")
        self.assertLess(result["elapsed"], 3)

    def test_idle_timeout_is_optional_and_distinct(self):
        job = self.start("import time; print('start', flush=True); time.sleep(30)",
                         timeout_seconds=10, idle_timeout_seconds=0.3)
        result = self.assert_final(job, "timed_out")
        self.assertEqual(result["timeout_reason"], "idle")
        self.assertIn("start", result["tail"])

    def test_quiet_command_with_idle_timeout_disabled_completes(self):
        job = self.start("import time; time.sleep(0.6); print('quiet build result')", idle_timeout_seconds=None)
        result = self.assert_final(job, "completed", 0)
        self.assertIn("quiet build result", result["tail"])

    def test_streaming_without_newline_resets_idle_deadline(self):
        job = self.start("import os,time\nfor i in range(12):\n os.write(1,b'x'); time.sleep(0.07)",
                         idle_timeout_seconds=0.3)
        self.until(lambda: "x" in job.poll()["tail"])
        self.assertNotIn(job.poll()["status"], jobs.TERMINAL_STATUSES)
        result = self.assert_final(job, "completed", 0)
        self.assertEqual(result["tail"], "x" * 12)

    def test_disk_log_and_memory_tail_bounded_under_large_output(self):
        limit = 32 * 1024
        job = self.start("import os\nfor i in range(512): os.write(1,b'x'*8192)\nos.write(1,b'FINAL-EVIDENCE')",
                         max_log_bytes=limit)
        sizes = []
        result = job.wait(on_progress=lambda snapshot: sizes.append(job.log_path.stat().st_size if job.log_path.exists() else 0))
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(result["log_truncated"])
        self.assertEqual(result["output_bytes"], 512 * 8192 + len(b"FINAL-EVIDENCE"))
        self.assertLessEqual(max(sizes), limit)
        self.assertLessEqual(job.log_path.stat().st_size, limit)
        self.assertLessEqual(len(result["tail"].encode()), jobs.TAIL_BYTES)
        self.assertTrue(result["tail"].endswith("FINAL-EVIDENCE"))

    def test_poll_and_cancel_are_responsive_for_silent_command(self):
        job = self.start("import time; time.sleep(30)")
        self.until(lambda: job.poll()["status"] == "running")
        started = time.monotonic()
        snapshot = job.cancel()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(snapshot["cancel_requested"])
        self.assert_final(job, "cancelled")
        self.assertLess(time.monotonic() - started, 5)

    def test_stop_callback_cancels_silent_command(self):
        job = self.start("import time; time.sleep(30)")
        started = time.monotonic()
        result = job.wait(poll_interval=10, should_stop=lambda: time.monotonic() - started >= 0.3)
        self.assertEqual(result["status"], "cancelled", result)
        self.assertLess(time.monotonic() - started, 5)

    def test_callback_exception_requests_cancel_before_propagating(self):
        job = self.start("import time; time.sleep(30)")
        def interrupted(_snapshot):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            job.wait(on_progress=interrupted)
        self.assert_final(job, "cancelled")

    def test_cancellation_kills_actual_child_and_grandchild(self):
        grandchild = "import os,time,pathlib; pathlib.Path('grandchild.pid').write_text(str(os.getpid())); time.sleep(30)"
        child = ("import os,time,pathlib,subprocess,sys; pathlib.Path('child.pid').write_text(str(os.getpid())); "
                 "subprocess.Popen([sys.executable,'-u','-c',%r]); time.sleep(30)" % grandchild)
        root = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-u','-c',%r]); time.sleep(30)" % child
        job = self.start(root)
        self.until(lambda: (self.root / "grandchild.pid").exists() and bool((self.root / "grandchild.pid").read_text()))
        pids = [int((self.root / name).read_text()) for name in ("child.pid", "grandchild.pid")]
        self.assertTrue(all(executing(pid) for pid in pids))
        job.cancel()
        self.assert_final(job, "cancelled")
        self.until(lambda: not any(executing(pid) for pid in pids))

    @unittest.skipIf(os.name == "nt", "POSIX explicit-shell convention; Windows caller supplies native argv")
    def test_explicit_shell_options_are_preserved(self):
        command = "%s -c %s" % (shlex.quote(sys.executable), shlex.quote("print('explicit shell')"))
        job = jobs.ManagedJob.start(command, cwd=self.root, env=os.environ.copy(), state_dir=self.root / "jobs",
                                    timeout_seconds=10, popen_options={"shell": True, "executable": "/bin/sh", "start_new_session": True})
        self.jobs.append(job)
        self.assertIn("explicit shell", self.assert_final(job, "completed", 0)["tail"])

    def test_no_implicit_shell_and_validation_before_launch(self):
        baseline = list(self.root.iterdir())
        for extra in ({"argv": "echo no"}, {"timeout_seconds": float("inf")},
                      {"timeout_seconds": 0}, {"idle_timeout_seconds": -1},
                      {"max_log_bytes": 1}, {"popen_options": {"stdout": subprocess.DEVNULL}}):
            with self.subTest(extra=extra):
                options = dict(argv=[sys.executable, "-c", "pass"], cwd=self.root, env=os.environ.copy(),
                               state_dir=self.root / "jobs", timeout_seconds=10)
                options.update(extra)
                with self.assertRaises(ValueError):
                    jobs.ManagedJob.start(**options)
        self.assertEqual(list(self.root.iterdir()), baseline)

    def test_days_long_bound_accepted_without_waiting_days(self):
        job = self.start("print('configured for days')", timeout_seconds=7 * 24 * 3600)
        self.assert_final(job, "completed", 0)
        self.assertEqual(json.loads(job.metadata_path.read_text())["timeout_seconds"], 7 * 24 * 3600)

    @unittest.skipUnless(os.name == "nt", "native Windows process flags")
    def test_windows_background_commands_do_not_allocate_console_hosts(self):
        _, _, _, options = jobs._validate_start([sys.executable, "-V"], self.root, 10, None, None, 8192)
        self.assertTrue(options["creationflags"] & subprocess.CREATE_NO_WINDOW)
        # Query the child's real console handle; a flag-only assertion does
        # not establish native behavior or clean supervisor shutdown.
        for _ in range(5):
            job = self.start("import ctypes; k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=ctypes.c_void_p; assert not k.GetConsoleWindow()")
            self.assert_final(job, "completed", 0)

    def test_history_never_assumes_ownership_of_persisted_pid(self):
        job = self.start("import time; time.sleep(30)")
        self.until(lambda: job.poll()["status"] == "running")
        record = jobs.ManagedJob.inspect_record(job.metadata_path)
        self.assertEqual(record["status"], "recover_unknown")
        self.assertEqual(record["original_status"], "running")
        self.assertFalse(record["terminal"])
        self.assertTrue(record["supervisor_active"])
        self.assertTrue(executing(record["pid"]))
        self.assertEqual(job.poll()["status"], "running")
        job.cancel()
        self.assert_final(job, "cancelled")

    def test_retained_job_cancel_uses_request_file_not_recorded_pid(self):
        job = self.start("import time; time.sleep(30)")
        self.until(lambda: job.poll()["status"] == "running")
        record = jobs.ManagedJob.request_cancel(job.metadata_path)
        self.assertTrue(record["cancel_requested"])
        self.assertTrue((job.directory / "cancel.request").is_file())
        self.assert_final(job, "cancelled")
        history = jobs.ManagedJob.inspect_record(job.metadata_path)
        self.assertTrue(history["terminal"])
        self.assertFalse(history["supervisor_active"])

    def test_request_cancel_rejects_mismatched_directory_identity(self):
        job = self.start("print('done')")
        self.assert_final(job, "completed", 0)
        forged = self.root / "job.json"
        forged.write_text(job.metadata_path.read_text())
        with self.assertRaises(ValueError):
            jobs.ManagedJob.request_cancel(forged)
        self.assertFalse((self.root / "cancel.request").exists())

    def test_supervisor_crash_is_unknown_not_ordinary_failure_or_success(self):
        # The fixture child ends itself, because a SIGKILL of the Linux
        # supervisor is explicitly outside its userspace watchdog guarantee.
        job = self.start("import os,time,pathlib; pathlib.Path('crash-child.pid').write_text(str(os.getpid())); time.sleep(1.2)")
        self.until(lambda: (self.root / "crash-child.pid").exists() and bool((self.root / "crash-child.pid").read_text()))
        pid = int((self.root / "crash-child.pid").read_text())
        job._worker.kill()  # Own Popen handle, not a recovered PID.
        job._worker.wait(timeout=5)
        result = job.wait()
        self.assertEqual(result["status"], "recover_unknown", result)
        self.assertIsNone(result["returncode"])
        history = jobs.ManagedJob.inspect_record(job.metadata_path)
        self.assertFalse(history["terminal"])
        self.assertFalse(history["supervisor_active"])
        self.until(lambda: not executing(pid))

    def test_worker_exclusive_claim_prevents_replaying_finished_action(self):
        job = self.start("import pathlib; pathlib.Path('effect-count').write_text('one')")
        self.assert_final(job, "completed", 0)
        duplicate = subprocess.run([sys.executable, "-B", str(MODULE), "--worker", str(job.directory / "launch.json")],
                                   capture_output=True, text=True, timeout=5)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("FileExistsError", duplicate.stderr)
        self.assertEqual((self.root / "effect-count").read_text(), "one")

    def test_cleanup_error_cannot_be_recovered_as_settled_failure(self):
        job = self.start("print('fixture')")
        self.assert_final(job, "completed", 0)
        record = json.loads(job.metadata_path.read_text())
        record.update(status="failed", cleanup_error="fixture: process tree could not be confirmed stopped")
        job.metadata_path.write_text(json.dumps(record))
        history = jobs.ManagedJob.inspect_record(job.metadata_path)
        self.assertFalse(history["terminal"])
        self.assertEqual(history["original_status"], "failed")
        self.assertEqual(history["status"], "recover_unknown")
        self.assertEqual(job.poll()["status"], "recover_unknown")

    def test_exited_root_does_not_leave_redirected_descendant_unbounded(self):
        child = "import os,time,pathlib; pathlib.Path('redirected-child.pid').write_text(str(os.getpid())); time.sleep(30)"
        source = ("import subprocess,sys,time,pathlib; subprocess.Popen([sys.executable,'-u','-c',%r],"
                  "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                  "\nwhile not pathlib.Path('redirected-child.pid').exists(): time.sleep(0.02)" % child)
        job = self.start(source)
        result = job.wait()
        self.assertIn(result["status"], ("failed", "recover_unknown"), result)
        self.assertIn("descendants", result["error"])
        pid = int((self.root / "redirected-child.pid").read_text())
        self.until(lambda: not executing(pid))

    def test_owner_abrupt_exit_does_not_remove_child_deadline(self):
        child = "import os,time,pathlib; pathlib.Path('orphan-child.pid').write_text(str(os.getpid())); time.sleep(30)"
        source = (
            "import importlib.util,os,pathlib,sys\n"
            "spec=importlib.util.spec_from_file_location('project_jobs',%r)\n"
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "job=module.ManagedJob.start([sys.executable,'-u','-c',%r],cwd=%r,env=dict(os.environ),state_dir=%r,timeout_seconds=0.6)\n"
            "pathlib.Path(%r).write_text(str(job.metadata_path))\n"
            "os._exit(0)\n"
        ) % (str(MODULE), child, str(self.root), str(self.root / "jobs"), str(self.root / "record-path"))
        owner = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True, text=True, timeout=10)
        self.assertEqual(owner.returncode, 0, owner.stderr)
        path = Path((self.root / "record-path").read_text())
        self.until(lambda: (self.root / "orphan-child.pid").exists())
        pid = int((self.root / "orphan-child.pid").read_text())
        self.until(lambda: jobs.ManagedJob.inspect_record(path)["status"] == "timed_out", timeout=6)
        self.until(lambda: not executing(pid))
        # Final command metadata can precede the detached worker releasing its
        # lock. Verify shutdown before the fixture removes its evidence tree.
        self.until(lambda: jobs.ManagedJob.inspect_record(path)["supervisor_active"] is False)
        record = jobs.ManagedJob.inspect_record(path)
        self.assertEqual(record["timeout_reason"], "wall")
        self.assertIsNone(record["cleanup_error"])
        self.assertLess(record["elapsed"], 5)
        self.assertFalse((path.parent / "launch.json").exists())
        self.assertEqual((path.parent / "supervisor.log").read_text(), "")


if __name__ == "__main__":
    unittest.main()
