"""Controller recovery regressions using real state, executor and command jobs.

Only the two explicit result-classification tests stub the job-result boundary;
their mutations and persisted in-flight intentions are real. Other tests execute
and cancel native processes, including deliberately losing an owned supervisor.
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_project_controller as fixtures
from test_project_jobs import executing
from project_controller import handle_command
from project_jobs import ManagedJob
from project_store import ProjectStore


class ProjectRecoveryTests(unittest.TestCase):
    # Reuse the fixture without inheriting/collecting its existing test cases.
    setUp = fixtures.ProjectControllerTests.setUp
    host = fixtures.ProjectControllerTests.host
    run_project = fixtures.ProjectControllerTests.run_project

    def until(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("condition did not become true within %.1f seconds" % timeout)

    def invoke(self, command):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            handle_command(command, self.cli)
        return output.getvalue()

    def snapshot(self):
        return ProjectStore(self.state_dir, self.workspace).read_snapshot()

    def classification(self, status, cleanup_error):
        command = 'python -c "print(\'boundary fixture\')"'
        evidence = {"job_id": "boundary-fixture", "log_path": str(self.root / "fixture.log"),
                    "status": status, "cleanup_error": cleanup_error,
                    "returncode": None, "tail": "partial side effect observed", "error": "lost supervisor"}

        def start(*_args, **_kwargs):
            # The crash/result boundary can follow a real mutation; losing its
            # result cannot turn the persisted intention into a safe retry.
            (self.workspace / "effect").write_text("already executed")
            return SimpleNamespace(poll=lambda: dict(evidence), wait=lambda **_kw: dict(evidence),
                                   cancel=lambda: dict(evidence))

        with patch("project_controller.ManagedJob.start", side_effect=start):
            state, events = self.run_project([{"name": "bash", "command": command}], {"allow_shell": True})
        self.assertEqual(state["status"], "BLOCKED", self.output.getvalue())
        self.assertTrue(state["recovery_required"])
        self.assertIsNotNone(state["inflight"])
        self.assertEqual(state["inflight"]["tool"], {"name": "bash", "command": command})
        self.assertEqual((self.workspace / "effect").read_text(), "already executed")
        self.assertEqual(self.host_calls, 1)
        self.assertFalse(any(event["kind"] == "action_finished" for event in events))

    def test_unknown_job_result_retains_inflight_and_does_not_retry(self):
        self.classification("recover_unknown", None)

    def test_cleanup_error_retains_inflight_even_with_failed_result(self):
        self.classification("failed", "process-tree termination could not be confirmed")

    def interrupted_command(self, control, expected_status):
        command = ('python -c "from pathlib import Path; import time; '
                   'Path(\'started\').write_text(\'yes\'); time.sleep(30); '
                   'Path(\'late\').write_text(\'must not execute\')"')
        observed_start = threading.Event()

        def stop():
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if (self.workspace / "started").exists():
                    observed_start.set()
                    break
                time.sleep(0.02)
            self.cli["INPUT_QUEUE"].put(control)

        stopper = threading.Thread(target=stop)
        stopper.start()
        try:
            state, events = self.run_project([{"name": "bash", "command": command}],
                                             {"allow_shell": True, "tool_timeout": 10})
        finally:
            stopper.join(9)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(observed_start.is_set(), self.output.getvalue())
        self.assertEqual(state["status"], expected_status, self.output.getvalue())
        self.assertTrue(state["recovery_required"])
        self.assertIsNotNone(state["inflight"])
        self.assertFalse((self.workspace / "late").exists())
        self.assertFalse(any(event["kind"] == "action_finished" for event in events))
        store = ProjectStore(self.state_dir, self.workspace)
        records = list((store.directory / "jobs").glob("*/job.json"))
        self.assertEqual(len(records), 1)
        record = ManagedJob.inspect_record(records[0])
        self.assertEqual(record["status"], "cancelled", record)
        self.assertFalse(record["supervisor_active"])

    def test_real_command_stop_persists_stopped_with_unknown_outcome(self):
        self.interrupted_command("/project stop", "STOPPED")

    def test_real_command_pause_persists_blocked_unknown_not_false_success(self):
        self.interrupted_command("/project pause", "BLOCKED")

    def active_unknown(self, recover=True):
        store = ProjectStore(self.state_dir, self.workspace)
        tool = {"name": "bash", "command": "fixture command still running"}
        with store:
            store.create("Review uncertain action", {"allow_shell": True})
            store.reserve_step()
            store.begin_action(tool)
            if recover:
                store.recover()
        job = ManagedJob.start([sys.executable, "-u", "-c", "import time; time.sleep(30)"],
                               cwd=self.workspace, env=os.environ.copy(), state_dir=store.directory / "jobs",
                               timeout_seconds=10)
        self.addCleanup(lambda: (job.cancel(), job.wait()))
        self.until(lambda: job.poll()["status"] == "running")
        return job, store

    def test_active_supervisor_blocks_explicit_acknowledgment(self):
        job, store = self.active_unknown()
        action = self.snapshot()["inflight"]
        output = self.invoke("/project acknowledge")
        self.assertIn("supervisor is still active", output)
        self.assertEqual(self.snapshot()["inflight"], action)
        self.assertTrue(self.snapshot()["recovery_required"])
        self.assertFalse((job.directory / "user-recovery-review.json").exists())
        self.assertEqual(job.poll()["status"], "running")
        self.assertEqual(self.host_calls, 0)

    def test_project_jobs_reads_live_history_without_replaying_or_cancelling(self):
        job, store = self.active_unknown()
        before = self.snapshot()
        output = self.invoke("/project jobs")
        record = json.loads(output.strip())
        self.assertEqual(record["job_id"], job.job_id)
        self.assertEqual(record["status"], "recover_unknown")
        self.assertEqual(record["original_status"], "running")
        self.assertTrue(record["supervisor_active"])
        self.assertEqual(record["log_path"], str(job.log_path))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((job.directory / "cancel.request").exists())

    def test_cancel_jobs_reports_request_and_stops_actual_retained_worker(self):
        job, store = self.active_unknown()
        before = self.snapshot()
        output = self.invoke("/project cancel-jobs")
        report = json.loads(output.strip())
        # The worker's prior snapshot can still say cancel_requested=false;
        # distinguish a new user request from worker-observed acknowledgment.
        self.assertTrue(report["cancellation_requested_by_user"], output)
        self.assertTrue((job.directory / "cancel.request").exists())
        result = job.wait()
        self.assertEqual(result["status"], "cancelled", result)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.host_calls, 0)

    def test_direct_acknowledge_recovers_stale_running_intention(self):
        job, store = self.active_unknown(recover=False)
        self.assertEqual(self.snapshot()["status"], "RUNNING")
        job.cancel()
        self.assertEqual(job.wait()["status"], "cancelled")
        output = self.invoke("/project acknowledge")
        state = self.snapshot()
        self.assertIsNone(state["inflight"], output)
        self.assertFalse(state["recovery_required"])
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(state["last_action"]["outcome"], {"status": "unknown", "user_acknowledged": True})
        self.assertEqual(self.host_calls, 0)

    def test_real_supervisor_loss_requires_review_then_resumes_without_replay(self):
        # The real child ends itself shortly after the forced Linux supervisor
        # loss. Windows containment may stop it immediately. We verify it is no
        # longer executing before supplying the explicit user acknowledgment.
        command = ('python -c "from pathlib import Path; import time,os; '
                   'Path(\'effect-child.pid\').write_text(str(os.getpid())); '
                   'p=Path(\'effect-count\'); p.write_text(p.read_text()+\'x\' if p.exists() else \'x\'); '
                   'time.sleep(1.2)"')
        launched = []
        failures = []
        real_start = ManagedJob.start

        def launch(*args, **kwargs):
            job = real_start(*args, **kwargs)
            launched.append(job)
            return job

        def lose_supervisor():
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if launched and (self.workspace / "effect-count").exists():
                    launched[0]._worker.kill()  # Own live handle; no recovered PID signalling.
                    return
                time.sleep(0.02)
            failures.append("command did not reach its real side effect")

        killer = threading.Thread(target=lose_supervisor)
        killer.start()
        try:
            with patch("project_controller.ManagedJob.start", side_effect=launch):
                state, _ = self.run_project([{"name": "bash", "command": command}], {"allow_shell": True})
        finally:
            killer.join(9)
        self.assertFalse(killer.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(launched), 1)
        job = launched[0]
        job._worker.wait(timeout=5)
        history = ManagedJob.inspect_record(job.metadata_path)
        # The worker can die between Popen and the first PID-bearing metadata
        # replace. Use the child's directly observed fixture PID in that case;
        # pid=null in history must never imply that no side effect happened.
        child_pid = int((self.workspace / "effect-child.pid").read_text())
        self.until(lambda: not executing(child_pid))
        if history["pid"] is not None:
            self.until(lambda: not executing(history["pid"]))
        self.assertEqual(state["status"], "BLOCKED", self.output.getvalue())
        self.assertTrue(state["recovery_required"])
        self.assertIsNotNone(state["inflight"])
        self.assertEqual(history["status"], "recover_unknown")
        self.assertFalse(history["supervisor_active"])
        self.assertEqual((self.workspace / "effect-count").read_text(), "x")

        declined = self.invoke("/project resume")
        self.assertIn("unknown outcome", declined)
        self.assertEqual(self.host_calls, 1)
        before = job.metadata_path.read_bytes()
        acknowledged = self.invoke("/project acknowledge")
        self.assertIn("acknowledged", acknowledged)
        reviewed = self.snapshot()
        self.assertEqual(reviewed["status"], "BLOCKED")
        self.assertIsNone(reviewed["inflight"])
        self.assertFalse(reviewed["recovery_required"])
        review = json.loads((job.directory / "user-recovery-review.json").read_text())
        self.assertEqual(review["metadata_sha256"], hashlib.sha256(before).hexdigest())
        self.assertTrue(review["user_acknowledged"])
        self.assertEqual(job.metadata_path.read_bytes(), before)

        self.host([{"name": "project_blocked", "reason": "Reviewed; no automatic retry"}])
        requests = []
        stream = self.cli["stream"]
        def capture(port, messages, control=None):
            requests.append(messages)
            return stream(port, messages, control=control)
        self.cli["stream"] = capture
        self.invoke("/project resume")
        after = self.snapshot()
        self.assertEqual(after["status"], "BLOCKED")
        self.assertEqual(after["steps_used"], 2)
        self.assertEqual(self.host_calls, 2)
        self.assertEqual((self.workspace / "effect-count").read_text(), "x")
        context = json.loads(requests[0][-1]["content"])
        self.assertEqual(len(context["reviewed_unknown_actions"]), 1)
        self.assertIn("do not blindly repeat", context["reviewed_unknown_actions"][0]["note"])


if __name__ == "__main__":
    unittest.main()
