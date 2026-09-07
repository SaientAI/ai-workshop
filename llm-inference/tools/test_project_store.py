#!/usr/bin/env python3
"""Real-file and real-process durable project tests; no model or network needed."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


RUNTIME = Path(__file__).resolve().parents[1] / "src-tauri/resources/saient"
sys.path.insert(0, str(RUNTIME))
from project_store import (BudgetExhausted, CorruptProject, DEFAULT_POLICY,
                           InvalidTransition, MAX_EVENTS, MAX_EVENT_BYTES,
                           ProjectLocked, ProjectStore, ProjectStoreError,
                           _path_identity, _path_within, validate_policy)


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saient-project-store-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "workspace ü"
        self.workspace.mkdir()
        self.state = self.root / "private-state"
        self.env = dict(os.environ, PYTHONPATH=str(RUNTIME), PYTHONUTF8="1",
                        PYTHONDONTWRITEBYTECODE="1")

    def store(self):
        return ProjectStore(self.state, self.workspace)

    def child(self, code, *args):
        return subprocess.run([sys.executable, "-B", "-c", code,
                               str(self.state), str(self.workspace), *map(str, args)],
                              env=self.env, capture_output=True, text=True, timeout=15)

    def test_create_defaults_and_canonical_workspace_identity(self):
        with self.store() as store:
            state = store.create("Build a calculator")
            self.assertEqual(state["policy"], DEFAULT_POLICY)
            self.assertFalse(state["policy"]["allow_shell"])
            self.assertFalse(state["policy"]["allow_write"])
            self.assertEqual(state["status"], "RUNNING")
            self.assertEqual(state["steps_used"], 0)
            self.assertEqual(state["deadline_at"] - state["created_at"], 86400)
            self.assertEqual(len(state["project_id"]), 64)
            self.assertFalse(store.directory.is_relative_to(self.workspace))
            alias = ProjectStore(self.state, self.workspace / ".")
            self.assertEqual(alias.project_id, store.project_id)
            self.assertEqual(store.usage()["steps_remaining"], 10000)
            self.assertEqual(store.recent_events()[0]["kind"], "created")

    def test_policy_allows_days_and_more_than_25_steps(self):
        with self.store() as store:
            store.create("Multi-day work", {"max_steps": 100000, "max_seconds": 7 * 86400,
                                           "tool_timeout": 2 * 86400,
                                           "verify_command": "python -m unittest"})
            for _ in range(40):
                store.reserve_step()
            self.assertEqual(store.load()["steps_used"], 40)
            self.assertGreater(store.usage()["seconds_remaining"], 6 * 86400)

    def test_invalid_policies_fail_closed(self):
        invalid = [None, True, False, -1, 0, float("inf"), float("nan"), "100"]
        for key in ("max_steps", "max_seconds", "max_no_progress", "tool_timeout"):
            for value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_policy({key: value})
        for key in ("allow_shell", "allow_write"):
            for value in (1, 0, "true", None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_policy({key: value})
        for policy in ({"unexpected": True}, {"max_steps": 1.1}, {"max_steps": 10000001},
                       {"max_seconds": 367 * 86400}, {"verify_command": True},
                       {"verify_command": "x" * 8001}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_policy(policy)

    def test_step_reservation_is_persisted_before_execution(self):
        with self.store() as store:
            store.create("Task", {"max_steps": 2})
            store.reserve_step()
        with self.store() as store:
            self.assertEqual(store.load()["steps_used"], 1)
            self.assertEqual(store.recover()["status"], "PAUSED")
            store.resume()
            store.reserve_step()
            with self.assertRaises(BudgetExhausted):
                store.reserve_step()
            self.assertEqual(store.load()["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(store.load()["steps_used"], 2)
        with self.store() as store:
            with self.assertRaises(InvalidTransition):
                store.resume()
            self.assertEqual(store.usage()["steps_remaining"], 0)

    def test_last_reserved_call_can_finish_its_action(self):
        with self.store() as store:
            store.create("Task", {"max_steps": 1})
            store.reserve_step()
            action = store.begin_action({"tool": "read", "path": "a.py"})
            store.finish_action(action, {"tool": "read", "ok": True})
            self.assertEqual(store.transition("COMPLETED", "Acceptance passed")["status"], "COMPLETED")

    def test_wall_deadline_includes_paused_time_and_resume_cannot_reset_it(self):
        with self.store() as store:
            with patch("project_store.time.time", return_value=1000.0):
                initial = store.create("Task", {"max_seconds": 60})
                store.transition("PAUSED", "User pause")
            with patch("project_store.time.time", return_value=1030.0):
                resumed = store.resume()
                self.assertEqual(resumed["deadline_at"], initial["deadline_at"])
                self.assertEqual(store.usage()["seconds_remaining"], 30)
                store.transition("PAUSED")
            with patch("project_store.time.time", return_value=1061.0):
                with self.assertRaises(BudgetExhausted):
                    store.resume()
                self.assertEqual(store.usage()["seconds_remaining"], 0)

    def test_unknown_action_after_real_crash_never_replays(self):
        effects = self.root / "side-effect.txt"
        result = self.child("""
import os, sys
from pathlib import Path
from project_store import ProjectStore
with ProjectStore(sys.argv[1], sys.argv[2]) as store:
    store.create('Write once')
    store.reserve_step()
    store.begin_action({'tool':'write', 'path':sys.argv[3]})
    Path(sys.argv[3]).write_text('executed once', encoding='utf-8')
    os._exit(23)
""", effects)
        self.assertEqual(result.returncode, 23, result.stderr)
        with self.store() as store:
            state = store.recover()
            self.assertEqual(state["status"], "BLOCKED")
            self.assertTrue(state["recovery_required"])
            self.assertIsNotNone(state["inflight"])
            self.assertEqual(state["steps_used"], 1)
            with self.assertRaises(InvalidTransition):
                store.resume()
            with self.assertRaises(InvalidTransition):
                store.finish_action(state["inflight"]["id"], {"ok": True})
            with self.assertRaises(InvalidTransition):
                store.reserve_step()
            state = store.resume(acknowledge_unknown=True)
            self.assertIsNone(state["inflight"])
            self.assertEqual(state["last_action"]["outcome"]["status"], "unknown")
            self.assertEqual(state["steps_used"], 1)
            self.assertEqual(effects.read_text(encoding="utf-8"), "executed once")

    def test_crashed_model_reservation_is_not_refunded(self):
        result = self.child("""
import os,sys
from project_store import ProjectStore
with ProjectStore(sys.argv[1],sys.argv[2]) as store:
    store.create('Task')
    store.reserve_step()
    os._exit(24)
""")
        self.assertEqual(result.returncode, 24, result.stderr)
        with self.store() as store:
            self.assertEqual(store.recover()["status"], "PAUSED")
            self.assertEqual(store.resume()["steps_used"], 1)

    def test_wrong_action_id_and_double_finish_are_rejected(self):
        with self.store() as store:
            store.create("Task")
            with self.assertRaises(InvalidTransition):
                store.begin_action("read")
            store.reserve_step()
            action = store.begin_action("read")
            for operation in (lambda: store.begin_action("write"), store.reserve_step,
                              lambda: store.finish_action("wrong-id", {"ok": True}),
                              lambda: store.transition("COMPLETED")):
                with self.assertRaises(InvalidTransition):
                    operation()
            self.assertEqual(store.load()["inflight"]["id"], action)
            store.finish_action(action, {"ok": True, "output": "é"})
            with self.assertRaises(InvalidTransition):
                store.finish_action(action, {"ok": True})
            self.assertEqual(store.recent_events()[-1]["data"]["outcome"]["output"], "é")

    def test_stopped_unknown_action_is_preserved_and_not_resumable(self):
        with self.store() as store:
            store.create("Task")
            store.reserve_step()
            action = store.begin_action("shell")
            store.transition("STOPPED", "User stop")
        with self.store() as store:
            state = store.recover()
            self.assertEqual(state["status"], "STOPPED")
            self.assertTrue(state["recovery_required"])
            self.assertEqual(state["inflight"]["id"], action)
            with self.assertRaises(InvalidTransition):
                store.resume(acknowledge_unknown=True)

    def test_no_progress_breaker_and_explicit_resume(self):
        with self.store() as store:
            store.create("Task", {"max_no_progress": 2})
            store.reserve_step()
            store.record_progress(False, "Same failed action")
            self.assertEqual(store.record_progress(False)["status"], "BLOCKED")
            with self.assertRaises(InvalidTransition):
                store.reserve_step()
            self.assertEqual(store.resume()["steps_used"], 1)
            self.assertEqual(store.load()["no_progress_count"], 0)
            store.record_progress(False)
            self.assertEqual(store.record_progress(True)["no_progress_count"], 0)

    def test_terminal_and_direct_running_transitions_rejected(self):
        for status in ("COMPLETED", "STOPPED", "BUDGET_EXHAUSTED"):
            workspace = self.root / status
            workspace.mkdir()
            with ProjectStore(self.state, workspace) as store:
                store.create("Task")
                with self.assertRaises(InvalidTransition):
                    store.transition("RUNNING")
                store.transition(status)
                for action in (store.resume, store.reserve_step,
                               lambda: store.transition("PAUSED"), lambda: store.create("Replacement")):
                    with self.subTest(status=status), self.assertRaises(InvalidTransition):
                        action()

    def test_existing_project_never_overwritten_and_permissions_unchanged(self):
        with self.store() as store:
            initial = store.create("Original", {"allow_write": False})
            with self.assertRaises(InvalidTransition):
                store.create("Replacement", {"allow_write": True})
            self.assertEqual(store.load(), initial)
            store.transition("PAUSED")
            self.assertFalse(store.resume()["policy"]["allow_write"])

    def test_state_and_event_commit_is_atomic_on_event_failure(self):
        with self.store() as store:
            before = store.create("Task")
            count = len(store.recent_events())
            # A real SQLite trigger aborts between the metadata and event writes.
            store._db.execute("CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'injected event failure'); END")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected event failure"):
                store.reserve_step()
            self.assertEqual(store.load(), before)
            self.assertEqual(len(store.recent_events()), count)
            store._db.execute("DROP TRIGGER fail_event")
            self.assertEqual(store.reserve_step()["steps_used"], 1)

    def test_context_and_persistent_journal_are_bounded(self):
        with self.store() as store:
            store.create("Task", {"max_steps": MAX_EVENTS + 30})
            # Seed historical journal volume in one real SQLite transaction;
            # the production save below must prune it back to the retention cap.
            with store._transaction():
                store._db.executemany("INSERT INTO events(at,kind,data) VALUES(0,'fixture','{}')",
                                      [()] * (MAX_EVENTS + 10))
            store.reserve_step()
            self.assertEqual(store._db.execute("SELECT COUNT(*) FROM events").fetchone()[0], MAX_EVENTS)
            recent = store.recent_events(5)
            self.assertEqual(len(recent), 5)
            self.assertEqual([row["seq"] for row in recent], sorted(row["seq"] for row in recent))
            for limit in (0, -1, True, 101, 1000000):
                with self.assertRaises(ValueError):
                    store.recent_events(limit)
            action = store.begin_action({"tool": "read", "payload": "é" * 100000})
            store.finish_action(action, {"output": "é" * 1000000})
            self.assertTrue(store.load()["last_action"]["outcome"]["truncated"])
            maximum = store._db.execute("SELECT MAX(length(CAST(data AS BLOB))) FROM events").fetchone()[0]
            self.assertLessEqual(maximum, MAX_EVENT_BYTES)

    def test_real_nonblocking_process_lock_repeated_and_release(self):
        code = """
import sys
from project_store import ProjectStore, ProjectLocked
try:
    with ProjectStore(sys.argv[1],sys.argv[2]):
        print('ACQUIRED')
except ProjectLocked:
    print('LOCKED')
"""
        for iteration in range(8):
            with self.subTest(iteration=iteration):
                with self.store() as store:
                    if iteration == 0:
                        store.create("Task")
                    result = self.child(code)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), "LOCKED")
                result = self.child(code)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "ACQUIRED")

    def test_readonly_status_works_while_another_process_owns_runner(self):
        with self.store() as store:
            state = store.create("Status remains readable")
            store.reserve_step()
            result = self.child("""
import json,sys
from project_store import ProjectStore
print(json.dumps(ProjectStore(sys.argv[1],sys.argv[2]).read_snapshot()))
""")
            self.assertEqual(result.returncode, 0, result.stderr)
            read = json.loads(result.stdout)
            self.assertEqual(read["project_id"], state["project_id"])
            self.assertEqual(read["steps_used"], 1)
            self.assertEqual(read["status"], "RUNNING")

    def test_readonly_absent_status_creates_nothing(self):
        self.assertIsNone(self.store().read_snapshot())
        self.assertFalse(self.state.exists())

    def test_missing_or_corrupt_metadata_fails_closed(self):
        corruptions = ["not json", "{}", "null", "[]"]
        for index, serialized in enumerate(corruptions):
            workspace = self.root / str(index)
            workspace.mkdir()
            target = ProjectStore(self.state, workspace)
            with target as store:
                store.create("Original")
                store._db.execute("UPDATE metadata SET state=?", (serialized,))
            with self.subTest(serialized=serialized), self.assertRaises(CorruptProject):
                with ProjectStore(self.state, workspace):
                    pass
            with self.assertRaises(CorruptProject):
                target.read_snapshot()

    def test_mutated_policy_identity_usage_deadline_fail_closed(self):
        mutators = [lambda value: value.update(schema_version=999),
                    lambda value: value.update(project_id="wrong"),
                    lambda value: value.update(steps_used=-1),
                    lambda value: value.update(steps_used=True),
                    lambda value: value.update(deadline_at=value["deadline_at"] + 100),
                    lambda value: value.update(recovery_required=True),
                    lambda value: value["policy"].pop("allow_write")]
        for index, mutate in enumerate(mutators):
            workspace = self.root / str(index)
            workspace.mkdir()
            with ProjectStore(self.state, workspace) as store:
                state = store.create("Original")
                mutate(state)
                store._db.execute("UPDATE metadata SET state=?", (json.dumps(state),))
                with self.subTest(index=index), self.assertRaises(CorruptProject):
                    store.load()

    def test_unsupported_database_schema_and_missing_state_not_reset(self):
        with self.store() as store:
            store.create("Original")
            store._db.execute("PRAGMA user_version=999")
        with self.assertRaises(CorruptProject):
            with self.store():
                pass
        connection = sqlite3.connect(self.store().db_path)
        with connection:
            connection.execute("PRAGMA user_version=1")
            connection.execute("DELETE FROM metadata")
        connection.close()
        with self.assertRaises(CorruptProject):
            with self.store():
                pass

    def test_state_inside_workspace_rejected(self):
        with self.assertRaises(ValueError):
            ProjectStore(self.workspace / ".saient", self.workspace)
        with self.assertRaises(ValueError):
            ProjectStore(self.workspace, self.workspace)

    def symlink_or_skip(self, target, link, is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=is_directory)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable")
            raise

    def test_state_directory_symlink_escape_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.symlink_or_skip(outside, self.state, True)
        with self.assertRaises(ProjectStoreError):
            with self.store():
                pass
        self.assertEqual(list(outside.iterdir()), [])

    def test_database_symlink_escape_is_rejected_without_modifying_target(self):
        with self.store() as store:
            store.create("Original")
        database = self.store().db_path
        saved = database.with_suffix(".saved")
        database.rename(saved)
        sentinel = self.root / "sentinel"
        sentinel.write_text("DO NOT MODIFY", encoding="utf-8")
        self.symlink_or_skip(sentinel, database)
        with self.assertRaises(ProjectStoreError):
            with self.store():
                pass
        with self.assertRaises(ProjectStoreError):
            self.store().read_snapshot()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "DO NOT MODIFY")

    def test_journal_symlink_escape_is_rejected(self):
        with self.store() as store:
            store.create("Original")
        sentinel = self.root / "sentinel"
        sentinel.write_text("DO NOT MODIFY", encoding="utf-8")
        self.symlink_or_skip(sentinel, Path(str(self.store().db_path) + "-journal"))
        with self.assertRaises(ProjectStoreError):
            with self.store():
                pass
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "DO NOT MODIFY")

    def test_hardlinked_state_file_is_rejected(self):
        with self.store() as store:
            store.create("Original")
        os.link(self.store().db_path, self.root / "hardlink")
        with self.assertRaises(ProjectStoreError):
            with self.store():
                pass

    def test_use_outside_context_is_rejected(self):
        with self.assertRaises(ProjectStoreError):
            self.store().load()

    def test_archive_retains_prior_state_events_and_allows_explicit_new_goal(self):
        with self.store() as store:
            store.create("Original", {"max_steps": 2})
            store.reserve_step()
            with self.assertRaises(InvalidTransition):
                store.archive()
            store.transition("WAITING")
            with self.assertRaises(InvalidTransition):
                store.archive()
            store.transition("PAUSED")
            before = store.load()
            events = store.recent_events()
            lock_inode = store.lock_path.stat().st_ino
            database_inode = store.db_path.stat().st_ino
            archive = store.archive()
            self.assertTrue(archive.is_file())
            self.assertIsNone(store.load())
            self.assertEqual(store.recent_events(), [])
            self.assertEqual(store.lock_path.stat().st_ino, lock_inode)
            self.assertEqual(store.db_path.stat().st_ino, database_inode)
            connection = sqlite3.connect(archive.as_uri() + "?mode=ro", uri=True)
            try:
                archived_state = json.loads(connection.execute("SELECT state FROM metadata").fetchone()[0])
                self.assertEqual(archived_state, before)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], len(events))
            finally:
                connection.close()
            created = store.create("Second goal", {"max_steps": 3})
            self.assertEqual(created["steps_used"], 0)
            self.assertEqual(created["policy"]["max_steps"], 3)
            store.transition("STOPPED")
            second_archive = store.archive()
            self.assertNotEqual(archive, second_archive)
            self.assertTrue(archive.is_file())

    def test_archive_cannot_discard_unknown_action(self):
        with self.store() as store:
            store.create("Task")
            store.reserve_step()
            store.begin_action("write")
            store.transition("PAUSED")
            with self.assertRaises(InvalidTransition):
                store.archive()
            store.recover()
            with self.assertRaises(InvalidTransition):
                store.archive()

    def test_failed_archive_cannot_reset_current_project(self):
        with self.store() as store:
            store.create("Original")
            store.reserve_step()
            before = store.transition("PAUSED")
            with patch("project_store.os.link", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    store.archive()
            self.assertEqual(store.load(), before)

    def test_explicit_unknown_acknowledgment_does_not_restart_stopped_project(self):
        with self.store() as store:
            store.create("Task")
            store.reserve_step()
            store.begin_action("write")
            with self.assertRaises(InvalidTransition):
                store.acknowledge_unknown()
            store.transition("STOPPED")
            store.recover()
            acknowledged = store.acknowledge_unknown()
            self.assertEqual(acknowledged["status"], "STOPPED")
            self.assertEqual(acknowledged["steps_used"], 1)
            self.assertIsNone(acknowledged["inflight"])
            self.assertFalse(acknowledged["recovery_required"])
            self.assertEqual(acknowledged["last_action"]["outcome"]["status"], "unknown")
            self.assertTrue(store.archive().is_file())

    def test_sqlite_recovers_uncommitted_transaction_after_real_process_crash(self):
        with self.store() as store:
            before = store.create("Original")
            count = len(store.recent_events())
        result = self.child("""
import os,sys,json
from project_store import ProjectStore
with ProjectStore(sys.argv[1],sys.argv[2]) as store:
    state=store.load()
    state['steps_used']=999
    store._db.execute('BEGIN IMMEDIATE')
    store._db.execute('UPDATE metadata SET state=?',(json.dumps(state),))
    store._db.execute("INSERT INTO events(at,kind,data) VALUES(0,'uncommitted','{}')")
    os._exit(25)
""")
        self.assertEqual(result.returncode, 25, result.stderr)
        with self.store() as store:
            self.assertEqual(store.load(), before)
            self.assertEqual(len(store.recent_events()), count)

    def test_archive_clear_transaction_failure_preserves_current_and_backup(self):
        with self.store() as store:
            store.create("Original")
            before = store.transition("PAUSED")
            store._db.execute("CREATE TRIGGER fail_clear BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'injected clear failure'); END")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected clear failure"):
                store.archive()
            self.assertEqual(store.load(), before)
            archives = list((store.directory / "archives").glob("*.sqlite3"))
            self.assertEqual(len(archives), 1)
            connection = sqlite3.connect(archives[0])
            try:
                self.assertEqual(json.loads(connection.execute("SELECT state FROM metadata").fetchone()[0]), before)
            finally:
                connection.close()

    def test_explicit_budget_extension_preserves_goal_usage_history_and_permissions(self):
        with self.store() as store:
            before = store.create("Original", {"max_steps": 1})
            store.reserve_step()
            with self.assertRaises(BudgetExhausted):
                store.reserve_step()
            count = len(store.recent_events())
            state = store.extend_budget(add_steps=3, add_seconds=2 * 86400)
            self.assertEqual(state["status"], "PAUSED")
            self.assertEqual(state["steps_used"], 1)
            self.assertEqual(state["goal"], "Original")
            self.assertEqual(state["policy"]["max_steps"], 4)
            self.assertEqual(state["policy"]["max_seconds"], 3 * 86400)
            self.assertEqual(state["deadline_at"], before["deadline_at"] + 2 * 86400)
            self.assertFalse(state["policy"]["allow_shell"])
            self.assertFalse(state["policy"]["allow_write"])
            self.assertEqual(len(store.recent_events()), count + 1)
            with self.assertRaises(InvalidTransition):
                store.reserve_step()
            store.resume()
            self.assertEqual(store.reserve_step()["steps_used"], 2)

    def test_budget_extension_rejects_active_unknown_and_invalid_allowances(self):
        with self.store() as store:
            store.create("Task")
            with self.assertRaises(InvalidTransition):
                store.extend_budget(add_steps=1)
            store.reserve_step()
            action = store.begin_action("write")
            store.transition("PAUSED")
            with self.assertRaises(InvalidTransition):
                store.extend_budget(add_steps=1)
            store.finish_action(action, {"ok": True})
            before = store.load()
            for args in ({}, {"add_steps": -1}, {"add_steps": True}, {"add_steps": 1.5},
                         {"add_steps": 10000000}, {"add_seconds": float("inf")},
                         {"add_seconds": float("nan")}, {"add_seconds": True},
                         {"add_seconds": -1}, {"add_seconds": 366 * 86400}):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    store.extend_budget(**args)
                self.assertEqual(store.load(), before)

    def test_automatic_wake_preserves_no_progress_usage_and_original_allowance(self):
        with self.store() as store:
            initial = store.create("Task", {"max_no_progress": 3})
            store.reserve_step()
            store.record_progress(False)
            store.record_progress(False)
            store.transition("WAITING", "Model unavailable")
            state = store.wake_from_wait()
            self.assertEqual(state["status"], "RUNNING")
            self.assertEqual(state["no_progress_count"], 2)
            self.assertEqual(state["steps_used"], 1)
            self.assertEqual(state["deadline_at"], initial["deadline_at"])
            self.assertEqual(state["policy"], initial["policy"])
            self.assertEqual(store.recent_events()[-1]["kind"], "availability_recovered")
            self.assertNotIn("User", state["reason"])
            self.assertEqual(store.record_progress(False)["status"], "BLOCKED")

    def test_automatic_wake_rejects_paused_and_unknown_projects(self):
        with self.store() as store:
            store.create("Task")
            with self.assertRaises(InvalidTransition):
                store.wake_from_wait()
            store.transition("PAUSED")
            with self.assertRaises(InvalidTransition):
                store.wake_from_wait()
            store.resume()
            store.reserve_step()
            store.begin_action("write")
            store.transition("WAITING")
            with self.assertRaises(InvalidTransition):
                store.wake_from_wait()
            self.assertIsNotNone(store.load()["inflight"])

    def test_automatic_wake_never_refunds_exhausted_call_or_time_budget(self):
        with self.store() as store:
            store.create("Task", {"max_steps": 1})
            store.reserve_step()
            store.transition("WAITING")
            with self.assertRaises(BudgetExhausted):
                store.wake_from_wait()
            self.assertEqual(store.load()["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(store.load()["steps_used"], 1)
            store.extend_budget(add_steps=1)
            store.resume()
            store.transition("WAITING")
            with patch("project_store.time.time", return_value=store.load()["deadline_at"] + 1):
                with self.assertRaises(BudgetExhausted):
                    store.wake_from_wait()
            self.assertEqual(store.load()["status"], "BUDGET_EXHAUSTED")

    def test_windows_extended_drive_and_unc_aliases_share_identity(self):
        for normal, extended in ((r"C:\Users\Me\Workspace", r"\\?\c:\USERS\me\Workspace"),
                                 (r"\\server\Share\Workspace", r"\\?\UNC\SERVER\share\Workspace")):
            with self.subTest(normal=normal):
                self.assertEqual(_path_identity(normal, windows=True),
                                 _path_identity(extended, windows=True))
        self.assertNotEqual(_path_identity("/tmp/Case", windows=False),
                            _path_identity("/tmp/case", windows=False))

    def test_windows_containment_uses_normalized_prefixes_and_path_boundaries(self):
        for candidate, root in ((r"\\?\C:\Users\me\Workspace\state", r"C:\Users\me\Workspace"),
                                (r"C:\Users\me\Workspace\state", r"\\?\C:\Users\me\Workspace"),
                                (r"\\?\UNC\server\share\workspace\state", r"\\server\share\workspace")):
            with self.subTest(candidate=candidate):
                self.assertTrue(_path_within(candidate, root, windows=True))
        self.assertFalse(_path_within(r"C:\work-other\state", r"\\?\C:\work", windows=True))
        self.assertFalse(_path_within(r"D:\work\state", r"C:\work", windows=True))
        self.assertFalse(_path_within(r"\\server\other\work", r"\\server\share\work", windows=True))

    @unittest.skipUnless(os.name == "nt", "native Windows extended-path lock and state containment")
    def test_native_windows_alias_uses_same_runner_lock_and_cannot_hide_state_in_workspace(self):
        normal = _path_identity(self.workspace)
        extended = "\\\\?\\UNC\\" + normal[2:] if normal.startswith("\\\\") else "\\\\?\\" + normal
        with self.store() as store:
            state = store.create("One workspace")
            alias = ProjectStore(self.state, extended)
            self.assertEqual(alias.project_id, store.project_id)
            self.assertEqual(alias.read_snapshot()["project_id"], state["project_id"])
            with self.assertRaises(ProjectLocked):
                with alias:
                    pass
        with self.assertRaises(ValueError):
            ProjectStore(extended + "\\state", normal)
        with self.assertRaises(ValueError):
            ProjectStore(normal + "\\state", extended)


if __name__ == "__main__":
    unittest.main()
