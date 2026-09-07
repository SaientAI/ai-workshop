#!/usr/bin/env python3
"""Real cross-process runtime locking tests, run on Linux and native Windows."""
import errno
import importlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


RUNTIME = Path(__file__).resolve().parents[1] / "src-tauri/resources/saient"
sys.path.insert(0, str(RUNTIME))


class RuntimeFileLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saient-lock-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
                        PYTHONPATH=str(RUNTIME), SAIENT_STATE_DIR=str(self.root / "state"))

    def child(self, code, *args):
        return subprocess.run([sys.executable, "-B", "-c", code, *map(str, args)],
                              env=self.env, capture_output=True, text=True, timeout=20)

    def helper(self):
        return importlib.import_module("runtime_file_lock")

    @staticmethod
    def child_line(child):
        incoming = queue.Queue()
        threading.Thread(target=lambda: incoming.put(child.stdout.readline()), daemon=True).start()
        return incoming.get(timeout=5).strip()

    def test_all_runtime_entry_modules_import_with_no_fcntl(self):
        result = self.child("""
import sys
sys.modules['fcntl'] = None
import conscience, state, orchestrator, desktop_bridge, binding_bridge, run_saient
import arc_gremlin.persistence
print('runtime imports passed')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("runtime imports passed", result.stdout)

    def test_native_runtime_imports_and_state_profile_audit_writes(self):
        result = self.child("""
import json, os
from pathlib import Path
import conscience, state, orchestrator, desktop_bridge, binding_bridge, run_saient
from arc_gremlin.persistence import locked_write, load_json
root = Path(os.environ['SAIENT_STATE_DIR'])
profile = conscience.IdentityProfile.default(root / 'profile.json')
profile.save()
assert conscience.IdentityProfile.load(root / 'profile.json').name == 'Saient'
audit = conscience.append_audit_record({'decision': 'allow', 'unicode': 'é'}, root / 'audit')
assert json.loads(audit.read_text(encoding='utf-8'))['unicode'] == 'é'
with state._tick_lock():
    locked_write(root / 'nested.json', {'verified': True})
assert load_json(root / 'nested.json')['verified'] is True
print('native runtime writes passed: ' + os.name)
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("native runtime writes passed: " + os.name, result.stdout)

    def test_real_process_contention_and_release_three_times(self):
        helper = self.helper()
        code = """
import errno, os, sys
from runtime_file_lock import exclusive_file_lock
with open(sys.argv[1], 'a+b') as probe:
    probe.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(probe.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise
        print('BLOCKED', flush=True)
    else:
        raise AssertionError('parent did not hold the native lock')
with exclusive_file_lock(sys.argv[1]):
    print('ACQUIRED', flush=True)
"""
        for iteration in range(3):
            with self.subTest(iteration=iteration):
                path = self.root / f"contention-{iteration}.lock"
                with helper.exclusive_file_lock(path):
                    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(path)],
                                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True)
                    try:
                        # A native nonblocking attempt proves actual contention,
                        # rather than inferring it from a scheduling delay.
                        self.assertEqual(self.child_line(child), "BLOCKED")
                        with self.assertRaises(subprocess.TimeoutExpired):
                            child.communicate(timeout=0.15)
                    except BaseException:
                        child.kill()
                        child.communicate(timeout=5)
                        raise
                try:
                    stdout, stderr = child.communicate(timeout=5)
                except BaseException:
                    child.kill()
                    child.communicate(timeout=5)
                    raise
                self.assertEqual(child.returncode, 0, stderr)
                self.assertIn("ACQUIRED", stdout)

    @unittest.skipUnless(os.name == "nt", "native Windows CRT contention deadline")
    def test_native_windows_waits_beyond_ten_second_crt_limit(self):
        path = self.root / "long-tick.lock"
        code = """
import sys
from runtime_file_lock import exclusive_file_lock
print('ATTEMPTING', flush=True)
with exclusive_file_lock(sys.argv[1]):
    print('ACQUIRED', flush=True)
"""
        child = None
        try:
            with self.helper().exclusive_file_lock(path):
                child = subprocess.Popen([sys.executable, "-B", "-c", code, str(path)],
                                         env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertEqual(self.child_line(child), "ATTEMPTING")
                with self.assertRaises(subprocess.TimeoutExpired):
                    child.communicate(timeout=11.1)
            stdout, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertIn("ACQUIRED", stdout)
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=5)

    def test_lock_releases_after_exception_and_preserves_sidecar_bytes(self):
        helper = self.helper()
        path = self.root / "existing.lock"
        path.write_bytes(b"existing sentinel")
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with helper.exclusive_file_lock(path):
                raise RuntimeError("body failed")
        result = self.child("from runtime_file_lock import exclusive_file_lock; import sys\n"
                            "with exclusive_file_lock(sys.argv[1]): print('reacquired')", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reacquired", result.stdout)
        self.assertEqual(path.read_bytes(), b"existing sentinel")

    def test_concurrent_audit_appends_preserve_all_records(self):
        code = """
import sys
from conscience import append_audit_record
for n in range(20):
    append_audit_record({'worker': int(sys.argv[2]), 'number': n, 'payload': 'é' * 4096}, sys.argv[1])
"""
        children = [subprocess.Popen([sys.executable, "-B", "-c", code, str(self.root / "audit"), str(worker)],
                                     env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for worker in range(4)]
        try:
            for child in children:
                _, stderr = child.communicate(timeout=20)
                self.assertEqual(child.returncode, 0, stderr)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=5)
        logs = list((self.root / "audit").glob("*.ndjson"))
        records = [json.loads(line) for path in logs for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 80)
        self.assertEqual({(row["worker"], row["number"]) for row in records},
                         {(worker, number) for worker in range(4) for number in range(20)})
        self.assertTrue(all(row["payload"] == "é" * 4096 for row in records))

    def test_windows_retries_contention_beyond_ten_attempts_then_unlocks_byte_zero(self):
        helper = self.helper()
        calls = []

        def lock(fd, mode, count):
            calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR)))
            if mode == 1 and len(calls) <= 12:
                raise OSError(errno.EACCES, "region held by another process")

        backend = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lock)
        with patch.object(helper, "_WINDOWS", True), patch.dict(sys.modules, {"msvcrt": backend}), \
                patch.object(helper.time, "sleep") as pause:
            with helper.exclusive_file_lock(self.root / "windows.lock"):
                pass
        self.assertEqual(pause.call_count, 12)
        self.assertEqual(calls, [(1, 1, 0)] * 13 + [(2, 1, 0)])

    def test_windows_lock_failure_does_not_enter_unlocked_body(self):
        helper = self.helper()
        backend = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2,
                                  locking=lambda *args: (_ for _ in ()).throw(OSError(errno.EBADF, "bad fd")))
        with patch.object(helper, "_WINDOWS", True), patch.dict(sys.modules, {"msvcrt": backend}), \
                patch.object(helper.time, "sleep") as pause:
            with self.assertRaises(OSError):
                with helper.exclusive_file_lock(self.root / "failure.lock"):
                    self.fail("unlocked body executed")
        pause.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
