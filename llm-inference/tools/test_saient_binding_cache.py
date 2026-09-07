"""Offline binding lifecycle regression checks; no host/model calls are made."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


RUNTIME = pathlib.Path(__file__).resolve().parents[1] / "src-tauri/resources/saient"
sys.path.insert(0, str(RUNTIME))
import binding_bridge as bridge


class BindingCacheTests(unittest.TestCase):
    MODEL = "Qwen2.5-Coder-7B-Instruct-Q4_K_M"
    ENDPOINT = "http://127.0.0.1:34919"
    FINGERPRINT = "matching-runtime-v4"

    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix="saient-binding-cache-test-")))
        self.enterContext(patch.object(bridge, "discover", return_value=self.MODEL))
        self.enterContext(patch.object(bridge, "runtime_fingerprint", return_value=self.FINGERPRINT))
        self.profile = self.enterContext(patch.object(
            bridge, "profile", side_effect=AssertionError("unexpected model profiling")))
        self.path = bridge._manifest_path(self.directory, self.MODEL, self.FINGERPRINT)

    def evidence(self, status):
        return {
            "binding_status": status,
            "dominant_failure": None if status == "bound" else "identity_or_record_authority_boundary",
            "minimum_interface": "L0_plain" if status == "bound" else None,
            "preferred_control": {"temperature": 0.55, "max_tokens": 160, "deheaded": False}
            if status == "bound" else None,
        }

    def save(self, status):
        evidence = self.evidence(status)
        manifest = {
            "manifest_version": 4,
            "runtime_fingerprint": self.FINGERPRINT,
            "model": self.MODEL,
            "binding_status": status,
            "profile_contract": list(bridge.PROFILE_CONTRACT),
            "minimum_interface": evidence["minimum_interface"],
            "preferred_control": evidence["preferred_control"],
            "profile": evidence,
            "last_verified_endpoint": "http://127.0.0.1:37665",
        }
        bridge._write_manifest(self.path, manifest)
        return manifest

    def test_v4_rejected_binding_is_reported_without_reprofiling(self):
        self.save("rejected")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(bridge.BindingError, "binding is rejected: identity_or_record_authority_boundary"):
            bridge.ensure_binding(self.ENDPOINT, self.directory)
        self.profile.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_require_and_bind_report_the_same_rejection(self):
        self.save("rejected")
        errors = []
        for operation in (bridge.ensure_binding, bridge.require_binding):
            with self.assertRaises(bridge.BindingError) as raised:
                operation(self.ENDPOINT, self.directory)
            errors.append(str(raised.exception))
        self.assertEqual(errors[0], errors[1])
        self.assertIn("rejected", errors[0])
        self.assertIn("rebind", errors[0])
        self.profile.assert_not_called()

    def test_unresolved_evidence_requires_explicit_retry(self):
        self.save("unresolved")
        with self.assertRaisesRegex(bridge.BindingError, "binding is unresolved"):
            bridge.ensure_binding(self.ENDPOINT, self.directory)
        self.profile.assert_not_called()

    def test_bound_manifest_survives_endpoint_change(self):
        expected = self.save("bound")
        for operation in (bridge.ensure_binding, bridge.require_binding):
            manifest, path = operation(self.ENDPOINT, self.directory)
            self.assertEqual(manifest, expected)
            self.assertEqual(path, self.path)
        self.profile.assert_not_called()

    def test_explicit_rebind_can_replace_rejected_evidence(self):
        self.save("rejected")
        self.profile.side_effect = None
        self.profile.return_value = self.evidence("bound")
        manifest, path = bridge.ensure_binding(self.ENDPOINT, self.directory, force=True)
        self.profile.assert_called_once_with(self.ENDPOINT, self.MODEL)
        self.assertEqual(manifest["binding_status"], "bound")
        self.assertEqual(json.loads(path.read_text())["binding_status"], "bound")
        self.assertEqual(manifest["profile_contract"], list(bridge.PROFILE_CONTRACT))

    def test_explicit_rebind_failure_is_saved_and_still_enforced(self):
        self.save("bound")
        self.profile.side_effect = None
        self.profile.return_value = self.evidence("rejected")
        with self.assertRaisesRegex(bridge.BindingError, "binding is rejected"):
            bridge.ensure_binding(self.ENDPOINT, self.directory, force=True)
        self.assertEqual(json.loads(self.path.read_text())["binding_status"], "rejected")
        with self.assertRaisesRegex(bridge.BindingError, "binding is rejected"):
            bridge.require_binding(self.ENDPOINT, self.directory)
        self.profile.assert_called_once()

    def test_user_operation_with_no_evidence_does_not_profile(self):
        with self.assertRaisesRegex(bridge.BindingError, "not finished binding"):
            bridge.require_binding(self.ENDPOINT, self.directory)
        self.profile.assert_not_called()

    def test_initial_model_setup_profiles_once_then_reuses_evidence(self):
        self.profile.side_effect = None
        self.profile.return_value = self.evidence("bound")
        first, path = bridge.ensure_binding(self.ENDPOINT, self.directory)
        second, _ = bridge.ensure_binding(self.ENDPOINT, self.directory)
        self.assertEqual(first, second)
        self.assertTrue(path.exists())
        self.profile.assert_called_once_with(self.ENDPOINT, self.MODEL)

    def test_chat_with_rejected_host_never_runs_a_tick_or_profile(self):
        self.save("rejected")
        with patch.object(bridge, "say", side_effect=AssertionError("unexpected user tick")) as say:
            with self.assertRaisesRegex(bridge.BindingError, "binding is rejected"):
                bridge.bound_chat(self.ENDPOINT, self.directory, "build a calculator")
        self.profile.assert_not_called()
        say.assert_not_called()

    def test_cli_rebind_is_an_explicit_profiling_path(self):
        self.save("rejected")
        self.profile.side_effect = None
        self.profile.return_value = self.evidence("bound")
        argv = ["binding_bridge.py", "rebind", "--endpoint", self.ENDPOINT,
                "--manifest-dir", str(self.directory)]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            result = bridge.main()
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["binding_status"], "bound")
        self.profile.assert_called_once_with(self.ENDPOINT, self.MODEL)

    def test_cached_rejection_is_a_nonzero_cli_result_for_rust(self):
        self.save("rejected")
        for operation in ("bind", "require"):
            with self.subTest(operation=operation):
                stdout, stderr = io.StringIO(), io.StringIO()
                argv = ["binding_bridge.py", operation, "--endpoint", self.ENDPOINT,
                        "--manifest-dir", str(self.directory)]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = bridge.main()
                self.assertEqual(result, 4)
                self.assertEqual(stdout.getvalue(), "")
                error = json.loads(stderr.getvalue())
                self.assertIn("binding is rejected", error["error"])
                self.assertIs(error["plain_llm_fallback"], False)
        self.profile.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
