#!/usr/bin/env python3
"""Offline release provenance regressions using isolated Git repositories."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import release_provenance as provenance


class ProvenanceTests(unittest.TestCase):
    VERSION = "1.2.3"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saient-provenance-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "app"
        self.engine = self.root / "engine"
        self.artifacts = self.root / "artifacts"
        self.output = self.root / "output"
        for directory in (self.repo, self.engine, self.artifacts):
            directory.mkdir()
        self.app = self.repo / "llm-inference"
        self.tauri = self.app / "src-tauri"
        self.runtime = self.tauri / "resources/saient"
        self.runtime.mkdir(parents=True)
        self.write(self.app / "package.json", {"version": self.VERSION})
        self.write(self.app / "package-lock.json", {"lockfileVersion": 3})
        self.write(self.tauri / "tauri.conf.json", {"version": self.VERSION})
        self.write(self.tauri / "tauri.linux.conf.json", {})
        self.write(self.tauri / "tauri.windows.conf.json", {})
        self.write(self.tauri / "Cargo.lock", "version = 3\n")
        self.write(self.tauri / "Cargo.toml", '[package]\nname = "app"\nversion = "1.2.3"\n')
        self.cli = b'#!/usr/bin/env python3\nprint("hello")\n'
        self.write(self.tauri / "src/pty.rs", b'const SAIENT_CLI_PY: &str = r####"' + self.cli + b'"####;\n')
        self.write(self.runtime / "runner.py", "print('runtime')\n")
        self.manifest = {"runtime_file_count": 1, "files": {
            "runner.py": {"bundled_sha256": provenance.sha256(self.runtime / "runner.py")}}}
        self.write(self.runtime / "RUNTIME_SOURCE.json", self.manifest)
        self.write(self.engine / "Cargo.toml", '[package]\nname = "engine"\nversion = "1.0.0"\n')
        for repo in (self.repo, self.engine):
            self.git(repo, "init", "-q")
            self.git(repo, "config", "core.autocrlf", "false")
            self.commit(repo)
        self.linux_names = [f"Saient_{self.VERSION}_amd64.deb", f"Saient_{self.VERSION}_amd64.AppImage"]
        self.windows_names = [f"Saient_{self.VERSION}_x64-setup.exe", f"Saient_{self.VERSION}_x64-setup.exe.sig"]
        self.bundles(self.linux_names)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict):
            value = json.dumps(value, indent=2) + "\n"
        path.write_bytes(value.encode() if isinstance(value, str) else value)

    @staticmethod
    def git(repo, *arguments):
        return subprocess.run(["git", "-C", str(repo), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "-c", "user.name=Provenance Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "test fixture")

    def bundles(self, names):
        for name in names:
            self.write(self.artifacts / name, b"fixture " + name.encode())

    def produce(self, platform="linux", artifact_dirs=None):
        return provenance.generate(self.repo, self.engine, platform,
                                   artifact_dirs or [self.artifacts], self.output)

    def test_linux_records_exact_source_and_final_artifact_hashes(self):
        document = self.produce()
        self.assertEqual(document["app"]["commit"], self.git(self.repo, "rev-parse", "HEAD"))
        self.assertEqual(document["engine"]["commit"], self.git(self.engine, "rev-parse", "HEAD"))
        self.assertEqual(document["app"]["embedded_terminal_cli_sha256"], hashlib.sha256(self.cli).hexdigest())
        self.assertEqual(document["runtime"]["file_count"], 1)
        self.assertEqual({row["filename"] for row in document["artifacts"]}, set(self.linux_names))
        encoded = (self.output / "release-provenance-linux.json").read_text()
        self.assertEqual(json.loads(encoded), document)
        self.assertNotIn(str(self.root), encoded)
        sums = (self.output / "SHA256SUMS-linux.txt").read_text()
        for name in self.linux_names:
            self.assertIn(f"{provenance.sha256(self.artifacts / name)}  {name}\n", sums)

    def test_windows_requires_and_hashes_final_installer_and_signature(self):
        windows = self.root / "windows"
        windows.mkdir()
        for name in self.windows_names:
            self.write(windows / name, b"final signed bytes " + name.encode())
        document = self.produce("windows", [windows])
        self.assertEqual({row["filename"] for row in document["artifacts"]}, set(self.windows_names))
        self.assertTrue((self.output / "SHA256SUMS-windows.txt").is_file())

    def test_same_artifact_and_output_directory_supports_identical_reruns(self):
        self.output = self.artifacts
        first = self.produce()
        encoded = (self.output / "release-provenance-linux.json").read_bytes()
        self.assertEqual(self.produce(), first)
        self.assertEqual((self.output / "release-provenance-linux.json").read_bytes(), encoded)

    def test_untracked_output_directory_inside_checkout_is_allowed(self):
        self.artifacts = self.repo / "release-provenance/linux"
        self.output = self.artifacts
        self.bundles(self.linux_names)
        self.assertEqual(len(self.produce()["artifacts"]), 2)
        self.assertEqual(self.git(self.repo, "status", "--porcelain", "--untracked-files=no"), "")

    def test_same_size_artifact_change_during_hashing_is_rejected(self):
        digest = provenance.sha256

        def change_during_hash(path):
            result = digest(path)
            if path.parent == self.artifacts:
                before = path.stat()
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            return result

        with patch.object(provenance, "sha256", side_effect=change_during_hash):
            with self.assertRaisesRegex(provenance.ProvenanceError, "artifact changed during hashing"):
                self.produce()

    def test_runtime_hash_mismatch_fails_before_output(self):
        self.write(self.runtime / "runner.py", "changed runtime\n")
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "runtime hash mismatch"):
            self.produce()
        self.assertFalse(self.output.exists())

    def test_runtime_count_mismatch_is_rejected(self):
        self.manifest["runtime_file_count"] = 2
        self.write(self.runtime / "RUNTIME_SOURCE.json", self.manifest)
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "runtime_file_count"):
            self.produce()

    def test_runtime_manifest_path_traversal_is_rejected(self):
        self.manifest["files"] = {"../outside.py": {"bundled_sha256": "0" * 64}}
        self.write(self.runtime / "RUNTIME_SOURCE.json", self.manifest)
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "unsafe runtime path"):
            self.produce()

    def test_unlisted_python_runtime_file_is_rejected(self):
        self.write(self.runtime / "unlisted.py", "print('unexpected')\n")
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "unlisted runtime"):
            self.produce()

    def test_dirty_tracked_app_is_rejected(self):
        self.write(self.app / "package-lock.json", "changed")
        with self.assertRaisesRegex(provenance.ProvenanceError, "dirty tracked source"):
            self.produce()

    def test_dirty_tracked_engine_is_rejected(self):
        self.write(self.engine / "Cargo.toml", "changed")
        with self.assertRaisesRegex(provenance.ProvenanceError, "dirty tracked source"):
            self.produce()

    def test_staged_app_change_is_rejected(self):
        self.write(self.app / "package-lock.json", "changed")
        self.git(self.repo, "add", ".")
        with self.assertRaisesRegex(provenance.ProvenanceError, "dirty tracked source"):
            self.produce()

    def test_missing_linux_artifact_is_rejected(self):
        (self.artifacts / self.linux_names[1]).unlink()
        with self.assertRaisesRegex(provenance.ProvenanceError, "missing required artifact"):
            self.produce()

    def test_missing_windows_signature_is_rejected(self):
        windows = self.root / "windows"
        self.write(windows / self.windows_names[0], b"installer")
        with self.assertRaisesRegex(provenance.ProvenanceError, "missing required artifact"):
            self.produce("windows", [windows])

    def test_duplicate_names_in_separate_directories_are_rejected(self):
        extra = self.root / "extra"
        self.write(extra / self.linux_names[0], b"another installer")
        with self.assertRaisesRegex(provenance.ProvenanceError, "duplicate artifact"):
            self.produce(artifact_dirs=[self.artifacts, extra])

    def test_repeated_artifact_directory_is_rejected(self):
        with self.assertRaisesRegex(provenance.ProvenanceError, "duplicate artifact"):
            self.produce(artifact_dirs=[self.artifacts, self.artifacts])

    def test_stale_version_mix_is_rejected(self):
        self.write(self.artifacts / "Saient_1.2.2_amd64.deb", b"stale")
        with self.assertRaisesRegex(provenance.ProvenanceError, "unexpected or stale artifact"):
            self.produce()

    def test_wrong_platform_artifact_is_rejected(self):
        self.write(self.artifacts / self.windows_names[0], b"wrong platform")
        with self.assertRaisesRegex(provenance.ProvenanceError, "unexpected or stale artifact"):
            self.produce()

    def test_empty_artifact_is_rejected(self):
        self.write(self.artifacts / self.linux_names[0], b"")
        with self.assertRaisesRegex(provenance.ProvenanceError, "empty artifact"):
            self.produce()

    def test_version_disagreement_is_rejected(self):
        self.write(self.app / "package.json", {"version": "1.2.4"})
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "version mismatch"):
            self.produce()

    def test_missing_embedded_cli_is_rejected(self):
        self.write(self.tauri / "src/pty.rs", "// no terminal controller\n")
        self.commit(self.repo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "embedded terminal CLI"):
            self.produce()

    def test_cli_returns_nonzero_without_misleading_output_on_failure(self):
        script = Path(provenance.__file__)
        result = subprocess.run([sys.executable, str(script), "--repo-root", str(self.repo),
                                 "--engine-root", str(self.engine), "--platform", "windows",
                                 "--artifact-dir", str(self.artifacts), "--output-dir", str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("provenance failed:", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
