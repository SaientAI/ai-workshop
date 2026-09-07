#!/usr/bin/env python3
"""Record final release bytes and their clean, immutable source revisions.

This emits independent build evidence, never the live website/update manifest.
It checks source identity and hashes; it does not claim installer execution or
cryptographic signature verification. Run after any installer signing step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


SOURCE_FILES = (
    "llm-inference/package.json",
    "llm-inference/package-lock.json",
    "llm-inference/src-tauri/Cargo.toml",
    "llm-inference/src-tauri/Cargo.lock",
    "llm-inference/src-tauri/tauri.conf.json",
    "llm-inference/src-tauri/tauri.linux.conf.json",
    "llm-inference/src-tauri/tauri.windows.conf.json",
    "llm-inference/src-tauri/src/pty.rs",
    "llm-inference/src-tauri/resources/saient/RUNTIME_SOURCE.json",
)
RUNTIME_PATH = "llm-inference/src-tauri/resources/saient"
ARTIFACT_SUFFIXES = (".deb", ".appimage", ".exe", ".sig", ".rpm", ".msi", ".dmg")


class ProvenanceError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        # Do not repeat environment, remote URLs, or potentially secret output.
        raise ProvenanceError(f"git {arguments[0]} failed (exit {result.returncode})")
    return result.stdout.strip()


def clean_revision(root):
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ProvenanceError("repository root must identify its own Git checkout")
    revision = git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ProvenanceError("invalid Git HEAD")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ProvenanceError("dirty tracked source; commit or restore intended changes before release")
    return revision


def regular_file(path, label):
    if path.is_symlink() or not path.is_file():
        raise ProvenanceError(f"missing or non-regular {label}")


def read_json(path):
    regular_file(path, path.name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid JSON in {path.name}") from error
    if not isinstance(value, dict):
        raise ProvenanceError(f"expected JSON object in {path.name}")
    return value


def verify_runtime(repo, tracked):
    runtime = repo / RUNTIME_PATH
    manifest = read_json(runtime / "RUNTIME_SOURCE.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ProvenanceError("runtime manifest has no files")
    if manifest.get("runtime_file_count") != len(files):
        raise ProvenanceError("runtime_file_count does not match manifest files")
    verified = {}
    for name, metadata in sorted(files.items()):
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or "\\" in name or ":" in name or name != relative.as_posix()):
            raise ProvenanceError("unsafe runtime path in manifest")
        path = runtime.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(runtime.resolve()):
            raise ProvenanceError("unsafe runtime path outside bundle")
        regular_file(path, f"runtime file {name}")
        if f"{RUNTIME_PATH}/{name}" not in tracked:
            raise ProvenanceError(f"runtime source is not tracked: {name}")
        expected = metadata.get("bundled_sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ProvenanceError(f"invalid bundled_sha256 for {name}")
        actual = sha256(path)
        if actual != expected:
            raise ProvenanceError(f"runtime hash mismatch: {name}")
        verified[name] = actual
    for path in runtime.rglob("*.py"):
        if path.relative_to(runtime).as_posix() not in files:
            raise ProvenanceError(f"unlisted runtime Python file: {path.name}")
    return {"manifest_sha256": sha256(runtime / "RUNTIME_SOURCE.json"),
            "file_count": len(verified), "bundled_files_sha256": verified}


def source_evidence(repo):
    tracked = set(git(repo, "ls-files", "-z").split("\0"))
    hashes = {}
    for name in SOURCE_FILES:
        path = repo / name
        regular_file(path, name)
        if name not in tracked:
            raise ProvenanceError(f"source file is not tracked: {name}")
        hashes[name] = sha256(path)
    tauri = repo / "llm-inference/src-tauri"
    version = read_json(tauri / "tauri.conf.json").get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise ProvenanceError("invalid application version")
    if read_json(repo / "llm-inference/package.json").get("version") != version:
        raise ProvenanceError("version mismatch between package.json and Tauri")
    cargo = (tauri / "Cargo.toml").read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s*\n(.*?)(?=^\[|\Z)", cargo)
    cargo_version = re.search(r'^version\s*=\s*"([^"]+)"\s*$', package[1], re.M) if package else None
    if not cargo_version or cargo_version[1] != version:
        raise ProvenanceError("version mismatch between Cargo.toml and Tauri")
    pty = (tauri / "src/pty.rs").read_bytes()
    marker = b'const SAIENT_CLI_PY: &str = r####"'
    if pty.count(marker) != 1 or b'"####;' not in pty.split(marker, 1)[-1]:
        raise ProvenanceError("missing or ambiguous embedded terminal CLI")
    cli = pty.split(marker, 1)[1].split(b'"####;', 1)[0]
    if not cli.strip():
        raise ProvenanceError("empty embedded terminal CLI")
    return version, hashes, hashlib.sha256(cli).hexdigest(), verify_runtime(repo, tracked)


def collect_artifacts(platform, version, directories):
    if platform == "linux":
        expected = {f"Saient_{version}_amd64.deb", f"Saient_{version}_amd64.AppImage"}
    elif platform == "windows":
        installer = f"Saient_{version}_x64-setup.exe"
        expected = {installer, installer + ".sig"}
    else:
        raise ProvenanceError("platform must be linux or windows")
    found = {}
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ProvenanceError("artifact directory does not exist")
        # Bundle folders also contain staging subdirectories; only final files
        # immediately inside each explicitly supplied directory are selected.
        for path in sorted(directory.iterdir()):
            if not path.name.lower().endswith(ARTIFACT_SUFFIXES):
                continue
            if path.name not in expected:
                raise ProvenanceError(f"unexpected or stale artifact: {path.name}")
            regular_file(path, f"artifact {path.name}")
            if path.name in found:
                raise ProvenanceError(f"duplicate artifact: {path.name}")
            before = path.stat()
            if before.st_size == 0:
                raise ProvenanceError(f"empty artifact: {path.name}")
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                raise ProvenanceError(f"artifact changed during hashing: {path.name}")
            found[path.name] = {"filename": path.name, "bytes": before.st_size, "sha256": digest}
    if expected - found.keys():
        raise ProvenanceError("missing required artifact: " + ", ".join(sorted(expected - found.keys())))
    return [found[name] for name in sorted(found)]


def atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def generate(repo_root, engine_root, platform, artifact_dirs, output_dir):
    repo = Path(repo_root).resolve()
    engine = Path(engine_root).resolve()
    app_commit, engine_commit = clean_revision(repo), clean_revision(engine)
    version, source_hashes, cli_hash, runtime = source_evidence(repo)
    artifacts = collect_artifacts(platform, version, artifact_dirs)
    # Detect source changes while reading/hashing; never label a mixed snapshot
    # as evidence for the initial revision.
    if clean_revision(repo) != app_commit or clean_revision(engine) != engine_commit:
        raise ProvenanceError("source revision changed while producing provenance")
    document = {
        "schema_version": 1,
        "platform": platform,
        "app": {"commit": app_commit, "version": version,
                "source_files_sha256": source_hashes,
                "embedded_terminal_cli_sha256": cli_hash},
        "engine": {"commit": engine_commit},
        "runtime": runtime,
        "artifacts": artifacts,
        "verification_scope": "Source/runtime hashes and final artifact bytes; not installer execution or signature validity.",
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"release-provenance-{platform}.json"
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sums = "".join(f"{row['sha256']}  {row['filename']}\n" for row in artifacts)
    sums += f"{hashlib.sha256(encoded).hexdigest()}  {name}\n"
    atomic_write(output / name, encoded)
    atomic_write(output / f"SHA256SUMS-{platform}.txt", sums.encode("utf-8"))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = generate(args.repo_root, args.engine_root, args.platform,
                            args.artifact_dir, args.output_dir)
    except (ProvenanceError, OSError) as error:
        print(f"provenance failed: {error}", file=sys.stderr)
        return 1
    print(f"Provenance recorded: {args.platform} {document['app']['version']} "
          f"app={document['app']['commit']} engine={document['engine']['commit']} "
          f"artifacts={len(document['artifacts'])} runtime_files={document['runtime']['file_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
