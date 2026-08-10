#!/usr/bin/env python3
"""Populate and Ed25519-sign Saient's Pi-hosted release manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def signed_message(release: str, artifact: dict[str, object]) -> bytes:
    return (
        "saient-update-v1\n"
        f"release={release.strip()}\n"
        f"platform={str(artifact['platform']).strip()}\n"
        f"url={str(artifact['url']).strip()}\n"
        f"bytes={int(artifact['bytes'])}\n"
        f"sha256={str(artifact['sha256']).strip().lower()}\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--platform", default="Debian/Ubuntu amd64")
    args = parser.parse_args()

    artifact_path = args.artifact.resolve(strict=True)
    key_path = args.private_key.resolve(strict=True)
    if not args.version or any(not part.isdigit() for part in args.version.split(".")):
        parser.error("--version must be a dotted numeric version")
    if not args.url.startswith("https://saient.co.uk/downloads/"):
        parser.error("--url must use https://saient.co.uk/downloads/")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact = manifest.setdefault("artifact", {})
    manifest["release"] = args.version
    artifact.update(
        {
            "platform": args.platform,
            "url": args.url,
            "bytes": artifact_path.stat().st_size,
            "sha256": digest,
        }
    )

    # OpenSSL's Ed25519 implementation is one-shot and must know the input
    # length. Supplying bytes through a pipe fails on OpenSSL 3.0, so use a
    # mode-0600 temporary file for the public manifest payload.
    with tempfile.NamedTemporaryFile("wb") as message_file:
        message_file.write(signed_message(args.version, artifact))
        message_file.flush()
        signature = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                os.fspath(key_path),
                "-in",
                message_file.name,
            ],
            check=True,
            capture_output=True,
        ).stdout
    if len(signature) != 64:
        raise RuntimeError(f"unexpected Ed25519 signature length: {len(signature)}")
    artifact["signature"] = base64.b64encode(signature).decode("ascii")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.manifest.parent,
        prefix=f".{args.manifest.name}.",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.manifest)
    args.manifest.chmod(0o644)

    print(f"release={args.version}")
    print(f"bytes={artifact['bytes']}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
