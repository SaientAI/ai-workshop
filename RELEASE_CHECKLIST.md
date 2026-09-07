# Release identity and verification

The Hugging Face local-app listing is maintained in a different repository. Its
PR commit is not a Saient application revision or an installer checksum.

Saient application versions must agree in `llm-inference/package.json`, its
lockfile, `src-tauri/Cargo.toml`, its lockfile and `src-tauri/tauri.conf.json`.
The website manifest describes the currently published binaries, not the newest
application source. Do not point old binaries at a newer source commit.

## Before publication

1. Commit reviewed changes and confirm the intended application SHA exists on
   the remote branch. Run CI, including the native Windows runtime/Rust checks.
2. Dispatch the Release workflow against that branch. Manual dispatch builds
   artifacts only; it cannot create a GitHub release. The workflow pins the same
   engine revision on both platforms and uses locked Cargo dependencies.
3. Read the results of both platform jobs. Each artifact archive must contain
   `release-provenance-<platform>.json` and `SHA256SUMS-<platform>.txt` alongside
   its final bundles. Verify all checksums after downloading/extracting the
   archive. Compare both JSON files: application and engine commits must match
   each other and the intended revisions.
4. Provenance checks reject dirty tracked source, mismatched bundled Python
   hashes, stale/missing/duplicate installers, and missing Windows updater
   signatures. A signature file's presence/hash does not verify its signature.
5. Test the new installer on the target OS: installation, startup, model load,
   workspace changes, terminal file creation and verification, cancellation,
   and `saient://` confirmation. Native CI is not a substitute for a Windows
   desktop/GPU installation check. Record the exact artifact hash tested.
6. If Authenticode signing is enabled, generate the updater signature only
   after Authenticode has finalized the installer bytes. Verify both signature
   types; do not describe an unsigned installer as Authenticode-signed.
7. Obtain publication approval. Publish new versioned binaries and their exact
   metadata together. Update website/download/update-feed metadata from the
   verified build outputs, not from a separate checkout or previous release.
   Re-download the public files and verify their size/hash/signature before
   claiming public release completion.

`tools/release_provenance.py` (under `llm-inference/`) generates build evidence;
it does not rewrite or publish `site/release-manifest.json`, sign files, verify
signature validity, or claim successful installer execution.

## Historical 1.0.23 identity

The public 1.0.23 artifacts were built by Actions run `33182952906` from app
commit `5e41ae412498e0831223f0b065931410249a4bdb` and engine commit
`5a2a82387b9548c6c165eb50880a5f49c54ef1ff`. A SHA audit on 2026-09-07 matched
the public Debian, AppImage and Windows installer bytes to those CI outputs.
Three stale source-file hashes in the repository's website manifest were
corrected against that historical commit. The artifact hashes were unchanged.

Version 1.0.24 contains subsequent desktop-terminal/workspace and Windows
portability repairs. A source commit or successful build alone must not be
reported as publication of those repairs.
