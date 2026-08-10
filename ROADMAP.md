# Saient — Ship Roadmap

Prioritized plan merging the fault scan + the feature backlog, with architecture decisions locked.
Source of truth for the v1.0 push. Tackle top-down; Tier 1 has no open decisions.

---

## Done (this session)
- Engine correctness: gpt-oss SWA, OLMoE routing, **per-arch RoPE** (fixed the whole llama family), full IQ/K quant GPU coverage, and the **KV-cache→VRAM sizing** fix (killed the long-context meltdown + empty-stream).
- **Engine bundling** (LM-Studio style): CUDA + CPU `tinyq4` + `libcudart` shipped in the app; runtime CUDA/CPU selection; `LD_LIBRARY_PATH`/`PATH` set at spawn. Wizard drops `pip install tinyq4` (Fast = instant).
- **Windows-compatible** backend (cfg-gated spawn, `taskkill`, `icon.ico`); `build.rs` cross-platform + `all-major` CUDA arch.
- **CI** (`.github/workflows/release.yml`): tag → 4 binaries → `.deb`/`.AppImage`/`.nsis` → draft Release.

## Decisions locked
- **Licensing:** 30-day local trial → **£20 one-time, signed offline license key** (Ed25519; app embeds public key, verifies offline; payment via Stripe/Gumroad checkout that emails the key). No license server.
- **Vision analyzer:** **Python pipeline** (managed venv + transformers), default **Moondream2** (tiny/fast), optional Qwen2-VL. Part of the "Full" stack, like image-gen/TTS.
- **Security:** **local launch password** (argon2 hash) + bind engine to `127.0.0.1` + per-session API token. No accounts.

---

## Tier 1 — quick wins, no decisions (do first)
- [ ] **Finish rebrand → Saient.** `TitleBar.svelte:36`, `SetupWizard.svelte:95` (`AI·Workshop` → `Saient`). Runtime config/models/cache paths now resolve under the project-local `data/` root.
- [ ] **Security bind:** tinyq4 server `0.0.0.0` → `127.0.0.1` (`tinyq4 server.rs`). Closes LAN exposure of the LLM + agent endpoints.
- [ ] **Stale text:** `engine.rs:723` error still says "Install it with `pip install tinyq4`" → bundled-engine wording.
- [x] **Path leak:** local runtime paths now resolve through `paths.rs` and `SAIENT_DATA_DIR`.
- [ ] **Hygiene:** remove my unused imports (`Ordering`, `bail`, `last`); strip 14 frontend `console.log`s; clear the `TTSScreen.svelte:44` TODO; pass over the ~50 real (non-lock) `unwrap()`s.

## Tier 2 — the three big features
- [ ] **Licensing (`license.rs` + paywall UI).** Trust-rooted first-run marker (signed) → 30-day trial; `verify(key)` Ed25519 against an embedded pubkey; states trial/expired/licensed; gate premium on expiry. Tiny `keygen` CLI (kept private) to sign keys after a Stripe/Gumroad sale. Pricing/Help copy ties in.
- [ ] **Vision analyzer.** `scripts/vision.py` (Moondream2 via transformers, VRAM-capped) + a "Vision" screen + a `vision_describe` Tauri command (mirrors `imggen`). Add to Full-setup install + walkthrough.
- [ ] **Launch password.** `auth.rs`: argon2 hash in config; lock screen on launch; set/reset flow in Settings. + engine `127.0.0.1` (Tier 1) + per-session bearer token on the tinyq4 API.

## Tier 3 — UX + web + legal
- [ ] **Toast tips system** (Svelte store + component) — contextual tips for new features.
- [ ] **Walkthrough refresh** — fold Vision, password, trial, toasts into `SetupWizard.svelte`.
- [ ] **Website** (`site/`): make crawlable (`robots.txt`, `sitemap.xml`, `<meta>`/OG, JSON-LD), add **Help** + **Pricing** pages (pricing reflects the £20 unlock).
- [ ] **Legal migration:** rebrand + move `staticplay-hub/public/{terms,privacy,privacy-manifesto,usage-responsibility-policy,threat-model,open-source-license-notice}.html` into the Saient site; link from footer + app.

---

## Release flow (CI)
Tag `vX.Y.Z` (or Actions → Release → Run) → builds Linux + Windows bundles → **draft** GitHub Release.
Tunables in `release.yml`: `CUDA_VERSION` (12.6.2; bump to 12.8 + a newer `Jimver/cuda-toolkit` tag for native Blackwell SASS), `TINYQ4_CUDA_ARCH=all-major`. First GPU-CI run may need a `cudart-dev`/symlink tweak or a Windows `nvcc` adjustment — both surface clearly.

## Notes / dependencies
- Pricing page + parts of the walkthrough depend on Licensing → build Licensing before they finalize.
- Vision + password are "Full"/optional; the bundled chat core stays zero-dependency.
- Windows ship still needs: build the Windows CUDA/CPU binaries in CI + gate the **PTY terminal** (Linux-only: `pty.rs` `/proc`, `killpg`) for Windows.
