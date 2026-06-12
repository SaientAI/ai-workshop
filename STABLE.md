# Saient — Definition of Stable

The map for crossing the minefield. These are the **load-bearing flows**: if one of
them breaks, a user notices and the product looks broken. Everything else is internals —
ugly-but-working code is *mud, not a mine*. Don't bleed for mud.

**The rule:** run `./smoke.sh` **before** you change anything (prove the ground is clean)
and **after** (prove you didn't trip a wire). Fix one thing at a time and commit after every
green. You always want a known-good to roll back to.

```
./smoke.sh          # Tier 1 — build + wire + parse + syntax. No GPU. ~3–5 min.
./smoke.sh --deep   # Tier 1 + boots CUDA & the video daemon. GPU. +~1–2 min.
```

---

## The load-bearing flows

| # | Flow | "Working" means | Auto-covered by | Only a human can confirm |
|---|------|-----------------|-----------------|--------------------------|
| 1 | **App launches** | Window opens, no panic on boot | `cargo check`, `cargo test` (compile + logic) | The window actually appears |
| 2 | **£20 unlock** | A valid key activates; invalid is rejected | `cargo test` (`license.rs`, `auth.rs`) | A real key end-to-end + live Stripe |
| 3 | **Chat** | Load a model → tokens stream back | `cargo check`, contract test, `--deep` boots tinyq4 | An actual coherent reply |
| 4 | **Image gen** | SDXL produces a PNG | `py_compile`, gen-env import, script contract | Real pixels, not melted |
| 5 | **Video gen** | Wan/Cog produces a clip, no OOM | `py_compile`, script contract, `--deep` daemon-ready | A real clip + 720p on a clean card |
| 6 | **Vision** | Describes an image | `py_compile`, script contract | A sensible description |
| 7 | **TTS** | Kokoro speaks text | `py_compile`, script contract | Audio actually plays |
| 8 | **LoRA train / Merge** | Training + checkpoint merge run | `py_compile`, `cargo test` (`merge.rs`) | A usable trained LoRA |
| 9 | **Agent** | Plans, edits files in the sandbox safely | `cargo test` (`planner`, `sandbox`, `fs_tool`, `patch`) | A real multi-step task |
| 10 | **In-app updater** | Detects + applies a new version | `cargo test` (`update.rs`) | An update against the live server |
| 11 | **Command wiring** | Every UI `invoke()` ↔ a real Rust command | `contract.test.js` (bidirectional) | — (fully automated) |
| 12 | **Build integrity** | Rust compiles, frontend typechecks + builds | `cargo check`, `svelte-check`, `vite build` | — (fully automated) |

---

## What the net canNOT see — test these by hand before a release

The smoke-net proves the code **builds, wires up, and parses**. It does **not** prove the app
*feels* right. Before you ship, click through:

- [ ] App **launches** and every screen opens without a blank panel
- [ ] **Chat**: load a model, send a message, get a coherent streamed reply
- [ ] **Image**: generate one picture — it looks like the prompt, not noise
- [ ] **Video**: load the 5B, hit **HD 720p**, generate — clip plays, no OOM (it frees chat first)
- [ ] **£20 unlock**: a fresh key activates and unlocks the paid surface
- [ ] **Updater**: the in-app update path still detects the live `version.json`
- [ ] **TTS / Vision**: one round-trip each

A passing `./smoke.sh` means *"I didn't break the structure."* This checklist means
*"it actually works."* You need both.

---

## Triage (when smoke.sh goes red)

1. **Read the failed check name** — it points at the flow, not just a file.
2. **Roll back if lost.** `git stash` or check out your last green commit. A known-good beats
   a half-fixed mess every time.
3. **Fix one thing. Re-run. Commit.** Don't batch fixes — you lose the ability to tell which
   change cleared which mine.
4. **Messy ≠ broken.** If the net is green and a user can't see the ugliness, leave it. Cleanup
   for its own sake is how you step on a mine you didn't have to walk near.
