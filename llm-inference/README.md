# Saient

A local AI desktop application built with Tauri v2. Runs entirely on your machine — no cloud, no API keys.

---

## Architecture

```
Saient (Tauri v2 desktop shell)
├── Frontend — Svelte 5 + TypeScript (Vite)
│   src/main.ts            — App entry point
│   src/App.svelte         — Root component, global CSS, event setup
│   src/lib/state.svelte.ts — Reactive state stores ($state runes)
│   src/lib/tauri.ts       — Typed Tauri invoke wrappers (all commands)
│   src/lib/events.ts      — Tauri event listeners → state
│   src/lib/types.ts       — Shared TypeScript types
│   src/lib/artifact.ts    — Artifact parser and iframe renderer
│   src/lib/format.ts      — Text formatting utilities
│   src/components/        — Svelte components (chat, screens, shared)
└── Backend — Rust (src-tauri/)
    ├── engine.rs  — Spawns and manages the tinyq4 inference server
    ├── gguf.rs    — GGUF v1/v2/v3 parser for model inspection
    ├── main.rs    — Tauri commands, event streaming, GPU stats, audit log
    ├── imggen.rs  — SDXL image generation (via Python subprocess)
    ├── tts.rs     — Kokoro TTS (via Python subprocess)
    ├── lora.rs    — LoRA trainer (via Python subprocess)
    ├── merge.rs   — Checkpoint merger (via Python subprocess)
    ├── resolve.rs — Runtime path discovery (Python, scripts, tinyq4)
    ├── tools/     — Agent filesystem, patch engine, sandbox
    ├── planner/   — Autonomous agent planner + verifier
    └── memory/    — Agent persistent memory store
```

**Inference backend:** [tinyq4](../tinyq4/) — a custom Rust GGUF inference server. CPU-only or CUDA, exposes `/health` + `/v1/chat/completions` SSE. Supports Q4_K, Q5_1, Q8_0, BF16, F16, F32. NOT llama.cpp.

---

## Features

| Feature | Description |
|---------|-------------|
| **Chat** | Streaming chat with any GGUF model. Artifact rendering (HTML/JS in iframe). Dual-agent drafter→critic mode. |
| **Image Gen** | SDXL/SD1.5 via diffusers. LoRA support. |
| **TTS** | Kokoro text-to-speech. Multiple voices. |
| **LoRA Trainer** | Fine-tune SDXL models on a local image dataset. |
| **Agent** | Autonomous file system agent with planner, sandbox execution, patch engine, and memory. |
| **Aria** | Goal-oriented agent runner with persistent plan state. |
| **Artifacts** | Model-generated HTML/JS rendered live in a sandboxed iframe. |

---

## Prerequisites

### Required

- **Rust** `>=1.75` — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Node.js** `>=18` — [nodejs.org](https://nodejs.org)
- **tinyq4** — build from `../data/llm-runtime/tinyq4/` or run `../scripts/build-engine.sh`:
  ```bash
  # CPU only
  cargo build --release
  # CUDA (RTX 40xx/50xx — set your compute cap)
  CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
  ```

### Optional (for Python features)

- Python venv managed by Saient at `../data/config/saient/venv/`
- Or set `PYTHON_PATH=/path/to/python` and `SCRIPTS_DIR=/path/to/scripts/`

---

## Running

```bash
cd llm-inference

# Install JS dependencies
npm install

# Development (hot reload)
CUDA_COMPUTE_CAP=89 npm run tauri dev

# Production build (.deb + .rpm)
CUDA_COMPUTE_CAP=89 npm run tauri build
```

The `.deb` package lands at `src-tauri/target/release/bundle/deb/`.

---

## Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| `npm run dev` | `vite` | Vite dev server only |
| `npm run build` | `vite build` | Frontend production build |
| `npm run check` | `svelte-check` | TypeScript + Svelte type checking |
| `npm run test` | `node src/artifact.test.js && node src/contract.test.js` | Unit + contract tests |
| `npm run asset:test` | `python3 tools/blender-pipeline/png_to_asset.py --self-test` | Smoke-test the PNG-to-relief-GLB pipeline; exits even if Blender is missing |
| `npm run asset:dry-run` | `python3 tools/blender-pipeline/png_to_asset.py --dry-run` | Scan `assets/source-png/` and show planned relief asset outputs |
| `npm run asset:build` | `python3 tools/blender-pipeline/png_to_asset.py` | Convert PNGs in `assets/source-png/` to prototype relief `.glb` files through Blender |
| `npm run local3d:setup` | `python3 tools/local-3d/setup_triposr.py` | Install isolated local TripoSR image-to-3D support |
| `npm run local3d:dry-run` | `python3 tools/local-3d/run_triposr.py --dry-run` | Show planned local image-to-3D GLB outputs |
| `npm run local3d:run` | `python3 tools/local-3d/run_triposr.py` | Generate real local image-to-3D `.glb` files from `assets/source-png/` |
| `npm run lint` | `svelte-check --fail-on-warnings` | Strict check (CI gate) |
| `npm run ci` | `check && test && build` | Full CI pipeline |

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TINYQ4_PATH` | auto-discovered | Override tinyq4 binary location |
| `SAIENT_DATA_DIR` | `../data` | Root for app-owned config, models, caches, and workspace |
| `SAIENT_MODELS_DIR` | `$SAIENT_DATA_DIR/models` | Managed model folder |
| `PYTHON_PATH` | managed venv, then `python3` | Python interpreter for imggen/TTS/LoRA |
| `SCRIPTS_DIR` | auto-discovered | Directory containing helper `.py` scripts |
| `CUDA_COMPUTE_CAP` | — | Required for CUDA builds (e.g. `89` for RTX 4090/5060 Ti) |

---

## Agent write mode

The agent can read files and run safe commands by default. To allow file writes, deletes, and arbitrary command execution, enable **Agent Write Mode** in the agent panel. A confirmation dialog is shown when enabling it.

All destructive agent actions (file writes, deletes, patches, command executions) are logged to `../data/share/saient/audit.jsonl` for review.

Safe read-only commands (always allowed): `ls`, `cat`, `head`, `tail`, `grep`, `find`, `wc`, `pwd`, `echo`, `file`, `stat`, `diff`, `which`, `env`, `git`, `cargo`, `python3`, `node`.

---

## Tests

```bash
# Rust unit tests (FsTool, PatchEngine, Planner)
cd src-tauri && cargo test

# JS tests (artifact parser + Tauri command-contract)
npm run test
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
