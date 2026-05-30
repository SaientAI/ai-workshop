# AI Workshop

**A local AI desktop app — chat, a coding agent, image generation, TTS, and LoRA training, all running on your own machine. No cloud, no API keys, no data leaving your computer.**

Built with Tauri + Svelte, powered by the [tinyq4](#inference-engine) GGUF inference engine.

---

## What it does

| | |
|---|---|
| 💬 **Chat** | Talk to local LLMs (GGUF). Renders live HTML **artifacts** in a side pane — ask it to build a tool or game and watch it run. |
| 🖥 **Kairo agent** | A real coding agent with a **PTY terminal** and a tool-use loop (`read · ls · write · edit · bash`). It plans, writes files, runs commands, and fixes its own errors — on *your* local model. |
| 🖼 **Image Gen** | SDXL text-to-image with LoRA support, schedulers, and live progress. |
| 🔊 **TTS** | Natural speech via Kokoro voices. |
| 🎛 **LoRA / Merge** | Train SDXL LoRAs and merge checkpoints, with a dataset cleaner. |

Everything is GPU-accelerated and fully offline.

---

## Quick start

### 1. Install
Grab the latest release (`.deb` / `.rpm` / `.AppImage`) from the [Releases](../../releases) page, or [build from source](#build-from-source).

### 2. First-run setup wizard
On first launch a wizard detects your system and sets everything up for you — **no dependency wrangling**:

- Reads your **GPU, driver, and CUDA version** and picks the matching PyTorch wheel automatically (the part that usually makes people rage-quit).
- Installs into a **managed Python venv** — it never touches your system Python.
- Offers two paths:
  - **Full setup** — Chat · Agent · Image · TTS · LoRA (installs the full ML stack, ~6 GB)
  - **Fast setup** — just Chat + Agent (small, ~1 minute)
- Optionally **downloads a starter model** for you (Qwen2.5-7B Instruct or Coder).

Already have everything? Hit **Skip** and it gets out of your way.

### 3. Start chatting
Pick a model in the sidebar → **Start server** → type. That's it.

> Need a model later? The sidebar has a **⬇ Download** button with curated, ready-to-run options — no need to hunt HuggingFace.

---

## Requirements

- **GPU:** NVIDIA recommended (CUDA). The app detects your CUDA version and installs a matched PyTorch — including **cu126/cu128 for RTX 50-series (Blackwell)**. No GPU works too (CPU fallback, slower).
- **OS:** Linux or Windows.
- **Python:** 3.10+ (only needed for Image/TTS/LoRA; the wizard builds its own venv).

The NVIDIA *driver* must already be installed (it's a kernel module) — the wizard checks for it and tells you if it's missing. Everything else it handles.

---

## Using it

### Keyboard shortcuts
Press <kbd>?</kbd> any time for the full list. Highlights:

| Shortcut | Action |
|---|---|
| <kbd>Ctrl</kbd>+<kbd>1</kbd>…<kbd>6</kbd> | Switch screen (Chat · Agent · Image · TTS · LoRA · Merge) |
| <kbd>Ctrl</kbd>+<kbd>Tab</kbd> | Cycle tabs within a screen |
| <kbd>Ctrl</kbd>+<kbd>K</kbd> | Clear the conversation |
| <kbd>Esc</kbd> | Stop generating |
| <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>K</kbd> | Toggle the Kairo agent |

### The Kairo agent
Open the **Agent** screen → **Terminal** tab and type `kairo` to launch the agent TUI, or use the **Planner** for autonomous runs. It works in a sandboxed workspace; **Write mode** (off by default) gates file writes and command execution, and destructive tools ask before running unless you flip on `/yolo`.

---

## Build from source

```bash
# prerequisites: Rust, Node 20+, and the Tauri Linux deps
#   sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev patchelf

cd llm-inference
npm install
npm run tauri dev      # run in dev
npm run tauri build    # produce installers
```

CI (GitHub Actions) runs type-checking, tests, and a Rust build on every push.

---

## Inference engine

The LLM backend is **[tinyq4](https://github.com/staticplayHub)** — a from-scratch Rust GGUF inference server (CPU or CUDA), **not llama.cpp**. It exposes an OpenAI-compatible `/v1/chat/completions` SSE endpoint and supports Q4_K, Q5_0/1, Q6_K, Q8_0, BF16, F16, F32, plus MoE (GPT-oss). Its CUDA GEMV kernels are hand-optimized — a 7B Q4 runs at ~50 tok/s and a 14B at ~30 tok/s on an RTX 5060 Ti.

Install it with `pip install tinyq4` (the setup wizard does this for you).

---

## Project layout

```
llm-inference/          the Tauri app
  src/                  Svelte 5 + TypeScript frontend
  src-tauri/src/        Rust backend (engine, agent, setup wizard, PTY, …)
scripts/                Python helpers for Image/TTS/LoRA (run in the managed venv)
```

See [`llm-inference/README.md`](llm-inference/README.md) for the full architecture breakdown.

---

## License

Apache-2.0.
