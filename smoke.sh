#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Saient smoke-net — the mine detector.
#
#   Run BEFORE you change anything (prove the ground is clean) and AFTER (prove you
#   didn't trip a wire). Green = the load-bearing flows still hold. See STABLE.md.
#
#   ./smoke.sh          Tier 1: build + wire + parse + syntax. No GPU. ~3-5 min.
#   ./smoke.sh --deep   Tier 1 + boot CUDA & the video daemon. GPU.  +~1-2 min.
#   ./smoke.sh --help
#
# What it can NOT see (test by hand — STABLE.md): the window opening, a real chat
# reply, real pixels, a live £20 key, the updater hitting the live server.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/llm-inference"
TAURI="$APP/src-tauri"
SCRIPTS="$ROOT/scripts"
DATA="${SAIENT_DATA_DIR:-$ROOT/data}"
CONFIG="${SAIENT_CONFIG_DIR:-$DATA/config/saient}"
CONFIG_DEV="$DATA/config/saient-dev"
MODELS="${SAIENT_MODELS_DIR:-$DATA/models}"
export SAIENT_DATA_DIR="$DATA"
export SAIENT_CONFIG_DIR="$CONFIG"
export SAIENT_MODELS_DIR="$MODELS"

DEEP=0
case "${1:-}" in
  --deep) DEEP=1 ;;
  --help|-h) sed -n '3,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
  "") ;;
  *) echo "unknown arg: $1 (try --help)"; exit 2 ;;
esac

# colours only on a tty
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; X=$'\033[0m'
else G=""; R=""; Y=""; B=""; X=""; fi

PASS=0; FAIL=0; WARN=0; FAILED=()
ok()    { printf "  ${G}✓${X} %s\n" "$1"; PASS=$((PASS+1)); }
bad()   { printf "  ${R}✗${X} %s\n" "$1"; FAIL=$((FAIL+1)); FAILED+=("$1"); }
warn()  { printf "  ${Y}!${X} %s ${Y}(warn)${X}\n" "$1"; WARN=$((WARN+1)); }
hdr()   { printf "\n${B}── %s${X}\n" "$1"; }
detail(){ echo "$1" | tail -6 | sed 's/^/      /'; }

# gate "label" cmd...  → ok/bad on exit code; show last lines on failure
gate() {
  local label="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ $rc -eq 0 ]; then ok "$label"; else bad "$label"; detail "$out"; fi
}

PY="${PYTHON_PATH:-$CONFIG/venv/bin/python}"
[ -x "$PY" ] || PY="$CONFIG_DEV/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"

printf "${B}Saient smoke-net${X}  (%s)\n" "$([ $DEEP = 1 ] && echo 'Tier 1 + deep' || echo 'Tier 1')"

# ── Cheap, high-signal checks first (fail in <2s on the dumb stuff) ────────────
hdr "Code ↔ resource contracts"
# Helper scripts the Rust spawns by name. The two video daemons are picked via a
# variable (find_script(script_name)) so a literal grep misses them — assert the known
# runtime set explicitly, then union with every literal find_script("...") reference.
missing=""
mapfile -t WANT < <( {
  printf '%s\n' generate_video.py generate_cogvideo.py enhance_video.py \
                generate_sdxl.py tts_kokoro.py vision.py
  grep -rhoE 'find_script\("[^"]+"' "$TAURI/src" 2>/dev/null | sed -E 's/.*find_script\("//; s/"$//'
} | sort -u )
for s in "${WANT[@]}"; do [ -f "$SCRIPTS/$s" ] || missing+=" $s"; done
if [ -z "$missing" ]; then ok "All ${#WANT[@]} referenced helper scripts present"
else bad "Missing helper scripts:$missing"; fi

hdr "Python integrity"
if [ -n "$PY" ] && [ -x "$PY" ]; then
  gate "Helper scripts compile (py_compile)" "$PY" -m py_compile "$SCRIPTS"/*.py
  gate "Gen env imports (torch/diffusers/transformers/peft/video export)" "$PY" -c \
       "import torch,diffusers,transformers,safetensors,peft,imageio,imageio_ffmpeg"
else
  bad "No Python interpreter found (set PYTHON_PATH or install python3)"
fi

hdr "Frontend ↔ Rust contract + parsers"
gate "Command wiring (contract.test.js)" bash -c "cd '$APP' && node src/contract.test.js"
gate "Artifact parser (artifact.test.js)"  bash -c "cd '$APP' && node src/artifact.test.js"

# ── Slower compile/build checks ───────────────────────────────────────────────
hdr "Build integrity"
gate "Frontend typecheck (svelte-check)" bash -c "cd '$APP' && npm run --silent check"
gate "Frontend builds (vite build)"       bash -c "cd '$APP' && npm run --silent build"
gate "Rust compiles (cargo check)"        bash -c "cd '$TAURI' && cargo check -q"
gate "Rust unit tests (cargo test)"       bash -c "cd '$TAURI' && cargo test -q"

# ── Environment (warn-only — env drift, not code breakage) ────────────────────
hdr "Environment (warn-only)"
[ -n "$PY" ] && [ -x "$PY" ] && ok "Python: $PY" || warn "Python interpreter missing"
# Mirror engine.rs find_tinyq4: bundled resources/engine/tinyq4-{cuda,cpu}, the managed
# venv (saient / saient-dev), or a pip install on PATH/.local/bin.
if ls "$TAURI"/resources/engine/tinyq4-* >/dev/null 2>&1 \
   || ls "$TAURI"/target/*/resources/engine/tinyq4-* >/dev/null 2>&1 \
   || [ -x "$CONFIG/venv/bin/tinyq4" ] \
   || [ -x "$CONFIG_DEV/venv/bin/tinyq4" ] \
   || command -v tinyq4 >/dev/null 2>&1 || [ -x "$HOME/.local/bin/tinyq4" ]; then
  ok "tinyq4 engine binary present"
else warn "tinyq4 engine not found (chat needs bundled tinyq4-cuda/-cpu or a pip install)"; fi
[ -f "$CONFIG/upscale/RealESRGAN_x2plus.pth" ] \
  && ok "RealESRGAN upscale weights present" \
  || warn "Upscale weights missing (video Enhance ▸ Upscale)"
[ -d "$MODELS" ] \
  && ok "Model directory present" || warn "No model directory found"

# ── Deep tier: boot real components (GPU) ─────────────────────────────────────
if [ "$DEEP" = 1 ]; then
  hdr "Deep — boots real components (GPU)"
  gate "CUDA visible to torch" "$PY" -c "import torch; assert torch.cuda.is_available()"
  M="$MODELS/projects-models/wan/Wan2.2-TI2V-5B-Diffusers"
  [ -d "$M" ] || M="$MODELS/wan/Wan2.2-TI2V-5B-Diffusers"
  if [ -d "$M" ]; then
    printf "  ${B}…${X} booting video daemon (load-only, may take ~1 min)\n"
    out="$(printf '{"model_path":"%s","device":"cuda","precision":"fast"}\n' "$M" \
           | timeout 180 "$PY" "$SCRIPTS/generate_video.py" 2>/dev/null || true)"
    if echo "$out" | grep -q '"ready"'; then ok "Video daemon loads to ready (5B)"
    else bad "Video daemon did NOT reach ready (5B)"; detail "$out"; fi
  else
    warn "5B model absent — skipped video daemon boot"
  fi
fi

# ── Verdict ───────────────────────────────────────────────────────────────────
hdr "Result"
printf "  ${G}%d passed${X}" "$PASS"
[ $WARN -gt 0 ] && printf " · ${Y}%d warn${X}" "$WARN"
[ $FAIL -gt 0 ] && printf " · ${R}%d FAILED${X}" "$FAIL"
printf "\n"
if [ $FAIL -gt 0 ]; then
  printf "\n${R}${B}✗ MINE TRIPPED${X} — load-bearing checks failed:\n"
  for f in "${FAILED[@]}"; do printf "    ${R}•${X} %s\n" "$f"; done
  printf "  Fix one, re-run, commit. Lost? Roll back to your last green.\n\n"
  exit 1
else
  printf "\n${G}${B}✓ STABLE${X} — load-bearing flows hold. Safe to commit.\n"
  [ $WARN -gt 0 ] && printf "  (%d env warning(s) above — not code breakage, worth a glance.)\n" "$WARN"
  printf "\n"
  exit 0
fi
