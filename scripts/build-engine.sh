#!/usr/bin/env bash
# build-engine.sh — build the tinyq4 inference engine (CUDA + CPU) and stage it,
# with its CUDA runtime lib, into the Tauri app's bundle resources.
#
# The app ships these binaries and selects CUDA at runtime when an NVIDIA driver is
# present (libcudart is bundled, so no system CUDA toolkit is needed on the user's box).
#
# Usage:   scripts/build-engine.sh
# Env:     TINYQ4_SRC   path to the tinyq4 source repo   (default: ./data/llm-runtime/tinyq4)
#          CUDA_HOME    CUDA toolkit for the CUDA build  (default: /usr/local/cuda-12.8)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # llm-inference-tauri/
TINYQ4_SRC="${TINYQ4_SRC:-$HERE/data/llm-runtime/tinyq4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
DEST="$HERE/llm-inference/src-tauri/resources/engine"
mkdir -p "$DEST"

echo "==> tinyq4 source: $TINYQ4_SRC"
echo "==> staging into:  $DEST"
cd "$TINYQ4_SRC"

# CPU build (always — the universal fallback)
echo "==> building tinyq4 (CPU)…"
CARGO_TARGET_DIR=/tmp/tq4-cpu cargo build --release
# The engine crate was renamed tinyq4 -> quartz; CI checks out the renamed
# repo while local copies may still be the old name. Accept either.
TQ4_CPU_BIN="$(ls /tmp/tq4-cpu/release/quartz /tmp/tq4-cpu/release/tinyq4 2>/dev/null | head -1)"
[ -n "$TQ4_CPU_BIN" ] || { echo "no engine binary in /tmp/tq4-cpu/release"; ls /tmp/tq4-cpu/release; exit 1; }
cp "$TQ4_CPU_BIN" "$DEST/tinyq4-cpu"
strip "$DEST/tinyq4-cpu" || true

# CUDA build (skipped if no nvcc; the app then falls back to the CPU binary)
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "==> building tinyq4 (CUDA, $CUDA_HOME)…"
  CUDA_HOME="$CUDA_HOME" cargo build --release --features cuda
  TQ4_CUDA_BIN="$(ls target/release/quartz target/release/tinyq4 2>/dev/null | head -1)"
  [ -n "$TQ4_CUDA_BIN" ] || { echo "no engine binary in target/release"; ls target/release; exit 1; }
  cp "$TQ4_CUDA_BIN" "$DEST/tinyq4-cuda"
  strip "$DEST/tinyq4-cuda" || true
  # Bundle libcudart so the binary loads without a system CUDA install.
  CUDART="$(ldd "$DEST/tinyq4-cuda" | grep -oP '=> \K[^ ]*libcudart\.so\.12' | head -1)"
  CUDART="${CUDART:-$CUDA_HOME/targets/x86_64-linux/lib/libcudart.so.12}"
  cp -L "$CUDART" "$DEST/libcudart.so.12"
  echo "==> bundled $(basename "$CUDART") ($(du -h "$DEST/libcudart.so.12" | cut -f1))"
else
  echo "!! no nvcc at $CUDA_HOME — skipping CUDA build (CPU-only bundle)"
fi

echo "==> done. Engine staged:"
ls -lh "$DEST"

# ── Windows ─────────────────────────────────────────────────────────────────────
# Build on a Windows box (or cross-compile) with the same two profiles and drop:
#   tinyq4-cuda.exe, tinyq4-cpu.exe, cudart64_12.dll   into the same resources/engine dir.
# The app uses PATH (instead of LD_LIBRARY_PATH) to locate cudart64_12.dll at spawn.
