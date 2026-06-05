#!/usr/bin/env bash
# build-engine.sh — build the tinyq4 inference engine (CUDA + CPU) and stage it,
# with its CUDA runtime lib, into the Tauri app's bundle resources.
#
# The app ships these binaries and selects CUDA at runtime when an NVIDIA driver is
# present (libcudart is bundled, so no system CUDA toolkit is needed on the user's box).
#
# Usage:   scripts/build-engine.sh
# Env:     TINYQ4_SRC   path to the tinyq4 source repo   (default: ~/llm-runtime/tinyq4)
#          CUDA_HOME    CUDA toolkit for the CUDA build  (default: /usr/local/cuda-12.8)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # llm-inference-tauri/
TINYQ4_SRC="${TINYQ4_SRC:-$HOME/llm-runtime/tinyq4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
DEST="$HERE/llm-inference/src-tauri/resources/engine"
mkdir -p "$DEST"

echo "==> tinyq4 source: $TINYQ4_SRC"
echo "==> staging into:  $DEST"
cd "$TINYQ4_SRC"

# CPU build (always — the universal fallback)
echo "==> building tinyq4 (CPU)…"
CARGO_TARGET_DIR=/tmp/tq4-cpu cargo build --release
cp /tmp/tq4-cpu/release/tinyq4 "$DEST/tinyq4-cpu"
strip "$DEST/tinyq4-cpu" || true

# CUDA build (skipped if no nvcc; the app then falls back to the CPU binary)
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "==> building tinyq4 (CUDA, $CUDA_HOME)…"
  CUDA_HOME="$CUDA_HOME" cargo build --release --features cuda
  cp target/release/tinyq4 "$DEST/tinyq4-cuda"
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
