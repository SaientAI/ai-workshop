#!/usr/bin/env python3
"""
merge_checkpoints.py  —  Blend two (or three) diffusion model checkpoints.

Memory-efficient: uses lazy tensor loading (safe_open) for .safetensors files
so only one tensor from each model is in RAM at a time.  Peak usage ≈ output
size rather than sum of all inputs.

Reads a JSON config from argv[1].
Emits JSON lines to stdout:
  {"type":"log","msg":"..."}
  {"type":"progress","done":N,"total":N}
  {"type":"done","output":"/path/to/merged.safetensors"}
  {"type":"error","msg":"..."}

Methods:
  weighted_sum   output = A*(1-w) + B*w
  add_difference output = A + (B - C)*w
"""

import sys, os, json, signal

if len(sys.argv) < 2:
    print(json.dumps({"type": "error", "msg": "Usage: merge_checkpoints.py <config.json>"}), flush=True)
    sys.exit(1)

with open(sys.argv[1]) as f:
    cfg = json.load(f)

def emit(obj):
    print(json.dumps(obj), flush=True)

def log(msg):
    emit({"type": "log", "msg": str(msg)})

def err(msg):
    emit({"type": "error", "msg": str(msg)})
    sys.exit(1)

_cancelled = False
def _handle_term(sig, frame):
    global _cancelled
    _cancelled = True
signal.signal(signal.SIGTERM, _handle_term)

# ── Config ─────────────────────────────────────────────────────────────────────

path_a   = cfg["model_a"]
path_b   = cfg["model_b"]
path_c   = cfg.get("model_c") or None
method   = cfg.get("method", "weighted_sum")
weight   = float(cfg.get("weight", 0.5))
out_path = cfg["output_path"]

# ── Validate ───────────────────────────────────────────────────────────────────

try:
    from safetensors.torch import save_file
    from safetensors import safe_open
except ImportError:
    err("safetensors not installed — run: pip install safetensors")

# ── Lazy accessor ──────────────────────────────────────────────────────────────

class LazySD:
    """Wraps safe_open (lazy) for .safetensors or a preloaded dict for .ckpt/.pt.
    Keeps only one tensor in memory at a time when backed by safe_open."""

    def __init__(self, path):
        if not os.path.exists(path):
            err(f"File not found: {path}")
        log(f"Opening {os.path.basename(path)} …")
        if path.endswith(".safetensors"):
            self._sf   = safe_open(path, framework="pt", device="cpu")
            self._dict = None
            self._keys = set(self._sf.keys())
        else:
            import torch
            log(f"  (loading .ckpt fully — only .safetensors supports lazy loading)")
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                if "state_dict" in data: data = data["state_dict"]
                elif "model" in data:    data = data["model"]
            self._sf   = None
            self._dict = data
            self._keys = set(data.keys())

    def keys(self):
        return self._keys

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        return self._sf.get_tensor(key) if self._sf else self._dict[key]

    def get(self, key, default=None):
        return self[key] if key in self._keys else default

# ── Open models (lazy) ─────────────────────────────────────────────────────────

try:
    sd_a = LazySD(path_a)
    sd_b = LazySD(path_b)
    sd_c = LazySD(path_c) if path_c else None
except SystemExit:
    raise
except Exception as e:
    err(f"Failed to open model: {e}")

if method == "add_difference" and sd_c is None:
    err("Model C is required for the add_difference method.")

keys_a = sd_a.keys()
keys_b = sd_b.keys()
shared = sorted(keys_a & keys_b)
only_a = sorted(keys_a - keys_b)
only_b = sorted(keys_b - keys_a)

log(f"Model A: {len(keys_a)} tensors  |  Model B: {len(keys_b)} tensors")
log(f"Shared: {len(shared)}  |  A-only: {len(only_a)}  |  B-only: {len(only_b)}")
log(f"Method: {method}  |  weight: {weight:.3f}")

# ── Merge (one tensor at a time) ───────────────────────────────────────────────

import torch

total  = len(shared) + len(only_a) + len(only_b)
result = {}
done   = 0

for key in shared:
    if _cancelled:
        log("Cancelled by user.")
        sys.exit(0)

    ta = sd_a[key]
    tb = sd_b[key]

    if ta.shape != tb.shape:
        result[key] = ta
    elif method == "weighted_sum":
        merged = ta.float() * (1.0 - weight) + tb.float() * weight
        result[key] = merged.to(ta.dtype)
    elif method == "add_difference":
        tc = sd_c.get(key, ta) if sd_c else ta
        if tc.shape != ta.shape:
            tc = ta
        merged = ta.float() + (tb.float() - tc.float()) * weight
        result[key] = merged.to(ta.dtype)
    else:
        result[key] = ta

    done += 1
    if done % 100 == 0 or done == total:
        emit({"type": "progress", "done": done, "total": total})

for key in only_a:
    if _cancelled: log("Cancelled by user."); sys.exit(0)
    result[key] = sd_a[key]
    done += 1
    if done % 100 == 0 or done == total:
        emit({"type": "progress", "done": done, "total": total})

for key in only_b:
    if _cancelled: log("Cancelled by user."); sys.exit(0)
    result[key] = sd_b[key]
    done += 1
    if done % 100 == 0 or done == total:
        emit({"type": "progress", "done": done, "total": total})

log(f"Merged {len(shared)} shared + {len(only_a)} A-only + {len(only_b)} B-only tensors")

# ── Save ───────────────────────────────────────────────────────────────────────

log(f"Saving to {out_path} …")
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

try:
    save_file(result, out_path)
except Exception as e:
    err(f"Failed to save: {e}")

emit({"type": "done", "output": out_path})
