#!/usr/bin/env python3
# Wan text-to-video daemon — STAGED loader (fits a 16 GB card / 40 GB box).
#
# Protocol (unchanged — Rust/UI don't care how we load):
#   stdin line 1 (load): {"model_path": "...", "device": "auto|cuda|cpu"}
#       -> {"loading_status": "..."} lines, then {"ready": true, "device": "cuda"}
#   then one request per line:
#       {"prompt","neg_prompt","num_frames","steps","cfg_scale","width","height","fps","seed"}
#       -> {"step":i,"total":t} progress lines, then {"base64_mp4":"...","frames":N,"elapsed":s}
#
# WHY STAGED:
#   The naive `DiffusionPipeline.from_pretrained(... PipelineQuantizationConfig)`
#   reads the ENTIRE model into CPU RAM in bf16 (~36 GB — the UMT5-XXL text
#   encoder alone is ~21 GB) and only quantizes once weights reach the GPU. On a
#   40 GB box that tips into swap and freezes the machine before anything loads.
#
#   Instead we never hold two heavy components in fp16 at once:
#     LOAD     -> transformer (4-bit) + VAE + tokenizer + scheduler, GPU-resident,
#                 pipeline built with text_encoder=None. Small + fast.
#     GENERATE -> load the text encoder 4-bit STRAIGHT onto the GPU (~5.5 GB VRAM,
#                 RAM stays flat), encode the prompt, then DELETE it + empty the
#                 CUDA cache before denoising with the resident transformer.
#   Peak VRAM ~8 GB (during encode), peak RAM < ~10 GB. No 27 GB CPU read, ever.
import base64, gc, io, json, os, re, signal, sys, tempfile, threading, time, traceback
from saient_paths import cache_dir, configure_hf_cache

configure_hf_cache()

# Reduce CUDA fragmentation so the RESIDENT daemon survives many generations.
# Without this, reserved-but-idle blocks from earlier denoise runs pile up and a
# later 5.5 GB text-encoder load OOMs even with "free" VRAM. Must be set before
# torch initialises CUDA — torch is imported lazily inside functions, so here is fine.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _watch_parent():
    """Self-terminate if the parent app dies, else this daemon orphans and holds
    ~5 GB of VRAM forever. getppid() flips (reparented to init) when the app exits —
    even on a hard crash — covering the case stdin-EOF misses (e.g. mid-generation)."""
    import threading, time
    ppid = os.getppid()
    def loop():
        while True:
            time.sleep(2)
            if os.getppid() != ppid or ppid == 1:
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()

PIPE = None        # WanPipeline with text_encoder=None, transformer+vae resident
TOKENIZER = None   # kept resident (tiny); text encoder is loaded per-encode
MODEL_PATH = ""
# Cache the ENCODED prompt embeddings keyed by (prompt, neg, cfg).
# Re-running the same prompt (seed/step/frame sweeps) then skips the ~90s text-
# encoder load entirely. We free the 5.5 GB encoder after each encode so it never
# competes with denoise activations — embeds are cached on CPU so long clips do
# not carry duplicate GPU copies through denoise/decode.
EMBED_CACHE = {"key": None, "pe": None, "ne": None}
I2V_PIPE = None    # WanImageToVideoPipeline, lazily assembled from the t2v components
# "fast" (default) = 4-bit transformer, GPU-resident — today's behaviour, unchanged.
# "quality" = bf16 transformer that LIVES in CPU RAM and is streamed onto the GPU only
# for the denoise loop. We deliberately avoid parking it back immediately after denoise
# (decode tries with weights resident). We only park (a) before a new TE encode or (b) on
# decode OOM. This reduces PCIe round-trips and targets 30s–1m gens. Set via "precision".
STREAM_TRANSFORMER = False
LORA_PATH = ""
LORA_STRENGTH = 1.0
SCHEDULER_BASE_CONFIG = None
MODEL_INDEX_CONFIG = {}
DUAL_EXPERT = False
TRANSFORMER2_CACHE_PATH = None
# Big single-transformer models (14B nf4 ~8 GB) can't hold the transformer AND the 5.5 GB
# UMT5 text encoder on a 16 GB card at once — encoding on top of the resident transformer
# once hard-froze the display GPU. When PARK_TRANSFORMER is set (decided at load by VRAM
# budget), generate() drops the nf4 transformer off the GPU for the text-encode and RELOADS
# it from the packed nf4 cache for denoise, so only one heavy piece is ever GPU-resident.
NF4_CACHE_PATH = None   # dir of the packed 4-bit transformer, for reload-after-encode
RESIDENT_GB = 0.0       # VRAM resident after load (≈ transformer) — gates PARK_TRANSFORMER
PARK_TRANSFORMER = False
LOW_VRAM_ACTIVE = False
LOW_VRAM_ATTN_BACKEND = None
VRAM_CAP_FRACTION = None
# Per-request "park to RAM": stream the transformer block-by-block from system RAM so a
# clip that overflows VRAM (e.g. native 5s@720p on the 16 GB card) fits. Slow (~160s/step
# vs ~69s), so it's opt-in per generation via req["block_offload"], NOT a launch env flag.
BLOCK_OFFLOAD_REQ = False
DENOISE_CACHE_MODE = "off"
DENOISE_CACHE_THRESHOLD = 0.10
VRAM_REPORT = {}
EMIT_LOCK = threading.Lock()


def emit(obj):
    # Preview workers emit alongside the denoise thread. Keep each JSON line atomic.
    with EMIT_LOCK:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _vram_limit_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    frac = VRAM_CAP_FRACTION
    if frac is None:
        frac = max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.98")), 0.98))
    return total * frac


def _vram_stage(stage, reset_peak=True, emit_line=True):
    """Record current/peak CUDA allocation for the acceptance-stage report."""
    import torch
    if not torch.cuda.is_available():
        stats = {"used": 0.0, "reserved": 0.0, "peak": 0.0, "free": 0.0, "limit": 0.0}
    else:
        free, total = torch.cuda.mem_get_info()
        stats = {
            "used": torch.cuda.memory_allocated() / 2**30,
            "reserved": torch.cuda.memory_reserved() / 2**30,
            "peak": torch.cuda.max_memory_allocated() / 2**30,
            "free": free / 2**30,
            "limit": _vram_limit_gb() or (total / 2**30),
        }
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
    VRAM_REPORT[stage] = stats
    if emit_line:
        headroom = max(stats["limit"] - stats["peak"], 0.0)
        emit({"loading_status": (
            f"VRAM {stage}: used {stats['used']:.2f} GB · peak {stats['peak']:.2f} GB · "
            f"reserved {stats['reserved']:.2f} GB · cap headroom {headroom:.2f} GB"
        )})
    return stats


def _emit_vram_report():
    order = ["model load", "text encoding", "denoising", "VAE decode", "video assembly", "cleanup"]
    lines = []
    for stage in order:
        s = VRAM_REPORT.get(stage)
        if not s:
            continue
        headroom = max(s["limit"] - s["peak"], 0.0)
        lines.append(f"{stage}: peak {s['peak']:.2f} GB, used {s['used']:.2f} GB, headroom {headroom:.2f} GB")
    if lines:
        emit({"loading_status": "VRAM report · " + " · ".join(lines)})


def _emit_video_result(b64, frames_count, elapsed, extended=False):
    _free_cuda()
    _vram_stage("video assembly")
    _free_cuda()
    _vram_stage("cleanup")
    _emit_vram_report()
    obj = {"base64_mp4": b64, "frames": frames_count, "elapsed": round(elapsed, 1)}
    if extended:
        obj["extended"] = True
    emit(obj)


def _set_attention_backend(model, low_vram):
    if model is None:
        return
    try:
        if low_vram:
            preferred = os.environ.get("SAIENT_WAN_ATTN_BACKEND", LOW_VRAM_ATTN_BACKEND or "_native_flash")
            for backend in dict.fromkeys((preferred, "_native_flash", "_native_efficient", "native")):
                try:
                    model.set_attention_backend(backend)
                    return
                except Exception:
                    continue
            raise RuntimeError("no low-VRAM attention backend accepted")
        else:
            model.reset_attention_backend()
    except Exception as e:
        if low_vram:
            emit({"loading_status": f"  ⚠ low-VRAM attention backend unavailable ({type(e).__name__}); using default"})


def _configure_denoise_cache(model, label="transformer"):
    """Enable Diffusers' native First Block Cache without consuming a LoRA slot."""
    if model is None:
        return False
    try:
        enabled = bool(getattr(model, "is_cache_enabled", False))
        current = getattr(model, "_cache_config", None)
        current_threshold = getattr(current, "threshold", None)
        wanted = DENOISE_CACHE_MODE == "balanced"
        if not wanted:
            if enabled:
                model.disable_cache()
            return False
        if enabled and current_threshold is not None and abs(float(current_threshold) - DENOISE_CACHE_THRESHOLD) < 1e-6:
            model._reset_stateful_cache()
            return True
        if enabled:
            model.disable_cache()
        from diffusers.hooks import FirstBlockCacheConfig
        model.enable_cache(FirstBlockCacheConfig(threshold=DENOISE_CACHE_THRESHOLD))
        emit({"loading_status": (
            f"denoise cache: First Block Cache on {label} · threshold {DENOISE_CACHE_THRESHOLD:g} "
            "· uses no LoRA slot"
        )})
        return True
    except Exception as e:
        emit({"loading_status": (
            f"  ⚠ denoise cache unavailable on {label} ({type(e).__name__}: {e}); running exact"
        )})
        return False


def _release_denoise_cache(model):
    """Drop cached block residuals before VAE decode so they do not hold VRAM."""
    if model is None or not bool(getattr(model, "is_cache_enabled", False)):
        return
    try:
        model.disable_cache()
    except Exception as e:
        emit({"loading_status": f"  ⚠ denoise cache cleanup failed ({type(e).__name__}: {e})"})


def _env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _raise_highres_vram_cap(enabled):
    if not enabled or not _env_flag("SAIENT_WAN_HIGHRES_VRAM_CAP", True):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        global VRAM_CAP_FRACTION
        current = VRAM_CAP_FRACTION
        if current is None:
            current = max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.98")), 0.98))
        target = max(current, float(os.environ.get("SAIENT_WAN_HIGHRES_VRAM_FRACTION", "0.98")))
        target = max(0.10, min(target, 0.98))
        if target <= current + 1e-4:
            return False
        torch.cuda.set_per_process_memory_fraction(target, 0)
        VRAM_CAP_FRACTION = target
        emit({"loading_status": f"high-res headroom: VRAM cap raised to {target:.0%} for 720p denoise"})
        return True
    except Exception as e:
        emit({"loading_status": f"  ⚠ high-res VRAM cap bump unavailable ({e})"})
        return False


def _rope_target_dtype():
    import torch
    raw = os.environ.get("SAIENT_WAN_ROPE_DTYPE", "bf16").strip().lower()
    if raw in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def _cast_rope_for_low_vram(model, label="transformer"):
    if model is None or not LOW_VRAM_ACTIVE or not _env_flag("SAIENT_WAN_ROPE_CAST", True):
        return False
    rope = getattr(model, "rope", None)
    target_dtype = _rope_target_dtype()
    marker = f"_saient_rope_{str(target_dtype).split('.')[-1]}"
    if rope is None or getattr(rope, marker, False):
        return False
    try:
        import torch
        saved = 0
        for name in ("freqs_cos", "freqs_sin"):
            t = getattr(rope, name, None)
            if torch.is_tensor(t) and t.is_floating_point() and t.dtype != target_dtype:
                new = t.to(dtype=target_dtype)
                saved += max(t.numel() * (t.element_size() - new.element_size()), 0)
                if hasattr(rope, "_buffers") and name in rope._buffers:
                    rope._buffers[name] = new
                else:
                    setattr(rope, name, new)
        setattr(rope, marker, True)
        if saved:
            _free_cuda()
            dtype_name = str(target_dtype).split(".")[-1]
            emit({"loading_status": (
                f"low-VRAM: {label} rotary tables {dtype_name} "
                f"(saved {saved / 2**20:.0f} MiB; smaller rope workspace)"
            )})
            return True
    except Exception as e:
        emit({"loading_status": f"  ⚠ low-VRAM rotary cast unavailable for {label} ({type(e).__name__}: {e})"})
    return False


def _denoise_latent_dtype(transformer_dtype, height, width, num_frames):
    import torch
    if not LOW_VRAM_ACTIVE or not _env_flag("SAIENT_WAN_LATENTS_BF16", True):
        return torch.float32
    if transformer_dtype not in (torch.float16, torch.bfloat16):
        return torch.float32
    high_res_pixels = int(os.environ.get("SAIENT_WAN_HIGHRES_PIXELS", "500000"))
    if height * width < high_res_pixels:
        return torch.float32
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = max(height // 8, 1)
    latent_w = max(width // 8, 1)
    saved_mib = (16 * latent_frames * latent_h * latent_w * 2) / 2**20
    emit({"loading_status": (
        f"low-VRAM: high-res denoise latents use {str(transformer_dtype).split('.')[-1]} "
        f"(saves ~{saved_mib:.0f} MiB vs fp32 master)"
    )})
    return transformer_dtype


def _model_resident_bytes(model):
    """Sum of parameter + buffer bytes = how much RAM this model occupies when parked."""
    total = 0
    try:
        for p in model.parameters():
            total += p.numel() * p.element_size()
        for b in model.buffers():
            total += b.numel() * b.element_size()
    except Exception:
        return 0
    return total


def _mem_available_bytes():
    """Physical RAM currently available (MemAvailable), EXCLUDING swap. 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _ram_can_park(model):
    """Guard against the swap-thrash freeze: parking a model bigger than physical RAM
    forces the OS into swap and locks the whole box solid (the fp32 14B → 54 GB in a
    39 GB box was a 3-hour hard-freeze). Returns (ok, model_gb, avail_gb, msg).

    We require the parked weights PLUS a 15% offload overhead PLUS 3 GB OS headroom to
    fit inside MemAvailable (physical RAM only — swap does NOT count as fittable)."""
    model_b = _model_resident_bytes(model)
    avail_b = _mem_available_bytes()
    gb = 2**30
    if model_b <= 0 or avail_b <= 0:
        # Can't measure — refuse rather than risk the freeze.
        return (False, model_b / gb, avail_b / gb,
                "cannot measure RAM/model size; refusing to park (would risk a swap freeze)")
    needed_b = model_b * 1.15 + 3 * gb
    ok = needed_b <= avail_b
    return (ok, model_b / gb, avail_b / gb,
            f"parked model ≈{model_b/gb:.1f} GB + overhead needs ≈{needed_b/gb:.1f} GB, "
            f"but only {avail_b/gb:.1f} GB physical RAM is free")


def _enable_transformer_group_offload(model, label="transformer"):
    """Low-VRAM mode: keep only active transformer blocks on the GPU during forward.

    This preserves weights, but it gridlocks this desktop setup in practice. It now
    requires both SAIENT_WAN_GROUP_OFFLOAD=1 and SAIENT_WAN_ALLOW_SLOW_BLOCK_OFFLOAD=1
    so normal low-VRAM mode and routine tests cannot enter this path.
    """
    # Allow either the per-request toggle (block_offload) OR the two legacy launch env
    # flags. The request toggle is the shipped path (a "park to RAM" checkbox / HD-5s-max
    # preset); the env flags stay for CLI/testing.
    allow_offload = BLOCK_OFFLOAD_REQ or (
        _env_flag("SAIENT_WAN_GROUP_OFFLOAD", False)
        and _env_flag("SAIENT_WAN_ALLOW_SLOW_BLOCK_OFFLOAD", False)
    )
    if model is None or not LOW_VRAM_ACTIVE or not allow_offload:
        return False
    if getattr(model, "_saient_group_offload_enabled", False):
        return True
    # HARD SAFETY GUARD: never park a model that won't fit in physical RAM. Parking into
    # swap thrashes the disk and freezes the whole machine (the fp32 14B freeze). Refuse
    # loudly instead — generate() turns this into a clean, actionable error.
    ok, model_gb, avail_gb, why = _ram_can_park(model)
    if not ok:
        emit({"loading_status": (
            f"  ✗ block offload REFUSED for {label}: {why}. "
            f"Use 480p + upscale for 5s@720p instead of parking."
        )})
        raise RuntimeError(
            f"block offload would freeze the PC: {why}. "
            f"This box can't park a {model_gb:.0f} GB model in {avail_gb:.0f} GB free RAM — "
            f"generate at 480p and upscale to 720p instead."
        )
    try:
        import torch
        blocks = max(1, int(os.environ.get("SAIENT_WAN_GROUP_OFFLOAD_BLOCKS", "1")))
        # Force sync offload: async streaming pins host memory (non-swappable) and makes
        # the freeze worse. The env flag is intentionally ignored for the shipped toggle.
        use_stream = False
        low_cpu = _env_flag("SAIENT_WAN_GROUP_OFFLOAD_LOW_CPU", True)
        model.enable_group_offload(
            onload_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=blocks,
            use_stream=use_stream,
            low_cpu_mem_usage=low_cpu,
        )
        setattr(model, "_saient_group_offload_enabled", True)
        _free_cuda()
        used = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        emit({"loading_status": (
            f"low-VRAM: {label} block offload enabled "
            f"({blocks} block/group, streamed={str(use_stream).lower()}) · {used:.1f} GB resident"
        )})
        return True
    except Exception as e:
        emit({"loading_status": f"  ⚠ low-VRAM block offload unavailable for {label} ({type(e).__name__}: {e})"})
        return False


def _wrap_condition_embedder_offload(model, label="transformer"):
    """Low-VRAM mode: park Wan condition_embedder after it has produced embeddings.

    The condition embedder is used at the start of transformer.forward(), before the
    expensive block loop where the 720p activation spike occurs. This is much
    lighter than block/group offload: one small embedding module moves per denoise
    step, while all transformer blocks stay GPU-resident.
    """
    if model is None or not LOW_VRAM_ACTIVE or not _env_flag("SAIENT_WAN_COND_OFFLOAD", True):
        return False
    ce = getattr(model, "condition_embedder", None)
    if ce is None or getattr(ce, "_saient_offload_wrapped", False):
        return False
    try:
        import torch
        bytes_total = sum(
            p.numel() * p.element_size()
            for p in ce.parameters()
            if torch.is_tensor(p) and p.device.type == "cuda"
        )
        orig_forward = ce.forward

        def _forward_then_park(*args, **kwargs):
            try:
                if next(ce.parameters()).device.type != "cuda":
                    ce.to("cuda:0")
            except Exception:
                # If the previous park succeeded but re-load fails, running with a
                # CPU submodule would throw a harder-to-read device mismatch below.
                raise
            out = orig_forward(*args, **kwargs)
            try:
                ce.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
            return out

        ce.forward = _forward_then_park
        ce._saient_offload_wrapped = True
        saved = bytes_total / 2**20
        emit({"loading_status": f"low-VRAM: {label} condition embedder parks after use ({saved:.0f} MiB)"})
        return True
    except Exception as e:
        emit({"loading_status": f"  ⚠ low-VRAM condition embedder offload unavailable ({type(e).__name__}: {e})"})
        return False


def _timestep_for_latents(t, latents):
    """Equivalent to Wan's all-ones mask path without allocating a full latent-sized mask."""
    seq_len = latents.shape[2] * ((latents.shape[3] + 1) // 2) * ((latents.shape[4] + 1) // 2)
    return t.expand(seq_len).unsqueeze(0).expand(latents.shape[0], -1)


# --- Extend / concat helpers (used when previous_video_b64 is supplied with an image) ---
def _b64_to_frames(video_b64: str):
    """Decode base64 mp4 into list of RGB uint8 frames. Uses the same imageio+ffmpeg
    path as enhance_video.py so the backend is consistent."""
    if not video_b64:
        return []
    import base64, tempfile, os
    try:
        import imageio.v2 as imageio
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"imageio not available for video concat: {e}")
    raw = base64.b64decode(video_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        reader = imageio.get_reader(tmp.name, "ffmpeg")
        frames = []
        for f in reader:
            a = np.asarray(f)
            if a.ndim == 3 and a.shape[2] >= 3:
                frames.append(a[:, :, :3].copy())
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _frame_to_uint8(frame, np):
    """Normalize Diffusers float frames and decoded uint8 frames for ffmpeg."""
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"video frame must be HxWxC, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        low = float(arr.min()) if arr.size else 0.0
        high = float(arr.max()) if arr.size else 0.0
        if low >= -1.01 and high <= 1.01:
            arr = (arr + 1.0) * 127.5 if low < -0.01 else arr * 255.0
    return np.ascontiguousarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _frames_to_b64(frames, fps: int) -> str:
    """Encode RGB frames as a high-quality H.264 MP4, then remove the temp file."""
    if frames is None or len(frames) == 0:
        return ""
    import base64, tempfile, os
    try:
        import imageio.v2 as imageio
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"imageio not available for video concat: {e}")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = None
    try:
        writer = imageio.get_writer(
            tmp.name,
            fps=fps,
            codec="libx264",
            quality=9,
            pixelformat="yuv420p",
            ffmpeg_params=["-preset", "medium", "-movflags", "+faststart"],
        )
        for f in frames:
            writer.append_data(_frame_to_uint8(f, np))
        writer.close()
        writer = None
        with open(tmp.name, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _match_extension_to_tail(prev_frames, ext_frames, window=24):
    """Match new chunk exposure/color to the previous tail using per-channel stats.

    This is deliberately simple and CPU-light: one global RGB gain/bias from the last
    frames of the previous chunk to the first frames of the next chunk. It reduces the
    common seam pop without changing model weights or adding another pass.
    Window is a bit longer than a single keyframe so multi-chunk 20–30s chains don't
    accumulate a visible brightness step every ~5s.
    """
    if not prev_frames or not ext_frames:
        return ext_frames
    try:
        import numpy as np
        n = max(1, min(int(window), len(prev_frames), len(ext_frames)))
        prev_sample = np.concatenate([np.asarray(f, dtype=np.float32).reshape(-1, 3) for f in prev_frames[-n:]], axis=0)
        ext_sample = np.concatenate([np.asarray(f, dtype=np.float32).reshape(-1, 3) for f in ext_frames[:n]], axis=0)
        prev_mean, prev_std = prev_sample.mean(axis=0), prev_sample.std(axis=0)
        ext_mean, ext_std = ext_sample.mean(axis=0), ext_sample.std(axis=0)
        # Slightly softer gain clamp than 0.75–1.35 so long chains don't over-correct
        # skin tones (which is where "lips vs labia" contrast mistakes get worse).
        gain = prev_std / np.maximum(ext_std, 1.0)
        gain = np.clip(gain, 0.82, 1.22)
        bias = prev_mean - ext_mean * gain
        out = []
        fade = min(12, len(ext_frames))  # ease the correction in over the first frames
        for i, f in enumerate(ext_frames):
            arr = np.asarray(f, dtype=np.float32)
            if fade > 1 and i < fade:
                t = (i + 1) / fade  # 0→1
                g = 1.0 + (gain - 1.0) * t
                b = bias * t
                arr = arr * g + b
            else:
                arr = arr * gain + bias
            out.append(np.clip(arr, 0, 255).astype(np.uint8))
        return out
    except Exception:
        return ext_frames


def _ffmpeg_concat(prev_b64: str, new_frames, fps: int) -> str | None:
    """Try to concat previous mp4 (as b64) + newly generated frames using the ffmpeg binary.
    This is the most reliable way to get correct total duration without re-encoding the
    entire history every time. Returns the combined base64 or None on failure.
    """
    if not prev_b64 or (len(new_frames) <= 0 if hasattr(new_frames, '__len__') else not bool(new_frames)):
        return None
    import base64, tempfile, os, subprocess, shutil
    if not shutil.which("ffmpeg"):
        return None
    prev_tmp = new_tmp = comb_tmp = None
    try:
        prev_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        prev_tmp.write(base64.b64decode(prev_b64))
        prev_tmp.close()

        new_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        new_tmp.close()
        with open(new_tmp.name, "wb") as f:
            f.write(base64.b64decode(_frames_to_b64(new_frames, fps)))

        comb_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        comb_tmp.close()

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", prev_tmp.name,
            "-i", new_tmp.name,
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            comb_tmp.name,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None

        with open(comb_tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None
    finally:
        for p in (prev_tmp, new_tmp, comb_tmp):
            if p and getattr(p, "name", None):
                try:
                    os.unlink(p.name)
                except Exception:
                    pass


def _shutdown():
    """Tear the model + CUDA context down IN-PROCESS so the driver hands VRAM back cleanly.
    A SIGKILL of a process holding multi-GB of CUDA makes the driver reclaim the context
    abruptly, which on a single GPU that also drives the display spikes + freezes the desktop
    (the "Clean VRAM" lock-up). Called on a `{"cmd":"quit"}` line or SIGTERM; after a quit the
    interpreter exits normally so CUDA's own atexit teardown also runs."""
    global PIPE, I2V_PIPE, TOKENIZER
    try:
        PIPE = None
        I2V_PIPE = None
        TOKENIZER = None
        EMBED_CACHE["key"] = None
        EMBED_CACHE["pe"] = None
        EMBED_CACHE["ne"] = None
    except Exception:
        pass
    _free_cuda()


def _vram():
    """(used_GB, peak_GB) on cuda:0, or (0,0) if no CUDA. Resets the peak counter."""
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return used, peak


def _cap_vram():
    """This GPU also drives the desktop. A full-VRAM encode once hard-froze the machine, so
    cap the process BELOW total VRAM: a shortfall then surfaces as a clean OOM instead of
    starving the display into a deadlock. Tune with SAIENT_VRAM_FRACTION (default 0.98)."""
    import torch
    if not torch.cuda.is_available():
        return
    try:
        global VRAM_CAP_FRACTION
        frac = max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.98")), 0.98))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        VRAM_CAP_FRACTION = frac
        emit({"loading_status": f"  · VRAM cap {frac:.0%} — headroom reserved for the display"})
    except Exception as e:
        emit({"loading_status": f"  ⚠ couldn't set VRAM cap ({e})"})


def _unload_transformer():
    """Free the GPU transformer so the 5.5 GB UMT5 encoder fits on a 16 GB card that can't
    hold both (14B nf4). We DELETE rather than move to CPU — bitsandbytes 4-bit weights don't
    change device cleanly; _reload_transformer() brings it back from the packed nf4 cache."""
    global PIPE, I2V_PIPE
    t = getattr(PIPE, "transformer", None)
    t2 = getattr(PIPE, "transformer_2", None)
    try:
        PIPE.transformer = None
        if hasattr(PIPE, "transformer_2"):
            PIPE.transformer_2 = None
        if I2V_PIPE is not None:
            I2V_PIPE.transformer = None
    except Exception:
        pass
    del t, t2
    _free_cuda()


def _reload_transformer():
    """Reload the packed nf4 transformer from cache straight onto the GPU (the reliable
    direction for bnb 4-bit) once the text encoder has been freed, ready for denoise."""
    import torch
    from diffusers import WanTransformer3DModel
    global PIPE, I2V_PIPE
    t = WanTransformer3DModel.from_pretrained(
        NF4_CACHE_PATH, torch_dtype=torch.bfloat16, device_map={"": 0})
    _set_attention_backend(t, LOW_VRAM_ACTIVE)
    _cast_rope_for_low_vram(t, "high-noise expert" if DUAL_EXPERT else "transformer")
    PIPE.transformer = t
    if hasattr(PIPE, "transformer_2") and DUAL_EXPERT:
        PIPE.transformer_2 = None
    if I2V_PIPE is not None:
        I2V_PIPE.transformer = t
    _free_cuda()
    if LORA_PATH:
        _apply_lora()
    _wrap_condition_embedder_offload(t, "high-noise expert" if DUAL_EXPERT else "transformer")
    _enable_transformer_group_offload(t, "high-noise expert" if DUAL_EXPERT else "transformer")
    _configure_denoise_cache(t, "high-noise expert" if DUAL_EXPERT else "transformer")


def _reload_transformer2():
    """Load Wan2.2's low-noise expert from its packed 4-bit cache. This is only used
    for dual-expert T2V; the high expert has already been freed before this runs."""
    global PIPE
    if not TRANSFORMER2_CACHE_PATH:
        raise RuntimeError("Wan2.2 low-noise transformer cache is not configured")
    # First run has no packed cache yet — the helper falls back to quantizing
    # MODEL_PATH/transformer_2 on the fly and saves the cache for next time.
    t = _load_transformer_cached("transformer_2", TRANSFORMER2_CACHE_PATH, "low-noise expert")
    _set_attention_backend(t, LOW_VRAM_ACTIVE)
    _cast_rope_for_low_vram(t, "low-noise expert")
    PIPE.transformer_2 = t
    _free_cuda()
    if LORA_PATH:
        _apply_lora(load_into_transformer_2=True)
    _wrap_condition_embedder_offload(t, "low-noise expert")
    if _env_flag("SAIENT_WAN_GROUP_OFFLOAD_LOW", False):
        _enable_transformer_group_offload(t, "low-noise expert")
    _configure_denoise_cache(t, "low-noise expert")


def _lora_state_dict(path):
    """Load a LoRA and normalise it to diffusers Wan keys. Many community/DiffSynth
    LoRAs use NATIVE Wan names (`blocks.N.cross_attn.k.lora_A.default.weight`) which
    diffusers' own converter rejects — remap them to
    `transformer.blocks.N.attn2.to_k.lora_A.weight`. Pass diffusers-format ones through."""
    from safetensors.torch import load_file
    sd = load_file(path)
    keys = list(sd.keys())

    if any(k.startswith("transformer.") for k in keys):
        return sd  # already diffusers-style

    try:
        from diffusers.loaders.lora_conversion_utils import (
            _convert_musubi_wan_lora_to_diffusers,
            _convert_non_diffusers_wan_lora_to_diffusers,
        )
        if any(k.startswith("diffusion_model.") for k in keys):
            return _convert_non_diffusers_wan_lora_to_diffusers(sd)
        if any(k.startswith("lora_unet_") for k in keys):
            return _convert_musubi_wan_lora_to_diffusers(sd)
        if any(k.startswith("blocks.") for k in keys):
            return _convert_non_diffusers_wan_lora_to_diffusers({
                f"diffusion_model.{k}": v for k, v in sd.items()
            })
    except Exception as ce:
        emit({"loading_status": f"  ⚠ Wan LoRA converter fallback ({type(ce).__name__})"})

    amap = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
    ffnmap = {"0": "net.0.proj", "2": "net.2"}

    def tail_name(tail):
        tail = tail.replace(".default.", ".")
        if tail in ("lora_A.weight", "lora_down.weight"):
            return "lora_A.weight"
        if tail in ("lora_B.weight", "lora_up.weight"):
            return "lora_B.weight"
        return None

    def target(blk, group, proj):
        if group in ("self_attn", "cross_attn"):
            attn = "attn1" if group == "self_attn" else "attn2"
            mapped = amap.get(proj)
            return f"transformer.blocks.{blk}.{attn}.{mapped}" if mapped else None
        if group == "ffn":
            mapped = ffnmap.get(proj)
            return f"transformer.blocks.{blk}.ffn.{mapped}" if mapped else None
        return None

    out = {}
    for k, v in sd.items():
        main = k
        if main.startswith("diffusion_model."):
            main = main[len("diffusion_model."):]

        if main.startswith("blocks."):
            p = main.split(".")
            if len(p) < 6:
                continue
            tail = tail_name(".".join(p[4:]))
            base = target(p[1], p[2], p[3])
            if tail and base:
                out[f"{base}.{tail}"] = v
            continue

        # Kohya/ai-toolkit Wan LoRA: lora_unet_blocks_0_cross_attn_k.lora_down.weight
        if main.startswith("lora_unet_blocks_"):
            head, _, tail_raw = main.partition(".")
            tail = tail_name(tail_raw)
            if not tail:
                continue
            parts = head[len("lora_unet_blocks_"):].split("_")
            if len(parts) >= 4 and parts[1] in ("self", "cross") and parts[2] == "attn":
                base = target(parts[0], f"{parts[1]}_attn", parts[3])
            elif len(parts) >= 3 and parts[1] == "ffn":
                base = target(parts[0], "ffn", parts[2])
            else:
                base = None
            if base:
                out[f"{base}.{tail}"] = v

    return out if out else sd


def _lora_noise_marker(path):
    """Return an explicit Wan2.2 expert marker from a LoRA filename, if present."""
    name = os.path.basename(path).lower()
    if re.search(r"(?:wan2[._-]?2|a14b)[._-]?low(?:[._-]|$)|low[._-]?noise|(?:^|[._-])low(?:[._-]|$)", name):
        return "low"
    if re.search(r"(?:wan2[._-]?2|a14b)[._-]?high(?:[._-]|$)|high[._-]?noise|(?:^|[._-])high(?:[._-]|$)", name):
        return "high"
    return None


def _paired_lora_path(path, for_low_expert):
    """Find the matching HIGH/LOW sibling for a paired Wan2.2 LoRA."""
    p = path
    if not p:
        return None
    d, b = os.path.split(p)
    swaps = [("high", "low"), ("HIGH", "LOW"), ("High", "Low"),
             ("low", "high"), ("LOW", "HIGH"), ("Low", "High")]
    want_low = bool(for_low_expert)
    for frm, to in swaps:
        if frm not in b:
            continue
        is_low_file = frm.lower() == "low"
        if is_low_file == want_low:
            opposite = os.path.join(d, b.replace(frm, to))
            return p if os.path.exists(opposite) else None
        cand = os.path.join(d, b.replace(frm, to))
        if os.path.exists(cand):
            return cand
    return None


def _expert_lora_path(for_low_expert, path=None):
    """Route a LoRA to the Wan2.2 denoiser it was made for.

    Diffusers loads an ordinary LoRA into Wan2.2's first (high-noise) denoiser by
    default. Explicit LOW/HIGH files target only that expert, while a real sibling
    pair supplies one file to each. Returning None means the live expert must stay
    on its base weights.
    """
    p = LORA_PATH if path is None else path
    if not p or not DUAL_EXPERT:
        return p

    paired = _paired_lora_path(p, for_low_expert)
    if paired:
        return paired

    marker = _lora_noise_marker(p)
    if marker == "low":
        return p if for_low_expert else None
    if marker == "high":
        return None if for_low_expert else p

    # Official Wan2.2/Diffusers behavior for an unscoped or Wan2.1 LoRA.
    return None if for_low_expert else p


def _lora_route_description(path=None):
    p = LORA_PATH if path is None else path
    if not p or not DUAL_EXPERT:
        return "single denoiser"
    if _paired_lora_path(p, False) and _paired_lora_path(p, True):
        return "paired HIGH/LOW experts"
    marker = _lora_noise_marker(p)
    if marker == "low":
        return "low-noise expert only"
    if marker == "high":
        return "high-noise expert only"
    return "high-noise expert only (Diffusers default for single/Wan2.1 LoRAs)"


def _low_noise_guidance(req, high_guidance):
    """Use Wan2.2 A14B's reference low-expert CFG unless explicitly overridden."""
    raw = req.get("cfg_scale_2")
    if raw not in (None, ""):
        return float(raw)
    return min(float(high_guidance), 3.0) if DUAL_EXPERT else float(high_guidance)


def _cast_lora_for_low_vram(sd, path):
    """In low-VRAM mode, store LoRA adapter weights in bf16 instead of fp32.

    The Wan Lightning HIGH adapter is roughly twice the size of the LOW adapter on
    disk, so this is the most direct ~1 GB headroom win without changing the base
    model, scheduler, frame count, or quantized transformer cache.
    """
    if not LOW_VRAM_ACTIVE or not _env_flag("SAIENT_WAN_LORA_BF16", True):
        return sd
    try:
        import torch
        before = sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "element_size"))
        converted = 0
        for k, v in list(sd.items()):
            if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.bfloat16:
                sd[k] = v.to(torch.bfloat16)
                converted += 1
        after = sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "element_size"))
        saved = max(before - after, 0) / 2**20
        if saved >= 1:
            emit({"loading_status": (
                f"low-VRAM: LoRA bf16 adapter storage saved {saved:.0f} MiB "
                f"({converted} tensors, {os.path.basename(path)})"
            )})
        return sd
    except Exception as e:
        emit({"loading_status": f"  ⚠ low-VRAM LoRA bf16 cast skipped ({type(e).__name__}: {e})"})
        return sd


def _cast_loaded_lora_params_for_low_vram(transformer, path):
    """PEFT may materialize adapter params as fp32 even if the state dict is bf16."""
    if transformer is None or not LOW_VRAM_ACTIVE or not _env_flag("SAIENT_WAN_LORA_BF16", True):
        return
    try:
        import torch
        before = converted = 0
        after = 0
        for name, param in transformer.named_parameters():
            lname = name.lower()
            if "lora_" not in lname or not torch.is_tensor(param):
                continue
            before += param.numel() * param.element_size()
            if param.is_floating_point() and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
                converted += 1
            after += param.numel() * param.element_size()
        for name, buf in transformer.named_buffers():
            lname = name.lower()
            if "lora_" not in lname or not torch.is_tensor(buf):
                continue
            before += buf.numel() * buf.element_size()
            if buf.is_floating_point() and buf.dtype != torch.bfloat16:
                buf.data = buf.data.to(torch.bfloat16)
                converted += 1
            after += buf.numel() * buf.element_size()
        _free_cuda()
        saved = max(before - after, 0) / 2**20
        if before:
            used = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            emit({"loading_status": (
                f"low-VRAM: live LoRA params bf16 saved {saved:.0f} MiB "
                f"({converted} tensors, {os.path.basename(path)}) · {used:.2f} GB resident"
            )})
    except Exception as e:
        emit({"loading_status": f"  ⚠ low-VRAM live LoRA bf16 cast skipped ({type(e).__name__}: {e})"})


def _active_transformer(prefer_low=False):
    """Return the currently loaded Wan expert. Dual-expert staging leaves one of
    transformer / transformer_2 as None — never touch .config on a None expert."""
    if PIPE is None:
        return None
    if prefer_low:
        return getattr(PIPE, "transformer_2", None) or getattr(PIPE, "transformer", None)
    return getattr(PIPE, "transformer", None) or getattr(PIPE, "transformer_2", None)


def _apply_lora(lora_path=None, lora_strength=None, load_into_transformer_2=False):
    """Attach a Wan LoRA as a PEFT adapter to the currently-loaded transformer.
    This is intentionally NOT a model merge: the base remains the cached 4-bit
    transformer and the LoRA stays as a small sidecar adapter."""
    global LORA_PATH, LORA_STRENGTH
    if lora_path is not None:
        LORA_PATH = lora_path
    if lora_strength is not None:
        LORA_STRENGTH = float(lora_strength)
    if not LORA_PATH:
        return
    import os as _os
    import contextlib
    import torch
    # After high→low expert swap, only transformer_2 is live. Auto-target it so we
    # never call load_lora_weights against a None PIPE.transformer (.config crash).
    if not load_into_transformer_2 and getattr(PIPE, "transformer", None) is None:
        if getattr(PIPE, "transformer_2", None) is not None:
            load_into_transformer_2 = True
        else:
            raise RuntimeError("cannot apply LoRA: no Wan expert is loaded")
    if load_into_transformer_2 and getattr(PIPE, "transformer_2", None) is None:
        raise RuntimeError("cannot apply LoRA to transformer_2: low-noise expert is not loaded")
    path = _expert_lora_path(load_into_transformer_2)
    expert = "low-noise" if load_into_transformer_2 else "high-noise"
    if not path:
        emit({"loading_status": f"LoRA routing: {expert} expert stays on base weights"})
        return False
    emit({"loading_status": f"applying LoRA {_os.path.basename(path)} @ {LORA_STRENGTH} to {expert} expert…"})
    sd = _lora_state_dict(path)
    sd = _cast_lora_for_low_vram(sd, path)
    ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else contextlib.nullcontext()
    with ctx:
        if load_into_transformer_2:
            PIPE.load_lora_into_transformer(
                sd, transformer=PIPE.transformer_2, adapter_name="extra", _pipeline=PIPE)
        else:
            PIPE.load_lora_weights(sd, adapter_name="extra")
        _set_lora_strength(LORA_STRENGTH)
    _cast_loaded_lora_params_for_low_vram(
        PIPE.transformer_2 if load_into_transformer_2 else PIPE.transformer,
        path,
    )
    emit({"loading_status": f"  ✓ LoRA adapter active on {expert} expert ({len(sd)} tensors)"})
    _free_cuda()
    return True


def _set_lora_strength(strength):
    """Change the active adapter weight without reloading the adapter."""
    global LORA_STRENGTH
    if not LORA_PATH:
        return
    import contextlib
    import torch
    LORA_STRENGTH = float(strength)
    # Prefer the live expert — Diffusers set_adapters walks PEFT modules on the pipe;
    # after dual-expert swap only one expert exists and the other is None.
    target = _active_transformer(prefer_low=getattr(PIPE, "transformer", None) is None)
    if target is None:
        raise RuntimeError("cannot set LoRA strength: no Wan expert is loaded")
    ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else contextlib.nullcontext()
    applied = False
    with ctx:
        # Prefer the live module: a diffusers MODEL's set_adapters takes weights= (the
        # pipeline's is adapter_weights=). After a dual-expert swap only one expert is
        # live and PIPE.transformer may be None, so a pipeline-level call would walk into
        # the missing expert — set strength on the module we actually hold. Never raise:
        # the adapter is already attached+active at its load-time weight (1.0 for
        # Lightning), so a missed strength tweak must not crash the whole generation.
        try:
            if hasattr(target, "set_adapter"):
                target.set_adapter("extra")
            if hasattr(target, "set_adapters"):
                target.set_adapters(["extra"], weights=[LORA_STRENGTH])
                applied = True
        except Exception as e:
            emit({"loading_status": f"  ⚠ LoRA strength on live expert failed ({type(e).__name__}); trying pipeline"})
        if not applied and PIPE is not None and getattr(PIPE, "transformer", None) is not None:
            try:
                PIPE.set_adapters(["extra"], adapter_weights=[LORA_STRENGTH])
                applied = True
            except Exception as e:
                emit({"loading_status": f"  ⚠ LoRA strength via pipeline failed ({type(e).__name__})"})
        if not applied:
            emit({"loading_status": "  ⚠ LoRA strength left at adapter default (could not restrike)"})
        if I2V_PIPE is not None and getattr(I2V_PIPE, "transformer", None) is not None:
            try:
                I2V_PIPE.set_adapters(["extra"], adapter_weights=[LORA_STRENGTH])
            except Exception:
                pass


def _ensure_lora_adapter(strength=None, load_into_transformer_2=False):
    """Make sure the selected LoRA sidecar is actually attached before strength changes.

    After an OOM or interrupted generation, the long-lived daemon can still have LORA_PATH
    set while Diffusers/PEFT reports no present adapters. Re-attach the sidecar instead of
    failing the next generation with "adapter not in present adapters: set()".
    """
    if not LORA_PATH:
        return False
    # Auto-route to the live expert after dual-expert high→low swap.
    if not load_into_transformer_2 and getattr(PIPE, "transformer", None) is None:
        if getattr(PIPE, "transformer_2", None) is not None:
            load_into_transformer_2 = True
        else:
            return False
    if _expert_lora_path(load_into_transformer_2) is None:
        return False
    target = LORA_STRENGTH if strength is None else float(strength)
    try:
        _set_lora_strength(target)
        return True
    except Exception as e:
        msg = str(e)
        recoverable = (
            "not in the list of present adapters" in msg
            or "present adapters" in msg
            or "requires_grad=True on inference tensor" in msg
            or "NoneType" in msg
            or "no Wan expert" in msg
        )
        if not recoverable:
            raise
        emit({"loading_status": "LoRA adapter state invalid — re-applying sidecar…"})
        try:
            if getattr(PIPE, "transformer", None) is not None:
                PIPE.unload_lora_weights()
        except Exception:
            pass
        _apply_lora(lora_strength=target, load_into_transformer_2=load_into_transformer_2)
        return True


def _configure_scheduler(req):
    """Apply per-generation scheduler overrides. Auto keeps the model's scheduler;
    Euler Beta uses FlowMatchEuler with beta sigmas, which is the Diffusers mapping
    for the common Wan 'Euler/Beta' recipe."""
    global PIPE, I2V_PIPE
    if PIPE is None:
        return
    cfg = dict(SCHEDULER_BASE_CONFIG or {})
    if not cfg:
        return
    mode = str(req.get("scheduler") or "auto").strip().lower().replace("-", "_")
    shift_raw = req.get("shift", None)
    shift = None
    if shift_raw not in (None, ""):
        try:
            shift = float(shift_raw)
        except Exception:
            shift = None
    try:
        if mode in ("euler_beta", "euler_beta57", "beta57", "flow_euler_beta"):
            from diffusers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                cfg, shift=shift if shift is not None else 8.0, use_beta_sigmas=True)
            label = f"Euler Beta · shift {float(scheduler.config.shift):g}"
        elif mode in ("euler", "flow_euler"):
            from diffusers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                cfg, shift=shift if shift is not None else 5.0, use_beta_sigmas=False)
            label = f"Euler · shift {float(scheduler.config.shift):g}"
        else:
            import diffusers as _df
            sched_cls = getattr(_df, cfg.get("_class_name", "UniPCMultistepScheduler"))
            kwargs = {}
            if shift is not None:
                if sched_cls.__name__ == "UniPCMultistepScheduler":
                    kwargs["flow_shift"] = shift
                elif sched_cls.__name__ == "FlowMatchEulerDiscreteScheduler":
                    kwargs["shift"] = shift
            scheduler = sched_cls.from_config(cfg, **kwargs)
            s = getattr(scheduler.config, "flow_shift", getattr(scheduler.config, "shift", None))
            label = scheduler.__class__.__name__.replace("Scheduler", "")
            if s is not None:
                label = f"{label} · shift {float(s):g}"
        PIPE.scheduler = scheduler
        if I2V_PIPE is not None:
            I2V_PIPE.scheduler = scheduler
        emit({"loading_status": f"scheduler: {label}"})
    except Exception as e:
        emit({"loading_status": f"  ⚠ scheduler override failed ({type(e).__name__}); using model default"})


def _load_transformer_cached(subfolder, cache_path, label, dev=0):
    """Load one Wan transformer subfolder through the packed 4-bit cache."""
    import torch
    from diffusers import WanTransformer3DModel
    from diffusers import BitsAndBytesConfig as DBnb

    transformer = None
    if os.path.exists(os.path.join(cache_path, "config.json")):
        emit({"loading_status": f"loading pre-quantized {label} (cached 4-bit)…"})
        try:
            transformer = WanTransformer3DModel.from_pretrained(
                cache_path, torch_dtype=torch.bfloat16, device_map={"": dev})
        except Exception as ce:
            emit({"loading_status": f"  ⚠ {label} cache load failed ({type(ce).__name__}); rebuilding…"})
            transformer = None
    if transformer is None:
        emit({"loading_status": f"loading {label} (4-bit nf4) + caching for next time…"})
        dbnb = DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = WanTransformer3DModel.from_pretrained(
            MODEL_PATH, subfolder=subfolder,
            quantization_config=dbnb, torch_dtype=torch.bfloat16, device_map={"": dev})
        try:
            os.makedirs(cache_path, exist_ok=True)
            transformer.save_pretrained(cache_path)
            emit({"loading_status": f"  ✓ {label} cached — fast loads from now on"})
        except Exception as se:
            emit({"loading_status": f"  ⚠ couldn't cache {label} ({type(se).__name__})"})
    return transformer


def _cache_dir(name):
    return str(cache_dir(name))


def _safetensors_gb(folder):
    """Total size (GB) of the *.safetensors shards in a model subfolder (0 if none)."""
    total = 0
    try:
        for f in os.listdir(folder):
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(folder, f))
    except Exception:
        return 0.0
    return total / 2**30


def _available_ram_gb():
    """Best-effort available system RAM (GB), or None if we can't tell (then don't block
    — better to attempt the load than to refuse one we couldn't size up)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    try:  # Linux fallback if psutil isn't present
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20   # kB → GB
    except Exception:
        pass
    return None


def _guard_loadable(model_path, quality):
    """Refuse loads that would swap-freeze the whole PC instead of failing cleanly — the
    "waited 15 min then the machine died" case. The bf16 Quality path reads the FULL
    transformer into RAM. If that won't fit with headroom, refuse rather than thrash
    swap to death. Wan2.2 dual-expert fast mode is allowed because only one 4-bit
    expert is loaded at a time."""
    if quality:
        need = _safetensors_gb(os.path.join(model_path, "transformer"))
        if os.path.isdir(os.path.join(model_path, "transformer_2")):
            need = max(need, _safetensors_gb(os.path.join(model_path, "transformer_2")))
        avail = _available_ram_gb()
        if avail is not None and need > 0 and need > avail - 6.0:
            raise RuntimeError(
                f"Quality mode loads the full-precision transformer (~{need:.0f} GB) into RAM, but "
                f"only ~{avail:.0f} GB is free — that would swap-freeze the machine. "
                f"Switch to Fast mode (untick Quality). For 14B T2V stay on fast+Lightning (no 5B fallbacks).")


def load(cfg):
    """Load the SMALL resident pieces (transformer 4-bit + VAE). Fast, low RAM."""
    global PIPE, TOKENIZER, MODEL_PATH, STREAM_TRANSFORMER
    global NF4_CACHE_PATH, RESIDENT_GB, PARK_TRANSFORMER, LORA_PATH, LORA_STRENGTH
    global SCHEDULER_BASE_CONFIG, MODEL_INDEX_CONFIG, DUAL_EXPERT, TRANSFORMER2_CACHE_PATH
    global VRAM_REPORT
    import torch
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers import BitsAndBytesConfig as DBnb
    from transformers import AutoTokenizer

    _cap_vram()   # reserve VRAM for the desktop — this GPU also drives the display
    VRAM_REPORT.clear()

    MODEL_PATH = cfg.get("model_path", "")
    MODEL_INDEX_CONFIG = {}
    try:
        with open(os.path.join(MODEL_PATH, "model_index.json")) as f:
            MODEL_INDEX_CONFIG = json.load(f)
    except Exception:
        MODEL_INDEX_CONFIG = {}
    DUAL_EXPERT = os.path.isdir(os.path.join(MODEL_PATH, "transformer_2"))
    TRANSFORMER2_CACHE_PATH = None
    lora_path = (cfg.get("lora_path") or "").strip()
    lora_strength = float(cfg.get("lora_strength", 1.0))
    use_lora = bool(lora_path)
    LORA_PATH = lora_path
    LORA_STRENGTH = lora_strength
    precision = (cfg.get("precision") or "fast").strip().lower()
    quality = precision == "quality"
    # Quality mode streams a bf16 transformer from RAM. LoRA does NOT imply bf16: Wan's
    # Diffusers/PEFT path can attach adapters to the cached 4-bit transformer, avoiding
    # a 53 GB full-precision read for 14B models.
    STREAM_TRANSFORMER = quality

    # Refuse loads that would tip the box into swap (A14B dual-expert / oversized bf16) with
    # a clear message, BEFORE reading any weights — never freeze the machine.
    _guard_loadable(MODEL_PATH, quality)
    dev = 0  # cuda:0

    t_load = time.time()
    _t = time.time()
    if quality:
        if DUAL_EXPERT:
            raise RuntimeError("Wan2.2 dual-expert Quality mode is not supported here yet. Use Fast mode so experts can stay staged 4-bit.")
        # bf16 transformer for the denoise loop. Left on CPU so generate() can stream
        # it onto the GPU after the text encoder has been freed.
        label = "bf16 — streamed from RAM (quality)"
        emit({"loading_status": f"loading transformer ({label})…"})
        place = {} if STREAM_TRANSFORMER else {"device_map": {"": dev}}
        transformer = WanTransformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer",
            torch_dtype=torch.bfloat16, **place)
    else:
        # PRE-QUANTIZED 4-bit cache (mirrors the umt5 TE cache). The bitsandbytes nf4
        # quantize-on-load reads the bf16 shards and packs them (~54s/shard → ~4.5 min
        # for the 5B). We do it ONCE, save the packed 4-bit to disk, and load THAT every
        # later time (no re-read, no re-quantize → seconds). Keyed per model.
        import hashlib
        key = os.path.basename(os.path.normpath(MODEL_PATH)) or hashlib.md5(MODEL_PATH.encode()).hexdigest()[:8]
        t_cache = os.path.join(_cache_dir("wan-transformer-4bit"), key)
        NF4_CACHE_PATH = t_cache   # remember for reload-after-encode on big models
        if DUAL_EXPERT:
            TRANSFORMER2_CACHE_PATH = os.path.join(_cache_dir("wan-transformer-4bit"), f"{key}--transformer_2")
            emit({"loading_status": "Wan2.2 T2V dual-expert: loading high-noise expert only…"})
        transformer = _load_transformer_cached("transformer", t_cache, "transformer", dev)
    emit({"loading_status": f"  ⏱ transformer: {time.time()-_t:.0f}s"})

    emit({"loading_status": "loading VAE…"})
    # Wan's VAE is small; fp32 keeps decode stable. Lands on GPU below.
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)

    emit({"loading_status": "loading tokenizer…"})
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")

    emit({"loading_status": "assembling pipeline (no text encoder yet)…"})
    # Build WanPipeline directly with text_encoder=None so the 21 GB component is
    # never loaded here. We construct via __init__ (not from_pretrained) to dodge
    # from_pretrained's "required component is None" validation; the scheduler is
    # the only thing left to load and it's a tiny JSON config.
    import diffusers as _df
    with open(os.path.join(MODEL_PATH, "scheduler", "scheduler_config.json")) as f:
        SCHEDULER_BASE_CONFIG = json.load(f)
        sched_cls = getattr(_df, SCHEDULER_BASE_CONFIG["_class_name"])
    scheduler = sched_cls.from_pretrained(MODEL_PATH, subfolder="scheduler")
    try:
        PIPE = WanPipeline(
            vae=vae, text_encoder=None, tokenizer=TOKENIZER,
            transformer=transformer, transformer_2=None,
            scheduler=scheduler,
            boundary_ratio=MODEL_INDEX_CONFIG.get("boundary_ratio"),
            expand_timesteps=bool(MODEL_INDEX_CONFIG.get("expand_timesteps", False)),
        )
    except Exception:
        if DUAL_EXPERT:
            raise
        # Fallback: let from_pretrained assemble, still skipping the text encoder.
        PIPE = WanPipeline.from_pretrained(
            MODEL_PATH, transformer=transformer, vae=vae, tokenizer=TOKENIZER,
            scheduler=scheduler, text_encoder=None, torch_dtype=torch.bfloat16)

    # Move only the non-quantized parts to GPU (the 4-bit transformer is already
    # placed by device_map; calling .to() on the whole pipe would error on it).
    try:
        PIPE.vae.to(f"cuda:{dev}")
    except Exception:
        pass
    if hasattr(PIPE, "vae") and hasattr(PIPE.vae, "enable_tiling"):
        # EXPLICIT tile sizes — the no-arg enable_tiling() did NOT actually bound the
        # Wan2.2-5B decode (measured +8.6 GB spike → OOM), but explicit tile+stride does:
        # 256px tiles drop the 480×832×49 decode spike to +1.6 GB (peak 4.3 GB) for ~+11s.
        # That's what makes the 5B fit 16 GB. Harmless on the 1.3B (its decode is tiny).
        try:
            PIPE.vae.enable_tiling(
                tile_sample_min_height=256, tile_sample_min_width=256,
                tile_sample_stride_height=192, tile_sample_stride_width=192)
        except TypeError:
            PIPE.vae.enable_tiling()   # older diffusers without the kwargs
    PIPE.set_progress_bar_config(disable=True)

    if use_lora:
        if DUAL_EXPERT:
            emit({"loading_status": f"LoRA routing: {_lora_route_description(lora_path)}"})
        try:
            _apply_lora(lora_path, lora_strength)
        except Exception as le:
            emit({"loading_status": f"  ⚠ LoRA failed ({le}); continuing without it"})

    if STREAM_TRANSFORMER:
        # Quality mode: the bf16 transformer stays in CPU RAM. Wan decodes OUTSIDE the pipe
        # call (see _decode_latents), so generate() can move the whole transformer GPU↔CPU
        # by hand — onto the GPU for denoise, back to RAM before the VAE decode. No accelerate
        # hooks (those tripped CogVideoX with a meta-device error); just plain .to() moves, so
        # only one heavy component is ever resident. Nothing to do here — it's left on CPU.
        emit({"loading_status": "quality: bf16 transformer parked in RAM (streamed for denoise)…"})

    _free_cuda()
    load_stats = _vram_stage("model load")
    used = load_stats["used"]
    RESIDENT_GB = used
    # Decide whether this model must free its transformer during the text-encode. Only the
    # resident-nf4 case needs it (bf16 stream already moves the transformer itself).
    # If transformer (~used) + the ~5.5 GB UMT5 encoder would exceed the VRAM cap, park it.
    _total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    _budget = _total * max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.98")), 0.98))
    PARK_TRANSFORMER = (not STREAM_TRANSFORMER) and (NF4_CACHE_PATH is not None) and (used + 5.5 > _budget)
    # Force for 14B test on 16GB
    if "A14B" in MODEL_PATH or "14B" in MODEL_PATH:
        PARK_TRANSFORMER = True
    if PARK_TRANSFORMER:
        emit({"loading_status": f"  ⓘ 16 GB fit: free transformer for encode ({used:.1f}+5.5 > {_budget:.1f} GB cap), reload for denoise"})
    EMBED_CACHE["key"] = None   # invalidate any stale embeds from a previous model
    emit({"loading_status": f"  ⏱ load total: {time.time()-t_load:.0f}s · VRAM {used:.1f} GB"})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emit({"ready": True, "device": device})


def encode(prompt, neg, do_cfg):
    """Load text encoder 4-bit on GPU, encode, then free it. Returns (pe, ne).

    Uses a PRE-QUANTIZED 4-bit cache: the first time we read the 22 GB fp16 UMT5 and
    quantize (~36s), then save the packed 4-bit (~5.5 GB) to disk. Every later run
    (any model, any session) loads that directly — no 22 GB read, no re-quantize
    (~10s). Wan's UMT5-XXL is shared across all models, so one cache serves all."""
    import torch
    from transformers import UMT5EncoderModel
    from transformers import BitsAndBytesConfig as TBnb

    u0, _ = _vram()
    emit({"loading_status": f"  · VRAM before text-encoder load: {u0:.1f} GB free-baseline"})
    cache = _cache_dir("umt5-xxl-4bit")
    cached = os.path.exists(os.path.join(cache, "config.json"))
    te = None
    if cached:
        emit({"loading_status": "loading pre-quantized text encoder (cached 4-bit)…"})
        try:
            te = UMT5EncoderModel.from_pretrained(cache, torch_dtype=torch.bfloat16, device_map={"": 0})
        except Exception as ce:
            emit({"loading_status": f"  ⚠ cache load failed ({ce}); rebuilding…"})
            te = None
    if te is None:
        emit({"loading_status": "loading text encoder (4-bit nf4) + caching for next time…"})
        tbnb = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
        te = UMT5EncoderModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder",
            quantization_config=tbnb, torch_dtype=torch.bfloat16, device_map={"": 0})
        try:
            os.makedirs(cache, exist_ok=True)
            te.save_pretrained(cache)
            emit({"loading_status": "  ✓ text encoder cached — fast loads from now on"})
        except Exception as se:
            emit({"loading_status": f"  ⚠ couldn't cache 4-bit TE ({se})"})

    PIPE.text_encoder = te
    try:
        # no_grad is load-bearing: without it encode_prompt builds an autograd graph and the
        # returned embeds keep a grad_fn chain back to the encoder's params — so `del te` can't
        # reclaim its ~5.5 GB, which then OOMs the 14B transformer reload. Inference never
        # needs the graph.
        with torch.inference_mode():
            pe, ne = PIPE.encode_prompt(
                prompt=prompt,
                negative_prompt=(neg or None),
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=1,
                device="cuda",
            )
    finally:
        # Drop the encoder before denoising so its ~5.5 GB VRAM is reclaimed.
        PIPE.text_encoder = None
        del te
        _free_cuda()
    return pe, ne


def _i2v_pipe():
    """Assemble a WanImageToVideoPipeline that REUSES the resident t2v components
    (same transformer/vae/scheduler — no extra weights loaded). Works because the
    Wan2.2 TI2V transformer has image_dim=None (no CLIP encoder) and expand_timesteps
    feeds the image straight in as the first-frame latent condition."""
    global I2V_PIPE
    if I2V_PIPE is None:
        from diffusers import WanImageToVideoPipeline
        I2V_PIPE = WanImageToVideoPipeline(
            vae=PIPE.vae, text_encoder=None, tokenizer=PIPE.tokenizer,
            transformer=PIPE.transformer, scheduler=PIPE.scheduler,
            image_encoder=None, image_processor=None,
            transformer_2=None, boundary_ratio=None, expand_timesteps=True,
        )
        I2V_PIPE.set_progress_bar_config(disable=True)
        # Park the heavy VAE on CPU the instant it finishes encoding the input image. The VAE's
        # ONLY use before decode is that single image-encode inside prepare_latents; keeping its
        # ~2.6 GB (5B) resident through the whole denoise loop is what OOM'd i2v at 480p/33f+.
        # Wrap encode → park, so denoise runs without it; _decode_latents moves it back. (t2v
        # never calls vae.encode, so this is i2v-only.) Re-arm on each call in case a prior gen
        # parked it.
        _vae = I2V_PIPE.vae
        _orig_encode = _vae.encode
        def _encode_then_park(*a, **k):
            import torch
            if next(_vae.parameters()).device.type != "cuda":
                _vae.to("cuda:0")
            out = _orig_encode(*a, **k)
            _vae.to("cpu"); torch.cuda.empty_cache()
            return out
        _vae.encode = _encode_then_park
    return I2V_PIPE


def _vae_is_heavy():
    """True for the Wan2.2-5B VAE (48 latent channels). Its fp32 weights are big enough
    that we park it on CPU whenever it isn't actively encoding/decoding, to leave VRAM
    for the ~5.5 GB text-encoder reload and the denoise activations on a 16 GB card."""
    try:
        return int(getattr(PIPE.vae.config, "z_dim", 16) or 16) >= 32
    except Exception:
        return False


def _vae_to(device):
    """Move the VAE if it isn't already there. Cheap no-op when already on `device`."""
    import torch
    try:
        if next(PIPE.vae.parameters()).device.type != ("cuda" if "cuda" in device else "cpu"):
            PIPE.vae.to(device); _free_cuda()
            return True
    except Exception:
        pass
    return False


def _use_untiled_vae_decode(z_dim, latent_frames, latent_height, latent_width):
    """Use a seam-free full-frame decode only when its measured-size proxy is safe."""
    if int(z_dim) >= 32:
        return False
    temporal_scale = max(1, int(getattr(PIPE, "vae_scale_factor_temporal", 4) or 4))
    spatial_scale = max(1, int(getattr(PIPE, "vae_scale_factor_spatial", 8) or 8))
    frames = max(1, (int(latent_frames) - 1) * temporal_scale + 1)
    sample_pixels = frames * int(latent_height) * spatial_scale * int(latent_width) * spatial_scale
    limit = int(os.environ.get("SAIENT_WAN_UNTILED_DECODE_PIXEL_FRAMES", "8000000"))
    return sample_pixels <= max(limit, 0)


def _decode_latents(latents):
    """Decode denoised latents → frames, OUTSIDE the pipeline — so two things are on us:

    1. torch.no_grad() — THE fix. The VAE's params are requires_grad=True, so calling
       vae.decode outside the pipeline's own @torch.no_grad() builds a full autograd graph
       and retains EVERY decoder activation → ~+11 GB and a guaranteed OOM on the 5B. (The
       inline pipeline decode never hit this; my output_type="latent" refactor moved the
       decode out of that no-grad context, which is what introduced the OOM.) Verified in
       isolation: grad-enabled OOMs, no_grad fits.
    2. output_type="latent" upstream + _free_cuda() here drop the denoise tensors before we
       decode, and with the explicit VAE tiling set in load() the 480×832×49 decode then
       peaks ~1.6 GB (fp32). Denorm replicates WanPipeline.__call__ exactly.

    For quality bf16: we *try* decode with the large transformer weights still on-GPU
    (they were left resident after denoise). If that OOMs we park only for this decode.
    This is the key to hitting ~30s–1m end-to-end gens instead of paying PCIe both ways."""
    import torch
    vae = PIPE.vae
    _free_cuda()                              # reclaim denoise tensors before the decode
    if next(vae.parameters()).device.type != "cuda":
        vae.to("cuda:0")                      # was parked on CPU during denoise → back for decode
    zc = vae.config.z_dim
    global STREAM_TRANSFORMER
    original_tiling = bool(getattr(vae, "use_tiling", False))
    original_tile_config = (
        getattr(vae, "tile_sample_min_height", 256),
        getattr(vae, "tile_sample_min_width", 256),
        getattr(vae, "tile_sample_stride_height", 192),
        getattr(vae, "tile_sample_stride_width", 192),
    )
    untiled = _use_untiled_vae_decode(zc, latents.shape[2], latents.shape[3], latents.shape[4])
    if untiled:
        vae.use_tiling = False
        emit({"loading_status": "VAE decode: full-frame short-clip path (no spatial tile seams)"})
    else:
        vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
        )
        emit({"loading_status": "VAE decode: memory-safe 256px tiles with 64px overlap"})

    def decode_once():
        with torch.inference_mode():
            lat = latents.to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean).view(1, zc, 1, 1, 1).to(lat.device, lat.dtype)
            std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zc, 1, 1, 1).to(lat.device, lat.dtype)
            lat = lat / std + mean
            video = vae.decode(lat, return_dict=False)[0]
        return PIPE.video_processor.postprocess_video(video, output_type="np")[0]

    def decode_or_oom():
        """Return after the OOM handler frame is gone, releasing its traceback tensors."""
        try:
            return decode_once(), None
        except RuntimeError as error:
            msg = str(error).lower()
            memory_error = "out of memory" in msg or "cuda error" in msg or "cublas" in msg
            if not memory_error:
                raise
            return None, str(error)

    try:
        decoded, oom_message = decode_or_oom()
        if oom_message is None:
            return decoded

        if STREAM_TRANSFORMER and getattr(PIPE, "transformer", None) is not None:
            emit({"loading_status": "quality: decode needed more room — parking bf16 transformer temporarily…"})
            PIPE.transformer.to("cpu")
            _free_cuda()
            if next(vae.parameters()).device.type != "cuda":
                vae.to("cuda:0")
            decoded, oom_message = decode_or_oom()
            if oom_message is None:
                return decoded

        if untiled:
            emit({"loading_status": "VAE full-frame decode exceeded headroom — retrying with 64px-overlap tiles…"})
            _free_cuda()
            vae.enable_tiling(
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                tile_sample_stride_height=192,
                tile_sample_stride_width=192,
            )
            decoded, oom_message = decode_or_oom()
            if oom_message is None:
                return decoded

        raise RuntimeError(f"VAE decode ran out of GPU memory: {oom_message}")
    finally:
        (
            vae.tile_sample_min_height,
            vae.tile_sample_min_width,
            vae.tile_sample_stride_height,
            vae.tile_sample_stride_width,
        ) = original_tile_config
        vae.use_tiling = original_tiling


def _preview_positions(num_frames, latent_frames, count, temporal_scale):
    """Return evenly spaced causal latent positions and approximate frame labels."""
    if num_frames <= 0 or latent_frames <= 0:
        return [], []
    count = max(1, min(int(count), int(latent_frames)))
    if count == 1:
        latent_indices = [0]
    else:
        latent_indices = [
            round(i * (latent_frames - 1) / (count - 1))
            for i in range(count)
        ]
    latent_indices = list(dict.fromkeys(int(index) for index in latent_indices))
    labels = [
        1 if index == 0 else min(index * temporal_scale + 1, num_frames)
        for index in latent_indices
    ]
    return labels, latent_indices


def _preview_contact_sheet(images):
    """Lay sparse 16:9 frames into a fixed 16:9 JPEG without touching disk."""
    from PIL import Image

    if not images:
        raise ValueError("preview decoder returned no frames")
    tile_w, tile_h = images[0].size
    cols = 3
    rows = 3 if len(images) > 6 else 2
    canvas = Image.new("RGB", (tile_w * cols, tile_h * 3), (5, 6, 10))
    y_gap = 0 if rows == 3 else tile_h // 3
    positions = []
    for index in range(len(images)):
        row = index // cols
        col = index % cols
        row_count = min(cols, len(images) - row * cols)
        x_offset = (cols - row_count) * tile_w // 2
        positions.append((x_offset + col * tile_w, y_gap + row * (tile_h + y_gap)))
    for frame, position in zip(images, positions):
        canvas.paste(frame, position)

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=76, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _decode_preview_sample(sample, vae, max_width, frame_indices):
    """Decode a spatially reduced full latent clip, then select preview frames."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    spatial_scale = max(1, int(getattr(PIPE, "vae_scale_factor_spatial", 8) or 8))
    latent_h, latent_w = int(sample.shape[-2]), int(sample.shape[-1])
    target_latent_w = min(latent_w, max(16, int(max_width) // spatial_scale))
    target_latent_h = min(
        latent_h,
        max(9, round(latent_h * target_latent_w / max(latent_w, 1))),
    )

    sample = sample.to(device=next(vae.parameters()).device, dtype=vae.dtype)
    if (target_latent_h, target_latent_w) != (latent_h, latent_w):
        sample = F.interpolate(
            sample,
            size=(sample.shape[2], target_latent_h, target_latent_w),
            mode="trilinear",
            align_corners=False,
        )

    zc = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean).view(1, zc, 1, 1, 1).to(sample.device, sample.dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zc, 1, 1, 1).to(sample.device, sample.dtype)
    decoded = vae.decode(sample / std + mean, return_dict=False)[0]
    indices = torch.tensor(
        [max(0, min(int(index), decoded.shape[2] - 1)) for index in frame_indices],
        device=decoded.device,
        dtype=torch.long,
    )
    selected = decoded[0].index_select(1, indices)
    rgb = (
        selected
        .add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    del decoded, selected, indices, sample, mean, std
    images = [Image.fromarray(frame, mode="RGB") for frame in rgb]
    return _preview_contact_sheet(images)


def _wait_for_preview(preview_state):
    worker = preview_state.get("worker")
    if worker is not None:
        worker.join()
        preview_state["worker"] = None


def _start_latent_preview(preview_state, latents, step, total, num_frames):
    """Clone sparse latent slices and decode them away from the denoise stream."""
    import torch

    if not preview_state.get("enabled") or preview_state.get("failed"):
        return
    worker = preview_state.get("worker")
    if worker is not None:
        if worker.is_alive():
            return  # Never build a preview backlog that can slow the actual generation.
        worker.join()
        preview_state["worker"] = None

    every = preview_state["every"]
    if step != 1 and step != total and step % every != 0:
        return

    count = 9 if step / max(total, 1) >= 0.8 else 5
    temporal_scale = max(1, int(getattr(PIPE, "vae_scale_factor_temporal", 4) or 4))
    labels, latent_indices = _preview_positions(
        num_frames,
        int(latents.shape[2]),
        count,
        temporal_scale,
    )
    if not latent_indices:
        return

    # Wan's temporal VAE is causal, so isolated temporal latents decode as noise.
    # Decode sparse positions together as one ordered sequence: this preserves
    # enough causal context without making preview cost grow with clip duration.
    # clone().detach() guarantees the scheduler's tensor remains read-only.
    sample = latents[:, :, latent_indices, :, :].clone().detach()
    frame_indices = [0 if index == 0 else index * temporal_scale for index in range(len(labels))]
    vae = PIPE.vae
    vae_device = next(vae.parameters()).device
    if vae_device.type != "cuda":
        preview_state["failed"] = True
        emit({
            "loading_status": (
                "live preview paused: low-VRAM mode parked the VAE on CPU "
                "(measured CPU decode is too slow for an in-progress preview)"
            )
        })
        del sample
        return
    ready_event = None
    ready_event = torch.cuda.Event()
    ready_event.record(torch.cuda.current_stream(device=vae_device))

    def run_preview():
        try:
            started = time.time()
            with torch.inference_mode():
                stream = preview_state.get("stream")
                if stream is None:
                    stream = torch.cuda.Stream(device=vae_device)
                    preview_state["stream"] = stream
                stream.wait_event(ready_event)
                with torch.cuda.stream(stream):
                    jpeg = _decode_preview_sample(
                        sample,
                        vae,
                        preview_state["max_width"],
                        frame_indices,
                    )
                stream.synchronize()
            emit({
                "preview_base64_jpeg": jpeg,
                "preview_step": step,
                "preview_total": total,
                "preview_frames": labels,
                "preview_seconds": round(time.time() - started, 2),
            })
        except Exception as e:
            preview_state["failed"] = True
            emit({"loading_status": f"live preview disabled after decode error: {type(e).__name__}: {e}"})

    if preview_state.get("async", False):
        worker = threading.Thread(target=run_preview, name="saient-latent-preview", daemon=True)
        preview_state["worker"] = worker
        worker.start()
    else:
        run_preview()


def _retrieve_latents_argmax(encoder_output):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from VAE encoder output")


def _frame_to_first_frame_condition(frame, height, width):
    """Encode one decoded RGB frame into Wan's first-frame latent condition.

    This is the same expand_timesteps conditioning path used by Diffusers'
    WanImageToVideoPipeline, but scoped to a single seam frame so 30s low-VRAM
    chunking does not restart each segment from pure noise.
    """
    import torch
    import numpy as np
    from PIL import Image

    if frame is None:
        return None
    vae = PIPE.vae
    if next(vae.parameters()).device.type != "cuda":
        vae.to("cuda:0")

    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
    else:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")

    with torch.inference_mode():
        image_tensor = PIPE.video_processor.preprocess(image, height=height, width=width).to(
            "cuda:0", dtype=torch.float32
        )
        video_condition = image_tensor.unsqueeze(2).to(device="cuda:0", dtype=vae.dtype)
        latent_condition = _retrieve_latents_argmax(vae.encode(video_condition))
        latent_condition = latent_condition.to(torch.float32)
        zc = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, zc, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zc, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = (latent_condition - latents_mean) * latents_std

    del image_tensor, video_condition
    return latent_condition.detach().to("cpu")


def _denoise_dual_expert(
    pe,
    ne,
    total,
    cfg_scale,
    cfg_scale_2,
    height,
    width,
    num_frames,
    generator,
    cb,
    first_frame_condition=None,
):
    """Wan2.2 T2V-A14B staged denoise. Diffusers' normal WanPipeline keeps both
    experts addressable; on this 16 GB display GPU we keep only one expert live:
    high-noise first, then unload it and load the low-noise expert at boundary_ratio."""
    import torch
    global PIPE

    if getattr(PIPE, "transformer", None) is None:
        if getattr(PIPE, "transformer_2", None) is not None:
            low = PIPE.transformer_2
            PIPE.transformer_2 = None
            del low
            _free_cuda()
        emit({"loading_status": "Wan2.2: reloading high-noise expert for new clip…"})
        _reload_transformer()
    if getattr(PIPE, "transformer", None) is None:
        raise RuntimeError("Wan2.2 high-noise expert failed to load (PIPE.transformer is None)")
    if PIPE.scheduler is None:
        raise RuntimeError("Wan2.2 scheduler is None — model load incomplete")
    _enable_transformer_group_offload(PIPE.transformer, "high-noise expert")
    if LORA_PATH:
        _ensure_lora_adapter(LORA_STRENGTH, load_into_transformer_2=False)
    _configure_denoise_cache(PIPE.transformer, "high-noise expert")

    device = "cuda:0"
    high = PIPE.transformer
    transformer_dtype = high.dtype
    pe = pe.to(transformer_dtype)
    if ne is not None:
        ne = ne.to(transformer_dtype)

    if num_frames % PIPE.vae_scale_factor_temporal != 1:
        num_frames = num_frames // PIPE.vae_scale_factor_temporal * PIPE.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    # Never do PIPE.transformer.config after a possible high→low swap mid-function;
    # capture config from the live high expert now.
    patch_size = high.config.patch_size
    h_multiple_of = PIPE.vae_scale_factor_spatial * patch_size[1]
    w_multiple_of = PIPE.vae_scale_factor_spatial * patch_size[2]
    height = height // h_multiple_of * h_multiple_of
    width = width // w_multiple_of * w_multiple_of

    PIPE.scheduler.set_timesteps(total, device=device)
    timesteps = PIPE.scheduler.timesteps
    num_channels_latents = high.config.in_channels
    latent_dtype = _denoise_latent_dtype(transformer_dtype, height, width, num_frames)
    latents = PIPE.prepare_latents(
        1, num_channels_latents, height, width, num_frames, latent_dtype,
        device, generator, None)
    condition = None
    first_frame_mask = None
    expand_ts = bool(getattr(getattr(PIPE, "config", None), "expand_timesteps", False))
    if first_frame_condition is not None:
        if not expand_ts:
            emit({"loading_status": "  ⚠ seam conditioning skipped: model does not use expand_timesteps"})
        else:
            condition = first_frame_condition.to(device=device, dtype=latents.dtype, non_blocking=True)
            if condition.shape[1] != latents.shape[1] or condition.shape[-2:] != latents.shape[-2:]:
                raise RuntimeError(
                    "seam conditioning latent shape mismatch: "
                    f"condition {tuple(condition.shape)} vs latents {tuple(latents.shape)}"
                )
            first_frame_mask = torch.ones(
                1, 1, latents.shape[2], latents.shape[3], latents.shape[4],
                dtype=latents.dtype, device=device,
            )
            first_frame_mask[:, :, 0] = 0

    # boundary_ratio=None means single-expert for the whole schedule (no low swap).
    br = getattr(getattr(PIPE, "config", None), "boundary_ratio", None)
    if br is None:
        boundary = None
        emit({"loading_status": "Wan2.2 T2V: no boundary_ratio — high-noise expert for all steps"})
    else:
        boundary = float(br) * float(PIPE.scheduler.config.num_train_timesteps)
        emit({"loading_status": f"Wan2.2 T2V: high-noise expert until t < {boundary:.0f}…"})
    switched = False
    PIPE._guidance_scale = cfg_scale
    PIPE._guidance_scale_2 = cfg_scale_2
    PIPE._attention_kwargs = None
    PIPE._current_timestep = None
    PIPE._interrupt = False
    PIPE._num_timesteps = len(timesteps)

    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            use_low = boundary is not None and float(t.detach().cpu()) < boundary
            if use_low and not switched:
                emit({"loading_status": "Wan2.2: switching high-noise → low-noise expert…"})
                torch.cuda.synchronize()
                # The previous iteration's loop ref pins the high expert on the GPU —
                # drop it or the del below frees nothing and the low-expert load OOMs.
                with torch.inference_mode(False):
                    current_model = None
                    high = PIPE.transformer
                    PIPE.transformer = None
                    del high
                    _free_cuda()
                    _reload_transformer2()
                    if getattr(PIPE, "transformer_2", None) is None:
                        raise RuntimeError("Wan2.2 low-noise expert failed to load (PIPE.transformer_2 is None)")
                    if _env_flag("SAIENT_WAN_GROUP_OFFLOAD_LOW", False):
                        _enable_transformer_group_offload(PIPE.transformer_2, "low-noise expert")
                    transformer_dtype = PIPE.transformer_2.dtype
                    pe = pe.to(transformer_dtype)
                    if ne is not None:
                        ne = ne.to(transformer_dtype)
                switched = True

            current_model = PIPE.transformer_2 if use_low else PIPE.transformer
            current_cfg_scale = cfg_scale_2 if use_low else cfg_scale
            if current_model is None:
                which = "low-noise (transformer_2)" if use_low else "high-noise (transformer)"
                raise RuntimeError(f"Wan2.2 expert switch failed: {which} is missing")

            if first_frame_mask is not None:
                latent_model_input = latents.clone()
                latent_model_input[:, :, :condition.shape[2]] = condition
                latent_model_input = latent_model_input.to(transformer_dtype)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = latents.to(transformer_dtype)
                if expand_ts:
                    timestep = _timestep_for_latents(t, latents)
                else:
                    timestep = t.expand(latents.shape[0])

            with current_model.cache_context("cond"):
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=pe,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

            if current_cfg_scale > 1.0:
                with current_model.cache_context("uncond"):
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=ne,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                noise_pred.sub_(noise_uncond).mul_(current_cfg_scale).add_(noise_uncond)
                del noise_uncond

            latents = PIPE.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            preview_latents = None
            if bool(getattr(PIPE.scheduler, "predict_x0", False)):
                model_outputs = getattr(PIPE.scheduler, "model_outputs", None)
                if model_outputs and torch.is_tensor(model_outputs[-1]):
                    candidate = model_outputs[-1]
                    if candidate.shape == latents.shape:
                        preview_latents = candidate
            del noise_pred, latent_model_input, timestep
            # empty_cache every step was ~free on short runs but multi-chunk Lightning
            # was paying PCIe/sync tax 80×; only scrub mid-run on long non-distill loops.
            if LOW_VRAM_ACTIVE and total >= 12 and i % 4 == 3:
                torch.cuda.empty_cache()
            callback_tensors = {"latents": latents}
            if preview_latents is not None:
                callback_tensors["_saient_preview_latents"] = preview_latents
            out = cb(PIPE, i, t, callback_tensors)
            latents = out.pop("latents", latents)

    if first_frame_mask is not None:
        latents[:, :, :condition.shape[2]] = condition
        del condition, first_frame_mask
    PIPE._current_timestep = None
    _release_denoise_cache(getattr(PIPE, "transformer", None))
    _release_denoise_cache(getattr(PIPE, "transformer_2", None))
    return latents


def generate(req):
    import torch
    global LOW_VRAM_ACTIVE, LOW_VRAM_ATTN_BACKEND, LORA_STRENGTH, BLOCK_OFFLOAD_REQ
    global DENOISE_CACHE_MODE, DENOISE_CACHE_THRESHOLD

    t0 = time.time()
    _free_cuda()  # release reserved blocks left over from the previous generation
    _vram()       # reset peak counter for this run
    total = int(req.get("steps", 30))
    cfg_scale = float(req.get("cfg_scale", 6.0))
    cfg_scale_2 = _low_noise_guidance(req, cfg_scale)
    cache_mode = str(req.get("denoise_cache") or "off").strip().lower()
    DENOISE_CACHE_MODE = "balanced" if cache_mode in ("balanced", "cache", "cached", "first_block") else "off"
    try:
        DENOISE_CACHE_THRESHOLD = max(0.001, min(float(req.get("cache_threshold", 0.10)), 0.25))
    except Exception:
        DENOISE_CACHE_THRESHOLD = 0.10
    do_cfg = max(cfg_scale, cfg_scale_2) > 1.0
    prompt = req.get("prompt", ""); neg = req.get("neg_prompt", "")
    height = int(req.get("height", 480)); width = int(req.get("width", 832))
    num_frames = int(req.get("num_frames", 49))
    fps = int(req.get("fps", 16))
    if DUAL_EXPERT and num_frames != 81:
        emit({"loading_status": (
            f"  ⚠ Wan2.2 A14B reference length is 81 frames; requested {num_frames} "
            f"({num_frames / max(fps, 1):.1f}s at {fps} FPS). Custom lengths can reduce temporal stability."
        )})
    preview_state = {
        "enabled": bool(req.get("preview", False)),
        "every": max(2, min(int(req.get("preview_every", 5)), max(total, 2))),
        "max_width": max(128, min(int(req.get("preview_max_width", 256)), 384)),
        "async": bool(DUAL_EXPERT),
        "worker": None,
        "stream": None,
        "failed": False,
    }
    image_b64 = (req.get("image_b64") or "").strip()
    wan_chunk_limit = 0
    if DUAL_EXPERT:
        wan_chunk_limit = int(os.environ.get("SAIENT_WAN_CHUNK_FRAMES", "121"))
        wan_chunk_limit = max(9, ((wan_chunk_limit - 1) // 4) * 4 + 1)
    needs_chunk = DUAL_EXPERT and num_frames > wan_chunk_limit
    high_res_pixels = int(os.environ.get("SAIENT_WAN_HIGHRES_PIXELS", "500000"))
    needs_highres_headroom = DUAL_EXPERT and (width * height) >= high_res_pixels
    long_clip = (num_frames / max(fps, 1)) >= 20
    BLOCK_OFFLOAD_REQ = bool(req.get("block_offload", False))
    LOW_VRAM_ACTIVE = (
        bool(req.get("low_vram", False))
        or BLOCK_OFFLOAD_REQ
        or needs_chunk
        or needs_highres_headroom
        or (DUAL_EXPERT and long_clip)
    )
    if BLOCK_OFFLOAD_REQ:
        emit({"loading_status": "park to RAM: transformer will stream block-by-block (fits over-VRAM clips; slower)"})
    LOW_VRAM_ATTN_BACKEND = (
        os.environ.get("SAIENT_WAN_HIGHRES_ATTN_BACKEND", "_native_flash")
        if needs_highres_headroom
        else None
    )
    if LOW_VRAM_ACTIVE:
        if needs_chunk and not req.get("low_vram", False):
            emit({"loading_status": (
                f"low-VRAM auto: {num_frames} frames exceeds safe {wan_chunk_limit}f 14B chunk limit"
            )})
        elif needs_highres_headroom and not req.get("low_vram", False):
            emit({"loading_status": (
                f"low-VRAM auto: {width}×{height} 14B needs high-res denoise headroom"
            )})
        emit({"loading_status": "low-VRAM mode: CPU prompt cache, low-VRAM attention, freed weights before VAE/video stages"})
        if needs_highres_headroom:
            emit({"loading_status": f"low-VRAM: high-res attention backend {LOW_VRAM_ATTN_BACKEND}"})
            _raise_highres_vram_cap(True)
        _set_attention_backend(getattr(PIPE, "transformer", None), True)
        _set_attention_backend(getattr(PIPE, "transformer_2", None), True)
        _cast_rope_for_low_vram(
            getattr(PIPE, "transformer", None),
            "high-noise expert" if DUAL_EXPERT else "transformer",
        )
        _cast_rope_for_low_vram(getattr(PIPE, "transformer_2", None), "low-noise expert")
        _wrap_condition_embedder_offload(
            getattr(PIPE, "transformer", None),
            "high-noise expert" if DUAL_EXPERT else "transformer",
        )
        _wrap_condition_embedder_offload(getattr(PIPE, "transformer_2", None), "low-noise expert")
    # The VAE is inactive during text encoding and T2V denoise. In low-VRAM mode,
    # park it even when it is not the "heavy" 5B VAE; the 30s A14B spike needs the
    # extra headroom and _decode_latents moves it back for decode.
    if (LOW_VRAM_ACTIVE or _vae_is_heavy()) and _vae_to("cpu"):
        emit({"loading_status": "headroom: VAE parked on CPU before encode/denoise…"})
    lora_profile = str(req.get("lora_profile") or "single").strip().lower()
    lora_split_step = max(1, min(total - 1, int(req.get("lora_split_step", max(1, total // 2))))) if total > 1 else 1
    lora_high = float(req.get("lora_strength_high", LORA_STRENGTH))
    lora_low = float(req.get("lora_strength_low", LORA_STRENGTH))
    lora_switched = False
    _configure_scheduler(req)
    if DUAL_EXPERT:
        emit({"loading_status": f"Wan2.2 guidance: high CFG {cfg_scale:g} · low CFG {cfg_scale_2:g}"})
    if LORA_PATH and lora_profile == "high_low":
        if DUAL_EXPERT:
            LORA_STRENGTH = lora_high
        else:
            _ensure_lora_adapter(lora_high)
        emit({"loading_status": f"LoRA high/low: {lora_high:g} for {lora_split_step} steps, then {lora_low:g}"})

    # ── Prompt embeddings: reuse cache, else load text encoder + encode ──────────
    # Return CLONES so the pipeline can never mutate the cached master in-place.
    key = (prompt, neg, do_cfg)
    if EMBED_CACHE["key"] == key and EMBED_CACHE["pe"] is not None:
        pe = EMBED_CACHE["pe"].to("cuda", non_blocking=True)
        ne = EMBED_CACHE["ne"].to("cuda", non_blocking=True) if EMBED_CACHE["ne"] is not None else None
        emit({"loading_status": "↻ reusing cached prompt embeds — text encoder skipped"})
    else:
        _t = time.time()
        # On a card too small for transformer + encoder at once (14B nf4), drop the
        # transformer off the GPU so the encoder fits, then reload it for denoise.
        if PARK_TRANSFORMER:
            emit({"loading_status": "16 GB fit: freeing transformer off GPU for the text encoder…"})
            _unload_transformer()
        # For quality bf16 offload: ensure transformer is in system RAM before we load the
        # ~5.5 GB text encoder. This is the only place we intentionally move it CPU-ward
        # between gens (saves a move if decode kept it on-GPU).
        if STREAM_TRANSFORMER:
            try:
                if next(PIPE.transformer.parameters()).device.type == "cuda":
                    emit({"loading_status": "quality: parking bf16 transformer for text-encoder headroom…"})
                    PIPE.transformer.to("cpu", non_blocking=True)
                    torch.cuda.synchronize()
                    _free_cuda()
            except Exception:
                pass
        pe, ne = encode(prompt, neg, do_cfg)
        if PARK_TRANSFORMER:
            _freed = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            emit({"loading_status": f"encoder freed ({_freed:.1f} GB resident) — reloading nf4 transformer → GPU…"})
            _reload_transformer()
        EMBED_CACHE["key"] = key
        EMBED_CACHE["pe"] = pe.detach().to("cpu").clone()
        EMBED_CACHE["ne"] = ne.detach().to("cpu").clone() if ne is not None else None
        emit({"loading_status": f"  ⏱ text encoder + encode (+park/reload): {time.time()-_t:.0f}s"})
    _free_cuda()
    _vram_stage("text encoding")

    # ── Denoise (time it via callback timestamps) ────────────────────────────────
    marks = {"first": None, "last": None, "step": None}
    progress_state = {"offset": 0, "total": total, "preview_frames": num_frames}

    def cb(_pipe, i, _t, kwargs):
        nonlocal lora_switched
        now = time.time()
        if marks["first"] is None:
            marks["first"] = now
        step_started = marks["step"] if marks["step"] is not None else t_dn
        marks["last"] = now
        marks["step"] = now
        if (
            LORA_PATH
            and lora_profile == "high_low"
            and not lora_switched
            and i + 1 >= lora_split_step
            and i + 1 < total
        ):
            _ensure_lora_adapter(
                lora_low,
                load_into_transformer_2=(
                    getattr(PIPE, "transformer_2", None) is not None
                    and getattr(PIPE, "transformer", None) is None
                ),
            )
            lora_switched = True
            emit({"loading_status": f"LoRA low-noise strength → {lora_low:g}"})
        emit({
            "step": progress_state["offset"] + i + 1,
            "total": progress_state["total"],
            "step_seconds": round(now - step_started, 1),
            "elapsed_seconds": round(now - t_dn, 1),
        })
        preview_latents = kwargs.pop("_saient_preview_latents", kwargs.get("latents"))
        if preview_latents is not None:
            _start_latent_preview(
                preview_state,
                preview_latents,
                progress_state["offset"] + i + 1,
                progress_state["total"],
                progress_state["preview_frames"],
            )
        return kwargs

    generator = None
    seed = req.get("seed", -1)
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    t_dn = time.time()
    if STREAM_TRANSFORMER:
        # Quality mode: stream the bf16 transformer from RAM onto the GPU for denoise.
        # Use non_blocking + sync for potentially faster PCIe move. We no longer park
        # immediately after — decode gets to try with weights resident.
        emit({"loading_status": "quality: streaming bf16 transformer → GPU for denoise…"})
        try:
            if next(PIPE.transformer.parameters()).device.type != "cuda":
                PIPE.transformer.to("cuda:0", non_blocking=True)
                torch.cuda.synchronize()
        except Exception:
            PIPE.transformer.to("cuda:0")
            _free_cuda()
        else:
            _free_cuda()
    # output_type="latent": the pipe RETURNS the denoised latents without decoding, so
    # the denoise activation set is freed before the VAE decode. The Wan2.2-5B VAE (48
    # latent ch) decode is heavy and was OOMing by ~130 MB when run inline (denoise
    # tensors still resident). We decode separately below with the GPU freed.
    frames = None
    chunked_denoise_s = None
    chunked_decode_s = 0.0
    if image_b64:
        if DUAL_EXPERT:
            emit({"error": "Wan2.2 T2V-A14B is text-to-video only — clear the image input or choose an I2V model."})
            return
        # ── Image-to-video ──────────────────────────────────────────────────────
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB").resize((width, height))
        # i2v encodes the input image through the VAE inside the pipe call, so the VAE must
        # be back on the GPU (we parked it before encode for text-encoder headroom).
        if _vae_to("cuda:0"):
            emit({"loading_status": "i2v: VAE back on GPU for image conditioning…"})
        pipe = _i2v_pipe()
        emit({"loading_status": "i2v: conditioning on your image, denoising…"})
        latents = pipe(
            image=img,
            prompt_embeds=pe, negative_prompt_embeds=ne,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=total, guidance_scale=cfg_scale,
            generator=generator, callback_on_step_end=cb,
            output_type="latent",
        ).frames
    else:
        # t2v doesn't use the VAE during denoise. A heavy (5B) VAE was already parked on CPU
        # before encode; this is a defensive re-park in case it's somehow still resident.
        # _decode_latents moves it back for decode. (1.3B VAE is tiny — left wherever it is.)
        if (LOW_VRAM_ACTIVE or _vae_is_heavy()) and _vae_to("cpu"):
            emit({"loading_status": "headroom: VAE parked on CPU during denoise…"})
        if not STREAM_TRANSFORMER and not DUAL_EXPERT and getattr(PIPE, "transformer", None) is None:
            emit({"loading_status": "low-VRAM: reloading transformer for denoise…"})
            _reload_transformer()
        if LOW_VRAM_ACTIVE and not STREAM_TRANSFORMER and not DUAL_EXPERT:
            _enable_transformer_group_offload(getattr(PIPE, "transformer", None), "transformer")
        if not STREAM_TRANSFORMER:
            emit({"loading_status": "denoising…"})
        if DUAL_EXPERT:
            chunk_limit = wan_chunk_limit
            # Fewer chunks = fewer high/low expert reloads. At SD/low res the latent
            # tensor is small enough that larger chunks still fit 16GB.
            pixels = max(width * height, 1)
            if pixels <= 640 * 480:
                chunk_limit = max(chunk_limit, int(os.environ.get("SAIENT_WAN_CHUNK_FRAMES_SD", "241")))
            elif pixels <= 832 * 480:
                chunk_limit = max(chunk_limit, int(os.environ.get("SAIENT_WAN_CHUNK_FRAMES_SD", "161")))
            chunk_limit = max(9, ((chunk_limit - 1) // 4) * 4 + 1)
            do_chunk = LOW_VRAM_ACTIVE and num_frames > chunk_limit
            if do_chunk:
                import math
                chunks = max(1, math.ceil((num_frames - 1) / max(chunk_limit - 1, 1)))
                progress_state["total"] = total * chunks
                emit({"loading_status": (
                    f"low-VRAM: chunked T2V denoise {chunks}×{chunk_limit}f "
                    f"(continues to {num_frames}f; fewer expert reloads)"
                )})
                combined_frames = []
                next_condition = None
                denoise_elapsed = 0.0
                decode_elapsed = 0.0
                denoise_stats = None
                decode_stats = None
                # expand_timesteps is required for first-frame seam cond; T2V-A14B usually
                # has it off — skip the encode entirely instead of paying VAE for a no-op.
                can_seam = bool(getattr(getattr(PIPE, "config", None), "expand_timesteps", False))
                if not can_seam:
                    emit({"loading_status": (
                        "low-VRAM: T2V has no expand_timesteps — skipping seam encode "
                        "(prompt continuity only; saves encode+reload tax)"
                    )})

                def merge_stats(prev, cur):
                    cur = dict(cur)
                    if prev is None:
                        return cur
                    cur["peak"] = max(prev.get("peak", 0.0), cur.get("peak", 0.0))
                    cur["reserved"] = max(prev.get("reserved", 0.0), cur.get("reserved", 0.0))
                    return cur

                for ci in range(chunks):
                    have = len(combined_frames)
                    need = num_frames - have
                    seg_frames = min(chunk_limit, need if ci == 0 else need + 1)
                    seg_frames = max(9, ((seg_frames - 1) // 4) * 4 + 1)
                    continuation = " · continuing from previous frame" if next_condition is not None else ""
                    emit({"loading_status": f"low-VRAM: chunk {ci + 1}/{chunks} · {seg_frames} frames{continuation}"})
                    progress_state["offset"] = ci * total
                    progress_state["preview_frames"] = seg_frames
                    if (LOW_VRAM_ACTIVE or _vae_is_heavy()) and _vae_to("cpu"):
                        emit({"loading_status": "headroom: VAE parked on CPU during chunk denoise…"})
                    seg_t0 = time.time()
                    seg_latents = _denoise_dual_expert(
                        pe, ne, total, cfg_scale, cfg_scale_2, height, width, seg_frames, generator, cb,
                        first_frame_condition=next_condition)
                    _wait_for_preview(preview_state)
                    next_condition = None
                    denoise_elapsed += time.time() - seg_t0
                    denoise_stats = merge_stats(
                        denoise_stats,
                        _vram_stage("denoising", reset_peak=True, emit_line=False),
                    )
                    # Always free the live expert before VAE decode. Dual-expert denoise
                    # ends on transformer_2; leaving it resident + poking LoRA/pipe APIs
                    # caused 'NoneType' .config crashes on the missing high expert.
                    if getattr(PIPE, "transformer", None) is not None or getattr(PIPE, "transformer_2", None) is not None:
                        emit({"loading_status": "low-VRAM: freeing expert before VAE decode…"})
                        _unload_transformer()
                    dec_t0 = time.time()
                    seg_video = _decode_latents(seg_latents)
                    del seg_latents
                    if hasattr(seg_video, "ndim") and getattr(seg_video, "ndim", 0) == 4:
                        seg_list = [seg_video[i] for i in range(seg_video.shape[0])]
                    else:
                        seg_list = list(seg_video)
                    if ci > 0 and seg_list:
                        seg_list = seg_list[1:]
                    append_count = max(num_frames - len(combined_frames), 0)
                    appended = seg_list[:append_count]
                    combined_frames.extend(appended)
                    if ci < chunks - 1 and appended and can_seam:
                        seam_frame = appended[-1]
                        fh, fw = seam_frame.shape[:2]
                        emit({"loading_status": "low-VRAM: encoding seam frame for next chunk continuity…"})
                        next_condition = _frame_to_first_frame_condition(seam_frame, fh, fw)
                    decode_elapsed += time.time() - dec_t0
                    decode_stats = merge_stats(
                        decode_stats,
                        _vram_stage("VAE decode", reset_peak=True, emit_line=False),
                    )
                    emit({"loading_status": f"low-VRAM: continued {len(combined_frames)}/{num_frames} frames"})
                    del seg_video, seg_list, appended
                    if LOW_VRAM_ACTIVE and _vae_to("cpu"):
                        emit({"loading_status": "low-VRAM: VAE parked on CPU after chunk decode…"})
                    _free_cuda()

                frames = combined_frames[:num_frames]
                if denoise_stats is not None:
                    VRAM_REPORT["denoising"] = denoise_stats
                if decode_stats is not None:
                    VRAM_REPORT["VAE decode"] = decode_stats
                chunked_denoise_s = denoise_elapsed
                chunked_decode_s = decode_elapsed
                latents = None
            else:
                latents = _denoise_dual_expert(
                    pe, ne, total, cfg_scale, cfg_scale_2, height, width, num_frames, generator, cb)
        else:
            _configure_denoise_cache(getattr(PIPE, "transformer", None), "transformer")
            latents = PIPE(
                prompt_embeds=pe, negative_prompt_embeds=ne,
                height=height, width=width, num_frames=num_frames,
                num_inference_steps=total, guidance_scale=cfg_scale,
                generator=generator, callback_on_step_end=cb,
                output_type="latent",
            ).frames
    _wait_for_preview(preview_state)
    t_dn_end = time.time()
    _release_denoise_cache(getattr(PIPE, "transformer", None))
    _release_denoise_cache(getattr(PIPE, "transformer_2", None))
    # For quality (STREAM_TRANSFORMER): leave the bf16 transformer on GPU for decode if it fits.
    # We park lazily (1) before the next prompt's text-encoder (to free headroom), or (2) inside
    # _decode_latents on OOM. This cuts one full ~10 GB PCIe roundtrip for single gens and when
    # decode headroom exists. Aim: total gen time in the 30s–60s range on capable cards.
    if STREAM_TRANSFORMER:
        resident = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        emit({"loading_status": f"denoise done ({t_dn_end - t_dn:.0f}s) · {resident:.1f} GB resident · decoding (weights left on GPU if room)…"})
    else:
        resident = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        emit({"loading_status": f"denoise done ({t_dn_end - t_dn:.0f}s) · {resident:.1f} GB resident · decoding…"})
    del pe, ne
    if frames is None:
        _free_cuda()
        _vram_stage("denoising")
        if (DUAL_EXPERT or LOW_VRAM_ACTIVE) and not STREAM_TRANSFORMER:
            if getattr(PIPE, "transformer", None) is not None or getattr(PIPE, "transformer_2", None) is not None:
                label = "finished dual expert" if DUAL_EXPERT else "inactive transformer"
                emit({"loading_status": f"freeing {label} before VAE decode…"})
                _unload_transformer()
        frames = _decode_latents(latents)
        del latents
        t_after = time.time()
        denoise_s = (marks["last"] - marks["first"]) if (marks["first"] and marks["last"]) else (t_dn_end - t_dn)
        decode_s = t_after - t_dn_end
        _free_cuda()
        _vram_stage("VAE decode")
    else:
        t_after = time.time()
        denoise_s = chunked_denoise_s if chunked_denoise_s is not None else (t_dn_end - t_dn)
        decode_s = chunked_decode_s
    sps = denoise_s / max(total - 1, 1)
    emit({"loading_status": f"  ⏱ denoise: {denoise_s:.0f}s ({sps:.1f}s/step) · decode: {decode_s:.0f}s"})
    if LOW_VRAM_ACTIVE and _vae_is_heavy() and _vae_to("cpu"):
        emit({"loading_status": "low-VRAM: VAE parked on CPU before video assembly…"})
    _free_cuda()

    prev_b64 = (req.get("previous_video_b64") or "").strip()
    image_b64_for_seed = (req.get("image_b64") or "").strip()

    # --- Extension stitching: try to return original + new segment ---
    if prev_b64 and frames is not None and (len(frames) > 0 if hasattr(frames, '__len__') else bool(frames)):
        # Force the very first frame of the *new* segment to be exactly the seed image the user
        # provided (the true last frame of the previous clip). This prevents the generated
        # frame0 from being a slightly brighter/different VAE reconstruction.
        try:
            if image_b64_for_seed:
                import io
                from PIL import Image
                import numpy as _np
                seed = Image.open(io.BytesIO(base64.b64decode(image_b64_for_seed))).convert("RGB")
                # resize to match what the model produced
                h, w = frames[0].shape[:2]
                seed = seed.resize((w, h), Image.LANCZOS)
                frames[0] = _np.asarray(seed, dtype=_np.uint8)
        except Exception:
            pass

        # Frame-level concat for extensions:
        # - drop the duplicate seed frame (frame 0 of the new chunk is the previous tail)
        # - color/exposure match the new chunk to the previous tail
        # - for pure T2V prompt-only continuation, keep the old crossfade fallback
        force_seam = req.get("force_seam_blend", False)
        do_force_blend = force_seam or not image_b64_for_seed
        try:
            prev_frames = _b64_to_frames(prev_b64)
            if prev_frames:
                ext = frames[1:] if len(frames) > 1 else frames
                if hasattr(ext, 'ndim') and getattr(ext, 'ndim', 0) == 4:
                    ext = [ext[i] for i in range(ext.shape[0])]
                ext = list(ext)
                if image_b64_for_seed:
                    ext = _match_extension_to_tail(prev_frames, ext)
                    emit({"loading_status": "seam: dropped duplicate seed frame + matched chunk color/exposure to previous tail"})
                elif do_force_blend:
                    if len(ext) > 0:
                        ext[0] = prev_frames[-1].copy()
                    join = min(12, len(prev_frames), len(ext) if isinstance(ext, (list, tuple)) else 0)
                    import numpy as _np
                    for i in range(join):
                        t = (i + 1) / float(join + 1)
                        alpha = t * t * (3 - 2 * t)
                        p = _np.asarray(prev_frames[-join + i], dtype=_np.float32)
                        e = _np.asarray(ext[i], dtype=_np.float32)
                        blended = (p * (1.0 - alpha) + e * alpha).astype(_np.uint8)
                        prev_frames[-join + i] = blended
                    emit({"loading_status": "seam: prompt-only T2V crossfade blend applied"})
                combined_frames = prev_frames + ext
                b64 = _frames_to_b64(combined_frames, fps)
                _emit_video_result(b64, len(combined_frames), time.time() - t0, extended=True)
                return
        except Exception as ce:
            emit({"loading_status": f"⚠ frame concat/color match failed ({ce})"})

        # Prefer fast container-level concat with ffmpeg (keeps original part bit-identical,
        # adds the new segment length). This makes "3s -> extend 3s" actually produce 6s.
        combined_b64 = _ffmpeg_concat(prev_b64, frames, fps)
        if combined_b64:
            # We don't decode the previous again just for the count (avoids imageio dep).
            # The mp4 itself is the correct longer duration.
            _emit_video_result(combined_b64, len(frames), time.time() - t0, extended=True)
            return
        else:
            # Fallback to in-memory frame concat (re-encodes everything but still gives correct length)
            try:
                prev_frames = _b64_to_frames(prev_b64)
                if prev_frames:
                    ext = frames[1:] if len(frames) > 1 else frames
                    if hasattr(ext, 'ndim') and getattr(ext, 'ndim', 0) == 4:  # ndarray (T,H,W,C)
                        ext = [ext[i] for i in range(ext.shape[0])]
                    # Force exact start from previous last frame for visual continuity
                    if len(ext) > 0 and len(prev_frames) > 0:
                        ext[0] = prev_frames[-1].copy()
                    # blend for this fallback too
                    join = min(12, len(prev_frames), len(ext) if isinstance(ext, (list, tuple)) else 0)
                    import numpy as _np
                    for i in range(join):
                        t = (i + 1) / float(join + 1)
                        alpha = t * t * (3 - 2 * t)
                        p = _np.asarray(prev_frames[-join + i], dtype=_np.float32)
                        e = _np.asarray(ext[i], dtype=_np.float32)
                        blended = (p * (1.0 - alpha) + e * alpha).astype(_np.uint8)
                        prev_frames[-join + i] = blended
                    combined_frames = prev_frames + (ext if isinstance(ext, (list, tuple)) else list(ext))
                    b64 = _frames_to_b64(combined_frames, fps)
                    _emit_video_result(b64, len(combined_frames), time.time() - t0, extended=True)
                    return
            except Exception as ce:
                emit({"loading_status": f"⚠ frame concat fallback failed ({ce})"})

    # Normal path: just the clip we generated (or extend concat failed)
    emit({"loading_status": "video export: H.264/libx264 high quality…"})
    b64 = _frames_to_b64(frames, fps)
    _emit_video_result(b64, len(frames), time.time() - t0)


def main():
    _watch_parent()   # die with the app — never orphan VRAM
    # Graceful teardown on SIGTERM (belt-and-braces for the quit-command path): free CUDA,
    # then exit — so the driver never has to reclaim a multi-GB context from a hard kill.
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), os._exit(0)))
    except Exception:
        pass
    load_line = sys.stdin.readline()
    if not load_line:
        return
    try:
        cfg = json.loads(load_line)
    except Exception as e:
        emit({"error": f"bad load config: {e}"})
        return

    try:
        emit({"loading_status": "importing torch + diffusers…"})
        load(cfg)
    except Exception as e:
        emit({"error": f"load failed: {e}", "trace": traceback.format_exc()[:1000]})
        return

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            continue
        if req.get("cmd") == "quit":   # graceful VRAM release (see _shutdown) — then exit clean
            _shutdown()
            break
        try:
            generate(req)
        except Exception as e:
            # Surface a short actionable message; full traceback still in "trace".
            tb = traceback.format_exc()
            emit({"error": f"{type(e).__name__}: {e}", "trace": tb[:2000]})


if __name__ == "__main__":
    main()
