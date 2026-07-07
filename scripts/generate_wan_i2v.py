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
import base64, gc, json, os, signal, sys, tempfile, time, traceback

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
# Cache the ENCODED prompt embeddings (a few MB on GPU) keyed by (prompt, neg, cfg).
# Re-running the same prompt (seed/step/frame sweeps) then skips the ~90s text-
# encoder load entirely. We free the 5.5 GB encoder after each encode so it never
# competes with denoise activations — embeds are tiny, the encoder is not.
EMBED_CACHE = {"key": None, "pe": None, "ne": None}
I2V_PIPE = None    # WanImageToVideoPipeline, lazily assembled from the t2v components
# "fast" (default) = 4-bit transformer, GPU-resident — today's behaviour, unchanged.
# "quality" = bf16 transformer that LIVES in CPU RAM and is streamed onto the GPU only
# for the denoise loop (then parked back), so the full-precision weights never have to
# share VRAM with the text encoder or VAE. Better fidelity at the cost of a ~10 GB PCIe
# round-trip per generation. Set from the load config's "precision" field.
STREAM_TRANSFORMER = False
LORA_PATH = ""
LORA_STRENGTH = 1.0
SCHEDULER_BASE_CONFIG = None
# Big single-transformer models (14B nf4 ~8 GB) can't hold the transformer AND the 5.5 GB
# UMT5 text encoder on a 16 GB card at once — encoding on top of the resident transformer
# once hard-froze the display GPU. When PARK_TRANSFORMER is set (decided at load by VRAM
# budget), generate() drops the nf4 transformer off the GPU for the text-encode and RELOADS
# it from the packed nf4 cache for denoise, so only one heavy piece is ever GPU-resident.
NF4_CACHE_PATH = None   # dir of the packed 4-bit transformer, for reload-after-encode
RESIDENT_GB = 0.0       # VRAM resident after load (≈ transformer) — gates PARK_TRANSFORMER
PARK_TRANSFORMER = False


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    starving the display into a deadlock. Tune with SAIENT_VRAM_FRACTION (default 0.90)."""
    import torch
    if not torch.cuda.is_available():
        return
    try:
        frac = max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.97")), 0.98))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        emit({"loading_status": f"  · VRAM cap {frac:.0%} — headroom reserved for the display"})
    except Exception as e:
        emit({"loading_status": f"  ⚠ couldn't set VRAM cap ({e})"})


def _unload_transformer():
    """Free the GPU transformer so the 5.5 GB UMT5 encoder fits on a 16 GB card that can't
    hold both (14B nf4). We DELETE rather than move to CPU — bitsandbytes 4-bit weights don't
    change device cleanly; _reload_transformer() brings it back from the packed nf4 cache."""
    global PIPE, I2V_PIPE
    t = getattr(PIPE, "transformer", None)
    try:
        PIPE.transformer = None
        if I2V_PIPE is not None:
            I2V_PIPE.transformer = None
    except Exception:
        pass
    del t
    _free_cuda()


def _reload_transformer():
    """Reload the packed nf4 transformer from cache straight onto the GPU (the reliable
    direction for bnb 4-bit) once the text encoder has been freed, ready for denoise."""
    import torch
    from diffusers import WanTransformer3DModel
    global PIPE, I2V_PIPE
    t = WanTransformer3DModel.from_pretrained(
        NF4_CACHE_PATH, torch_dtype=torch.bfloat16, device_map={"": 0})
    PIPE.transformer = t
    if I2V_PIPE is not None:
        I2V_PIPE.transformer = t
    _free_cuda()
    if LORA_PATH:
        _apply_lora()


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


def _apply_lora(lora_path=None, lora_strength=None):
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
    emit({"loading_status": f"applying LoRA {_os.path.basename(LORA_PATH)} @ {LORA_STRENGTH}…"})
    sd = _lora_state_dict(LORA_PATH)
    PIPE.load_lora_weights(sd, adapter_name="extra")
    _set_lora_strength(LORA_STRENGTH)
    emit({"loading_status": f"  ✓ LoRA adapter active ({len(sd)} tensors)"})
    _free_cuda()


def _set_lora_strength(strength):
    """Change the active adapter weight without reloading the adapter."""
    global LORA_STRENGTH
    if not LORA_PATH:
        return
    LORA_STRENGTH = float(strength)
    if PIPE is not None:
        PIPE.set_adapters(["extra"], adapter_weights=[LORA_STRENGTH])
    if I2V_PIPE is not None:
        I2V_PIPE.set_adapters(["extra"], adapter_weights=[LORA_STRENGTH])


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


def _cache_dir(name):
    """Managed cache dir ~/.config/saient/<name>. If a pre-rebrand ~/.config/ai-workshop/<name>
    exists and the new one doesn't, migrate it in place on first use (instant rename, same
    filesystem) so we never re-quantize a cache the old brand already built."""
    new = os.path.expanduser(f"~/.config/saient/{name}")
    old = os.path.expanduser(f"~/.config/ai-workshop/{name}")
    if not os.path.exists(new) and os.path.exists(old):
        try:
            os.makedirs(os.path.dirname(new), exist_ok=True)
            os.rename(old, new)
        except Exception:
            return old
    return new


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
    "waited 15 min then the machine died" case. Two killers on a 16 GB / 40 GB box:

    1. Dual-expert MoE (Wan2.2 A14B): a `transformer_2` subfolder means two ~14B experts.
       This single-transformer pipeline can't even drive the second expert, and the bf16
       read alone (~28-54 GB) tips the box into swap. Point the user at the model that DOES
       do HD here.
    2. bf16 Quality mode reads the FULL transformer into RAM. If that won't fit with
       headroom, refuse rather than thrash swap to death."""
    if os.path.isdir(os.path.join(model_path, "transformer_2")):
        raise RuntimeError(
            "This is a dual-expert (A14B) model — not supported on a 16 GB card yet: it "
            "needs both ~14B experts and a load big enough to swap-freeze the PC. "
            "Use Wan2.2-TI2V-5B for stable HD / 720p instead.")
    if quality:
        need = _safetensors_gb(os.path.join(model_path, "transformer"))
        avail = _available_ram_gb()
        if avail is not None and need > 0 and need > avail - 6.0:
            raise RuntimeError(
                f"Quality mode loads the full-precision transformer (~{need:.0f} GB) into RAM, but "
                f"only ~{avail:.0f} GB is free — that would swap-freeze the machine. Switch to "
                f"Fast mode (untick Quality) or pick a smaller model.")


def load(cfg):
    """Load the SMALL resident pieces (transformer 4-bit + VAE). Fast, low RAM."""
    global PIPE, TOKENIZER, MODEL_PATH, STREAM_TRANSFORMER
    global NF4_CACHE_PATH, RESIDENT_GB, PARK_TRANSFORMER, LORA_PATH, LORA_STRENGTH
    global SCHEDULER_BASE_CONFIG
    import torch
    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers import BitsAndBytesConfig as DBnb
    from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

    _cap_vram()   # reserve VRAM for the desktop — this GPU also drives the display

    MODEL_PATH = cfg.get("model_path", "")
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
        transformer = None
        if os.path.exists(os.path.join(t_cache, "config.json")):
            emit({"loading_status": "loading pre-quantized transformer (cached 4-bit)…"})
            try:
                transformer = WanTransformer3DModel.from_pretrained(
                    t_cache, torch_dtype=torch.bfloat16, device_map={"": dev})
            except Exception as ce:
                emit({"loading_status": f"  ⚠ transformer cache load failed ({ce}); rebuilding…"})
                transformer = None
        if transformer is None:
            emit({"loading_status": "loading transformer (4-bit nf4) + caching for next time…"})
            dbnb = DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16)
            transformer = WanTransformer3DModel.from_pretrained(
                MODEL_PATH, subfolder="transformer",
                quantization_config=dbnb, torch_dtype=torch.bfloat16, device_map={"": dev})
            try:
                os.makedirs(t_cache, exist_ok=True)
                transformer.save_pretrained(t_cache)
                emit({"loading_status": "  ✓ transformer cached — fast loads from now on"})
            except Exception as se:
                emit({"loading_status": f"  ⚠ couldn't cache 4-bit transformer ({se})"})
    emit({"loading_status": f"  ⏱ transformer: {time.time()-_t:.0f}s"})

    emit({"loading_status": "loading VAE…"})
    # Wan's VAE is small; fp32 keeps decode stable. Lands on GPU below.
    vae = AutoencoderKLWan.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.float32)

    emit({"loading_status": "loading tokenizer…"})
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")

    emit({"loading_status": "loading CLIP image encoder…"})
    # Wan2.1-I2V conditions on the input frame via a CLIP-H image encoder (image_dim=1280
    # cross-attn) PLUS the VAE-encoded first frame. ~1.2 GB fp16, kept GPU-resident.
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_PATH, subfolder="image_encoder", torch_dtype=torch.float16)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_PATH, subfolder="image_processor")

    emit({"loading_status": "assembling i2v pipeline (no text encoder yet)…"})
    # Build WanImageToVideoPipeline with text_encoder=None (21 GB UMT5 loaded 4-bit at encode
    # time from the shared cache). expand_timesteps=False → the native Wan2.1-I2V path (uses the
    # CLIP image_encoder), NOT the Wan2.2-TI2V first-frame-latent trick.
    import diffusers as _df
    with open(os.path.join(MODEL_PATH, "scheduler", "scheduler_config.json")) as f:
        SCHEDULER_BASE_CONFIG = json.load(f)
        sched_cls = getattr(_df, SCHEDULER_BASE_CONFIG["_class_name"])
    scheduler = sched_cls.from_pretrained(MODEL_PATH, subfolder="scheduler")
    PIPE = WanImageToVideoPipeline(
        vae=vae, text_encoder=None, tokenizer=TOKENIZER, transformer=transformer,
        scheduler=scheduler, image_processor=image_processor, image_encoder=image_encoder,
        transformer_2=None, boundary_ratio=None, expand_timesteps=False)

    # Move only the non-quantized parts to GPU (the 4-bit transformer is already
    # placed by device_map; calling .to() on the whole pipe would error on it).
    try:
        PIPE.vae.to(f"cuda:{dev}")
        PIPE.image_encoder.to(f"cuda:{dev}")   # CLIP encoder resident (~1.2 GB) for image conditioning
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
                tile_sample_stride_height=224, tile_sample_stride_width=224)
        except TypeError:
            PIPE.vae.enable_tiling()   # older diffusers without the kwargs
    PIPE.set_progress_bar_config(disable=True)

    if use_lora:
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
    used, _ = _vram()
    RESIDENT_GB = used
    # Decide whether this model must free its transformer during the text-encode. Only the
    # resident-nf4 case needs it (bf16 stream already moves the transformer itself).
    # If transformer (~used) + the ~5.5 GB UMT5 encoder would exceed the VRAM cap, park it.
    _total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    _budget = _total * max(0.10, min(float(os.environ.get("SAIENT_VRAM_FRACTION", "0.97")), 0.98))
    PARK_TRANSFORMER = (not STREAM_TRANSFORMER) and (NF4_CACHE_PATH is not None) and (used + 6.0 > _budget)
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
        with torch.no_grad():
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
       peaks ~1.6 GB (fp32). Denorm replicates WanPipeline.__call__ exactly."""
    import torch
    vae = PIPE.vae
    _free_cuda()                              # reclaim denoise tensors before the decode
    if next(vae.parameters()).device.type != "cuda":
        vae.to("cuda:0")                      # was parked on CPU during denoise → back for decode
    zc = vae.config.z_dim
    with torch.no_grad():
        lat = latents.to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, zc, 1, 1, 1).to(lat.device, lat.dtype)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zc, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat / std + mean
        video = vae.decode(lat, return_dict=False)[0]
    return PIPE.video_processor.postprocess_video(video, output_type="np")[0]


def generate(req):
    import torch
    from diffusers.utils import export_to_video

    t0 = time.time()
    _free_cuda()  # release reserved blocks left over from the previous generation
    _vram()       # reset peak counter for this run
    # The previous generation's decode left a heavy VAE resident on the GPU. The text
    # encoder we may reload below is the single biggest transient (~5.5 GB), so on a 16 GB
    # card a 2nd generation with a NEW prompt (cache miss → TE reload) OOMed on top of it.
    # Park the heavy VAE on CPU now; it's brought back only for i2v conditioning / decode.
    if _vae_is_heavy() and _vae_to("cpu"):
        emit({"loading_status": "headroom: VAE parked on CPU before encode…"})
    total = int(req.get("steps", 30))
    cfg_scale = float(req.get("cfg_scale", 6.0))
    do_cfg = cfg_scale > 1.0
    prompt = req.get("prompt", ""); neg = req.get("neg_prompt", "")
    lora_profile = str(req.get("lora_profile") or "single").strip().lower()
    lora_split_step = max(1, min(total - 1, int(req.get("lora_split_step", max(1, total // 2))))) if total > 1 else 1
    lora_high = float(req.get("lora_strength_high", LORA_STRENGTH))
    lora_low = float(req.get("lora_strength_low", LORA_STRENGTH))
    lora_switched = False
    _configure_scheduler(req)
    if LORA_PATH and lora_profile == "high_low":
        _set_lora_strength(lora_high)
        emit({"loading_status": f"LoRA high/low: {lora_high:g} for {lora_split_step} steps, then {lora_low:g}"})

    # ── Prompt embeddings: reuse cache, else load text encoder + encode ──────────
    # Return CLONES so the pipeline can never mutate the cached master in-place.
    key = (prompt, neg, do_cfg)
    if EMBED_CACHE["key"] == key and EMBED_CACHE["pe"] is not None:
        pe = EMBED_CACHE["pe"].clone()
        ne = EMBED_CACHE["ne"].clone() if EMBED_CACHE["ne"] is not None else None
        emit({"loading_status": "↻ reusing cached prompt embeds — text encoder skipped"})
    else:
        _t = time.time()
        # On a card too small for transformer + encoder at once (14B nf4), drop the
        # transformer off the GPU so the encoder fits, then reload it for denoise.
        if PARK_TRANSFORMER:
            emit({"loading_status": "16 GB fit: freeing transformer off GPU for the text encoder…"})
            _unload_transformer()
        pe, ne = encode(prompt, neg, do_cfg)
        if PARK_TRANSFORMER:
            _freed = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            emit({"loading_status": f"encoder freed ({_freed:.1f} GB resident) — reloading nf4 transformer → GPU…"})
            _reload_transformer()
        EMBED_CACHE["key"] = key
        EMBED_CACHE["pe"] = pe.detach().clone()
        EMBED_CACHE["ne"] = ne.detach().clone() if ne is not None else None
        emit({"loading_status": f"  ⏱ text encoder + encode (+park/reload): {time.time()-_t:.0f}s"})

    # ── Denoise (time it via callback timestamps) ────────────────────────────────
    marks = {"first": None, "last": None}

    def cb(_pipe, i, _t, kwargs):
        nonlocal lora_switched
        now = time.time()
        if marks["first"] is None:
            marks["first"] = now
        marks["last"] = now
        if (
            LORA_PATH
            and lora_profile == "high_low"
            and not lora_switched
            and i + 1 >= lora_split_step
            and i + 1 < total
        ):
            _set_lora_strength(lora_low)
            lora_switched = True
            emit({"loading_status": f"LoRA low-noise strength → {lora_low:g}"})
        emit({"step": i + 1, "total": total})
        return kwargs

    generator = None
    seed = req.get("seed", -1)
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    height = int(req.get("height", 480)); width = int(req.get("width", 832))
    num_frames = int(req.get("num_frames", 49))
    image_b64 = (req.get("image_b64") or "").strip()

    t_dn = time.time()
    if STREAM_TRANSFORMER:
        # Quality mode: stream the bf16 transformer from RAM onto the GPU for denoise. Encode
        # is done (text encoder freed) and the heavy VAE is parked on CPU, so it has the card
        # to itself. Moved back to RAM before decode (below).
        emit({"loading_status": "quality: streaming bf16 transformer → GPU for denoise…"})
        PIPE.transformer.to("cuda:0"); _free_cuda()
    # output_type="latent": the pipe RETURNS the denoised latents without decoding, so
    # the denoise activation set is freed before the VAE decode. The Wan2.2-5B VAE (48
    # latent ch) decode is heavy and was OOMing by ~130 MB when run inline (denoise
    # tensors still resident). We decode separately below with the GPU freed.
    # ── Image-to-video (native Wan2.1-I2V; PIPE already IS the i2v pipeline) ──────
    if not image_b64:
        emit({"error": "Wan2.1-I2V is image-to-video — add an image (the 'Add image → animate' button)."})
        return
    import io
    from PIL import Image
    # LANCZOS = high-quality downscale. A real photo is usually >480p; a cheap resize here
    # softens the input frame → and i2v carries frame 0's quality through the whole clip.
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB").resize((width, height), Image.LANCZOS)
    emit({"loading_status": "i2v: encoding your image (CLIP + VAE), denoising…"})
    # PIPE encodes the image via CLIP + VAE internally; transformer(8.5)+CLIP(1.2)+acts ≈ 13.7 GB.
    latents = PIPE(
        image=img,
        prompt_embeds=pe, negative_prompt_embeds=ne,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=total, guidance_scale=cfg_scale,
        generator=generator, callback_on_step_end=cb,
        output_type="latent",
    ).frames
    t_dn_end = time.time()
    if STREAM_TRANSFORMER:
        # Park the bf16 transformer back in RAM so its ~10 GB of VRAM is freed for the VAE
        # decode and the next prompt's text-encoder load. Only one heavy component resident.
        PIPE.transformer.to("cpu"); _free_cuda()
        emit({"loading_status": "quality: transformer parked back in RAM — VRAM freed for decode…"})
    del pe, ne
    resident = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    emit({"loading_status": f"denoise done ({t_dn_end - t_dn:.0f}s) · {resident:.1f} GB resident · decoding (GPU freed)…"})
    frames = _decode_latents(latents)
    del latents
    t_after = time.time()
    denoise_s = (marks["last"] - marks["first"]) if (marks["first"] and marks["last"]) else (t_dn_end - t_dn)
    decode_s = t_after - t_dn_end
    _, peak = _vram()
    sps = denoise_s / max(total - 1, 1)
    emit({"loading_status": f"  ⏱ denoise: {denoise_s:.0f}s ({sps:.1f}s/step) · decode: {decode_s:.0f}s · peak VRAM {peak:.1f} GB"})
    _free_cuda()

    fps = int(req.get("fps", 16))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    export_to_video(frames, tmp.name, fps=fps)
    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    os.unlink(tmp.name)
    emit({"base64_mp4": b64, "frames": len(frames), "elapsed": round(time.time() - t0, 1)})


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
            emit({"error": str(e), "trace": traceback.format_exc()[:1000]})


if __name__ == "__main__":
    main()
