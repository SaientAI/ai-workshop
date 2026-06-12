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
import base64, gc, json, os, sys, tempfile, time, traceback

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


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _vram():
    """(used_GB, peak_GB) on cuda:0, or (0,0) if no CUDA. Resets the peak counter."""
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return used, peak


def _lora_state_dict(path):
    """Load a LoRA and normalise it to diffusers Wan keys. Many community/DiffSynth
    LoRAs use NATIVE Wan names (`blocks.N.cross_attn.k.lora_A.default.weight`) which
    diffusers' own converter rejects — remap them to
    `transformer.blocks.N.attn2.to_k.lora_A.weight`. Pass diffusers-format ones through."""
    from safetensors.torch import load_file
    sd = load_file(path)
    keys = list(sd.keys())
    if any(k.startswith("transformer.") or "diffusion_model" in k for k in keys):
        return sd  # already diffusers-style
    if not any(("self_attn" in k or "cross_attn" in k or "ffn" in k) for k in keys):
        return sd  # unknown layout — let diffusers try
    amap = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
    ffnmap = {"0": "net.0.proj", "2": "net.2"}
    out = {}
    for k, v in sd.items():
        p = k.replace(".default.", ".").split(".")   # strip peft adapter infix
        if len(p) < 6 or p[0] != "blocks":
            continue
        blk = p[1]; tail = ".".join(p[4:])           # e.g. lora_A.weight
        if p[2] in ("self_attn", "cross_attn"):
            attn = "attn1" if p[2] == "self_attn" else "attn2"
            proj = amap.get(p[3])
            if proj:
                out[f"transformer.blocks.{blk}.{attn}.{proj}.{tail}"] = v
        elif p[2] == "ffn":
            sub = ffnmap.get(p[3])
            if sub:
                out[f"transformer.blocks.{blk}.ffn.{sub}.{tail}"] = v
    return out if out else sd


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


def _guard_loadable(model_path, use_lora, quality):
    """Refuse loads that would swap-freeze the whole PC instead of failing cleanly — the
    "waited 15 min then the machine died" case. Two killers on a 16 GB / 40 GB box:

    1. Dual-expert MoE (Wan2.2 A14B): a `transformer_2` subfolder means two ~14B experts.
       This single-transformer pipeline can't even drive the second expert, and the bf16
       read alone (~28-54 GB) tips the box into swap. Point the user at the model that DOES
       do HD here.
    2. bf16 path (Quality mode / LoRA) reads the FULL transformer into RAM. If that won't
       fit with headroom, refuse rather than thrash swap to death."""
    if os.path.isdir(os.path.join(model_path, "transformer_2")):
        raise RuntimeError(
            "This is a dual-expert (A14B) model — not supported on a 16 GB card yet: it "
            "needs both ~14B experts and a load big enough to swap-freeze the PC. "
            "Use Wan2.2-TI2V-5B for stable HD / 720p instead.")
    if use_lora or quality:
        need = _safetensors_gb(os.path.join(model_path, "transformer"))
        avail = _available_ram_gb()
        if avail is not None and need > 0 and need > avail - 6.0:
            mode = "Quality mode" if quality else "LoRA"
            raise RuntimeError(
                f"{mode} loads the full-precision transformer (~{need:.0f} GB) into RAM, but "
                f"only ~{avail:.0f} GB is free — that would swap-freeze the machine. Switch to "
                f"Fast mode (untick Quality) or pick a smaller model.")


def load(cfg):
    """Load the SMALL resident pieces (transformer 4-bit + VAE). Fast, low RAM."""
    global PIPE, TOKENIZER, MODEL_PATH, STREAM_TRANSFORMER
    import torch
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers import BitsAndBytesConfig as DBnb
    from transformers import AutoTokenizer

    MODEL_PATH = cfg.get("model_path", "")
    lora_path = (cfg.get("lora_path") or "").strip()
    lora_strength = float(cfg.get("lora_strength", 1.0))
    use_lora = bool(lora_path)
    precision = (cfg.get("precision") or "fast").strip().lower()
    quality = precision == "quality"
    # The transformer goes bf16 for BOTH quality mode and LoRA (4-bit breaks adapter scaling).
    # On a 16 GB card a resident bf16 5B (~10 GB) + the 5.5 GB text encoder at encode time OOMs,
    # so STREAM it: keep the bf16 transformer in CPU RAM and move it to the GPU only for the
    # denoise loop (see generate()), freeing the card for the encoder. Cheap CPU↔GPU hop for the
    # 1.3B (~2.6 GB); the win is heavy models (5B) that otherwise OOM at encode with a LoRA on.
    STREAM_TRANSFORMER = quality or use_lora

    # Refuse loads that would tip the box into swap (A14B dual-expert / oversized bf16) with
    # a clear message, BEFORE reading any weights — never freeze the machine.
    _guard_loadable(MODEL_PATH, use_lora, quality)
    dev = 0  # cuda:0

    t_load = time.time()
    _t = time.time()
    if use_lora or quality:
        # bf16 transformer (4-bit makes LoRA adapter scaling unreliable, and is the
        # quality ceiling for the denoise loop). GPU-resident for fast/LoRA; left on CPU
        # for quality so it can be streamed in/out around denoise.
        if quality:
            label = "bf16 — streamed from RAM (quality)"
        elif use_lora:
            label = "bf16 — for LoRA"
        else:
            label = "bf16"
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

    emit({"loading_status": "assembling pipeline (no text encoder yet)…"})
    # Build WanPipeline directly with text_encoder=None so the 21 GB component is
    # never loaded here. We construct via __init__ (not from_pretrained) to dodge
    # from_pretrained's "required component is None" validation; the scheduler is
    # the only thing left to load and it's a tiny JSON config.
    import diffusers as _df
    with open(os.path.join(MODEL_PATH, "scheduler", "scheduler_config.json")) as f:
        sched_cls = getattr(_df, json.load(f)["_class_name"])
    scheduler = sched_cls.from_pretrained(MODEL_PATH, subfolder="scheduler")
    try:
        PIPE = WanPipeline(vae=vae, text_encoder=None, tokenizer=TOKENIZER,
                           transformer=transformer, scheduler=scheduler)
    except Exception:
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
                tile_sample_stride_height=224, tile_sample_stride_width=224)
        except TypeError:
            PIPE.vae.enable_tiling()   # older diffusers without the kwargs
    PIPE.set_progress_bar_config(disable=True)

    if use_lora:
        import os as _os
        emit({"loading_status": f"applying LoRA {_os.path.basename(lora_path)} @ {lora_strength}…"})
        try:
            sd = _lora_state_dict(lora_path)
            PIPE.load_lora_weights(sd, adapter_name="extra")
            PIPE.set_adapters(["extra"], adapter_weights=[lora_strength])
            emit({"loading_status": f"  ✓ LoRA applied ({len(sd)} keys)"})
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

    # ── Prompt embeddings: reuse cache, else load text encoder + encode ──────────
    # Return CLONES so the pipeline can never mutate the cached master in-place.
    key = (prompt, neg, do_cfg)
    if EMBED_CACHE["key"] == key and EMBED_CACHE["pe"] is not None:
        pe = EMBED_CACHE["pe"].clone()
        ne = EMBED_CACHE["ne"].clone() if EMBED_CACHE["ne"] is not None else None
        emit({"loading_status": "↻ reusing cached prompt embeds — text encoder skipped"})
    else:
        _t = time.time()
        pe, ne = encode(prompt, neg, do_cfg)
        EMBED_CACHE["key"] = key
        EMBED_CACHE["pe"] = pe.detach().clone()
        EMBED_CACHE["ne"] = ne.detach().clone() if ne is not None else None
        emit({"loading_status": f"  ⏱ text encoder + encode: {time.time()-_t:.0f}s"})

    # ── Denoise (time it via callback timestamps) ────────────────────────────────
    marks = {"first": None, "last": None}

    def cb(_pipe, i, _t, kwargs):
        now = time.time()
        if marks["first"] is None:
            marks["first"] = now
        marks["last"] = now
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
    if image_b64:
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
        if _vae_is_heavy() and _vae_to("cpu"):
            emit({"loading_status": "headroom: VAE parked on CPU during denoise…"})
        if not STREAM_TRANSFORMER:
            emit({"loading_status": "denoising…"})
        latents = PIPE(
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
        try:
            generate(req)
        except Exception as e:
            emit({"error": str(e), "trace": traceback.format_exc()[:1000]})


if __name__ == "__main__":
    main()
