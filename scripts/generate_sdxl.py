#!/usr/bin/env python3
"""SDXL / SD1.5 image generation daemon for Saient.

Daemon protocol — newline-delimited JSON on stdin/stdout:
  stdin line 1:  {"model_path":"...","lora_path":"...","device":"auto"}
  stdout line 1: {"ready":true,"device":"cuda"}   OR  {"error":"..."}

  For each generation request:
  stdin:  {"prompt":"...","neg_prompt":"...","steps":20,"cfg_scale":7,"seed":42,"width":1024,"height":1024,"scheduler":"..."}
  stdout: {"step":1,"total":20}  (repeated — progress)
          {"base64_png":"...","device":"cuda","elapsed":3.5}  (final result)
          {"error":"..."}  (on failure — daemon keeps running)
"""
import base64, hashlib, io, json, os, re, sys, time
from saient_paths import configure_hf_cache, model_scan_dirs, shared_cache_dir

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
configure_hf_cache()
# Lets PyTorch reuse reserved-but-unallocated blocks instead of OOMing on fragmentation —
# the face-detail img2img pass needs ~0.5 GB on top of the resident base pipeline and was
# failing with hundreds of MB "reserved but unallocated". Must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_MODEL_DIRS = [str(p) for p in model_scan_dirs()]

_MONOCHROME_TERMS = (
    "black and white", "black-and-white", "b&w", "bw", "monochrome",
    "grayscale", "greyscale", "sepia", "pencil", "sketch", "line art",
    "lineart", "ink drawing", "manga panel", "coloring book",
)

_COLOR_PROMPT_TERMS = "full color, vibrant colors"
_COLOR_NEGATIVE_TERMS = (
    "monochrome, grayscale, greyscale, black and white, b&w, sepia, "
    "sketch, line art"
)
_ASSET_NEGATIVE_TERMS = (
    "cropped, close-up, portrait, bust, out of frame, duplicate, two characters, "
    "multiple views, character sheet, color palette, text, logo, blurry, lowres, "
    "bad anatomy, extra limbs"
)
_CREATURE_NEGATIVE_TERMS = "extra heads, malformed creature, unclear silhouette"
_BUILDING_NEGATIVE_TERMS = "person, humanoid, character, creature, face, arms, legs"
_PROP_NEGATIVE_TERMS = "person, humanoid, character, creature, face, arms, legs"
_ASSET_PROMPT_REPLACEMENTS = (
    ("(style of lords mobile:1.3)", "lords mobile style"),
    ("3d stylized digital art", "stylized 3d art"),
    ("mobile strategy game icon", "mobile game asset"),
    ("single game character asset", "single character asset"),
    ("exaggerated cartoon proportions", "cartoon proportions"),
    ("hand-painted textures", "hand-painted texture"),
    ("clean studio lighting", "studio lighting"),
    ("full body shot", "full body"),
    ("solid neutral gray background", "plain gray background"),
    ("solid neutral grey background", "plain gray background"),
)
_NEGATIVE_PRIORITY = (
    "lowres", "blurry", "bad anatomy", "cropped", "close-up", "portrait", "bust",
    "out of frame", "duplicate", "two characters", "multiple views",
    "character sheet", "color palette", "text", "watermark", "monochrome",
    "grayscale", "black and white", "sketch", "line art", "extra limbs",
)


def _find_local_sdxl_base():
    for d in _MODEL_DIRS:
        if not os.path.isdir(d):
            continue
        for entry in os.scandir(d):
            idx = os.path.join(entry.path, "model_index.json")
            if os.path.isfile(idx):
                try:
                    with open(idx) as f:
                        cls = json.load(f).get("_class_name", "")
                    if "XL" in cls:
                        return entry.path
                except Exception:
                    pass
    return None


def _weights_variant(model_path):
    """Return "fp16" for a diffusers dir that ships only fp16-variant weights.

    from_pretrained() looks for `diffusion_pytorch_model.safetensors` unless it is
    handed a variant. Repos mirrored fp16-only (SDXL base is commonly fetched that
    way) carry `diffusion_pytorch_model.fp16.safetensors` instead and fail to load
    with "no file named diffusion_pytorch_model.bin found". Decide from the main
    weight folder — unet for SD/SDXL, transformer for SD3-style pipelines.
    """
    for sub in ("unet", "transformer"):
        d = os.path.join(model_path, sub)
        if not os.path.isdir(d):
            continue
        names = [n for n in os.listdir(d) if n.endswith((".safetensors", ".bin"))]
        if any(".fp16." not in n for n in names):
            return None
        return "fp16" if names else None
    return None


def _is_sdxl_checkpoint(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return any(k.startswith("conditioner.") for k in f.keys())
    except Exception:
        name = os.path.basename(path).lower()
        return "xl" in name or "sdxl" in name


# ── Architecture descriptor ─────────────────────────────────────────────────────
# Single source of truth for "what model family is this and how should we run it".
# Computed once at load time from the pipe's real class name + component configs
# (never from filenames beyond the last-resort fallbacks below), cached on the pipe
# as pipe._saient_arch, and READ everywhere else in this file instead of re-derived.
# Do not add another isinstance()/type(pipe).__name__/cls_name-substring check
# anywhere else in this file — extend _DESCRIPTORS and architecture_of() instead.

#   clamp_resolution: whether max_resolution is a hard ceiling worth enforcing. SD1.x/SD2.x
#   U-Nets and their Turbo distillations visibly degrade (duplication/melting) well past
#   their trained resolution class, so clamping protects the user from a bad-by-construction
#   setting. SDXL and SD3 are trained multi-resolution and stay coherent well past their
#   "native" size, so max_resolution is informational there only — clamping them would take
#   away a legitimate use, not protect against a broken one.
_DESCRIPTORS = {
    "sd15":       dict(family="sd15",       default_cfg=7.5, default_steps=25,
                        supports_face_detail=True,  scheduler_mode="standard",
                        prompt_token_limit=77,  max_resolution=512,  clamp_resolution=True),
    "sd2":        dict(family="sd2",        default_cfg=7.5, default_steps=25,
                        supports_face_detail=True,  scheduler_mode="v_prediction",
                        prompt_token_limit=77,  max_resolution=768,  clamp_resolution=True),
    "sd_turbo":   dict(family="sd_turbo",   default_cfg=1.0, default_steps=2,
                        supports_face_detail=False, scheduler_mode="standard",
                        prompt_token_limit=77,  max_resolution=512,  clamp_resolution=True),
    "sdxl":       dict(family="sdxl",       default_cfg=7.0, default_steps=30,
                        supports_face_detail=True,  scheduler_mode="standard",
                        prompt_token_limit=77,  max_resolution=1536, clamp_resolution=False),
    "sdxl_turbo": dict(family="sdxl_turbo", default_cfg=1.0, default_steps=4,
                        supports_face_detail=False, scheduler_mode="standard",
                        prompt_token_limit=77,  max_resolution=1024, clamp_resolution=True),
    "sd3":        dict(family="sd3",        default_cfg=4.5, default_steps=28,
                        supports_face_detail=False, scheduler_mode="flow_match",
                        prompt_token_limit=256, max_resolution=1536, clamp_resolution=False),
    "unknown":    dict(family="unknown",    default_cfg=7.0, default_steps=20,
                        supports_face_detail=False, scheduler_mode="standard",
                        prompt_token_limit=77,  max_resolution=1024, clamp_resolution=False),
}


def _scheduler_config(model_path):
    sched_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    if not os.path.isfile(sched_path):
        return {}
    try:
        with open(sched_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _scheduler_signals_turbo(model_path):
    """Turbo/LCM-distilled SDXL and SD1.5 checkpoints reuse their base model's exact
    pipeline class name — _class_name alone can't tell them apart. The distillation
    shows up in the shipped scheduler config instead (trailing timestep spacing with
    linear interpolation is the signature diffusers itself uses for these configs)."""
    cfg = _scheduler_config(model_path)
    if not cfg:
        return False
    return cfg.get("timestep_spacing") == "trailing" and cfg.get("interpolation_type") == "linear"


def _scheduler_signals_v_prediction(model_path):
    """SD2.1-768 ships v_prediction in its own scheduler config; SD2.0-base and all
    SD1.x/SDXL models ship epsilon (diffusers' default). Config-based, not a filename
    guess — this is the actual field apply_scheduler()'s from_config() calls read."""
    return _scheduler_config(model_path).get("prediction_type") == "v_prediction"


def architecture_of(model_path, cls_name):
    """Classify a diffusers-folder model from its model_index.json class name plus,
    where the class name alone is ambiguous, its scheduler config. Returns a copy of
    the matching _DESCRIPTORS entry — treat the result as read-only."""
    if "StableDiffusion3" in cls_name:
        family = "sd3"
    elif "XL" in cls_name:
        family = "sdxl_turbo" if _scheduler_signals_turbo(model_path) else "sdxl"
    elif "StableDiffusion" in cls_name:
        if _scheduler_signals_turbo(model_path):
            family = "sd_turbo"
        elif _scheduler_signals_v_prediction(model_path):
            family = "sd2"
        else:
            family = "sd15"
    else:
        family = "unknown"
    return dict(_DESCRIPTORS[family])


def architecture_of_checkpoint(is_xl):
    """Single-file .safetensors checkpoints have no sidecar scheduler_config.json to
    read, so Turbo/v-prediction detection (which both depend on that file) isn't
    possible here — only the SD1.5-vs-SDXL split from _is_sdxl_checkpoint() is.
    This is a real, disclosed gap, not a silent misdetection: a Turbo checkpoint will
    get base-SDXL/SD1.5 defaults, which are wrong for it but at least won't crash."""
    return dict(_DESCRIPTORS["sdxl" if is_xl else "sd15"])


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _contains_any(text, terms):
    t = f" {text.lower().replace('-', ' ')} "
    return any(term.replace("-", " ") in t for term in terms)


def _append_terms(text, terms):
    text = text.strip()
    if not text:
        return terms
    existing = {term.strip().lower() for term in text.split(",") if term.strip()}
    missing = [term.strip() for term in terms.split(",") if term.strip() and term.strip().lower() not in existing]
    if not missing:
        return text
    return text.rstrip(" ,") + ", " + ", ".join(missing)


def _prepend_terms(text, terms):
    text = text.strip()
    if not text:
        return terms
    existing = {term.strip().lower() for term in text.split(",") if term.strip()}
    missing = [term.strip() for term in terms.split(",") if term.strip() and term.strip().lower() not in existing]
    if not missing:
        return text
    return ", ".join(missing) + ", " + text.lstrip(" ,")


def _append_if_missing_concept(text, term, aliases):
    if _contains_any(text, aliases):
        return text
    return _append_terms(text, term)


def _compact_asset_prompt(prompt):
    out = prompt
    for src, dst in _ASSET_PROMPT_REPLACEMENTS:
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip(" ,")


def _prioritize_negative_prompt(neg_prompt):
    terms = [t.strip() for t in neg_prompt.split(",") if t.strip()]
    seen = set()
    deduped = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)

    ordered = []
    for wanted in _NEGATIVE_PRIORITY:
        for term in deduped:
            if term.lower() == wanted and term not in ordered:
                ordered.append(term)
                break
    ordered.extend(term for term in deduped if term not in ordered)
    return ", ".join(ordered)


def _token_count(pipe, text, use_t5=False):
    # SD3's T5-XXL branch (tokenizer_3) has a real budget far beyond CLIP's 77 tokens —
    # measuring it against the CLIP tokenizers would silently reapply the wrong limit.
    if use_t5:
        t5 = getattr(pipe, "tokenizer_3", None)
        tokenizers = [t5] if t5 is not None else []
    else:
        tokenizers = [
            tok for tok in (getattr(pipe, "tokenizer", None), getattr(pipe, "tokenizer_2", None))
            if tok is not None
        ]
    if not tokenizers:
        return len(text.split())
    try:
        from transformers import logging as hf_logging
        previous = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            return max(len(tok(text, truncation=False).input_ids) for tok in tokenizers)
        finally:
            hf_logging.set_verbosity(previous)
    except Exception:
        return max(len(tok(text, truncation=False).input_ids) for tok in tokenizers)


def _fit_to_token_limit(pipe, text, limit=77, use_t5=False):
    count = _token_count(pipe, text, use_t5=use_t5)
    if count <= limit:
        return text, count, False

    parts = [p.strip() for p in text.split(",") if p.strip()]
    while len(parts) > 1:
        candidate = ", ".join(parts)
        count = _token_count(pipe, candidate, use_t5=use_t5)
        if count <= limit:
            return candidate, count, True
        parts.pop()
    return ", ".join(parts), _token_count(pipe, ", ".join(parts), use_t5=use_t5), True


def _chunk_token_ids(tokenizer, text, chunks_wanted=None):
    """Split text into CLIP-sized blocks of token ids, A1111/Comfy style.

    CLIP's context is 77 tokens: 75 of text plus BOS/EOS. Encoding each block
    separately and concatenating keeps a long prompt whole; the old behaviour threw
    away everything past the cap, which turned a photo prompt into whatever its first
    75 tokens happened to describe. `chunks_wanted` pads the short side so positive
    and negative come back the same length — CFG subtracts one from the other.
    """
    body = tokenizer.model_max_length - 2
    ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    blocks = [ids[i:i + body] for i in range(0, len(ids), body)] or [[]]
    if chunks_wanted is not None:
        blocks = blocks[:chunks_wanted]
        blocks += [[]] * (chunks_wanted - len(blocks))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = []
    for block in blocks:
        full = [tokenizer.bos_token_id] + block + [tokenizer.eos_token_id]
        full += [pad] * (tokenizer.model_max_length - len(full))
        out.append(full)
    return out


def _encode_long_prompt(pipe, prompt, neg_prompt, device):
    """Encode prompts of any length into embeddings, dropping nothing.

    Returns kwargs for pipe() in place of prompt/negative_prompt. Verified to match
    diffusers' own encode_prompt exactly (0.0 max abs difference) when the prompt fits
    in one chunk, so short prompts behave identically to before. SDXL concatenates the
    two encoders' penultimate hidden states (768 + 1280 = 2048) and takes the pooled
    embedding from the first chunk; SD1.5 has one encoder and uses its final state.
    """
    import torch

    is_xl = hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None
    tokenizers = [pipe.tokenizer] + ([pipe.tokenizer_2] if is_xl else [])
    encoders   = [pipe.text_encoder] + ([pipe.text_encoder_2] if is_xl else [])

    n_chunks = max(len(_chunk_token_ids(tok, text))
                   for tok in tokenizers for text in (prompt, neg_prompt))

    def encode(text):
        per_encoder, pooled = [], None
        for tok, enc in zip(tokenizers, encoders):
            ids = torch.tensor(_chunk_token_ids(tok, text, chunks_wanted=n_chunks),
                               dtype=torch.long, device=device)
            if is_xl:
                out = enc(ids, output_hidden_states=True)
                seq = out.hidden_states[-2]
                if enc is encoders[-1]:
                    pooled = out[0][:1]
            else:
                seq = enc(ids)[0]
            per_encoder.append(seq.reshape(1, -1, seq.shape[-1]))
        return (torch.cat(per_encoder, dim=-1) if is_xl else per_encoder[0]), pooled

    prompt_embeds, pooled = encode(prompt)
    neg_embeds, neg_pooled = encode(neg_prompt)
    kwargs = {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": neg_embeds}
    if is_xl:
        kwargs["pooled_prompt_embeds"] = pooled
        kwargs["negative_pooled_prompt_embeds"] = neg_pooled
    return kwargs, n_chunks


def apply_color_guard(prompt, neg_prompt):
    """Default normal prompts toward color, while preserving explicit sketch/mono requests."""
    if _contains_any(prompt, _MONOCHROME_TERMS):
        return prompt, neg_prompt
    # If the user already asked for color, don't add extra color prose to the
    # positive prompt. Long SDXL prompts get truncated at 77 CLIP tokens.
    if _contains_any(prompt, ("color", "colour", "vibrant", "richly colored", "colorful")):
        return prompt, _append_terms(neg_prompt, _COLOR_NEGATIVE_TERMS)
    return (
        _append_terms(prompt, _COLOR_PROMPT_TERMS),
        _append_terms(neg_prompt, _COLOR_NEGATIVE_TERMS),
    )


def apply_asset_guard(prompt, neg_prompt, asset_kind="humanoid", model_path=""):
    kind = (asset_kind or "humanoid").lower()
    prompt = _compact_asset_prompt(prompt)
    neg_prompt = _append_terms(neg_prompt, _ASSET_NEGATIVE_TERMS)

    # Keep prompt guard terms very short. If the user's prompt already contains
    # the concept, leave it alone; otherwise prepend critical layout terms so
    # they survive CLIP truncation.
    prompt = _append_if_missing_concept(prompt, "game asset", ("game asset", "character asset", "building asset", "prop asset", "game icon", "icon"))
    prompt = _append_if_missing_concept(prompt, "plain background", ("plain background", "clean background", "neutral background", "gray background", "grey background", "solid neutral", "studio background"))
    prompt = _prepend_terms(prompt, "" if _contains_any(prompt, ("centered", "centre", "center")) else "centered")

    if kind not in ("building", "prop"):
        prompt = _prepend_terms(prompt, "" if _contains_any(prompt, ("full body", "full-body", "whole body")) else "full body")
        prompt = _prepend_terms(prompt, "" if _contains_any(prompt, ("solo", "single", "1 character", "one character")) else "solo")

    if kind == "humanoid":
        prompt = _append_if_missing_concept(prompt, "humanoid", ("humanoid", "human", "warrior", "hero", "knight", "armor", "armour"))
    elif kind == "creature":
        prompt = _append_if_missing_concept(prompt, "creature", ("creature", "monster", "beast", "dragon"))
        neg_prompt = _append_terms(neg_prompt, _CREATURE_NEGATIVE_TERMS)
    elif kind == "building":
        prompt = _append_if_missing_concept(prompt, "building", ("building", "structure", "castle", "tower", "house", "barracks"))
        neg_prompt = _append_terms(neg_prompt, _BUILDING_NEGATIVE_TERMS)
    elif kind == "prop":
        prompt = _append_if_missing_concept(prompt, "prop", ("prop", "item", "object", "weapon", "tool"))
        neg_prompt = _append_terms(neg_prompt, _PROP_NEGATIVE_TERMS)

    return prompt, _prioritize_negative_prompt(neg_prompt)


def _component_cache_key(model_path, subfolder):
    """Key a generated cache by source path and weight metadata."""
    source = os.path.join(os.path.realpath(model_path), subfolder)
    digest = hashlib.sha256(f"{source}\n".encode())
    try:
        for name in sorted(os.listdir(source)):
            path = os.path.join(source, name)
            if not os.path.isfile(path) or not name.endswith((".json", ".safetensors", ".bin")):
                continue
            stat = os.stat(path)
            digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    except OSError:
        pass
    model_name = os.path.basename(os.path.normpath(model_path)) or "model"
    return f"{model_name}-{subfolder}-{digest.hexdigest()[:12]}"


def _cache_ready(path):
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        return any(
            name.endswith((".safetensors", ".bin"))
            and os.path.getsize(os.path.join(path, name)) > 0
            for name in os.listdir(path)
        )
    except OSError:
        return False


def _folder_size_gib(path):
    total = 0
    try:
        for root, _, files in os.walk(path):
            total += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    except OSError:
        return 0.0
    return total / 2**30


def _load_quantized_component(
    model_path, subfolder, cache_path, component_cls, quantization_config,
    dtype, label, status,
):
    if _cache_ready(cache_path):
        size = _folder_size_gib(cache_path)
        started = time.monotonic()
        status(f"Loading cached {label} ({size:.1f} GB local 4-bit)…")
        try:
            component = component_cls.from_pretrained(
                cache_path,
                torch_dtype=dtype,
                device_map={"": 0},
                local_files_only=True,
            )
            status(f"Loaded cached {label} in {time.monotonic() - started:.0f}s")
            return component
        except Exception as cache_error:
            status(
                f"Cached {label} could not load ({type(cache_error).__name__}); "
                "rebuilding it once…"
            )

    started = time.monotonic()
    status(f"Quantizing {label} to 4-bit (one-time local cache build)…")
    component = component_cls.from_pretrained(
        model_path,
        subfolder=subfolder,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    status(f"Quantized {label} in {time.monotonic() - started:.0f}s")

    try:
        os.makedirs(cache_path, exist_ok=True)
        started = time.monotonic()
        status(f"Saving packed {label} cache locally…")
        component.save_pretrained(cache_path)
        if _cache_ready(cache_path):
            size = _folder_size_gib(cache_path)
            status(
                f"Saved {label} cache ({size:.1f} GB) in "
                f"{time.monotonic() - started:.0f}s"
            )
        else:
            status(f"Warning: {label} cache write was incomplete; this load will still continue")
    except Exception as cache_error:
        status(
            f"Warning: could not save {label} cache ({type(cache_error).__name__}: "
            f"{cache_error}); this load will still continue"
        )
    return component


def _load_sd3(model_path, device, status_fn=None):
    """Load SD3 / SD3.5 with 4-bit (nf4) quantization of the T5-XXL text encoder and
    the transformer so it fits a 16 GB card. T5-XXL alone is ~9.5 GB in fp16; nf4 drops
    it to ~3 GB. bitsandbytes places quantized modules on the GPU itself — they must NOT
    be moved with pipe.to() (that raises), so we move only the non-quantized modules."""
    def status(msg):
        if status_fn:
            status_fn(msg)
    import torch
    from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBnb

    if device != "cuda":
        # bitsandbytes requires CUDA; fall back to unquantized fp32 on CPU (slow but works).
        status("Loading SD3 on CPU (fp32, unquantized)…")
        return StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch.float32, local_files_only=True,
        ).to("cpu")

    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    cache_root = str(shared_cache_dir("sd3-4bit"))
    transformer_cache = os.path.join(
        cache_root, _component_cache_key(model_path, "transformer")
    )
    text_encoder_cache = os.path.join(
        cache_root, _component_cache_key(model_path, "text_encoder_3")
    )

    load_started = time.monotonic()
    transformer = _load_quantized_component(
        model_path, "transformer", transformer_cache, SD3Transformer2DModel,
        DiffusersBnb(**nf4), torch.float16, "SD3 transformer", status,
    )
    text_encoder_3 = _load_quantized_component(
        model_path, "text_encoder_3", text_encoder_cache, T5EncoderModel,
        TransformersBnb(**nf4), torch.float16, "SD3 T5-XXL text encoder", status,
    )
    started = time.monotonic()
    status("Assembling SD3 pipeline…")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, transformer=transformer, text_encoder_3=text_encoder_3,
        torch_dtype=torch.float16, local_files_only=True,
    )
    status(f"Assembled SD3 pipeline in {time.monotonic() - started:.0f}s")
    # Quantized transformer + T5 are already GPU-resident. Move the remaining
    # (unquantized) modules onto the GPU too — a whole-pipe .to() would crash on the
    # 4-bit modules, so place them individually.
    for name in ("vae", "text_encoder", "text_encoder_2"):
        comp = getattr(pipe, name, None)
        if comp is not None:
            comp.to("cuda")
    status(f"SD3 load complete in {time.monotonic() - load_started:.0f}s")
    return pipe


def load_pipeline(model_path, lora_path, req_device, status_fn=None):
    def status(msg):
        if status_fn:
            status_fn(msg)

    status("Importing torch…")
    import torch
    status("Importing diffusers…")
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

    is_ckpt = model_path.endswith(".safetensors") and os.path.isfile(model_path)
    if not is_ckpt and not os.path.isdir(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    req_device = req_device.lower()
    if req_device == "cpu":
        device, dtype = "cpu", torch.float32
    elif req_device == "cuda":
        device, dtype = "cuda", torch.float16
    elif torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        device = "cuda" if free_vram >= 5.0 else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
    else:
        device, dtype = "cpu", torch.float32

    dtype_label = "fp16" if dtype == torch.float16 else "fp32"
    status(f"Loading pipeline on {device} ({dtype_label})…")

    if is_ckpt:
        is_xl = _is_sdxl_checkpoint(model_path)
        arch = architecture_of_checkpoint(is_xl)
        if is_xl:
            local_base = _find_local_sdxl_base()
            if not local_base:
                raise RuntimeError(
                    "No local SDXL diffusers model found for config. "
                    "Download sdxl-base-1.0 to Saient's managed data/models folder first."
                )
            status("Loading SDXL checkpoint (this may take a few minutes)…")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, config=local_base,
                torch_dtype=dtype, safety_checker=None,
            ).to(device)
            # from_single_file takes the checkpoint's *weights* but the local base's
            # *configs* — and that base ships the fp16-fix VAE, whose config sets
            # force_upcast=false. That flag is only true of those weights: a single-file
            # checkpoint bakes in its own (usually stock) VAE, which overflows fp16 and
            # decodes to a flat NaN image with the upcast disabled. Restore the stock
            # SDXL setting so diffusers upcasts for the decode as it does upstream.
            pipe.vae.config.force_upcast = True
        else:
            status("Loading SD1.5 checkpoint…")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=dtype, safety_checker=None,
            ).to(device)
    else:
        index_path = os.path.join(model_path, "model_index.json")
        cls_name = ""
        if os.path.exists(index_path):
            with open(index_path) as f:
                cls_name = json.load(f).get("_class_name", "")
            # Reject non-image pipelines (video etc.) with a clear message rather
            # than letting diffusers fail with a cryptic "no unet" error.
            if "StableDiffusion" not in cls_name:
                raise RuntimeError(
                    f"'{os.path.basename(model_path)}' is a '{cls_name}' model, "
                    "not a Stable Diffusion image model. Use the Video tab for "
                    "video models, or pick an SD1.5 / SDXL model here."
                )
        arch = architecture_of(model_path, cls_name)
        if arch["family"] == "sd3":
            # SD3 / SD3.5 — separate loader (transformer + 3 text encoders, quantized).
            status("Loading SD3 (4-bit quantized)…")
            pipe = _load_sd3(model_path, device, status)
        else:
            pipeline_cls = StableDiffusionXLPipeline if arch["family"] in ("sdxl", "sdxl_turbo") else StableDiffusionPipeline
            status(f"Loading {pipeline_cls.__name__} weights (this may take a few minutes)…")
            pipe = pipeline_cls.from_pretrained(
                model_path, torch_dtype=dtype, variant=_weights_variant(model_path),
                safety_checker=None, local_files_only=True,
            ).to(device)

    # Hard-disable the NSFW safety checker on every path (single_file / pretrained,
    # SD1.5 + SDXL). Passing safety_checker=None at load isn't always enough — some
    # SD1.5 checkpoints re-attach it or keep requires_safety_checker=True, which
    # black-outs "flagged" frames. Null it out explicitly so output is never censored.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    if lora_path and os.path.isfile(lora_path):
        status("Loading LoRA weights…")
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
        )

    pipe.set_progress_bar_config(disable=True)
    # Stash the descriptor computed above so generate_image()/_detail_faces() read
    # capability flags instead of re-deriving "what model is this" their own way.
    pipe._saient_arch = arch
    status(
        f"Architecture: {arch['family']} (cfg={arch['default_cfg']}, "
        f"steps={arch['default_steps']}, face_detail={arch['supports_face_detail']})"
    )
    return pipe, device


def resolve_scheduler_id(scheduler_id, model_path):
    scheduler_id = (scheduler_id or "auto").lower()
    if scheduler_id == "auto":
        return "dpm++2m_karras"
    return scheduler_id


def apply_scheduler(pipe, scheduler_id, model_path=""):
    from diffusers import (DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                           EulerAncestralDiscreteScheduler, DDIMScheduler,
                           UniPCMultistepScheduler, PNDMScheduler,
                           LMSDiscreteScheduler)
    scheduler_id = resolve_scheduler_id(scheduler_id, model_path)
    cfg = pipe.scheduler.config
    if scheduler_id == "dpm++2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    elif scheduler_id == "dpm++2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++")
    elif scheduler_id == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
    elif scheduler_id == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(cfg)
    elif scheduler_id == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(cfg)
    elif scheduler_id == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(cfg)
    elif scheduler_id == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(cfg)
    elif scheduler_id == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(cfg)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
    return scheduler_id


def _detail_faces(pipe, device, req, image):
    """ADetailer-style face fix. Base SDXL renders small faces (full-body shots → the face
    is often <10% of the frame) as soft, 'melted' blobs because there just aren't enough
    pixels. We detect each face, regenerate JUST that region at ~1024px via img2img at low
    strength (adds detail, keeps identity/pose), and feather it back. Reuses the already-
    loaded pipeline weights — no extra model load. Best-effort: any failure returns the base
    image untouched. Returns (image, n_fixed, reason) where reason is "fixed", "no_faces",
    or "unsupported" — callers should tell "nothing to fix" and "can't fix this model" apart."""
    import torch, numpy as np, cv2
    from PIL import Image, ImageFilter, ImageDraw
    from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)

    # Architectures like SD3/3.5 have no unet (they have `transformer`) and a 16-channel
    # VAE, not 4-channel — building StableDiffusionImg2ImgPipeline.from_pipe() from one
    # is not a same-architecture reuse like the SDXL/SD1.5 case below, it's cross-
    # architecture and unsound. Skip cleanly rather than hand it a mismatched pipeline.
    # Read the capability flag computed at load time, not a re-derived identity check.
    if not pipe._saient_arch["supports_face_detail"]:
        return image, 0, "unsupported"

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    casc = cv2.data.haarcascades
    detected = []
    for xml in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        c = cv2.CascadeClassifier(casc + xml)
        for f in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24)):
            detected.append(tuple(int(v) for v in f))
    # De-dup overlapping detections (frontal+profile can both fire) and skip close-ups that
    # are already detailed enough (face wider than ~45% of the frame).
    faces = []
    for (x, y, fw, fh) in sorted(detected, key=lambda f: -f[2] * f[3]):
        if fw >= 0.45 * w:
            continue
        cx, cy = x + fw / 2, y + fh / 2
        if any(abs(cx - (px + pw / 2)) < pw * 0.6 and abs(cy - (py + ph / 2)) < ph * 0.6
               for (px, py, pw, ph) in faces):
            continue
        faces.append((x, y, fw, fh))
    if not faces:
        return image, 0, "no_faces"

    # Reclaim the base generation's leftover activation/VAE buffers before the face pass —
    # without this the img2img can OOM by a few hundred MB on a full 16 GB card.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_xl = isinstance(pipe, StableDiffusionXLPipeline)
    I2I = StableDiffusionXLImg2ImgPipeline if is_xl else StableDiffusionImg2ImgPipeline
    # CACHE a shared img2img on the pipe. `from_pipe` was leaking a full ~6.5 GB UNet duplicate
    # on EVERY generation (never freed) → VRAM stacked to OOM by the 2nd run. Build it ONCE from
    # the base pipe's exact module objects (same tensors, nothing copied) and reuse it.
    img2img = getattr(pipe, "_adetailer_i2i", None)
    if img2img is None:
        if is_xl:
            img2img = I2I(vae=pipe.vae, unet=pipe.unet, scheduler=pipe.scheduler,
                          text_encoder=pipe.text_encoder, text_encoder_2=pipe.text_encoder_2,
                          tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2)
        else:
            img2img = I2I.from_pipe(pipe)  # SD1.5 path (small; rare)
        img2img.set_progress_bar_config(disable=True)
        pipe._adetailer_i2i = img2img
    img2img.scheduler = pipe.scheduler  # keep in sync with the current generation's scheduler
    # VAE slicing keeps the per-face decode cheap so the detail pass fits alongside the base pipe.
    for fn in ("enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(img2img, fn):
            try: getattr(img2img, fn)()
            except Exception: pass

    strength = max(0.2, min(0.7, float(req.get("face_detail_strength", 0.45))))
    fsteps = max(20, int(req.get("steps", pipe._saient_arch["default_steps"])))
    cfg = float(req.get("cfg_scale", pipe._saient_arch["default_cfg"]))
    base_prompt = req.get("prompt", "").strip()
    fprompt = (base_prompt + ", detailed face, sharp eyes, detailed skin, sharp focus").strip(", ")
    fneg = req.get("neg_prompt", "").strip()
    seed = int(req.get("seed", 42))

    out = arr.copy()
    n = 0
    for (x, y, fw, fh) in faces:
        pad = int(fw * 0.6)                                  # include hair / jaw / neck context
        cx0, cy0 = max(0, x - pad), max(0, y - pad)
        cx1, cy1 = min(w, x + fw + pad), min(h, y + fh + pad)
        crop = Image.fromarray(out[cy0:cy1, cx0:cx1])
        cw, ch = crop.size
        if cw < 16 or ch < 16:
            continue
        # Detail the face at high res, but CAP by free VRAM: the base pipeline stays resident
        # (~6.5 GB) and a full 1024px img2img peaks ~8.8 GB → together they OOM a 16 GB card.
        # 768-896px is already plenty of detail for a face and leaves comfortable headroom.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 2**30
        else:
            free_gb = 99.0
        target = 1024 if free_gb > 9.5 else (896 if free_gb > 7.0 else 768)
        scale = target / max(cw, ch)                          # face detail, fit to VRAM
        tw = max(8, int(cw * scale) // 8 * 8)
        th = max(8, int(ch * scale) // 8 * 8)
        work = crop.resize((tw, th), Image.LANCZOS)
        g = torch.Generator(device=device).manual_seed(seed + 1 + n)
        kw = dict(image=work, strength=strength, num_inference_steps=fsteps,
                  guidance_scale=cfg, generator=g, prompt=fprompt)
        if is_xl:
            kw["prompt_2"] = fprompt
        if fneg:
            kw["negative_prompt"] = fneg
            if is_xl:
                kw["negative_prompt_2"] = fneg
        fixed = img2img(**kw).images[0].resize((cw, ch), Image.LANCZOS)
        # Feathered ellipse so the regenerated face blends in with no box seam.
        mask = Image.new("L", (cw, ch), 0)
        m = int(min(cw, ch) * 0.10)
        ImageDraw.Draw(mask).ellipse((m, m, cw - m, ch - m), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(4, min(cw, ch) // 12)))
        out[cy0:cy1, cx0:cx1] = np.array(Image.composite(fixed, crop, mask))
        n += 1

    # NOTE: img2img is cached on pipe (shares weights) — do NOT delete it. Just drop the
    # face-pass activations.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Image.fromarray(out), n, "fixed"


def _validate_output_image(image, family):
    """Reject obviously-broken output instead of reporting success on it. This is a
    mechanical sanity check (finite pixels, not a solid/near-solid color), not an
    aesthetic judgment — it exists to catch the class of failure where a pipeline runs
    to completion without raising but the decoded image itself is garbage (NaNs
    collapsing to flat black, a corrupted cross-architecture pass, etc). Raises rather
    than returning a bool so a caller can't accidentally downgrade this to a warning."""
    import numpy as np
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"generation produced non-finite pixel values (family={family})")
    std = float(arr.std())
    if std < 1.0:
        raise RuntimeError(
            f"generation produced a degenerate image (family={family}, pixel std={std:.3f} "
            "— solid/near-solid color, not real content)"
        )


def generate_image(pipe, device, req):
    import torch, gc
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

    # Release whatever the previous generation left reserved BEFORE we start. Without this,
    # each run's peak (the base gen + the hi-res face-detailer img2img) stacks on the last
    # run's leftovers and eventually OOMs — the "worked the first run then dies" symptom.
    if torch.cuda.is_available():
        gc.collect(); torch.cuda.empty_cache()

    prompt      = req.get("prompt", "").strip()
    neg_prompt  = req.get("neg_prompt", "").strip()
    if req.get("asset_guard", True):
        prompt, neg_prompt = apply_asset_guard(
            prompt,
            neg_prompt,
            req.get("asset_kind", "humanoid"),
            req.get("model_path", ""),
        )
    prompt, neg_prompt = apply_color_guard(prompt, neg_prompt)
    arch        = pipe._saient_arch
    steps       = int(req.get("steps", arch["default_steps"]))
    cfg_scale   = float(req.get("cfg_scale", arch["default_cfg"]))
    seed        = int(req.get("seed", 42))
    width       = int(req.get("width", 1024))
    height      = int(req.get("height", 1024))
    scheduler_id = req.get("scheduler", "auto")
    model_path = req.get("model_path", "")

    # Fail loudly here instead of letting a 0/negative/non-multiple-of-8 size hit a
    # cryptic shape-mismatch deep inside the VAE/UNet — every supported architecture's
    # latent space downsamples by a factor of 8.
    if width <= 0 or height <= 0 or width % 8 != 0 or height % 8 != 0:
        raise RuntimeError(
            f"invalid resolution {width}x{height}: width and height must be positive "
            "multiples of 8"
        )
    if steps <= 0:
        raise RuntimeError(f"invalid steps={steps}: must be a positive integer")

    # SD3 uses its own flow-matching scheduler — don't override it with the SD/SDXL
    # samplers. Everything else gets the requested/auto scheduler (v-prediction models
    # get it transparently: apply_scheduler seeds from the model's own loaded scheduler
    # config, which already carries prediction_type through from_config()).
    if arch["scheduler_mode"] == "flow_match":
        scheduler_id = "flow-match (native)"
    else:
        scheduler_id = apply_scheduler(pipe, scheduler_id, model_path)

    # SD3's T5-XXL branch supports a much longer prompt than CLIP's 77-token cap — measure
    # and compact against the architecture's real limit (and its longest encoder) instead of
    # a hardcoded CLIP assumption, or SD3 loses the prompt length it was built to use.
    limit = arch["prompt_token_limit"]
    use_t5 = arch["family"] == "sd3"
    neg_prompt = _prioritize_negative_prompt(neg_prompt)
    encoder_label = "T5" if use_t5 else "CLIP"
    prompt_tokens = _token_count(pipe, prompt, use_t5=use_t5)
    neg_tokens = _token_count(pipe, neg_prompt, use_t5=use_t5)

    # Past CLIP's context, encode in chunks instead of deleting the tail. Cutting at 77
    # silently removed "photograph / studio lighting / photorealism" from long prompts and
    # left the anatomy word-list behind, which base SDXL renders as an anatomy plate.
    # SD3's T5 branch has its own budget and encode path, so it keeps compacting.
    chunked = (not use_t5) and max(prompt_tokens, neg_tokens) > limit
    if not chunked:
        prompt, prompt_tokens, prompt_trimmed = _fit_to_token_limit(pipe, prompt, limit=limit, use_t5=use_t5)
        neg_prompt, neg_tokens, neg_trimmed = _fit_to_token_limit(pipe, neg_prompt, limit=limit, use_t5=use_t5)
        if prompt_trimmed or neg_trimmed:
            emit({
                "loading_status": (
                    f"prompt compacted for {encoder_label}: prompt {prompt_tokens}/{limit}, "
                    f"negative {neg_tokens}/{limit}"
                )
            })
    emit({
        "loading_status": (
            f"generate: scheduler={scheduler_id}, steps={steps}, "
            f"cfg={cfg_scale:.1f}, prompt={prompt_tokens}/{limit}"
        )
    })

    # Only clamp families whose U-Net visibly degrades past its trained resolution class
    # (see clamp_resolution in _DESCRIPTORS) — this used to be an isinstance(pipe,
    # StableDiffusionPipeline) check, which also (wrongly) caught SD2.1-768 and forced it
    # down to SD1.5's 512 ceiling since both share the same diffusers pipeline class.
    if arch["clamp_resolution"]:
        width  = min(width, arch["max_resolution"])
        height = min(height, arch["max_resolution"])

    generator = torch.Generator(device=device).manual_seed(seed)

    def callback(p, i, t, kwargs):
        emit({"step": i + 1, "total": steps})
        return kwargs

    kwargs = dict(
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
        width=width,
        height=height,
        callback_on_step_end=callback,
    )
    if chunked:
        embed_kwargs, n_chunks = _encode_long_prompt(pipe, prompt, neg_prompt, device)
        kwargs.update(embed_kwargs)
        emit({
            "loading_status": (
                f"long prompt: {prompt_tokens} {encoder_label} tokens encoded as "
                f"{n_chunks} chunks (nothing dropped)"
            )
        })
    else:
        kwargs["prompt"] = prompt
        if isinstance(pipe, StableDiffusionXLPipeline):
            kwargs["prompt_2"] = prompt
        if neg_prompt:
            kwargs["negative_prompt"] = neg_prompt
            if isinstance(pipe, StableDiffusionXLPipeline):
                kwargs["negative_prompt_2"] = neg_prompt

    t0 = time.time()
    image = pipe(**kwargs).images[0]

    # ADetailer-style face pass — re-detail small faces at hi-res. Base SDXL produces soft/
    # "melted" faces in full-body shots because the face is only ~9% of the frame. On by
    # default; auto-skips when no (small) face is found, so close-up portraits are untouched.
    # "unsupported" (architecture can't do this) and "no_faces" (nothing to fix) are reported
    # differently — a silent identical no-op for both would hide that SD3 never even tries.
    if req.get("face_detail", True):
        try:
            image, n_fixed, reason = _detail_faces(pipe, device, req, image)
            if reason == "fixed":
                emit({"loading_status": f"face-detail: refined {n_fixed} face(s) at hi-res"})
            elif reason == "unsupported":
                emit({"loading_status": f"face-detail: not supported for {arch['family']}, skipped"})
        except Exception as e:
            emit({"loading_status": f"face-detail skipped ({e})"})

    elapsed = time.time() - t0

    # No fake success: reject NaN/Inf and degenerate (solid/near-solid color) output here,
    # before it's PNG-encoded and handed back as a result. A pipeline that ran without
    # raising is not the same thing as a pipeline that produced a real image.
    _validate_output_image(image, arch["family"])

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    # Leave a clean slate so the next generation starts from the resident weights, not a
    # fragmented heap full of this run's activations.
    if torch.cuda.is_available():
        gc.collect(); torch.cuda.empty_cache()
    return {"base64_png": b64, "device": device, "elapsed": round(elapsed, 1)}


def main():
    load_line = sys.stdin.readline()
    if not load_line.strip():
        emit({"error": "No load config received"})
        return

    try:
        load_cfg = json.loads(load_line)
    except Exception as e:
        emit({"error": f"Bad load config JSON: {e}"})
        return

    model_path = load_cfg.get("model_path", "")
    lora_path  = load_cfg.get("lora_path", "")
    device_req = load_cfg.get("device", "auto")

    def on_status(msg):
        emit({"loading_status": msg})

    try:
        pipe, device = load_pipeline(model_path, lora_path, device_req, on_status)
    except Exception as e:
        emit({"error": str(e)})
        return

    # Carry the architecture descriptor back so the UI can reset CFG/steps to what this
    # specific model actually wants instead of a stale value from whatever was loaded
    # before — the frontend always sends an explicit cfg_scale/steps, so a backend-only
    # default is otherwise dead code from the user's point of view.
    emit({"ready": True, "device": device, "arch": pipe._saient_arch})

    # Generation loop — one request per line
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as e:
            emit({"error": f"Bad request JSON: {e}"})
            continue
        try:
            result = generate_image(pipe, device, req)
            emit(result)
        except Exception as e:
            emit({"error": str(e)})


if __name__ == "__main__":
    main()
