#!/usr/bin/env python3
"""SDXL / SD1.5 image generation daemon for AI Workshop.

Daemon protocol — newline-delimited JSON on stdin/stdout:
  stdin line 1:  {"model_path":"...","lora_path":"...","device":"auto"}
  stdout line 1: {"ready":true,"device":"cuda"}   OR  {"error":"..."}

  For each generation request:
  stdin:  {"prompt":"...","neg_prompt":"...","steps":20,"cfg_scale":7,"seed":42,"width":1024,"height":1024,"scheduler":"..."}
  stdout: {"step":1,"total":20}  (repeated — progress)
          {"base64_png":"...","device":"cuda","elapsed":3.5}  (final result)
          {"error":"..."}  (on failure — daemon keeps running)
"""
import base64, io, json, os, sys, time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Lets PyTorch reuse reserved-but-unallocated blocks instead of OOMing on fragmentation —
# the face-detail img2img pass needs ~0.5 GB on top of the resident base pipeline and was
# failing with hundreds of MB "reserved but unallocated". Must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_MODEL_DIRS = ["/home/tiny/models", "/home/tiny/projects/models"]


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


def _is_sdxl_checkpoint(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return any(k.startswith("conditioner.") for k in f.keys())
    except Exception:
        name = os.path.basename(path).lower()
        return "xl" in name or "sdxl" in name


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
        if is_xl:
            local_base = _find_local_sdxl_base()
            if not local_base:
                raise RuntimeError(
                    "No local SDXL diffusers model found for config. "
                    "Download sdxl-base-1.0 to ~/models/ first."
                )
            status("Loading SDXL checkpoint (this may take a few minutes)…")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, config=local_base,
                torch_dtype=dtype, safety_checker=None,
            ).to(device)
        else:
            status("Loading SD1.5 checkpoint…")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=dtype, safety_checker=None,
            ).to(device)
    else:
        index_path = os.path.join(model_path, "model_index.json")
        pipeline_cls = StableDiffusionXLPipeline
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
            if "XL" not in cls_name:
                pipeline_cls = StableDiffusionPipeline
        status(f"Loading {pipeline_cls.__name__} weights (this may take a few minutes)…")
        pipe = pipeline_cls.from_pretrained(
            model_path, torch_dtype=dtype,
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
    return pipe, device


def apply_scheduler(pipe, scheduler_id):
    from diffusers import (DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                           EulerAncestralDiscreteScheduler, DDIMScheduler,
                           UniPCMultistepScheduler)
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
    elif scheduler_id == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(cfg)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)


def _detail_faces(pipe, device, req, image):
    """ADetailer-style face fix. Base SDXL renders small faces (full-body shots → the face
    is often <10% of the frame) as soft, 'melted' blobs because there just aren't enough
    pixels. We detect each face, regenerate JUST that region at ~1024px via img2img at low
    strength (adds detail, keeps identity/pose), and feather it back. Reuses the already-
    loaded pipeline weights — no extra model load. Best-effort: any failure returns the base
    image untouched. Returns (image, n_fixed)."""
    import torch, numpy as np, cv2
    from PIL import Image, ImageFilter, ImageDraw
    from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)

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
        return image, 0

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
    fsteps = max(20, int(req.get("steps", 20)))
    cfg = float(req.get("cfg_scale", 7.0))
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
    return Image.fromarray(out), n


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
    steps       = int(req.get("steps", 20))
    cfg_scale   = float(req.get("cfg_scale", 7.0))
    seed        = int(req.get("seed", 42))
    width       = int(req.get("width", 1024))
    height      = int(req.get("height", 1024))
    scheduler_id = req.get("scheduler", "dpm++2m_karras")

    apply_scheduler(pipe, scheduler_id)

    if isinstance(pipe, StableDiffusionPipeline):
        width  = min(width, 512)
        height = min(height, 512)

    generator = torch.Generator(device=device).manual_seed(seed)

    def callback(p, i, t, kwargs):
        emit({"step": i + 1, "total": steps})
        return kwargs

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
        width=width,
        height=height,
        callback_on_step_end=callback,
    )
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
    if req.get("face_detail", True):
        try:
            image, n_fixed = _detail_faces(pipe, device, req, image)
            if n_fixed:
                emit({"loading_status": f"face-detail: refined {n_fixed} face(s) at hi-res"})
        except Exception as e:
            emit({"loading_status": f"face-detail skipped ({e})"})

    elapsed = time.time() - t0

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

    emit({"ready": True, "device": device})

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
