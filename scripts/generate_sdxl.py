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
            if "XL" not in cls_name:
                pipeline_cls = StableDiffusionPipeline
        status(f"Loading {pipeline_cls.__name__} weights (this may take a few minutes)…")
        pipe = pipeline_cls.from_pretrained(
            model_path, torch_dtype=dtype,
            safety_checker=None, local_files_only=True,
        ).to(device)

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


def generate_image(pipe, device, req):
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

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
    elapsed = time.time() - t0

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
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
