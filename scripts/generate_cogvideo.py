#!/usr/bin/env python3
# CogVideoX image-to-video daemon. Same stdin/stdout protocol as generate_video.py
# (the Rust side picks this script when the model is a CogVideoX family model), but
# CogVideoX is a different architecture (T5 text encoder, AutoencoderKLCogVideoX,
# CogVideoXTransformer3DModel) so it gets its own daemon.
#
# Staging mirrors the Wan path — 4-bit transformer resident, T5 loaded 4-bit only to
# encode then freed, NO cpu offload (we quantise hard enough to fit resident, which
# avoids the RAM-pinning offload that froze the box). CogVideoX-5b-I2V REQUIRES an
# input image.
import base64, gc, io, json, os, sys, tempfile, time, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PIPE = None
MODEL_PATH = ""
EMBED_CACHE = {"key": None, "pe": None, "ne": None}


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _watch_parent():
    import threading, time as _t
    ppid = os.getppid()
    def loop():
        while True:
            _t.sleep(2)
            if os.getppid() != ppid or ppid == 1:
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def _vram():
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    u = torch.cuda.memory_allocated() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return u, p


def load(cfg):
    global PIPE, MODEL_PATH
    import torch
    from diffusers import (CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel,
                           AutoencoderKLCogVideoX, BitsAndBytesConfig as DBnb)
    from transformers import AutoTokenizer
    import diffusers as _df

    MODEL_PATH = cfg.get("model_path", "")
    dev = 0
    t_load = time.time()

    emit({"loading_status": "loading transformer (4-bit nf4) onto GPU…"})
    _t = time.time()
    dbnb = DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", quantization_config=dbnb,
        torch_dtype=torch.bfloat16, device_map={"": dev})
    emit({"loading_status": f"  ⏱ transformer: {time.time()-_t:.0f}s"})

    emit({"loading_status": "loading VAE…"})
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")

    with open(os.path.join(MODEL_PATH, "scheduler", "scheduler_config.json")) as f:
        sched_cls = getattr(_df, json.load(f)["_class_name"])
    scheduler = sched_cls.from_pretrained(MODEL_PATH, subfolder="scheduler")

    emit({"loading_status": "assembling pipeline (no text encoder yet)…"})
    PIPE = CogVideoXImageToVideoPipeline(
        tokenizer=tokenizer, text_encoder=None, vae=vae,
        transformer=transformer, scheduler=scheduler)
    try:
        PIPE.vae.to(f"cuda:{dev}")
    except Exception:
        pass
    for fn in ("enable_tiling", "enable_slicing"):
        if hasattr(PIPE.vae, fn):
            getattr(PIPE.vae, fn)()
    PIPE.set_progress_bar_config(disable=True)
    _free_cuda()
    used, _ = _vram()
    EMBED_CACHE["key"] = None
    emit({"loading_status": f"  ⏱ load total: {time.time()-t_load:.0f}s · VRAM {used:.1f} GB"})
    emit({"ready": True, "device": "cuda" if torch.cuda.is_available() else "cpu"})


def encode(prompt, neg, do_cfg):
    """Load T5 4-bit (cached after first build), encode, free it. Returns (pe, ne)."""
    import torch
    from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb

    cache = os.path.expanduser("~/.config/ai-workshop/t5-cogvideo-4bit")
    te = None
    if os.path.exists(os.path.join(cache, "config.json")):
        emit({"loading_status": "loading pre-quantized T5 (cached 4-bit)…"})
        try:
            te = T5EncoderModel.from_pretrained(cache, torch_dtype=torch.bfloat16, device_map={"": 0})
        except Exception:
            te = None
    if te is None:
        emit({"loading_status": "loading T5 text encoder (4-bit) + caching…"})
        tbnb = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        te = T5EncoderModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", quantization_config=tbnb,
            torch_dtype=torch.bfloat16, device_map={"": 0})
        try:
            os.makedirs(cache, exist_ok=True); te.save_pretrained(cache)
            emit({"loading_status": "  ✓ T5 cached — fast loads from now on"})
        except Exception as se:
            emit({"loading_status": f"  ⚠ couldn't cache T5 ({se})"})

    PIPE.text_encoder = te
    try:
        pe, ne = PIPE.encode_prompt(
            prompt=prompt, negative_prompt=(neg or ""),
            do_classifier_free_guidance=do_cfg, num_videos_per_prompt=1,
            max_sequence_length=226, device="cuda", dtype=torch.bfloat16)
    finally:
        PIPE.text_encoder = None
        del te
        _free_cuda()
    return pe, ne


def generate(req):
    import torch
    from PIL import Image
    from diffusers.utils import export_to_video

    t0 = time.time(); _free_cuda(); _vram()
    total = int(req.get("steps", 50))
    cfg_scale = float(req.get("cfg_scale", 6.0))
    do_cfg = cfg_scale > 1.0
    prompt = req.get("prompt", ""); neg = req.get("neg_prompt", "")
    image_b64 = (req.get("image_b64") or "").strip()
    if not image_b64:
        emit({"error": "CogVideoX-5b-I2V needs an input image — add one (it's image-to-video only)."})
        return

    width = int(req.get("width", 720)); height = int(req.get("height", 480))
    num_frames = int(req.get("num_frames", 49))

    key = (prompt, neg, do_cfg)
    if EMBED_CACHE["key"] == key and EMBED_CACHE["pe"] is not None:
        pe = EMBED_CACHE["pe"].clone()
        ne = EMBED_CACHE["ne"].clone() if EMBED_CACHE["ne"] is not None else None
        emit({"loading_status": "↻ reusing cached prompt embeds — T5 skipped"})
    else:
        _t = time.time()
        pe, ne = encode(prompt, neg, do_cfg)
        EMBED_CACHE["key"] = key
        EMBED_CACHE["pe"] = pe.detach().clone()
        EMBED_CACHE["ne"] = ne.detach().clone() if ne is not None else None
        emit({"loading_status": f"  ⏱ T5 + encode: {time.time()-_t:.0f}s"})

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB").resize((width, height))

    marks = {"first": None, "last": None}
    def cb(_p, i, _t, kw):
        now = time.time()
        if marks["first"] is None: marks["first"] = now
        marks["last"] = now
        emit({"step": i + 1, "total": total})
        return kw

    generator = None
    seed = req.get("seed", -1)
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    emit({"loading_status": "i2v: conditioning on your image, denoising…"})
    t_dn = time.time()
    frames = PIPE(
        image=img, prompt_embeds=pe, negative_prompt_embeds=ne,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=total, guidance_scale=cfg_scale,
        generator=generator, callback_on_step_end=cb,
    ).frames[0]
    t_after = time.time()
    if marks["first"] and marks["last"]:
        dn = marks["last"] - marks["first"]; dec = t_after - marks["last"]
    else:
        dn, dec = t_after - t_dn, 0.0
    _, peak = _vram()
    emit({"loading_status": f"  ⏱ denoise: {dn:.0f}s ({dn/max(total-1,1):.1f}s/step) · decode: {dec:.0f}s · peak VRAM {peak:.1f} GB"})
    del pe, ne
    _free_cuda()

    fps = int(req.get("fps", 8))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
    export_to_video(frames, tmp.name, fps=fps)
    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    os.unlink(tmp.name)
    emit({"base64_mp4": b64, "frames": len(frames), "elapsed": round(time.time() - t0, 1)})


def main():
    _watch_parent()
    load_line = sys.stdin.readline()
    if not load_line:
        return
    try:
        cfg = json.loads(load_line)
    except Exception as e:
        emit({"error": f"bad load config: {e}"}); return
    try:
        emit({"loading_status": "importing torch + diffusers…"})
        load(cfg)
    except Exception as e:
        emit({"error": f"load failed: {e}", "trace": traceback.format_exc()[:1000]}); return
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
