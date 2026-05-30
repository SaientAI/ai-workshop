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

PIPE = None        # WanPipeline with text_encoder=None, transformer+vae resident
TOKENIZER = None   # kept resident (tiny); text encoder is loaded per-encode
MODEL_PATH = ""


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load(cfg):
    """Load the SMALL resident pieces (transformer 4-bit + VAE). Fast, low RAM."""
    global PIPE, TOKENIZER, MODEL_PATH
    import torch
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers import BitsAndBytesConfig as DBnb
    from transformers import AutoTokenizer

    MODEL_PATH = cfg.get("model_path", "")
    dev = 0  # cuda:0

    emit({"loading_status": "loading transformer (4-bit nf4) onto GPU…"})
    dbnb = DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer",
        quantization_config=dbnb, torch_dtype=torch.bfloat16, device_map={"": dev})

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
        PIPE.vae.enable_tiling()
    PIPE.set_progress_bar_config(disable=True)
    _free_cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emit({"ready": True, "device": device})


def encode(prompt, neg, do_cfg):
    """Load text encoder 4-bit on GPU, encode, then free it. Returns (pe, ne)."""
    import torch
    from transformers import UMT5EncoderModel
    from transformers import BitsAndBytesConfig as TBnb

    emit({"loading_status": "loading text encoder (4-bit nf4) onto GPU…"})
    tbnb = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
    te = UMT5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder",
        quantization_config=tbnb, torch_dtype=torch.bfloat16, device_map={"": 0})

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


def generate(req):
    import torch
    from diffusers.utils import export_to_video

    t0 = time.time()
    total = int(req.get("steps", 30))
    cfg_scale = float(req.get("cfg_scale", 6.0))
    do_cfg = cfg_scale > 1.0

    pe, ne = encode(req.get("prompt", ""), req.get("neg_prompt", ""), do_cfg)

    def cb(_pipe, i, _t, kwargs):
        emit({"step": i + 1, "total": total})
        return kwargs

    generator = None
    seed = req.get("seed", -1)
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    emit({"loading_status": "denoising…"})
    frames = PIPE(
        prompt_embeds=pe,
        negative_prompt_embeds=ne,
        height=int(req.get("height", 480)),
        width=int(req.get("width", 832)),
        num_frames=int(req.get("num_frames", 49)),
        num_inference_steps=total,
        guidance_scale=cfg_scale,
        generator=generator,
        callback_on_step_end=cb,
    ).frames[0]
    del pe, ne
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
