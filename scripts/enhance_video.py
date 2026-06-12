#!/usr/bin/env python3
# Quality-pass enhancer — runs AFTER generation as separate, unloaded steps.
#
# Philosophy (the user's): the generator and the enhancer never coexist. The Rust
# side drops the video daemon first (freeing all VRAM), then spawns this one-shot
# script. Each STAGE here loads its own model, runs, and frees it before the next
# stage — so a pass owns the whole GPU and we never stack two heavy models.
#
# Protocol (one-shot, not a resident daemon):
#   stdin: one JSON line:
#     {"video_b64": "...", "fps": 16, "stages": ["refine","upscale","interpolate"],
#      "model_path": "...",                      # for refine (Wan v2v, bf16)
#      "refine_strength": 0.35, "refine_steps": 20, "prompt": "...", "neg_prompt": "...",
#      "cfg_scale": 6.0,
#      "upscale_model": "/.../RealESRGAN_x2plus.pth",
#      "interp_factor": 2}
#   stdout: {"loading_status": "..."} / {"step": i, "total": t} progress lines,
#           then {"enhanced_b64": "...", "frames": N, "width": W, "height": H, "elapsed": s}
import base64, gc, json, os, sys, tempfile, time, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _watch_parent():
    """Die with the app so the enhancer never orphans and holds VRAM."""
    import threading, time
    ppid = os.getppid()
    def loop():
        while True:
            time.sleep(2)
            if os.getppid() != ppid or ppid == 1:
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _cache_dir(name):
    """Managed cache dir ~/.config/saient/<name>. Migrate a pre-rebrand
    ~/.config/ai-workshop/<name> in place on first use so we never re-quantize."""
    new = os.path.expanduser(f"~/.config/saient/{name}")
    old = os.path.expanduser(f"~/.config/ai-workshop/{name}")
    if not os.path.exists(new) and os.path.exists(old):
        try:
            os.makedirs(os.path.dirname(new), exist_ok=True)
            os.rename(old, new)
        except Exception:
            return old
    return new


def to_uint8(frame):
    """Normalise any frame (PIL / float[0,1] / float[-1,1] / CHW / uint8) to
    HxWx3 uint8 RGB. Stages produce different formats — Wan v2v returns float
    [0,1], the decoder returns uint8 — so we canonicalise between every stage
    (a mismatch here is what turned the refined clip into a black screen)."""
    import numpy as np
    from PIL import Image
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    a = np.asarray(frame)
    # CHW → HWC if it came through as channels-first
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = a.astype("float32")
        if float(a.min()) < -0.01:        # [-1,1] → [0,1]
            a = (a + 1.0) / 2.0
        if float(a.max()) <= 1.5:         # [0,1] → [0,255]
            a = a * 255.0
        a = np.clip(a, 0, 255).round().astype("uint8")
    if a.ndim == 2:                        # grayscale → RGB
        a = np.stack([a] * 3, axis=-1)
    return a[..., :3]


# ── frame I/O ─────────────────────────────────────────────────────────────────

def b64_to_frames(video_b64):
    """Decode an mp4 (base64) to a list of HxWx3 uint8 RGB numpy frames."""
    import imageio.v2 as imageio   # ffmpeg backend (imageio-ffmpeg), not pyav
    import numpy as np
    raw = base64.b64decode(video_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw); tmp.close()
    try:
        reader = imageio.get_reader(tmp.name, "ffmpeg")
        frames = [np.asarray(f)[:, :, :3] for f in reader]
        reader.close()
    finally:
        os.unlink(tmp.name)
    return frames


def frames_to_b64(frames, fps):
    import imageio.v2 as imageio   # ffmpeg backend
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    w = imageio.get_writer(tmp.name, fps=fps, codec="libx264",
                           quality=8, macro_block_size=1)
    for f in frames:
        w.append_data(f)
    w.close()
    with open(tmp.name, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    os.unlink(tmp.name)
    return b64


# ── Stage: refine (Wan vid2vid, full bf16 — fits because generator is unloaded) ─

def stage_refine(frames, req):
    import torch
    from PIL import Image
    from diffusers import WanVideoToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan
    from transformers import UMT5EncoderModel, AutoTokenizer
    from transformers import BitsAndBytesConfig as TBnb

    model_path = req["model_path"]
    # Refine is Wan-specific (Wan v2v pipeline). On a non-Wan model (e.g. CogVideoX)
    # skip it cleanly so upscale/interpolate still run — those are model-agnostic and
    # are the bigger wins for a CogVideoX clip anyway (720→1440, 8→16 fps).
    try:
        with open(os.path.join(model_path, "model_index.json")) as f:
            mi = f.read()
    except Exception:
        mi = ""
    if "Wan" not in mi:
        emit({"loading_status": "refine: Wan-only for now — skipping (upscale/interpolate still run)"})
        return frames
    emit({"loading_status": "refine: loading transformer (bf16 — full precision)…"})
    # No quantization: the 1.3B transformer is ~2.6 GB in bf16 and the GPU is all
    # ours now, so we run it at full quality instead of 4-bit.
    transformer = WanTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map={"": 0})
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    import diffusers as _df
    with open(os.path.join(model_path, "scheduler", "scheduler_config.json")) as f:
        sched_cls = getattr(_df, json.load(f)["_class_name"])
    scheduler = sched_cls.from_pretrained(model_path, subfolder="scheduler")
    try:
        pipe = WanVideoToVideoPipeline(vae=vae, text_encoder=None, tokenizer=tokenizer,
                                       transformer=transformer, scheduler=scheduler)
    except Exception:
        pipe = WanVideoToVideoPipeline.from_pretrained(
            model_path, transformer=transformer, vae=vae, tokenizer=tokenizer,
            scheduler=scheduler, text_encoder=None, torch_dtype=torch.bfloat16)
    try:
        pipe.vae.to("cuda:0")
    except Exception:
        pass
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    cfg_scale = float(req.get("cfg_scale", 6.0))
    do_cfg = cfg_scale > 1.0

    # Encode prompt with a 4-bit text encoder, then free it (same staged trick).
    # Reuse the shared pre-quantized 4-bit UMT5 cache built by generate_video.py.
    cache = _cache_dir("umt5-xxl-4bit")
    te = None
    if os.path.exists(os.path.join(cache, "config.json")):
        emit({"loading_status": "refine: loading pre-quantized text encoder (cached)…"})
        try:
            te = UMT5EncoderModel.from_pretrained(cache, torch_dtype=torch.bfloat16, device_map={"": 0})
        except Exception:
            te = None
    if te is None:
        emit({"loading_status": "refine: loading text encoder (4-bit) to encode…"})
        tbnb = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        te = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", quantization_config=tbnb,
            torch_dtype=torch.bfloat16, device_map={"": 0})
    pipe.text_encoder = te
    try:
        pe, ne = pipe.encode_prompt(prompt=req.get("prompt", ""),
                                    negative_prompt=(req.get("neg_prompt", "") or None),
                                    do_classifier_free_guidance=do_cfg,
                                    num_videos_per_prompt=1, device="cuda")
    finally:
        pipe.text_encoder = None
        del te
        _free_cuda()

    pil = [Image.fromarray(f) for f in frames]
    total = int(req.get("refine_steps", 20))

    def cb(_p, i, _t, kw):
        emit({"step": i + 1, "total": total})
        return kw

    emit({"loading_status": f"refine: vid2vid @ strength {req.get('refine_strength', 0.35)}…"})
    out = pipe(
        video=pil,
        prompt_embeds=pe, negative_prompt_embeds=ne,
        strength=float(req.get("refine_strength", 0.35)),
        num_inference_steps=total,
        guidance_scale=cfg_scale,
        callback_on_step_end=cb,
    ).frames[0]

    result = [to_uint8(im) for im in out]   # Wan v2v → float[0,1]; canonicalise to uint8
    del pipe, transformer, vae, pe, ne
    _free_cuda()
    return result


# ── Stage: upscale (spandrel-loaded RealESRGAN — clean on modern torch) ─────────

def stage_upscale(frames, req):
    import torch, numpy as np
    import spandrel
    path = req.get("upscale_model", "")
    emit({"loading_status": f"upscale: loading {os.path.basename(path)}…"})
    model = spandrel.ModelLoader().load_from_file(path).to("cuda").eval()
    scale = getattr(model, "scale", 2)
    n = len(frames)
    out = []
    with torch.inference_mode():
        for i, f in enumerate(frames):
            f = to_uint8(f)   # guard: never divide a float[0,1] frame by 255 again
            t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to("cuda")
            y = model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
            out.append(y)
            emit({"step": i + 1, "total": n})
    del model
    _free_cuda()
    emit({"loading_status": f"upscale: {scale}× done ({out[0].shape[1]}×{out[0].shape[0]})"})
    return out


# ── Stage: interpolate (robust optical-flow, motion-compensated) ────────────────
# Neural RIFE won't build on this box (ncnn wheel fails, no real Vulkan device), so
# we interpolate with OpenCV DIS optical flow: estimate bidirectional flow between
# each adjacent pair, warp both toward the intermediate time, and blend — true
# motion interpolation (content moves along the flow), zero downloads.
#
# ANTI-STROBE (the "disco lights" fix): raw flow warping sprays colored speckle
# wherever the flow is WRONG — occlusion boundaries (a moving silhouette reveals/
# hides background) and low-texture regions (sky, grass, walls) where DIS has nothing
# to lock onto. Frame to frame that speckle flickers = strobing. Three guards:
#   1. blur the flow field → kills isolated speckle vectors,
#   2. forward/backward consistency check → flow that doesn't round-trip is untrusted,
#   3. where flow is untrusted, fall back to a plain cross-dissolve (smooth, never
#      speckly) instead of a garbage warp.
# Real motion-compensation where flow is reliable; graceful cross-fade where it isn't.
# (The warp sign is `gx - t*flow`: DIS returns prev→next displacement and remap is a
# backward map, so to advance frame i forward by t we SAMPLE at gx - t*flow. Verified.)

def _flow(prev_gray, next_gray):
    import cv2
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
        return dis.calc(prev_gray, next_gray, None)
    except Exception:
        return cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)


def _flow_confidence(fa, fb, gx, gy, sigma=3.0):
    """Confidence in [0,1]: ~1 where forward flow `fa` and backward flow `fb` cancel
    (reliable), ~0 where they don't (occlusion / textureless region — where the
    strobe speckle is born). `sigma` px sets how much round-trip error we tolerate."""
    import cv2, numpy as np
    mapx = (gx + fa[..., 0]).astype(np.float32)
    mapy = (gy + fa[..., 1]).astype(np.float32)
    fb_at = cv2.remap(fb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(fa + fb_at, axis=2)        # round-trip residual, px
    return np.exp(-(err / sigma) ** 2)


def stage_interpolate(frames, req):
    import cv2, numpy as np
    factor = int(req.get("interp_factor", 2))
    frames = [to_uint8(f) for f in frames]
    n = len(frames)
    if factor < 2 or n < 2:
        return frames

    emit({"loading_status": f"interpolate: robust optical-flow {factor}× ({n}→{factor*(n-1)+1} frames)…"})
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    h, w = frames[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    out = []
    for i in range(n - 1):
        f0, f1 = frames[i].astype(np.float32), frames[i + 1].astype(np.float32)
        f01 = cv2.GaussianBlur(_flow(grays[i], grays[i + 1]), (0, 0), 1.2)
        f10 = cv2.GaussianBlur(_flow(grays[i + 1], grays[i]), (0, 0), 1.2)
        conf = np.minimum(_flow_confidence(f01, f10, gx, gy),     # trust only where
                          _flow_confidence(f10, f01, gx, gy))     # both directions agree
        conf = cv2.GaussianBlur(conf, (0, 0), 2.0)[..., None]     # soften mask edges
        out.append(frames[i])
        for k in range(1, factor):
            t = k / factor
            m0 = cv2.remap(f0, gx - t * f01[..., 0], gy - t * f01[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m1 = cv2.remap(f1, gx - (1 - t) * f10[..., 0], gy - (1 - t) * f10[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            warped = m0 * (1 - t) + m1 * t                        # motion-compensated
            cross = f0 * (1 - t) + f1 * t                         # safe fallback
            mid = warped * conf + cross * (1 - conf)
            out.append(np.clip(mid, 0, 255).astype(np.uint8))
        emit({"step": i + 1, "total": n - 1})
    out.append(frames[-1])
    emit({"loading_status": f"interpolate: {len(out)} frames @ {factor}× fps (flow-guarded)"})
    return out


STAGES = {"refine": stage_refine, "upscale": stage_upscale, "interpolate": stage_interpolate}


def main():
    _watch_parent()
    line = sys.stdin.readline()
    if not line:
        return
    try:
        req = json.loads(line)
    except Exception as e:
        emit({"error": f"bad request: {e}"})
        return
    try:
        t0 = time.time()
        emit({"loading_status": "decoding input video…"})
        frames = b64_to_frames(req["video_b64"])
        fps = int(req.get("fps", 16))
        # Canonical order regardless of how the UI listed them: refine (detail at
        # base res) → interpolate (flow is robust on small frames) → upscale (enlarge
        # the already-smooth sequence). Interpolating AFTER upscaling makes flow noisy
        # on 4× pixels → juddery in-betweens.
        order = ["refine", "interpolate", "upscale"]
        stages = sorted(req.get("stages", []),
                        key=lambda s: order.index(s) if s in order else 99)
        emit({"loading_status": f"pass order: {' → '.join(stages)}"})
        for name in stages:
            fn = STAGES.get(name)
            if fn is None:
                emit({"loading_status": f"unknown stage '{name}' — skipped"})
                continue
            try:                                   # one bad stage must not kill the pass
                frames = fn(frames, req)
                if name == "interpolate":
                    fps *= int(req.get("interp_factor", 2))
            except Exception as se:
                emit({"loading_status": f"  ⚠ {name} failed ({se}); skipping — other stages continue"})
                _free_cuda()
        emit({"loading_status": "encoding result…"})
        frames = [to_uint8(f) for f in frames]   # final guard before mp4 encode
        b64 = frames_to_b64(frames, fps)
        h, w = frames[0].shape[0], frames[0].shape[1]
        emit({"enhanced_b64": b64, "frames": len(frames), "width": w, "height": h,
              "elapsed": round(time.time() - t0, 1)})
    except Exception as e:
        emit({"error": str(e), "trace": traceback.format_exc()[:1200]})


if __name__ == "__main__":
    main()
