#!/usr/bin/env python3
# Wan text-to-video daemon. Protocol mirrors generate_sdxl.py:
#
#   stdin line 1 (load): {"model_path": "...", "device": "auto|cuda|cpu"}
#       -> {"loading_status": "..."} lines, then {"ready": true, "device": "cuda"}
#   then one request per line:
#       {"prompt","neg_prompt","num_frames","steps","cfg_scale","width","height","fps","seed"}
#       -> {"step":i,"total":t} progress lines, then {"base64_mp4":"...","frames":N,"elapsed":s}
#
# Loaded once; the 27 GB model stays resident across requests.
import base64, json, os, sys, tempfile, time, traceback


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    load_line = sys.stdin.readline()
    if not load_line:
        return
    try:
        cfg = json.loads(load_line)
    except Exception as e:
        emit({"error": f"bad load config: {e}"})
        return

    model_path = cfg.get("model_path", "")
    try:
        emit({"loading_status": "importing torch + diffusers…"})
        import torch, gc, json as _json
        from diffusers.utils import export_to_video

        # Which pipeline does this model use? (model_index.json _class_name)
        pipe_cls = ""
        try:
            pipe_cls = _json.load(open(os.path.join(model_path, "model_index.json"))).get("_class_name", "")
        except Exception:
            pass

        pipe = None
        if "Wan" in pipe_cls:
            # ── 4-bit (nf4) — the "runs on a phone" trick ──────────────────────
            # Wan's UMT5-XXL text encoder is ~21 GB at fp16 (~5.5 GB at 4-bit) and
            # the transformer ~2.6 GB (~1 GB). At fp16 + pinned offload it froze a
            # 39 GB box on load. Quantized, the WHOLE model fits resident in ~8 GB
            # VRAM — same idea tinyq4 already uses for LLM weights. Falls back to a
            # plain load if bitsandbytes isn't available.
            try:
                from diffusers import WanPipeline, WanTransformer3DModel
                from diffusers import BitsAndBytesConfig as DiffBnb
                from transformers import UMT5EncoderModel
                from transformers import BitsAndBytesConfig as TfBnb
                d_nf4 = DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                t_nf4 = TfBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                emit({"loading_status": "loading text encoder (4-bit nf4)…"})
                te = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder",
                        quantization_config=t_nf4, torch_dtype=torch.bfloat16)
                emit({"loading_status": "loading transformer (4-bit nf4)…"})
                tr = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer",
                        quantization_config=d_nf4, torch_dtype=torch.bfloat16)
                emit({"loading_status": "assembling pipeline…"})
                pipe = WanPipeline.from_pretrained(model_path, text_encoder=te, transformer=tr,
                        torch_dtype=torch.bfloat16)
                pipe.vae.to("cuda")                 # quantized parts already on GPU
                pipe.vae.enable_tiling()
                gc.collect()
            except Exception as qe:
                emit({"loading_status": f"4-bit load failed ({qe}); using fallback…"})
                pipe = None

        if pipe is None:
            # Generic / lighter models (LTX, CogVideoX) — sequential offload, no pin.
            from diffusers import DiffusionPipeline
            emit({"loading_status": f"loading {os.path.basename(model_path)} (sequential offload)…"})
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_model_cpu_offload()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

        pipe.set_progress_bar_config(disable=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emit({"ready": True, "device": device})
    except Exception as e:
        emit({"error": f"load failed: {e}", "trace": traceback.format_exc()[:800]})
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
            t0 = time.time()
            total = int(req.get("steps", 30))

            def cb(_pipe, i, _t, kwargs):
                emit({"step": i + 1, "total": total})
                return kwargs

            generator = None
            seed = req.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                generator = torch.Generator(device="cpu").manual_seed(int(seed))

            frames = pipe(
                prompt=req.get("prompt", ""),
                negative_prompt=(req.get("neg_prompt", "") or None),
                height=int(req.get("height", 480)),
                width=int(req.get("width", 832)),
                num_frames=int(req.get("num_frames", 49)),
                num_inference_steps=total,
                guidance_scale=float(req.get("cfg_scale", 6.0)),
                generator=generator,
                callback_on_step_end=cb,
            ).frames[0]

            fps = int(req.get("fps", 16))
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            export_to_video(frames, tmp.name, fps=fps)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            os.unlink(tmp.name)

            emit({"base64_mp4": b64, "frames": len(frames), "elapsed": round(time.time() - t0, 1)})
        except Exception as e:
            emit({"error": str(e), "trace": traceback.format_exc()[:800]})


if __name__ == "__main__":
    main()
