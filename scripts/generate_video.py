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
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video

        # ── Single-pass 4-bit (nf4) quant ──────────────────────────────────────
        # PipelineQuantizationConfig loads + quantizes the heavy components in ONE
        # read pass (no double-read). Wan's UMT5-XXL text encoder is ~21 GB fp16 /
        # ~5.5 GB at 4-bit; quantized, the whole model fits resident in ~8 GB VRAM
        # with RAM staying flat — same trick tinyq4 uses for 4-bit LLM weights.
        pipe = None
        try:
            from diffusers import PipelineQuantizationConfig
            emit({"loading_status": "loading + quantizing to 4-bit (nf4) — reads the 27 GB once, ~5 min…"})
            quant = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
            pipe = DiffusionPipeline.from_pretrained(
                model_path, quantization_config=quant, torch_dtype=torch.bfloat16
            )
        except Exception as qe:
            emit({"loading_status": f"4-bit unavailable ({qe}); plain load…"})
            pipe = None

        if pipe is None:
            emit({"loading_status": f"loading {os.path.basename(model_path)} (sequential offload)…"})
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

        # Offload is safe now — quantized components are ~6.5 GB, not the 21 GB that
        # pinned RAM and froze the box. Try model offload, fall back to sequential.
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pass
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
