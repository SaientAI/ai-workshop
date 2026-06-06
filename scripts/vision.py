#!/usr/bin/env python3
"""Vision analyzer daemon for Saient — local image understanding via Moondream2.

Mirrors the generate_sdxl.py daemon protocol (newline-delimited JSON on stdin/stdout):
  stdin line 1:  {"model":"moondream2","device":"auto"}     (load)
  stdout line 1: {"ready":true,"device":"cuda"}             OR  {"error":"..."}
  Per request:
  stdin:  {"image_b64":"<base64 PNG/JPEG>","question":"Describe this image."}
          (empty/blank question → a general caption)
  stdout: {"answer":"...","elapsed":1.4}                    OR  {"error":"..."}  (daemon keeps running)

The model downloads from HuggingFace on first use, then caches. Subsequent loads are local.
"""
import base64, io, json, os, sys, time

# Reuse reserved-but-unallocated CUDA blocks (set before torch/CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODELS = {
    "moondream2": "vikhyatk/moondream2",
}

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    try:
        cfg = json.loads(sys.stdin.readline() or "{}")
    except Exception as e:
        emit({"error": f"bad load config: {e}"}); return

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = MODELS.get(cfg.get("model", "moondream2"), MODELS["moondream2"])
        want = cfg.get("device", "auto")
        device = "cuda" if (want != "cpu" and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=dtype,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    except Exception as e:
        emit({"error": f"load failed: {e}"}); return

    emit({"ready": True, "device": device})

    def describe(image, question):
        # Moondream's API has shifted across releases — try the current one, then fall back.
        q = (question or "").strip()
        if hasattr(model, "query"):                        # moondream 2025+ API
            out = model.query(image, q or "Describe this image in detail.")
            return out["answer"] if isinstance(out, dict) else str(out)
        if not q and hasattr(model, "caption"):            # caption convenience
            out = model.caption(image)
            return out["caption"] if isinstance(out, dict) else str(out)
        enc = model.encode_image(image)                    # classic API
        return model.answer_question(enc, q or "Describe this image in detail.", tokenizer)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            raw = base64.b64decode(req["image_b64"])
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            t0 = time.time()
            answer = describe(image, req.get("question", ""))
            emit({"answer": answer.strip(), "elapsed": round(time.time() - t0, 2)})
        except Exception as e:
            emit({"error": str(e)})

if __name__ == "__main__":
    main()
