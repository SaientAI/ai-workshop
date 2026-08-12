#!/usr/bin/env python3
"""Vision analyzer daemon for Saient — local image understanding via Moondream2.

Mirrors the generate_sdxl.py daemon protocol (newline-delimited JSON on stdin/stdout):
  stdin line 1:  {"model":"moondream2","device":"auto"}     (load)
  stdout line 1: {"ready":true,"device":"cuda"}             OR  {"error":"..."}
  Per request:
  stdin:  {"image_b64":"<base64 PNG/JPEG>","question":"Describe this image."}
          (empty/blank question → a general caption)
  stdout: {"answer":"...","elapsed":1.4}                    OR  {"error":"..."}  (daemon keeps running)

Full Setup downloads the model into Saient's cache. Runtime loading is strictly
local; it never starts an implicit download.
"""
import base64, io, json, os, shutil, sys, time
from pathlib import Path

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

        model_name = cfg.get("model", "moondream2")
        if model_name not in MODELS:
            raise RuntimeError(f"unsupported local vision model: {model_name}")
        assets = Path(os.environ.get("SAIENT_RUNTIME_ASSETS_DIR", ""))
        repo = assets / "vision" / "moondream2"
        starmie = assets / "vision" / "starmie-v1" / "tokenizer.json"
        required = (repo / "config.json", repo / "model.safetensors", starmie)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Managed Moondream assets are missing. Open Settings > Setup and run Full setup. "
                + "Missing: " + ", ".join(str(path) for path in missing)
            )

        # Transformers' dynamic-module resolver does not discover every nested
        # relative import in this Moondream revision. Stage the complete pinned
        # Python package in run-only state before loading; app startup clears it.
        hf_home = Path(os.environ.get("HF_HOME", assets / ".." / "runtime-tmp" / "huggingface"))
        module_dir = hf_home / "modules" / "transformers_modules" / repo.name
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / "__init__.py").touch()
        for source in repo.glob("*.py"):
            shutil.copy2(source, module_dir / source.name)
        want = cfg.get("device", "auto")
        device = "cuda" if (want != "cpu" and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # This pinned Moondream revision calls Tokenizer.from_pretrained for its
        # separate Starmie tokenizer inside trusted model code. Redirect that
        # one request to the explicitly managed local file, then restore it.
        from tokenizers import Tokenizer
        original_tokenizer_loader = Tokenizer.from_pretrained
        def local_tokenizer(identifier, *args, **kwargs):
            if identifier == "moondream/starmie-v1":
                return Tokenizer.from_file(str(starmie))
            raise RuntimeError(f"blocked unexpected tokenizer download: {identifier}")
        Tokenizer.from_pretrained = staticmethod(local_tokenizer)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(repo), trust_remote_code=True, torch_dtype=dtype, local_files_only=True,
            ).to(device).eval()
        finally:
            Tokenizer.from_pretrained = original_tokenizer_loader
        tokenizer = AutoTokenizer.from_pretrained(
            str(repo), trust_remote_code=True, local_files_only=True,
        )
    except Exception as e:
        emit({"error": f"load failed: {e}"}); raise SystemExit(1)

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
