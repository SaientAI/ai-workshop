#!/usr/bin/env python3
"""Pull stabilityai/stable-diffusion-3.5-medium into Saient's managed data/models folder."""
import json
import os
import sys

from huggingface_hub import HfApi, snapshot_download
from saient_paths import configure_hf_cache, models_dir

REPO = "stabilityai/stable-diffusion-3.5-medium"
DEST = str(models_dir() / "image" / "stable-diffusion-3.5-medium")
ALLOW = [
    "model_index.json", "*.json", "*.txt", "*.model",
    "scheduler/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
    "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
    "transformer/*", "vae/*",
]
IGNORE = ["*.bin", "*.ckpt", "*.pt", "*.pth", "*.onnx", "*workflow*", "*example*", "*.jpg", "*.png", "LICENSE*"]


def main():
    configure_hf_cache()
    try:
        HfApi().auth_check(REPO)
    except Exception:
        print(
            f"Gated — open https://huggingface.co/{REPO} while logged in as Xlbully and click Agree.",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {REPO} -> {DEST}", flush=True)
    snapshot_download(repo_id=REPO, local_dir=DEST, allow_patterns=ALLOW, ignore_patterns=IGNORE)
    ok = os.path.isfile(os.path.join(DEST, "model_index.json"))
    size_gb = round(
        sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DEST) for f in fs) / 1e9,
        2,
    )
    print(json.dumps({"ok": ok, "dest": DEST, "size_gb": size_gb}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
