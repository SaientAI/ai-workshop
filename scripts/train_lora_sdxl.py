#!/usr/bin/env python3
"""SDXL LoRA trainer for AI Workshop.
Reads JSON config from the path given as argv[1].
Streams JSON progress lines to stdout.
"""
import json, sys, os, signal, glob, math
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_stop_requested = False

def emit(obj):
    print(json.dumps(obj), flush=True)

def log(msg):    emit({"type": "log",   "msg": str(msg)})
def err(msg):    emit({"type": "error", "msg": str(msg)}); sys.exit(1)
def done(path):  emit({"type": "done",  "output": str(path)})
def progress(step, total_steps, epoch, total_epochs, loss):
    emit({
        "type": "progress",
        "step": step,
        "total_steps": total_steps,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": round(float(loss), 6),
    })

def _handle_stop(*_):
    global _stop_requested
    _stop_requested = True
    log("Stop requested — will save after current batch")

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT,  _handle_stop)

# ── Local SDXL config discovery (same logic as generate_sdxl.py) ────────────
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

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        err("Usage: train_lora_sdxl.py <config.json>")

    cfg = json.loads(Path(sys.argv[1]).read_text())
    model_path  = cfg["model_path"]
    dataset_dir = cfg["dataset_dir"]
    output_path = Path(cfg["output_path"])
    rank        = int(cfg.get("rank", 16))
    alpha       = float(cfg.get("alpha", rank))
    lr          = float(cfg.get("lr", 1e-4))
    epochs      = int(cfg.get("epochs", 10))
    batch_size  = int(cfg.get("batch_size", 1))
    resolution  = int(cfg.get("resolution", 1024))

    # ── Dependency checks ─────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        err("torch not installed")

    try:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    except ImportError:
        err("peft not installed — run: pip install peft")

    try:
        from safetensors.torch import save_file
    except ImportError:
        err("safetensors not installed — run: pip install safetensors")

    from PIL import Image
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    log(f"Device: {device}  |  dtype: {dtype}")

    # ── Load SDXL components ──────────────────────────────────────────────────
    is_ckpt = os.path.isfile(model_path) and \
              model_path.endswith((".safetensors", ".ckpt"))

    if is_ckpt:
        log("Loading from checkpoint — extracting SDXL components...")
        from diffusers import StableDiffusionXLPipeline, DDPMScheduler
        local_base = _find_local_sdxl_base()
        kwargs = {"torch_dtype": dtype, "safety_checker": None}
        if local_base:
            kwargs["config"] = local_base
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, **kwargs)
        unet            = pipe.unet
        vae             = pipe.vae
        text_encoder    = pipe.text_encoder
        text_encoder_2  = pipe.text_encoder_2
        tokenizer       = pipe.tokenizer
        tokenizer_2     = pipe.tokenizer_2
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        del pipe
    else:
        from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        log(f"Loading diffusers model from {model_path} ...")
        load_kw = {"torch_dtype": dtype, "local_files_only": True}
        unet           = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", **load_kw)
        vae            = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **load_kw)
        text_encoder   = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", **load_kw)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", **load_kw)
        tokenizer      = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tokenizer_2    = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler",
                                                        local_files_only=True)

    # Freeze VAE + text encoders; move to device
    vae.requires_grad_(False).to(device)
    text_encoder.requires_grad_(False).to(device)
    text_encoder_2.requires_grad_(False).to(device)

    # ── Apply LoRA to UNet ────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in unet.parameters())
    log(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    class LoraDataset(Dataset):
        def __init__(self, folder, res):
            self.pairs = []
            for ext in ("*.png","*.jpg","*.jpeg","*.webp","*.PNG","*.JPG","*.JPEG"):
                for img_p in sorted(glob.glob(os.path.join(folder, ext))):
                    cap_p = os.path.splitext(img_p)[0] + ".txt"
                    cap   = Path(cap_p).read_text().strip() if os.path.exists(cap_p) else ""
                    self.pairs.append((img_p, cap))
            self.tfm = transforms.Compose([
                transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(res),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            path, cap = self.pairs[idx]
            return {"pixel_values": self.tfm(Image.open(path).convert("RGB")), "caption": cap}

    dataset = LoraDataset(dataset_dir, resolution)
    if not dataset.pairs:
        err(f"No images found in {dataset_dir}")
    log(f"Dataset: {len(dataset)} images")

    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer  = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=lr
    )
    total_steps = epochs * math.ceil(len(dataset) / batch_size)
    global_step = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    log("Starting training…")

    for epoch in range(epochs):
        if _stop_requested:
            break
        epoch_loss, steps_ep = 0.0, 0

        for batch in loader:
            if _stop_requested:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            captions     = batch["caption"]

            with torch.no_grad():
                # Tokenise both CLIP encoders
                tok1 = tokenizer(captions, padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
                tok2 = tokenizer_2(captions, padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt").input_ids.to(device)
                out1 = text_encoder(tok1, output_hidden_states=True)
                out2 = text_encoder_2(tok2, output_hidden_states=True)
                # SDXL expects concatenated hidden states from both encoders
                prompt_embeds        = torch.cat([out1.hidden_states[-2],
                                                  out2.hidden_states[-2]], dim=-1)
                pooled_prompt_embeds = out2[0]

                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample() \
                          * vae.config.scaling_factor

            # Forward diffusion
            noise     = torch.randn_like(latents)
            bsz       = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL size / crop conditioning
            add_time_ids = torch.tensor(
                [[resolution, resolution, 0, 0, resolution, resolution]] * bsz,
                dtype=dtype, device=device)
            added_cond = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            noise_pred = unet(noisy, timesteps, prompt_embeds,
                              added_cond_kwargs=added_cond).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            steps_ep    += 1
            epoch_loss  += loss.item()
            progress(global_step, total_steps, epoch + 1, epochs,
                     epoch_loss / steps_ep)

        log(f"Epoch {epoch+1}/{epochs} complete — avg loss: "
            f"{epoch_loss/max(steps_ep,1):.6f}")

    # ── Save LoRA weights ─────────────────────────────────────────────────────
    log(f"Saving LoRA → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state_dict = get_peft_model_state_dict(unet)
    save_file(lora_state_dict, str(output_path))
    log("Saved.")
    done(output_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        emit({"type": "error", "msg": f"{e}\n{traceback.format_exc()}"})
        sys.exit(1)
