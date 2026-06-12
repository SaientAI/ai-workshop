#!/usr/bin/env python3
"""LoRA dataset cleaner for Saient.
Streams JSON progress lines to stdout.

What it does:
  - Removes known junk files (.DS_Store, Thumbs.db, etc.)
  - Removes files with wrong extensions that aren't images or captions
  - Validates each image (corrupt → remove, too small → remove)
  - Removes orphaned .txt files that have no matching image
  - Creates an empty .txt caption for every image that's missing one
"""

import json, sys, os
from pathlib import Path

def emit(obj):
    print(json.dumps(obj), flush=True)

def log(msg):       emit({"type": "log",    "msg": str(msg)})
def warn(msg):      emit({"type": "warn",   "msg": str(msg)})
def done(stats):    emit({"type": "done",   "stats": stats})
def err(msg):       emit({"type": "error",  "msg": str(msg)}); sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

JUNK_NAMES = {
    ".ds_store", "thumbs.db", "desktop.ini", ".gitkeep", ".gitignore",
    "picasa.ini", ".directory", "ehthumbs.db", "zone.identifier",
}
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
KEEP_EXTS   = IMAGE_EXTS | {".txt"}   # everything else is junk
MIN_SIZE    = 64                        # pixels — smaller than this is useless

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        err("Usage: clean_lora_dataset.py <folder>")

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        err(f"Not a directory: {folder}")

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        err("Pillow not installed — run: pip install Pillow")

    stats = {
        "images_found":      0,
        "images_valid":      0,
        "images_removed":    0,
        "captions_existing": 0,
        "captions_created":  0,
        "orphans_removed":   0,
        "junk_removed":      0,
    }

    # ── Pass 1: remove junk / non-image-non-caption files ────────────────────
    log("Pass 1 — scanning for junk files…")
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.name.lower() in JUNK_NAMES:
            f.unlink()
            log(f"  Removed junk: {f.name}")
            stats["junk_removed"] += 1
        elif f.suffix.lower() not in KEEP_EXTS:
            f.unlink()
            log(f"  Removed unsupported file: {f.name}")
            stats["junk_removed"] += 1

    # ── Pass 2: validate images ───────────────────────────────────────────────
    log("Pass 2 — validating images…")
    valid_stems = set()

    image_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    stats["images_found"] = len(image_files)

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img.verify()            # catches truncated/corrupt files
        except Exception as e:
            img_path.unlink()
            log(f"  Removed corrupt: {img_path.name} ({e})")
            stats["images_removed"] += 1
            continue

        # Re-open after verify (verify() exhausts the file object)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            img_path.unlink()
            log(f"  Removed unreadable: {img_path.name} ({e})")
            stats["images_removed"] += 1
            continue

        if w < MIN_SIZE or h < MIN_SIZE:
            img_path.unlink()
            log(f"  Removed too-small ({w}x{h}): {img_path.name}")
            stats["images_removed"] += 1
            continue

        valid_stems.add(img_path.stem)
        stats["images_valid"] += 1

    # ── Pass 3: remove orphaned .txt captions (no matching image) ────────────
    log("Pass 3 — checking captions…")
    for txt in sorted(folder.glob("*.txt")):
        if txt.stem not in valid_stems:
            txt.unlink()
            log(f"  Removed orphaned caption: {txt.name}")
            stats["orphans_removed"] += 1

    # ── Pass 4: create missing captions for valid images ─────────────────────
    for stem in sorted(valid_stems):
        cap = folder / f"{stem}.txt"
        if cap.exists():
            content = cap.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                warn(f"  Empty caption: {cap.name} (ok, but add a description for best results)")
            stats["captions_existing"] += 1
        else:
            cap.write_text("", encoding="utf-8")
            log(f"  Created caption: {cap.name}")
            stats["captions_created"] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    log(
        f"Done — {stats['images_valid']} valid images | "
        f"{stats['images_removed']} removed | "
        f"{stats['captions_created']} captions created | "
        f"{stats['orphans_removed']} orphans removed | "
        f"{stats['junk_removed']} junk removed"
    )
    done(stats)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        emit({"type": "error", "msg": f"{e}\n{traceback.format_exc()}"})
        sys.exit(1)
