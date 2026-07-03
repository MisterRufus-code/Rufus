#!/usr/bin/env python3
"""
comfy_client.py — ComfyUI + FLUX.1-dev image generator for Rufus.

Generates photoreal portrait images via a local ComfyUI server (FLUX.1-dev),
then reuses sd_client's PIL/FFmpeg helpers to upscale → crop 1080×1920 → Ken
Burns mp4. Produces the same deliverable as sd_client / hyperframes — a list of
.mp4 paths — so the rest of the pipeline is untouched.

Why this exists: with a 24GB GPU (RTX 3090) FLUX.1-dev is the best free local
image model. ComfyUI exposes a headless HTTP API (POST /prompt, poll /history,
GET /view) — no websocket dependency needed.

Requirements (Windows 11 / Linux):
  ComfyUI running with --listen, FLUX checkpoint in models/checkpoints/:
    flux1-dev-fp8.safetensors   (single-file fp8 build — fits 24GB comfortably)

Environment:
  COMFY_HOST    (default: http://localhost:8188)
  COMFY_MODEL   (default: flux1-dev-fp8.safetensors)
  COMFY_STEPS   (default: 20)
  SD_CLIPS      (clip count — shared with sd_client)
  RUFUS_DEBUG=1 (keep keyframes + prompts under media_library/debug/<stamp>/)

Usage:
  RUFUS_VIDEO_SOURCE=comfy python scripts/main.py --skip-upload
  python scripts/comfy_client.py "modern luxury kitchen, golden hour, wide angle"
"""

import json
import os
import random
import time
import uuid
from pathlib import Path

import requests

# Reuse the proven, dependency-free (PIL/FFmpeg-only) helpers from sd_client so
# animation and the negative prompt behave identically across backends.
from sd_client import (
    _animate_to_clip,
    NEGATIVE_PROMPT,
    OUT_W,
    OUT_H,
)


def _fit_to_portrait(img_bytes: bytes, out_path: Path) -> bool:
    """Cover-resize a FLUX frame to exactly 1080×1920, preserving composition.

    832×1472 (0.5652) vs 1080×1920 (0.5625) are near-identical aspect ratios, so
    we Lanczos-scale to just cover the target and trim the ~0.5% sliver. This
    keeps ~99% of the frame FLUX composed — unlike sd_client's fixed 2× upscale
    + center-crop, which at this generation size would discard 35% of the image.
    """
    from PIL import Image
    import io as _io

    img  = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = max(OUT_W / w, OUT_H / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - OUT_W) // 2
    top  = (new_h - OUT_H) // 2
    img  = img.crop((left, top, left + OUT_W, top + OUT_H))

    img.save(str(out_path), format="PNG", optimize=False)
    return out_path.exists() and out_path.stat().st_size > 20_000

# Perceptual de-duplication so no two scenes in a video look alike. Kept local
# (sd_client doesn't expose these in this build) — PIL-only, no numpy.
DUP_THRESHOLD   = 6    # max Hamming distance (of 64 bits) to treat two frames as "same"
MAX_DUP_RETRIES = 2    # regenerations allowed before accepting a near-duplicate


def _avg_hash(png_path: Path) -> int | None:
    """64-bit average hash (aHash) of an image — visually similar images produce
    hashes only a few bits apart."""
    try:
        from PIL import Image
        img = Image.open(str(png_path)).convert("L").resize((8, 8), Image.LANCZOS)
        px  = list(img.tobytes())
        avg = sum(px) / len(px)
        bits = 0
        for p in px:
            bits = (bits << 1) | (1 if p >= avg else 0)
        return bits
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")

# FLUX likes ~1MP. 832×1472 is ÷16 on both axes, ~9:16, then we upscale+crop to
# exactly 1080×1920. (Generating native 1080×1920 wastes VRAM/time for no gain.)
GEN_W, GEN_H = 832, 1472

POLL_INTERVAL = 1.5    # seconds between /history polls
GEN_TIMEOUT   = 300    # max seconds to wait for one image (FLUX ~20-30s on a 3090)


def _host() -> str:
    return os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")


def is_available() -> bool:
    """Return True if a ComfyUI server is up and responsive."""
    try:
        r = requests.get(f"{_host()}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _build_flux_graph(prompt: str, seed: int, model: str, steps: int) -> dict:
    """A minimal FLUX.1-dev txt2img graph (no custom nodes required).

    CheckpointLoaderSimple → CLIPTextEncode(+/-) → FluxGuidance → KSampler
    → VAEDecode → SaveImage. Mirrors the tutorial workflow but strips the WAS
    style-CSV nodes (which need extra installs) and injects our own prompt.
    """
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEGATIVE_PROMPT, "clip": ["4", 1]}},
        "15": {"class_type": "FluxGuidance",
               "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
        "10": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": GEN_W, "height": GEN_H, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": 1.0,
                  "sampler_name": "dpmpp_2m", "scheduler": "sgm_uniform",
                  "denoise": 1.0,
                  "model": ["4", 0], "positive": ["15", 0],
                  "negative": ["7", 0], "latent_image": ["10", 0],
              }},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "rufus", "images": ["8", 0]}},
    }


def _submit(graph: dict, client_id: str) -> str | None:
    """POST a workflow to /prompt. Returns prompt_id or None."""
    try:
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": client_id},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("prompt_id")
    except Exception as e:
        print(f"[comfy] submit failed: {e}")
        return None


def _await_image(prompt_id: str) -> bytes | None:
    """Poll /history/<id> until the SaveImage node reports an output, then fetch
    the PNG via /view. Returns raw bytes or None on timeout/failure."""
    deadline = time.time() + GEN_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(f"{_host()}/history/{prompt_id}", timeout=15)
            r.raise_for_status()
            hist = r.json()
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue

        entry = hist.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            for node in outputs.values():
                images = node.get("images") or []
                if images:
                    img = images[0]
                    try:
                        vr = requests.get(
                            f"{_host()}/view",
                            params={"filename": img.get("filename", ""),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")},
                            timeout=30,
                        )
                        vr.raise_for_status()
                        return vr.content
                    except Exception as e:
                        print(f"[comfy] image fetch failed: {e}")
                        return None
            # entry exists but no images → generation errored in ComfyUI
            if "status" in entry and entry["status"].get("status_str") == "error":
                print("[comfy] ComfyUI reported a generation error for this prompt")
                return None
        time.sleep(POLL_INTERVAL)

    print(f"[comfy] timed out after {GEN_TIMEOUT}s waiting for image")
    return None


def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0,
                   niche_cfg: dict | None = None) -> list[Path]:
    """Generate one Ken Burns clip per query via ComfyUI + FLUX.1-dev, in order.

    Pipeline per clip:
      query → FLUX 832×1472 → Lanczos 2× → crop 1080×1920 → Ken Burns mp4

    Matches sd_client.generate_clips' contract: returns a list of 1080×1920 mp4
    Paths (one per query), or [] if ComfyUI is not running / all images fail, so
    main.py can fall through to the next backend.
    """
    if not is_available():
        print(f"[comfy] ComfyUI not running at {_host()} — start it with --listen, "
              f"or set COMFY_HOST. Falling back.")
        return []

    model = (os.environ.get("COMFY_MODEL")
             or (niche_cfg or {}).get("comfy_model")
             or "flux1-dev-fp8.safetensors")
    steps = int(os.environ.get("COMFY_STEPS", "20"))

    tmp_dir = Path(__file__).parent.parent / "media_library" / "temp" / "comfy"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(queries or ["cinematic establishing shot"])
    if len(prompts) < n:
        base = prompts[:]
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + ", different angle, wider shot")
    prompts = prompts[:n]

    stamp        = int(time.time())
    client_id    = uuid.uuid4().hex
    master_seed  = random.randint(1, 2_000_000_000)
    accepted_hashes: list[int] = []
    clips: list[Path] = []
    print(f"[comfy] FLUX model={model} steps={steps} base_seed={master_seed}")

    debug_dir = None
    if os.environ.get("RUFUS_DEBUG"):
        debug_dir = Path(__file__).parent.parent / "media_library" / "debug" / str(stamp)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[comfy] DEBUG on — keeping keyframes in {debug_dir}")

    for i, prompt in enumerate(prompts):
        print(f"[comfy] {i+1}/{len(prompts)}: {prompt[:70]}")
        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False

        for retry in range(MAX_DUP_RETRIES + 1):
            # %(2**31) keeps the seed in range for any backend; offset per clip/retry.
            seed  = (master_seed + i + 1000 * retry) % (2**31 - 1)
            graph = _build_flux_graph(prompt, seed, model, steps)
            pid   = _submit(graph, client_id)
            if not pid:
                continue
            img_bytes = _await_image(pid)
            if not img_bytes:
                continue

            if not _fit_to_portrait(img_bytes, png_path):  # → exactly 1080×1920
                continue

            h = _avg_hash(png_path)
            is_dup = (h is not None and accepted_hashes
                      and min(_hamming(h, p) for p in accepted_hashes) < DUP_THRESHOLD)
            if is_dup and retry < MAX_DUP_RETRIES:
                print(f"[comfy] dup on clip {i+1} → regen (retry {retry+1})")
                continue
            if is_dup:
                print(f"[comfy] clip {i+1} still near-dup after retries — keeping")
            if h is not None:
                accepted_hashes.append(h)
            accepted = True
            break

        if not accepted:
            print(f"[comfy] no usable image for clip {i+1} — skipping")
            continue

        if debug_dir is not None:
            try:
                (debug_dir / f"{i+1:02d}.png").write_bytes(png_path.read_bytes())
                (debug_dir / f"{i+1:02d}.txt").write_text(
                    f"FLUX PROMPT:\n{prompt}\n", encoding="utf-8")
            except Exception as e:
                print(f"[comfy] debug-save failed for clip {i+1}: {e}")

        clip_path = tmp_dir / f"{stamp}_{i}.mp4"
        if _animate_to_clip(png_path, clip_path, duration=clip_duration, idx=i):
            clips.append(clip_path)
            print(f"[comfy] clip {i+1} ready")
        else:
            print(f"[comfy] animation failed for clip {i+1}")
        png_path.unlink(missing_ok=True)

    print(f"[comfy] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    qs = sys.argv[1:] or ["modern luxury kitchen interior, golden hour light, wide angle",
                          "sunlit living room, floor to ceiling windows, city view"]
    for p in generate_clips(qs, n=len(qs)):
        print(f"CLIP={p}")
