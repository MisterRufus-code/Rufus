#!/usr/bin/env python3
"""
wan_client.py — Wan 2.2 14B image-to-video for Rufus (via ComfyUI).

The successor to svd_client as the PRIMARY motion engine: Wan 2.2 (2025) has
dramatically better temporal consistency than SVD (2023) — structures keep
their shape while moving (verified on the rig: 81 frames, zero warping on the
exact fine-detail content SVD melted), it takes a TEXT motion prompt per clip,
and it handles faces well enough that the SVD-era face-skip heuristics don't
apply here.

Needs the six files the ComfyUI "Wan 2.2 14B Image to Video" template installs
(~35GB total, one-time):
    models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
    models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
    models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
    models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
    models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
    models/vae/wan_2.1_vae.safetensors

Contract: animate_image(png_path, out_path, duration, idx, prompt) -> bool —
same deliverable as svd_client.animate_image, so comfy_client can walk an
ordered engine chain (wan → svd → Ken Burns) and a failure here never costs
a clip.

FAIL-SAFE BY DESIGN: this graph is reconstructed from ComfyUI's native Wan 2.2
nodes rather than a captured API export, so any node/param mismatch with the
installed ComfyUI version must cost nothing — ready() gates on /object_info,
and any submit/generation error returns False (caller falls back to SVD).
When a submit is rejected, the ComfyUI error text is printed so the graph can
be corrected from the run log.

Graph (two-stage MoE — high-noise expert then low-noise expert):
  UNETLoader(high) → LoraLoaderModelOnly(lightx2v high) → ModelSamplingSD3 ┐
  UNETLoader(low)  → LoraLoaderModelOnly(lightx2v low)  → ModelSamplingSD3 ┤
  CLIPLoader(umt5, wan) → CLIPTextEncode(motion prompt / negative)         ├→
  LoadImage + VAELoader → WanImageToVideo → KSamplerAdvanced(high, steps
  0..N/2, leftover noise) → KSamplerAdvanced(low, N/2..end) → VAEDecode →
  SaveImage frames → svd_client._assemble (16fps → 30fps ping-pong loop
  at 1080×1920).

The lightx2v LoRAs distill generation to 4 steps — that's what makes a 14B
model practical per-beat on a 3090 (~2-5 min/clip).

Environment:
  RUFUS_WAN          1 (default) — 0 disables Wan (chain continues with SVD)
  RUFUS_WAN_FRAMES   81   (Wan's native 5s at 16fps; ping-pong+loop covers the
                           requested clip duration downstream)
  RUFUS_WAN_STEPS    4    (lightx2v distilled step count)
  RUFUS_WAN_SHIFT    5.0  (ModelSamplingSD3 shift)
  RUFUS_WAN_TIMEOUT  900  (seconds to wait for one clip's frames)
  WAN_HIGH_MODEL / WAN_LOW_MODEL / WAN_HIGH_LORA / WAN_LOW_LORA /
  WAN_CLIP_NAME / WAN_VAE_NAME — filename overrides for the six files above.
"""

import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import requests

from comfy_client import _host
# Reuse the proven SVD helpers: portrait init prep (576×1024 is a valid Wan
# bucket too), history polling, and the frames→mp4 assembly pipeline.
from svd_client import _prep_init_image, _await_frames, _assemble, _upload_image, SVD_W, SVD_H

WAN_FPS = 16   # Wan 2.2's native frame rate (81 frames = ~5s)

HIGH_MODEL = os.environ.get("WAN_HIGH_MODEL", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
LOW_MODEL  = os.environ.get("WAN_LOW_MODEL",  "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
HIGH_LORA  = os.environ.get("WAN_HIGH_LORA",  "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")
LOW_LORA   = os.environ.get("WAN_LOW_LORA",   "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")
CLIP_NAME  = os.environ.get("WAN_CLIP_NAME",  "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
VAE_NAME   = os.environ.get("WAN_VAE_NAME",   "wan_2.1_vae.safetensors")

NEGATIVE_PROMPT = (
    "static image, no motion, blurry, distortion, warping, morphing, "
    "deformed anatomy, extra limbs, flickering, text, watermark, "
    "jpeg artifacts, oversaturated colors"
)


def enabled() -> bool:
    return os.environ.get("RUFUS_WAN", "1").strip().lower() not in ("0", "false", "no", "off")


def _node_choices(class_type: str, field: str) -> list[str] | None:
    """The valid choices for a node's dropdown field per /object_info, or
    None when the node/endpoint can't be read (server down, node missing)."""
    try:
        r = requests.get(f"{_host()}/object_info/{class_type}", timeout=10)
        if r.status_code != 200:
            return None
        info = r.json().get(class_type)
        if not info:
            return None
        choices = info["input"]["required"][field][0]
        return [c for c in choices if isinstance(c, str)] if isinstance(choices, list) else None
    except Exception:
        return None


def ready() -> tuple[bool, str]:
    """Fail-closed preflight: the WanImageToVideo node must exist and BOTH
    diffusion models must be in UNETLoader's list. (LoRA/CLIP/VAE mismatches
    are caught by the submit-level fallback instead — two probes keep this
    fast while covering the likeliest misinstalls.)"""
    try:
        r = requests.get(f"{_host()}/object_info/WanImageToVideo", timeout=10)
        if r.status_code != 200 or not r.json().get("WanImageToVideo"):
            return False, "ComfyUI has no WanImageToVideo node (update ComfyUI?)"
    except Exception as e:
        return False, f"ComfyUI unreachable ({e})"

    unets = _node_choices("UNETLoader", "unet_name")
    if unets is None:
        return False, "couldn't read UNETLoader's model list"
    missing = [m for m in (HIGH_MODEL, LOW_MODEL) if m not in unets]
    if missing:
        return False, (f"missing in models\\diffusion_models\\: {', '.join(missing)}")
    return True, "Wan 2.2 14B loadable (high+low noise)"


def _motion_prompt(beat_prompt: str) -> str:
    """A Wan motion prompt from the beat's image prompt: the subject (so the
    model knows what it's animating) + a consistent subtle-motion direction.
    Wan follows motion text well — this is where SVD's blind drift becomes
    directed, gentle documentary movement."""
    subject = " ".join((beat_prompt or "").split())[:220]
    return (f"{subject}. The camera moves slowly and subtly — a gentle push-in "
            f"or drift. Natural, restrained motion true to the scene; stable "
            f"composition; cinematic documentary feel; no sudden movements.")


def _build_wan_graph(image_name: str, prompt: str, seed: int,
                     frames: int, steps: int, shift: float) -> dict:
    """Native-node Wan 2.2 i2v graph. Two experts split one denoise schedule:
    high-noise model takes the first half of the steps (with leftover noise),
    low-noise model finishes. cfg 1.0 + euler/simple per the lightx2v recipe."""
    half = max(1, steps // 2)
    return {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": HIGH_MODEL, "weight_dtype": "default"}},
        "2":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": LOW_MODEL, "weight_dtype": "default"}},
        "3":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": HIGH_LORA, "strength_model": 1.0}},
        "4":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["2", 0], "lora_name": LOW_LORA, "strength_model": 1.0}},
        "5":  {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["3", 0], "shift": shift}},
        "6":  {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["4", 0], "shift": shift}},
        "7":  {"class_type": "CLIPLoader",
               "inputs": {"clip_name": CLIP_NAME, "type": "wan", "device": "default"}},
        "8":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["7", 0]}},
        "9":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": NEGATIVE_PROMPT, "clip": ["7", 0]}},
        "10": {"class_type": "VAELoader",
               "inputs": {"vae_name": VAE_NAME}},
        "11": {"class_type": "LoadImage",
               "inputs": {"image": image_name}},
        "12": {"class_type": "WanImageToVideo",
               "inputs": {
                   "width": SVD_W, "height": SVD_H, "length": frames,
                   "batch_size": 1,
                   "positive": ["8", 0], "negative": ["9", 0],
                   "vae": ["10", 0], "start_image": ["11", 0],
               }},
        "13": {"class_type": "KSamplerAdvanced",
               "inputs": {
                   "add_noise": "enable", "noise_seed": seed,
                   "steps": steps, "cfg": 1.0,
                   "sampler_name": "euler", "scheduler": "simple",
                   "start_at_step": 0, "end_at_step": half,
                   "return_with_leftover_noise": "enable",
                   "model": ["5", 0],
                   "positive": ["12", 0], "negative": ["12", 1],
                   "latent_image": ["12", 2],
               }},
        "14": {"class_type": "KSamplerAdvanced",
               "inputs": {
                   "add_noise": "disable", "noise_seed": seed,
                   "steps": steps, "cfg": 1.0,
                   "sampler_name": "euler", "scheduler": "simple",
                   "start_at_step": half, "end_at_step": 10000,
                   "return_with_leftover_noise": "disable",
                   "model": ["6", 0],
                   "positive": ["12", 0], "negative": ["12", 1],
                   "latent_image": ["13", 0],
               }},
        "15": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["10", 0]}},
        "16": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "rufus_wan", "images": ["15", 0]}},
    }


def _submit_verbose(graph: dict, client_id: str) -> str | None:
    """comfy_client._submit, but on a validation reject also print ComfyUI's
    error body — this graph is blind-wired, so the exact node/param complaint
    in the run log is how it gets corrected."""
    try:
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": client_id},
                          timeout=30)
        if r.status_code != 200:
            print(f"[wan] graph rejected (HTTP {r.status_code}): {r.text[:400]}")
            return None
        return r.json().get("prompt_id")
    except Exception as e:
        print(f"[wan] submit failed: {e}")
        return None


def animate_image(png_path: Path, out_path: Path,
                  duration: float = 8.0, idx: int = 0,
                  prompt: str = "") -> bool:
    """FLUX still → Wan 2.2 motion clip at out_path. False on ANY failure —
    the caller walks down the engine chain (SVD, then Ken Burns), so a clip
    is never lost to this engine. No face-skip here: Wan's temporal
    consistency handles faces (that heuristic was an SVD-weakness patch)."""
    try:
        frames  = int(os.environ.get("RUFUS_WAN_FRAMES", "81"))
        steps   = int(os.environ.get("RUFUS_WAN_STEPS", "4"))
        shift   = float(os.environ.get("RUFUS_WAN_SHIFT", "5.0"))
        timeout = float(os.environ.get("RUFUS_WAN_TIMEOUT", "900"))
        seed    = random.randint(1, 2**31 - 1)

        with tempfile.TemporaryDirectory(prefix="rufus_wan_") as td:
            tmp = Path(td)
            init_png = tmp / f"init_{uuid.uuid4().hex[:8]}.png"
            if not _prep_init_image(png_path, init_png):
                return False
            image_name = _upload_image(init_png)
            if not image_name:
                return False

            graph = _build_wan_graph(image_name, _motion_prompt(prompt),
                                     seed, frames, steps, shift)
            pid = _submit_verbose(graph, uuid.uuid4().hex)
            if not pid:
                return False

            t0 = time.time()
            frame_bytes = _await_frames(pid, timeout=timeout)
            if len(frame_bytes) < 2:
                return False
            for j, fb in enumerate(frame_bytes):
                (tmp / f"frame_{j:04d}.png").write_bytes(fb)

            print(f"[wan] {len(frame_bytes)} frames in {time.time() - t0:.0f}s "
                  f"(steps={steps}, {SVD_W}x{SVD_H})")
            return _assemble(tmp, WAN_FPS, out_path, duration)
    except Exception as e:
        print(f"[wan] animate failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not src or not src.exists():
        print("usage: python wan_client.py <still.png> [out.mp4] [motion prompt]")
        sys.exit(1)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".wan.mp4")
    ok, why = ready()
    print(f"ready: {ok} ({why})")
    if ok and animate_image(src, out, prompt=" ".join(sys.argv[3:])):
        print(f"OK → {out}")
