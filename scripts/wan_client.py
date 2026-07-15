#!/usr/bin/env python3
"""
wan_client.py — Wan 2.2 14B image-to-video for Rufus (via ComfyUI).

The successor to svd_client as the PRIMARY motion engine: Wan 2.2 (2025) has
dramatically better temporal consistency than SVD (2023) — structures keep
their shape while moving (verified on the rig: 81 frames, zero warping on the
exact fine-detail content SVD melted), and it takes a TEXT motion prompt per
clip. CORRECTION from an earlier version of this note: faces still degrade
(blur/swim) under Wan motion even though rigid objects don't — a live glitch
report caught this — so the SVD-era face-skip heuristics are REINSTATED here
too (see animate_image), default on, opt-out via RUFUS_WAN_FACE_MOTION=1.

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

QUALITY MODE, verified against an actual API export of the channel owner's
proven-good test run (a clean 81-frame clip, zero warping): that run used the
template's DEFAULT toggle state — "Enable 4steps LoRA?" = false — meaning
real classifier-free guidance (cfg 3.5) with NO lightx2v LoRA, 20 steps split
10/10 between the high/low-noise experts. That combination is what's actually
proven, not a guess. The lightx2v 4-step/cfg-1.0 path this file used to
hardcode was UNVERIFIED against that quality bar.

Per explicit direction ("good quality and not so long"), the default here
keeps the proven quality driver (no LoRA, real cfg) but trims steps 20 -> 12
(RUFUS_WAN_STEPS) — a standard diffusion tradeoff (most models are
well-converged well before 20 steps; the split point scales with it, so the
two experts still split the schedule evenly). Measured on the rig: 20 steps
took ~19 min of sampling per clip; 12 steps should land around ~11-12 min —
roughly 1.5-2h of motion generation for a 9-10 clip video instead of 3h+.
The fast lightx2v path (4 steps, cfg 1.0) is kept as an explicit opt-in via
RUFUS_WAN_LORA=1 for when speed matters more than the verified-quality bar.

Environment:
  RUFUS_WAN          1 (default) — 0 disables Wan (chain continues with SVD)
  RUFUS_WAN_FRAMES   81   (Wan's native 5s at 16fps; ping-pong+loop covers the
                           requested clip duration downstream)
  RUFUS_WAN_STEPS    12   (total steps across both experts, split evenly)
  RUFUS_WAN_CFG      3.5  (real classifier-free guidance — the proven setting;
                           only used when RUFUS_WAN_LORA=0, the default)
  RUFUS_WAN_LORA     0 (default) — 1 switches to the fast lightx2v path
                           (forces steps=4, cfg=1.0 regardless of the two
                           settings above — that's the distilled LoRA's only
                           supported operating point)
  RUFUS_WAN_SHIFT    5.0  (ModelSamplingSD3 shift)
  RUFUS_WAN_TIMEOUT  1800 (seconds to wait for one clip's frames — real CFG
                           at 12+ steps is much slower than the 4-step path)
  RUFUS_WAN_FACE_MOTION  0 (default) — 1 re-allows Wan to animate faces
                           (blurs/glitches under motion per a live report;
                           default skips to the next engine, same detectors
                           as SVD's face-skip)
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
# bucket too), history polling, and the frames→mp4 assembly pipeline. Also the
# face-detection pair — see the RUFUS_WAN_FACE_MOTION note on animate_image.
from svd_client import (_prep_init_image, _await_frames, _assemble, _upload_image,
                        SVD_W, SVD_H, _prompt_likely_shows_a_face, _image_shows_a_face)

WAN_FPS = 16   # Wan 2.2's native frame rate (81 frames = ~5s)

HIGH_MODEL = os.environ.get("WAN_HIGH_MODEL", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
LOW_MODEL  = os.environ.get("WAN_LOW_MODEL",  "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
HIGH_LORA  = os.environ.get("WAN_HIGH_LORA",  "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")
LOW_LORA   = os.environ.get("WAN_LOW_LORA",   "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")
CLIP_NAME  = os.environ.get("WAN_CLIP_NAME",  "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
VAE_NAME   = os.environ.get("WAN_VAE_NAME",   "wan_2.1_vae.safetensors")

# The model's own tuned negative prompt (from the ComfyUI Wan 2.2 template's
# default, in the API export) — this is what the checkpoint was validated
# against, not a generic English guess. Wan's text encoder (umt5_xxl) is
# multilingual; using the reference-language prompt is the safer bet for
# matching known-good behavior.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
    directed, gentle documentary movement.

    CRITICAL constraint, found from a real glitch report: every clip is
    assembled forward-then-REVERSED for a seamless loop (see svd_client.
    _assemble) — great for ambient drift (a reversed gentle sway still looks
    natural), but a one-way, completing action (a page turning, a hand
    finishing a gesture) played backward looks exactly like rewinding a
    tape. The motion prompt must steer Wan toward camera/ambient motion only,
    never a subject action with a clear start-and-finish, or the ping-pong
    loop itself will visibly repeat/undo that action every cycle."""
    subject = " ".join((beat_prompt or "").split())[:220]
    return (f"{subject}. The camera moves slowly and subtly — a gentle push-in "
            f"or drift, or ambient environmental motion (drifting dust, light "
            f"flicker, fabric sway). Natural, restrained motion true to the "
            f"scene; stable composition; cinematic documentary feel. NEVER "
            f"animate a one-way completing action — no page turning, no object "
            f"handling, no hand gesture that starts and finishes, no walking "
            f"steps — this clip loops by playing forward then reversed, and any "
            f"one-way action will visibly undo itself every cycle. No sudden "
            f"movements.")


def _build_wan_graph(image_name: str, prompt: str, seed: int, frames: int,
                     steps: int, cfg: float, shift: float,
                     use_lora: bool) -> dict:
    """Native-node Wan 2.2 i2v graph. Two experts split one denoise schedule:
    high-noise model takes the first half of the steps (with leftover noise),
    low-noise model finishes.

    use_lora=False (default): the model output feeds ModelSamplingSD3
    DIRECTLY — matches the proven-good export's default toggle state exactly
    (real cfg, no LoRA in the loop at all, not just strength=0).
    use_lora=True: lightx2v LoRAs applied, forced steps=4/cfg=1.0 (their only
    supported operating point) regardless of the steps/cfg passed in."""
    if use_lora:
        steps, cfg = 4, 1.0
    half = max(1, steps // 2)

    g = {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": HIGH_MODEL, "weight_dtype": "default"}},
        "2":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": LOW_MODEL, "weight_dtype": "default"}},
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
        "15": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["10", 0]}},
        "16": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "rufus_wan", "images": ["15", 0]}},
    }

    if use_lora:
        g["3"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": HIGH_LORA, "strength_model": 1.0}}
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["2", 0], "lora_name": LOW_LORA, "strength_model": 1.0}}
        high_model_src, low_model_src = ["3", 0], ["4", 0]
    else:
        high_model_src, low_model_src = ["1", 0], ["2", 0]

    g["5"] = {"class_type": "ModelSamplingSD3",
              "inputs": {"model": high_model_src, "shift": shift}}
    g["6"] = {"class_type": "ModelSamplingSD3",
              "inputs": {"model": low_model_src, "shift": shift}}
    g["13"] = {"class_type": "KSamplerAdvanced",
               "inputs": {
                   "add_noise": "enable", "noise_seed": seed,
                   "steps": steps, "cfg": cfg,
                   "sampler_name": "euler", "scheduler": "simple",
                   "start_at_step": 0, "end_at_step": half,
                   "return_with_leftover_noise": "enable",
                   "model": ["5", 0],
                   "positive": ["12", 0], "negative": ["12", 1],
                   "latent_image": ["12", 2],
               }}
    g["14"] = {"class_type": "KSamplerAdvanced",
               "inputs": {
                   "add_noise": "disable", "noise_seed": seed,
                   "steps": steps, "cfg": cfg,
                   "sampler_name": "euler", "scheduler": "simple",
                   "start_at_step": half, "end_at_step": 10000,
                   "return_with_leftover_noise": "disable",
                   "model": ["6", 0],
                   "positive": ["12", 0], "negative": ["12", 1],
                   "latent_image": ["13", 0],
               }}
    return g


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


def _skip_face_motion() -> bool:
    """RUFUS_WAN_FACE_MOTION=1 opts back INTO animating faces with Wan. Default
    is skip (False) — see animate_image's docstring for why this changed from
    the original "Wan handles faces fine" assumption."""
    return os.environ.get("RUFUS_WAN_FACE_MOTION", "0").strip().lower() not in ("1", "true", "yes", "on")


def animate_image(png_path: Path, out_path: Path,
                  duration: float = 8.0, idx: int = 0,
                  prompt: str = "") -> bool:
    """FLUX still → Wan 2.2 motion clip at out_path. False on ANY failure —
    the caller walks down the engine chain (SVD, then Ken Burns), so a clip
    is never lost to this engine.

    Face-skip REINSTATED here after a real glitch report: faces that looked
    clean in the FLUX still came out blurred/swimming once animated — Wan's
    stronger temporal consistency on rigid objects doesn't fully extend to
    fine facial detail under motion. Reuses the exact same detectors built
    for SVD (prompt-text "portrait"/"face", plus a pixel-level Haar-cascade
    check on the actual image) rather than duplicating that logic. Opt back
    in with RUFUS_WAN_FACE_MOTION=1 if you want to re-test this once Wan
    settings/versions change."""
    if _skip_face_motion():
        if _prompt_likely_shows_a_face(prompt):
            print(f"[wan] clip {idx+1}: prompt shows a face — skipping Wan motion "
                  f"(known to blur/glitch faces under motion), trying next engine")
            return False
        if _image_shows_a_face(png_path):
            print(f"[wan] clip {idx+1}: image contains a detected face — "
                  f"skipping Wan motion, trying next engine")
            return False
    try:
        frames   = int(os.environ.get("RUFUS_WAN_FRAMES", "81"))
        steps    = int(os.environ.get("RUFUS_WAN_STEPS", "12"))
        cfg      = float(os.environ.get("RUFUS_WAN_CFG", "3.5"))
        shift    = float(os.environ.get("RUFUS_WAN_SHIFT", "5.0"))
        use_lora = os.environ.get("RUFUS_WAN_LORA", "0").strip().lower() in ("1", "true", "yes", "on")
        timeout  = float(os.environ.get("RUFUS_WAN_TIMEOUT", "1800"))
        seed     = random.randint(1, 2**31 - 1)

        with tempfile.TemporaryDirectory(prefix="rufus_wan_") as td:
            tmp = Path(td)
            init_png = tmp / f"init_{uuid.uuid4().hex[:8]}.png"
            if not _prep_init_image(png_path, init_png):
                return False
            image_name = _upload_image(init_png)
            if not image_name:
                return False

            graph = _build_wan_graph(image_name, _motion_prompt(prompt),
                                     seed, frames, steps, cfg, shift, use_lora)
            pid = _submit_verbose(graph, uuid.uuid4().hex)
            if not pid:
                return False

            t0 = time.time()
            frame_bytes = _await_frames(pid, timeout=timeout)
            if len(frame_bytes) < 2:
                return False
            for j, fb in enumerate(frame_bytes):
                (tmp / f"frame_{j:04d}.png").write_bytes(fb)

            eff_steps, eff_cfg = (4, 1.0) if use_lora else (steps, cfg)
            print(f"[wan] {len(frame_bytes)} frames in {time.time() - t0:.0f}s "
                  f"(steps={eff_steps}, cfg={eff_cfg}, lora={use_lora}, {SVD_W}x{SVD_H})")
            # ping_pong=False: the motion prompt now asks for camera/ambient
            # motion only (see _motion_prompt), but a reversed loop still
            # visibly "undoes" anything with directionality — one-way playback
            # + freeze-extend has no loop point to go wrong at all.
            return _assemble(tmp, WAN_FPS, out_path, duration, ping_pong=False)
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
