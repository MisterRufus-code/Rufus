#!/usr/bin/env python3
"""
sd_client.py — Stable Diffusion image generator + Ken Burns animator for Rufus.

Generates photorealistic portrait images via Automatic1111 WebUI API, upscales
with Real-ESRGAN, crops to 1080×1920, then animates each still into a Ken Burns
video clip. Produces the same deliverable as Pexels — a list of .mp4 paths —
so the rest of the pipeline is untouched.

Requirements:
  Automatic1111 WebUI running with --api flag:
    ./webui.sh --api --xformers --medvram --listen

  Recommended model: Realistic Vision v5.1 (photorealistic, 4GB VRAM)
  Download to stable-diffusion-webui/models/Stable-diffusion/

Environment:
  SD_HOST   (default: http://localhost:7860)
  SD_CLIPS  (default: 4 clips per run)
  SD_STEPS  (default: 20 — lower = faster, higher = sharper)

Usage:
  RUFUS_VIDEO_SOURCE=sd python scripts/main.py --skip-upload
"""

import base64
import io
import json
import os
import random
import subprocess
import time
from pathlib import Path

import requests

CONFIG_DIR = Path(__file__).parent.parent / "config"

# ── Constants ────────────────────────────────────────────────────────────────

IMG_W = 576    # safe for GTX 1060 6GB
IMG_H = 1024   # 9:16 portrait ratio
OUT_W = 1080   # final Shorts width
OUT_H = 1920   # final Shorts height
FPS   = 30

TIMEOUT_GEN = 180   # A1111 generation can be slow on 6GB
TIMEOUT_UPX = 120   # R-ESRGAN upscale timeout

# Tuned for Realistic Vision v5.1 — surgical quality tokens, no bloat.
# "RAW photo" activator is included in every prompt directly; these tokens
# reinforce sharpness, film realism, and proper optics — not redundant hype.
QUALITY_SUFFIX = (
    "photorealistic, hyperrealistic, "
    "ultra-sharp skin pores, subsurface scattering, micro skin texture, "
    "natural skin imperfections, real hair strands, fabric weave visible, "
    "Fujifilm Portra 400 film emulation, Sigma 50mm f/1.4 Art lens, "
    "shallow depth of field, creamy out-of-focus background bokeh, "
    "natural chromatic falloff, fine film grain overlay, "
    "natural lighting falloff, motivated key light source, "
    "catch light in eyes, real shadow gradients, "
    "tack-sharp focus on subject, crisp foreground detail, "
    "professional editorial photography, award-winning photography"
)
NEGATIVE_PROMPT = (
    # Quality failures
    "(worst quality:2), (low quality:2), (normal quality:2), lowres, "
    "(blurry:1.3), (soft focus:1.3), hazy, out of focus, unfocused, "
    "jpeg artifacts, noise, color banding, "
    "(overexposed:1.2), (underexposed:1.2), blown highlights, crushed blacks, "
    "flat lighting, no shadows, even lighting, shadowless, "
    # Anatomy failures
    "(bad anatomy:1.5), (deformed:1.5), (disfigured:1.4), mutated, "
    "(bad hands:1.6), (bad fingers:1.6), (extra fingers:1.5), missing fingers, "
    "fused fingers, too many fingers, extra limbs, extra arms, extra legs, "
    "mutated hands, long neck, cross-eyed, lazy eye, "
    "(asymmetrical eyes:1.3), floating limbs, disconnected limbs, "
    # Face failures
    "(bad face:1.4), (distorted face:1.4), (ugly face:1.3), uncanny valley, "
    "plastic skin, waxy skin, mannequin face, doll face, "
    "skin spots, acnes, skin blemishes, age spot, "
    "fake smile, forced smile, unnatural expression, vacant stare, "
    "poorly drawn face, misaligned features, (asymmetrical face:1.2), "
    # Stock photo look
    "(stock photo:1.4), posed, smiling at camera, looking at camera, "
    "corporate headshot, generic businessman, glossy advertisement, "
    "clip art, illustration, cartoon, anime, sketch, painting, "
    "3d render, cgi, digital art, concept art, "
    "staged scene, fake background, green screen look, composite, "
    "model pose, catalog pose, perfect teeth, overly polished, "
    "oversaturated, neon colors, Instagram filter, "
    # Technical artifacts
    "watermark, text, signature, logo, border, frame, "
    "duplicate, split image, collage, out of frame, tiled, "
    # Content bans
    "nudity, nsfw"
)

# Per-scene-slot cinematographic specs — each slot has a distinct visual language.
# Anchored to the 4-slot rotation in _build_sd_prompts so every clip in a run
# looks like a different camera angle from the same film.
ANCHOR_PHOTO_SPECS = [
    # Slot 0: EXTREME CLOSE-UP — intimate, textural, faces/hands/objects
    {
        "composition": (
            "extreme close-up macro detail, subject fills entire frame, "
            "razor-thin depth of field, bokeh background, tactile texture visible, "
            "subject isolation"
        ),
        "lighting": (
            "single hard tungsten light source at 45-degree angle, "
            "deep chiaroscuro shadows, specular highlight on subject edge, "
            "no fill light, inky blacks"
        ),
        "camera": (
            "Canon EOS R5, Canon 100mm f/2.8L Macro IS USM, "
            "1/250s, ISO 400, tripod-mounted, tack-sharp focus"
        ),
    },
    # Slot 1: WIDE ESTABLISHING — world-building, scale, environmental
    {
        "composition": (
            "wide establishing shot, subject small in vast environment, "
            "rule of thirds, strong leading lines converging to subject, "
            "deep depth of field, sky fills upper third"
        ),
        "lighting": (
            "natural ambient light, golden hour sun at 15-degree angle, "
            "long warm directional shadows, atmospheric haze in distance, "
            "warm amber highlights, cool blue shadows"
        ),
        "camera": (
            "Sony A7R IV, Sony FE 24mm f/1.4 GM, "
            "f/8, 1/500s, ISO 200, circular polarizer filter"
        ),
    },
    # Slot 2: MEDIUM SHOT — human, emotional, action
    {
        "composition": (
            "medium shot, waist-up, subject slightly off-center left third, "
            "negative space right, environmental context visible behind, "
            "portrait orientation, subject in motion or reaction"
        ),
        "lighting": (
            "3-point lighting: key soft box at 45 degrees camera left, "
            "fill reflector at 2:1 ratio camera right, "
            "warm rim light from behind right creating edge separation"
        ),
        "camera": (
            "Nikon Z7II, NIKKOR Z 85mm f/1.4 S, "
            "f/1.8, 1/200s, ISO 640, selective focus on face"
        ),
    },
    # Slot 3: AERIAL / ABSTRACT — pattern, overview, symbolic
    {
        "composition": (
            "overhead bird's-eye nadir view, symmetrical composition, "
            "abstract geometric pattern, flat lay, "
            "minimalist negative space, graphic and bold"
        ),
        "lighting": (
            "diffused overhead daylight, even studio exposure, "
            "soft shadows revealing texture and depth, no harsh highlights, "
            "clean even illumination showing pattern"
        ),
        "camera": (
            "DJI Mavic 3 Pro, 24mm equivalent, "
            "f/5.6, 1/800s, ISO 200, nadir shot, zero parallax"
        ),
    },
]


# ── Host ─────────────────────────────────────────────────────────────────────

_host_cache: str | None = None

def _host() -> str:
    global _host_cache
    if _host_cache is None:
        import os
        _host_cache = os.environ.get("SD_HOST", "http://localhost:7860").rstrip("/")
    return _host_cache


def is_available() -> bool:
    """Return True if Automatic1111 WebUI is running and responsive."""
    try:
        r = requests.get(f"{_host()}/sdapi/v1/sd-models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Image generation ─────────────────────────────────────────────────────────

def _niche_style_suffix() -> str:
    """Per-niche cinematic style from niches.json — keeps all images on-brand."""
    import os
    try:
        data   = json.loads((CONFIG_DIR / "niches.json").read_text())
        active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
        return data["niches"][active].get("style_suffix", "") or QUALITY_SUFFIX
    except Exception:
        return QUALITY_SUFFIX


def _query_to_prompt(query: str, style: str = "", idx: int = 0) -> str:
    """Build a full SD prompt: subject + per-anchor cinematographic spec + quality.

    idx selects the ANCHOR_PHOTO_SPECS slot (0-3) so each scene in a run uses a
    distinct camera angle, lighting setup, and composition — the four clips read
    as coverage of one film, not four random photos.
    """
    spec = ANCHOR_PHOTO_SPECS[idx % len(ANCHOR_PHOTO_SPECS)]
    parts = [
        f"RAW photo, ({query}:1.35)",
        spec["composition"],
        spec["lighting"],
        spec["camera"],
    ]
    if style and style != QUALITY_SUFFIX:
        parts.append(style)
    parts.append(QUALITY_SUFFIX)
    return ", ".join(parts)


def _generate_image(prompt: str, seed: int = -1) -> bytes | None:
    """Call A1111 txt2img at 576×1024. Returns raw PNG bytes or None.

    A fixed seed across a run keeps lighting/palette consistent between images
    so the four scenes read as one film instead of four random photos.
    """
    import os
    steps = int(os.environ.get("SD_STEPS", "32"))  # 32 = sharper detail vs 28
    payload = {
        "prompt":          prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width":           IMG_W,
        "height":          IMG_H,
        "steps":           steps,
        "cfg_scale":       7.0,              # better prompt adherence for complex subjects
        "sampler_name":    "DPM++ SDE Karras",  # sharper fine detail for portraits
        "restore_faces":   True,             # GFPGAN pass fixes uncanny valley
        "enable_hr":       False,            # upscale separately — safer on 6GB VRAM
        "batch_size":      1,
        "n_iter":          1,
        "seed":            seed,
        # ADetailer: automatically detects and redraws faces/hands at high res.
        # Silently ignored if the A1111 ADetailer extension is not installed.
        "alwayson_scripts": {
            "ADetailer": {
                "args": [{
                    "ad_model":                 "face_yolov8n.pt",
                    "ad_denoising_strength":    0.35,
                    "ad_inpaint_only_masked":   True,
                    "ad_mask_blur":             4,
                    "ad_prompt":                "",
                    "ad_negative_prompt":       "(worst quality), (low quality), cartoon, cgi, blurry",
                }]
            }
        },
    }
    try:
        r = requests.post(f"{_host()}/sdapi/v1/txt2img", json=payload,
                          timeout=TIMEOUT_GEN)
        r.raise_for_status()
        img_b64 = r.json()["images"][0]
        data = base64.b64decode(img_b64)
        if len(data) < 50_000:
            print("[sd] image too small — likely generation failure")
            return None
        return data
    except Exception as e:
        print(f"[sd] generate failed: {e}")
        return None


# ── Upscaling ─────────────────────────────────────────────────────────────────

def _upscale_realesrgan(img_bytes: bytes) -> bytes | None:
    """Upscale 2× via A1111 extras API (tile-based → low VRAM usage)."""
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "resize_mode":      0,     # scale-factor mode
        "upscaling_resize": 2.0,   # 576×1024 → 1152×2048
        "upscaler_1":       "R-ESRGAN 4x+",
        "image":            img_b64,
    }
    try:
        r = requests.post(f"{_host()}/sdapi/v1/extra-single-image",
                          json=payload, timeout=TIMEOUT_UPX)
        r.raise_for_status()
        result_b64 = r.json().get("image", "")
        if not result_b64:
            return None
        data = base64.b64decode(result_b64)
        return data if len(data) > 100_000 else None
    except Exception as e:
        print(f"[sd] R-ESRGAN failed ({e}) — will use Lanczos")
        return None


def _upscale_lanczos(img_bytes: bytes) -> bytes:
    """PIL Lanczos 2× resize — fallback when R-ESRGAN unavailable."""
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes))
    big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return buf.getvalue()


def _upscale(img_bytes: bytes) -> bytes:
    result = _upscale_realesrgan(img_bytes)
    if result:
        return result
    print("[sd] falling back to Lanczos upscale")
    return _upscale_lanczos(img_bytes)


# ── Crop to portrait ──────────────────────────────────────────────────────────

def _crop_to_portrait(img_bytes: bytes, out_path: Path) -> bool:
    """Center-crop upscaled image (1152×2048) to 1080×1920."""
    from PIL import Image
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    # If somehow smaller than target — pad with black
    if w < OUT_W or h < OUT_H:
        bg = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
        bg.paste(img, ((OUT_W - w) // 2, (OUT_H - h) // 2))
        img = bg
    else:
        left = (w - OUT_W) // 2
        top  = (h - OUT_H) // 2
        img  = img.crop((left, top, left + OUT_W, top + OUT_H))

    img.save(str(out_path), format="PNG", optimize=False)
    return out_path.exists() and out_path.stat().st_size > 20_000


# ── Ken Burns animation ───────────────────────────────────────────────────────

def _animate_to_clip(img_path: Path, out_path: Path,
                     duration: float = 8.0, idx: int = 0) -> bool:
    """Animate a 1080×1920 PNG into a Ken Burns mp4 via FFmpeg zoompan.

    4-pattern rotation so consecutive clips always feel different:
      0: zoom in, pan right   (subject entering frame)
      1: zoom out, pan left   (pull-back reveal)
      2: zoom in, pan up      (upward momentum)
      3: zoom out, pan down   (downward weight)
    """
    total_frames = int(duration * FPS)
    step = round(0.20 / total_frames, 7)  # 20% zoom range — more cinematic than 15%

    pattern = idx % 4
    if pattern == 0:   # push-in + right drift (subject entering)
        zoom_expr = f"min(zoom+{step},1.20)"
        x_expr    = f"iw/2-(iw/zoom/2)+({step*total_frames:.4f}*on/{total_frames}*iw/7)"
        y_expr    = "ih/2-(ih/zoom/2)"
    elif pattern == 1: # pull-back + left drift (reveal)
        zoom_expr = f"if(eq(on,1),1.20,max(zoom-{step},1.0))"
        x_expr    = f"iw/2-(iw/zoom/2)-({step*total_frames:.4f}*on/{total_frames}*iw/7)"
        y_expr    = "ih/2-(ih/zoom/2)"
    elif pattern == 2: # push-in + upward drift (momentum)
        zoom_expr = f"min(zoom+{step},1.20)"
        x_expr    = "iw/2-(iw/zoom/2)"
        y_expr    = f"ih/2-(ih/zoom/2)-({step*total_frames:.4f}*on/{total_frames}*ih/10)"
    else:              # pull-back + downward drift (weight)
        zoom_expr = f"if(eq(on,1),1.20,max(zoom-{step},1.0))"
        x_expr    = "iw/2-(iw/zoom/2)"
        y_expr    = f"ih/2-(ih/zoom/2)+({step*total_frames:.4f}*on/{total_frames}*ih/10)"

    vf = (
        f"zoompan=z='{zoom_expr}':d={total_frames}:"
        f"x='{x_expr}':y='{y_expr}':"
        f"s={OUT_W}x{OUT_H}:fps={FPS},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-framerate", str(FPS),
        "-i", str(img_path),
        "-vf", vf,
        "-vframes", str(total_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"[sd] ffmpeg animate failed: {r.stderr[-300:]}")
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50_000


# ── Perceptual de-duplication ───────────────────────────────────────────────────

DUP_THRESHOLD   = 6   # max Hamming distance (of 64 bits) to treat two images as "same"
MAX_DUP_RETRIES = 2   # regenerations allowed before accepting a near-duplicate


def _avg_hash(png_path: Path) -> int | None:
    """64-bit average hash (aHash) of an image — PIL only, no numpy.

    Downscale to 8×8 grayscale, then set each bit by whether its pixel is at or
    above the mean. Visually similar images produce hashes a few bits apart.
    """
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


# ── Public API ────────────────────────────────────────────────────────────────

def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0,
                   prebuilt: bool = False) -> list[Path]:
    """
    Generate one animated clip per query via local Stable Diffusion, in order.

    Pipeline per clip:
      query → SD 576×1024 → R-ESRGAN 2× → 1152×2048 → crop 1080×1920 → Ken Burns mp4

    prebuilt=True: queries are already complete SD token prompts from _build_sd_prompts
    (GPT-written, 60-80 words, contain their own camera/lighting/color specs). Skip the
    _query_to_prompt wrapping — just ensure the RV5.1 "RAW photo" activator and the
    core quality tail are present, then send directly to A1111.

    prebuilt=False (default): queries are short keyword strings; _query_to_prompt builds
    the full SD token prompt with anchor specs + quality suffix.

    Each finished image is perceptual-hashed; if it collides with an already-
    accepted image it is regenerated with a fresh seed (up to MAX_DUP_RETRIES)
    so no image visibly repeats within a video. A render never fails on this —
    after the retries the closest attempt is kept.

    Returns list of .mp4 Paths (one per query), or [] if SD not running.
    """
    if not is_available():
        print(f"[sd] A1111 not running at {_host()} — start with: ./webui.sh --api --medvram --xformers")
        return []

    tmp_dir = Path(__file__).parent.parent / "media_library" / "temp" / "sd"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    clips:  list[Path] = []
    stamp   = int(time.time())
    prompts = list(queries or ["cinematic scene"])
    if len(prompts) < n:
        base = prompts[:]
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + ", different composition, wider shot")
    prompts = prompts[:n]

    style          = _niche_style_suffix()
    master_seed    = random.randint(1, 2_000_000_000)
    accepted_hashes: list[int] = []
    print(f"[sd] base seed {master_seed} — each image offset for variety")

    # RUFUS_DEBUG=1 keeps a copy of every accepted keyframe + its prompt under
    # media_library/debug/<stamp>/ so a run can be inspected/critiqued afterwards
    # (see scripts/inspect_run.py). Off by default — zero overhead on real runs.
    debug_dir = None
    if os.environ.get("RUFUS_DEBUG"):
        debug_dir = Path(__file__).parent.parent / "media_library" / "debug" / str(stamp)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sd] DEBUG on — keeping keyframes in {debug_dir}")

    for i, query in enumerate(prompts):
        print(f"[sd] {i+1}/{len(prompts)}: {query[:70]}")

        if prebuilt:
            # GPT already wrote a complete SD token prompt with specs baked in.
            # Only ensure the RV5.1 activator leads and the quality tail is present.
            prompt = query if query.startswith("RAW photo") else f"RAW photo, {query}"
            if "8k uhd" not in prompt:
                prompt = f"{prompt}, {QUALITY_SUFFIX}"
        else:
            prompt = _query_to_prompt(query, style, idx=i)

        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False

        # Generate → upscale → crop → de-dup check. Regenerate with a new seed
        # if the result is too close to an earlier image; keep the last try if
        # all retries still collide (never block a render).
        for retry in range(MAX_DUP_RETRIES + 1):
            seed = master_seed + i + 1000 * retry
            img_bytes = _generate_image(prompt, seed=seed)
            if not img_bytes:
                continue
            img_bytes = _upscale(img_bytes)
            if not _crop_to_portrait(img_bytes, png_path):
                continue

            h = _avg_hash(png_path)
            is_dup = (
                h is not None and accepted_hashes
                and min(_hamming(h, prev) for prev in accepted_hashes) < DUP_THRESHOLD
            )
            if is_dup and retry < MAX_DUP_RETRIES:
                print(f"[sd] dup detected on clip {i+1} → regen (retry {retry+1})")
                continue
            if is_dup:
                print(f"[sd] clip {i+1} still near-dup after {MAX_DUP_RETRIES} retries — keeping")
            if h is not None:
                accepted_hashes.append(h)
            accepted = True
            break

        if not accepted:
            print(f"[sd] no usable image for clip {i+1} — skipping")
            continue

        # Keep a debug copy of the keyframe + the exact prompt before it's deleted.
        if debug_dir is not None:
            try:
                (debug_dir / f"{i+1:02d}.png").write_bytes(png_path.read_bytes())
                (debug_dir / f"{i+1:02d}.txt").write_text(
                    f"BEAT QUERY:\n{query}\n\nFULL SD PROMPT:\n{prompt}\n", encoding="utf-8")
            except Exception as e:
                print(f"[sd] debug-save failed for clip {i+1}: {e}")

        # Ken Burns → mp4
        clip_path = tmp_dir / f"{stamp}_{i}.mp4"
        if _animate_to_clip(png_path, clip_path, duration=clip_duration, idx=i):
            clips.append(clip_path)
            print(f"[sd] clip {i+1} ready")
        else:
            print(f"[sd] animation failed for clip {i+1}")

        png_path.unlink(missing_ok=True)

    print(f"[sd] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] or ["stock market trading floor", "financial charts red decline"]
    result  = generate_clips(queries, n=len(queries))
    for p in result:
        print(f"CLIP={p}")
