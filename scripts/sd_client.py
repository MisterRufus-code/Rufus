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
import subprocess
import time
from pathlib import Path

import requests

# ── Constants ────────────────────────────────────────────────────────────────

IMG_W = 576    # safe for GTX 1060 6GB
IMG_H = 1024   # 9:16 portrait ratio
OUT_W = 1080   # final Shorts width
OUT_H = 1920   # final Shorts height
FPS   = 30

TIMEOUT_GEN = 180   # A1111 generation can be slow on 6GB
TIMEOUT_UPX = 120   # R-ESRGAN upscale timeout

QUALITY_SUFFIX = (
    "photorealistic, cinematic, professional photography, sharp focus, "
    "dramatic lighting, high detail, 8k uhd, DSLR, film grain"
)
NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, sketch, painting, watermark, text, "
    "logo, blurry, low quality, deformed, bad anatomy, jpeg artifacts, "
    "signature, frame, border, split image, collage"
)


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

def _query_to_prompt(query: str) -> str:
    return f"({query}:1.3), {QUALITY_SUFFIX}"


def _generate_image(prompt: str) -> bytes | None:
    """Call A1111 txt2img at 576×1024. Returns raw PNG bytes or None."""
    import os
    steps = int(os.environ.get("SD_STEPS", "20"))
    payload = {
        "prompt":          prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width":           IMG_W,
        "height":          IMG_H,
        "steps":           steps,
        "cfg_scale":       7.0,
        "sampler_name":    "DPM++ 2M Karras",
        "restore_faces":   False,
        "enable_hr":       False,   # upscale separately — safer on 6GB
        "batch_size":      1,
        "n_iter":          1,
        "seed":            -1,
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

    Even clips zoom in (1.0 → 1.15), odd clips zoom out (1.15 → 1.0).
    Matches the direction alternation in audio_gen.py's Ken Burns filter.
    """
    total_frames = int(duration * FPS)
    step = round(0.15 / total_frames, 7)   # zoom delta per frame for full-range motion

    if idx % 2 == 0:
        zoom_expr = f"min(zoom+{step},1.15)"
        x_expr    = "iw/2-(iw/zoom/2)"
        y_expr    = "ih/2-(ih/zoom/2)"
    else:
        # Start at 1.15 on frame 1, then decrement
        zoom_expr = f"if(eq(on,1),1.15,max(zoom-{step},1.0))"
        x_expr    = "iw/2-(iw/zoom/2)"
        y_expr    = "ih/2-(ih/zoom/2)"

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


# ── Public API ────────────────────────────────────────────────────────────────

def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0) -> list[Path]:
    """
    Generate n animated clips from scene queries via local Stable Diffusion.

    Pipeline per clip:
      query → SD 576×1024 → R-ESRGAN 2× → 1152×2048 → crop 1080×1920 → Ken Burns mp4

    Returns list of .mp4 Paths, or [] if SD not running.
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

    for i, query in enumerate(prompts):
        print(f"[sd] {i+1}/{len(prompts)}: {query[:70]}")
        prompt = _query_to_prompt(query)

        # 1. Generate at 576×1024 (~3-5s on GTX 1060)
        img_bytes = _generate_image(prompt)
        if not img_bytes:
            continue

        # 2. Upscale 2× → 1152×2048 (R-ESRGAN tile-based, ~5s)
        img_bytes = _upscale(img_bytes)

        # 3. Crop to 1080×1920
        png_path = tmp_dir / f"{stamp}_{i}.png"
        if not _crop_to_portrait(img_bytes, png_path):
            print(f"[sd] crop failed for clip {i+1}")
            continue

        # 4. Ken Burns → mp4
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
