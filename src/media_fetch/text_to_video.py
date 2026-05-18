"""
Text-to-Video generation for Rufus.

Backends (tried in order):
  1. CogVideoX-2B via diffusers  — works on T4 16GB (Colab Pro)
  2. Wan2.1-1.3B via diffusers   — works on L4 24GB (Colab Pro)
  3. ComfyUI API                 — if ComfyUI server is running
  4. Skip                        — falls back to Pexels footage

Usage:
  from src.media_fetch.text_to_video import generate_clip, is_available
  if is_available():
      path = generate_clip("concrete being poured into foundation", duration=5)
"""

from __future__ import annotations

import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Output directory for generated clips
_OUTPUT_DIR = Path(os.getenv("MEDIA_LIBRARY_PATH", "media_library")) / "ai_generated"

# ComfyUI server URL (if running)
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

_backend: Optional[str] = None  # detected once


def _detect_backend() -> Optional[str]:
    """Detect which text-to-video backend is available."""
    global _backend
    if _backend is not None:
        return _backend

    # Check ComfyUI first (fastest if already running)
    try:
        import httpx
        r = httpx.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        if r.status_code == 200:
            _backend = "comfyui"
            return _backend
    except Exception:
        pass

    # Check GPU VRAM
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 20:
                _backend = "wan2"     # L4 / A100
            elif vram_gb >= 14:
                _backend = "cogvideo" # T4
            else:
                _backend = None       # not enough VRAM
        else:
            _backend = None
    except ImportError:
        _backend = None

    return _backend


def is_available() -> bool:
    """Return True if any text-to-video backend is usable."""
    return _detect_backend() is not None


def _clip_path(prompt: str) -> Path:
    """Deterministic output path for a given prompt."""
    slug = hashlib.md5(prompt.encode()).hexdigest()[:12]
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR / f"t2v_{slug}.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# Backend: CogVideoX-2B (T4 — 16GB)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_cogvideo(prompt: str, duration: int, output: Path) -> Path:
    console.print(f"[cyan]T2V CogVideoX-2B[/cyan] generating: {prompt[:60]}...")
    from diffusers import CogVideoXPipeline
    import torch

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=30,
        num_frames=duration * 8,  # ~8 fps
        guidance_scale=6.0,
        generator=torch.Generator().manual_seed(42),
    )

    # Export frames to MP4
    _frames_to_mp4(result.frames[0], output, fps=8)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Wan2.1-1.3B (L4 / A100 — 20GB+)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_wan2(prompt: str, duration: int, output: Path) -> Path:
    console.print(f"[cyan]T2V Wan2.1[/cyan] generating: {prompt[:60]}...")
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    import torch

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    pipe.enable_model_cpu_offload()

    result = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
        height=480,
        width=832,
        num_frames=duration * 16,  # 16fps
        guidance_scale=5.0,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(42),
    )

    _frames_to_mp4(result.frames[0], output, fps=16)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Backend: ComfyUI API
# ─────────────────────────────────────────────────────────────────────────────

def _generate_comfyui(prompt: str, duration: int, output: Path) -> Path:
    """Submit a Wan2.1 workflow to ComfyUI and wait for the result."""
    import httpx, json, uuid, time, shutil

    console.print(f"[cyan]T2V ComfyUI[/cyan] submitting: {prompt[:60]}...")

    workflow = {
        "6": {
            "class_type": "WanTextToVideo",
            "inputs": {
                "model": ["4", 0],
                "positive": prompt,
                "negative": "low quality, blurry, distorted",
                "width": 832, "height": 480,
                "num_frames": duration * 16,
                "steps": 30, "cfg": 5.0, "seed": 42,
            }
        },
        "4": {"class_type": "WanModelLoader", "inputs": {"model_name": "Wan2.1-T2V-1.3B"}},
        "9": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["6", 0], "frame_rate": 16,
              "format": "video/h264-mp4", "filename_prefix": "rufus_t2v"}},
    }

    client_id = str(uuid.uuid4())
    r = httpx.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=10)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    # Poll for completion
    for _ in range(120):  # max 10 minutes
        time.sleep(5)
        hist = httpx.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10).json()
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_out in outputs.values():
                for vid in node_out.get("videos", []):
                    url = f"{COMFYUI_URL}/view?filename={vid['filename']}&type={vid.get('type','output')}"
                    data = httpx.get(url, timeout=60).content
                    output.write_bytes(data)
                    return output
    raise TimeoutError("ComfyUI generation timed out after 10 minutes")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frames_to_mp4(frames, output: Path, fps: int = 8) -> None:
    """Convert a list of PIL frames to MP4 via FFmpeg."""
    import subprocess, tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                from PIL import Image as _PIL
                frame = _PIL.fromarray(frame)
            frame.save(f"{tmp}/frame_{i:05d}.png")

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-r", str(fps),
            "-i", f"{tmp}/frame_%05d.png",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(output),
        ], check=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_clip(
    prompt: str,
    duration: int = 5,
    force: bool = False,
) -> Optional[Path]:
    """
    Generate a video clip from a text prompt.

    Args:
        prompt:   Text description of the scene to generate.
        duration: Target clip length in seconds (4-8 recommended).
        force:    Re-generate even if cached clip exists.

    Returns:
        Path to the generated MP4, or None if generation failed.
    """
    backend = _detect_backend()
    if backend is None:
        return None

    output = _clip_path(prompt)
    if output.exists() and not force:
        console.print(f"[dim]T2V cache hit: {output.name}[/dim]")
        return output

    t0 = time.time()
    try:
        if backend == "comfyui":
            _generate_comfyui(prompt, duration, output)
        elif backend == "wan2":
            _generate_wan2(prompt, duration, output)
        elif backend == "cogvideo":
            _generate_cogvideo(prompt, duration, output)

        elapsed = time.time() - t0
        size_mb = output.stat().st_size / 1e6
        console.print(
            f"[green]✓ T2V clip generated[/green] "
            f"({elapsed:.0f}s, {size_mb:.1f}MB) → {output.name}"
        )
        return output

    except Exception as exc:
        console.print(f"[yellow]T2V generation failed ({backend}): {exc}[/yellow]")
        if output.exists():
            output.unlink(missing_ok=True)
        return None


def generate_scene_clips(
    scene_prompts: list[str],
    duration_each: int = 5,
) -> list[Path]:
    """
    Generate one clip per scene prompt. Returns list of successful clip paths.
    Skips scenes where generation fails.
    """
    clips = []
    total = len(scene_prompts)
    for i, prompt in enumerate(scene_prompts, 1):
        console.print(f"[cyan]T2V scene {i}/{total}[/cyan]")
        path = generate_clip(prompt, duration=duration_each)
        if path:
            clips.append(path)
    return clips
