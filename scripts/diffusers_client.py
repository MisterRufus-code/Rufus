#!/usr/bin/env python3
"""
diffusers_client.py — HuggingFace Diffusers in-process image generator for Rufus.

Generates images directly in Python — no A1111 server, no external API.
Runs on CPU or GPU. First run downloads the model (~2-6GB, cached to
~/.cache/huggingface/).

Activate with: RUFUS_VIDEO_SOURCE=diffusers

Model options (set RUFUS_DIFFUSERS_MODEL):
  sdxl-turbo   (default) — 4-step, fast, 1080×1920, no negative prompt
  sdxl         — 25-step, full quality, negative prompt supported
  flux-schnell — FLUX.1-schnell, best quality, needs 12GB+ VRAM

Install: pip install diffusers transformers accelerate

Output: same mp4 clips as sd_client.py (Ken Burns animation over each image).
"""

import os
import random
import subprocess
import tempfile
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
MEDIA_DIR  = Path(__file__).parent.parent / "media_library"
CACHE_DIR  = MEDIA_DIR / "diffusers_cache"

W, H          = 1080, 1920
KEN_BURNS_DUR = 8.0
FPS           = 30

# Model IDs for each variant
_MODELS = {
    "sdxl-turbo":   "stabilityai/sdxl-turbo",
    "sdxl":         "stabilityai/stable-diffusion-xl-base-1.0",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}
DEFAULT_MODEL_KEY = os.environ.get("RUFUS_DIFFUSERS_MODEL", "sdxl-turbo").lower()

NEGATIVE_PROMPT = (
    "(worst quality:2), (low quality:2), lowres, jpeg artifacts, "
    "(blurry:1.3), flat lighting, (bad anatomy:1.5), (bad hands:1.6), "
    "(bad face:1.4), (distorted face:1.4), plastic skin, "
    "(stock photo:1.4), posed, smiling at camera, cartoon, anime, "
    "watermark, text overlay, duplicate"
)

_pipe = None   # lazy singleton — loaded once per process


def _load_pipe():
    """Load the diffusion pipeline once. GPU if RUFUS_GPU=1, else CPU."""
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import AutoPipelineForText2Image, FluxPipeline

    model_key = DEFAULT_MODEL_KEY
    model_id  = _MODELS.get(model_key, _MODELS["sdxl-turbo"])
    gpu = os.environ.get("RUFUS_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    gpu = gpu and torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    dtype  = torch.float16 if gpu else torch.float32

    print(f"[diffusers] loading {model_key} ({model_id}) on {device.upper()}…")
    print("[diffusers] first run downloads model — subsequent runs use cache")

    if model_key == "flux-schnell":
        _pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        _pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)

    if gpu:
        _pipe = _pipe.to(device)
        # Memory optimizations for ≤8GB VRAM (e.g. GTX 1060 6GB)
        if hasattr(_pipe, "enable_attention_slicing"):
            _pipe.enable_attention_slicing(1)
        if hasattr(_pipe, "enable_vae_slicing"):
            _pipe.enable_vae_slicing()
        if hasattr(_pipe, "enable_vae_tiling"):
            _pipe.enable_vae_tiling()
    elif hasattr(_pipe, "enable_sequential_cpu_offload"):
        _pipe.enable_sequential_cpu_offload()

    print(f"[diffusers] {model_key} ready")
    return _pipe


def _generate_image(prompt: str, seed: int = -1) -> Path | None:
    """Generate one image. Returns temp PNG path or None."""
    try:
        import torch
        pipe = _load_pipe()

        gen_kwargs: dict = {"prompt": prompt, "width": W, "height": H}
        model_key = DEFAULT_MODEL_KEY

        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator().manual_seed(seed)

        if model_key == "sdxl-turbo":
            gen_kwargs.update({"num_inference_steps": 4, "guidance_scale": 0.0})
        elif model_key == "sdxl":
            gen_kwargs.update({
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
            })
        elif model_key == "flux-schnell":
            gen_kwargs.update({"num_inference_steps": 4, "guidance_scale": 0.0})

        result = pipe(**gen_kwargs)
        image  = result.images[0]

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image.save(tmp.name)
        return Path(tmp.name)

    except Exception as e:
        print(f"[diffusers] generation failed: {e}")
        return None


def _ken_burns(img_path: Path, out_path: Path, duration: float = KEN_BURNS_DUR) -> bool:
    """Apply a random Ken Burns effect to a still image → mp4 clip."""
    frames = int(duration * FPS)
    patterns = [
        f"zoompan=z='min(zoom+0.0010,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}",
        f"zoompan=z='if(lte(zoom,1.0),1.5,max(1.0,zoom-0.0010))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}",
        f"zoompan=z='min(zoom+0.0008,1.4)':x='if(gte(zoom,1.4),iw/2-(iw/zoom/2),x+1)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}",
        f"zoompan=z='min(zoom+0.0008,1.4)':x='ih/2-(iw/zoom/2)':y='if(gte(zoom,1.4),ih/2-(ih/zoom/2),y+1)':d={frames}:s={W}x{H}",
    ]
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(img_path),
         "-vf", random.choice(patterns),
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-r", str(FPS), "-pix_fmt", "yuv420p",
         str(out_path)],
        capture_output=True, timeout=90,
    )
    return r.returncode == 0 and out_path.exists()


def generate_clips(prompts: list[str], master_seed: int = 42) -> list[Path]:
    """Generate one Ken Burns clip per prompt using diffusers in-process.

    Returns as many clips as successfully generated (partial success OK).
    Returns [] only if diffusers is not installed or all images fail.

    Args:
        prompts:     List of SD prompts (one per video beat).
        master_seed: Base seed for reproducibility (each image gets master_seed + i).
    """
    try:
        import diffusers  # noqa: F401
    except ImportError:
        print("[diffusers] not installed — pip install diffusers transformers accelerate")
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for i, prompt in enumerate(prompts):
        img_seed = master_seed + i
        print(f"[diffusers] generating image {i+1}/{len(prompts)} (seed={img_seed}): {prompt[:70]}…")

        img_path = _generate_image(prompt, seed=img_seed)
        if img_path is None:
            print(f"[diffusers] image {i+1} failed — skipping")
            continue

        clip_path = CACHE_DIR / f"diff_{i:02d}.mp4"
        if _ken_burns(img_path, clip_path):
            clips.append(clip_path)
            print(f"[diffusers] clip {i+1}: {clip_path.name}")
        else:
            print(f"[diffusers] Ken Burns failed for clip {i+1}")

        # Clean up temp PNG
        Path(img_path).unlink(missing_ok=True)

    print(f"[diffusers] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    prompts = sys.argv[1:] or [
        "RAW photo, person reviewing bank statements at 2am, dramatic side lighting, "
        "Nikon Z7II 85mm f/1.4, cinematic color grade, photorealistic",
    ]
    result = generate_clips(prompts)
    for p in result:
        print(f"CLIP={p}")
