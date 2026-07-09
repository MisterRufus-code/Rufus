#!/usr/bin/env python3
"""
svd_client.py — Stable Video Diffusion image-to-video for Rufus (via ComfyUI).

Animates each FLUX still into REAL motion (drifting smoke, flickering light,
shifting fabric) instead of the Ken Burns zoom, using the same ComfyUI server
that generated the image. Needs one extra checkpoint (one-time ~9GB download):

    svd_xt.safetensors  →  ComfyUI/models/checkpoints/
    https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt

Contract: animate_image(png_path, out_path, duration, idx) -> bool — the same
deliverable as sd_client._animate_to_clip, so comfy_client tries SVD first and
falls back to Ken Burns per image on ANY failure.

TWO ENGINES, resolved once per run by resolve_engine():
  comfy      — built-in ComfyUI nodes (ImageOnlyCheckpointLoader /
               SVD_img2vid_Conditioning / VideoLinearCFGGuidance), no custom
               node installs. Preferred: the server is already up for FLUX.
  diffusers  — in-process StableVideoDiffusionPipeline (HuggingFace). No
               ComfyUI checkpoint placement needed — downloads to the HF cache
               on first use. Needs CUDA (SVD on CPU takes hours) and
               `pip install diffusers transformers accelerate`.
               NOTE: SVD is image-only conditioning — it takes NO text prompt
               (model-card boilerplate showing a prompt is misleading; the
               correct class is StableVideoDiffusionPipeline).

Pipeline per clip:
  1080×1920 FLUX still → 576×1024 init image (SVD's portrait size) → 25 frames
  → minterpolate to 30fps → ping-pong (forward+reverse, seamless loop) →
  Lanczos upscale 1080×1920 → looped to the requested clip duration.

Environment:
  RUFUS_IMG2VID            1 (default) — set 0 to skip SVD and keep Ken Burns stills
  RUFUS_IMG2VID_ENGINE     auto (default) / comfy / diffusers
  COMFY_SVD_MODEL          svd_xt.safetensors           (comfy engine)
  RUFUS_SVD_DIFFUSERS_MODEL stabilityai/stable-video-diffusion-img2vid-xt
  RUFUS_SVD_FRAMES         25   (SVD-XT's native frame count)
  RUFUS_SVD_FPS            8    (conditioning fps; assembled output is 30fps)
  RUFUS_SVD_MOTION         70   (motion_bucket_id 1-255: higher = more motion
                                 but also more warping/deformation — keep low)
  RUFUS_SVD_STEPS          30
"""

import os
import random
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests

from comfy_client import _host, _submit, POLL_INTERVAL
from sd_client import FPS, OUT_W, OUT_H

# SVD is trained at 1024×576; the swapped portrait orientation is the standard
# vertical-video recipe. Frames are Lanczos-upscaled to 1080×1920 on assembly.
SVD_W, SVD_H = 576, 1024

SVD_TIMEOUT = 420   # 25 frames on a 3090 ≈ 60-120s; generous headroom

# SVD is trained on general scene/object motion, NOT faces — it has no concept
# of facial structure, so animating a shot with a visible face frequently
# warps/melts features (eyes drifting, mouth morphing) across the 25 frames.
# Ken Burns (plain camera pan/zoom on the static image) can't distort anything,
# so beats that likely show a face skip SVD entirely rather than risk it.
# Heuristic, not a vision check: the FLUX/SD prompt-writer's own framing
# vocabulary uses "portrait" for person-focused beats; "face" is the other
# direct tell. False positives just mean a real object shot loses motion for
# no reason (safe); false negatives mean an occasional face still gets
# animated (same as today) — biased toward the safe side either way.
_FACE_HINT_RE = re.compile(r"\bportraits?\b|\bfaces?\b", re.IGNORECASE)


def _prompt_likely_shows_a_face(prompt: str) -> bool:
    return bool(_FACE_HINT_RE.search(prompt or ""))


_face_cascade = None   # lazy singleton — loaded once per process


def _load_face_cascade():
    """Load OpenCV's bundled frontal-face Haar cascade once. No model download —
    the XML ships inside the opencv-python-headless package itself."""
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade


def _image_shows_a_face(png_path: Path) -> bool:
    """Pixel-level backstop for _prompt_likely_shows_a_face: catches a face the
    PROMPT TEXT didn't mention by name — e.g. GPT wrote "a banknote depicting
    Mao Zedong" (no "portrait"/"face" keyword) but the rendered image is a
    close-up of an engraved portrait, which Haar cascades still fire on.

    Only counts a face above a minimum size (minSize relative to image
    dimensions) — an enormous crowd shot with dozens of tiny, distant faces
    can still get real SVD motion; anything large enough to actually read as
    a face on a phone screen does not. Tuned toward "skip motion when in
    doubt" per the priority set by the channel owner (perfect over fast) —
    lower minSize catches more borderline/smaller faces at the cost of
    losing SVD motion on a few more shots. Fails open (False) on any error —
    this is a bonus safety net on top of the prompt check, never a hard
    requirement, and OpenCV may not be installed yet on every machine."""
    try:
        import cv2
        img = cv2.imread(str(png_path))
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        faces = _load_face_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(1, w // 10), max(1, h // 10)))
        return len(faces) > 0
    except Exception:
        return False


def img2vid_enabled() -> bool:
    return os.environ.get("RUFUS_IMG2VID", "1").strip().lower() not in ("0", "false", "no", "off")


def _svd_model() -> str:
    return os.environ.get("COMFY_SVD_MODEL", "svd_xt.safetensors")


def _parse_image_checkpoint_list(obj_info: dict) -> list[str]:
    """Names loadable by ImageOnlyCheckpointLoader (the SVD loader node).
    Same response shape as CheckpointLoaderSimple; [] on any mismatch."""
    try:
        names = obj_info["ImageOnlyCheckpointLoader"]["input"]["required"]["ckpt_name"][0]
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
    except (KeyError, IndexError, TypeError):
        return []


def list_svd_checkpoints() -> list[str]:
    try:
        r = requests.get(f"{_host()}/object_info/ImageOnlyCheckpointLoader", timeout=10)
        r.raise_for_status()
        return _parse_image_checkpoint_list(r.json())
    except Exception:
        return []


def svd_ready() -> tuple[bool, str]:
    """One preflight per run. Unlike the FLUX preflight this fails CLOSED on an
    unreadable list: SVD is an optional enhancement with a guaranteed fallback
    (Ken Burns), so 'can't verify' should never spend doomed submits."""
    model = _svd_model()
    names = list_svd_checkpoints()
    if not names:
        return False, "couldn't read ComfyUI's image-checkpoint list"
    if model not in names:
        return False, (f"'{model}' not in ComfyUI's checkpoints — download "
                       f"svd_xt.safetensors into ComfyUI\\models\\checkpoints\\ "
                       f"or set COMFY_SVD_MODEL")
    return True, f"{model} loadable"


def diffusers_ready() -> tuple[bool, str]:
    """Is the in-process Diffusers SVD engine usable? Requires the packages and
    a CUDA GPU — SVD-XT on CPU takes hours per clip, which would hang a run."""
    try:
        import diffusers  # noqa: F401
        import torch
    except ImportError:
        return False, ("diffusers not installed — "
                       "pip install diffusers transformers accelerate")
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA GPU — SVD via diffusers is CPU-impractical"
    except Exception as e:
        return False, f"torch CUDA probe failed: {e}"
    return True, "diffusers SVD in-process (first run downloads ~9GB to the HF cache)"


def resolve_engine() -> tuple[str | None, str]:
    """Pick the img2vid engine once per run: ComfyUI if its SVD checkpoint is
    loadable (the server is already up for FLUX — fastest path), else
    in-process Diffusers. RUFUS_IMG2VID_ENGINE=comfy|diffusers forces one.
    Returns (engine_name | None, human-readable reason)."""
    if not img2vid_enabled():
        return None, "disabled (RUFUS_IMG2VID=0)"
    forced = os.environ.get("RUFUS_IMG2VID_ENGINE", "auto").strip().lower()
    if forced not in ("auto", "comfy", "diffusers"):
        return None, f"unknown RUFUS_IMG2VID_ENGINE '{forced}' (use auto/comfy/diffusers)"

    comfy_why = ""
    if forced in ("auto", "comfy"):
        ok, comfy_why = svd_ready()
        if ok:
            return "comfy", comfy_why
        if forced == "comfy":
            return None, comfy_why

    ok, diff_why = diffusers_ready()
    if ok:
        return "diffusers", diff_why
    if forced == "diffusers":
        return None, diff_why
    return None, f"comfy: {comfy_why}; diffusers: {diff_why}"


def _upload_image(png_path: Path) -> str | None:
    """POST the init image to ComfyUI's input folder; returns its server-side
    name for LoadImage, or None on failure."""
    try:
        with open(png_path, "rb") as f:
            r = requests.post(
                f"{_host()}/upload/image",
                files={"image": (png_path.name, f, "image/png")},
                data={"overwrite": "true"},
                timeout=60,
            )
        r.raise_for_status()
        return r.json().get("name") or png_path.name
    except Exception as e:
        print(f"[svd] init-image upload failed: {e}")
        return None


def _build_svd_graph(image_name: str, seed: int, model: str,
                     frames: int, fps: int, motion: int, steps: int) -> dict:
    """Minimal SVD img2vid graph, built-in nodes only:
    ImageOnlyCheckpointLoader → LoadImage → SVD_img2vid_Conditioning
    → VideoLinearCFGGuidance → KSampler → VAEDecode → SaveImage (all frames)."""
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader",
              "inputs": {"ckpt_name": model}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": image_name}},
        "3": {"class_type": "SVD_img2vid_Conditioning",
              "inputs": {
                  "width": SVD_W, "height": SVD_H,
                  "video_frames": frames, "motion_bucket_id": motion,
                  "fps": fps, "augmentation_level": 0.0,
                  "clip_vision": ["1", 1], "init_image": ["2", 0], "vae": ["1", 2],
              }},
        "4": {"class_type": "VideoLinearCFGGuidance",
              "inputs": {"min_cfg": 1.0, "model": ["1", 0]}},
        "5": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": 2.5,
                  "sampler_name": "euler", "scheduler": "karras", "denoise": 1.0,
                  "model": ["4", 0], "positive": ["3", 0],
                  "negative": ["3", 1], "latent_image": ["3", 2],
              }},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "rufus_svd", "images": ["6", 0]}},
    }


def _await_frames(prompt_id: str) -> list[bytes]:
    """Poll /history until the SaveImage node lists the frame batch, then fetch
    every frame in order. [] on timeout or any fetch failure."""
    deadline = time.time() + SVD_TIMEOUT
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
            for node in entry.get("outputs", {}).values():
                images = node.get("images") or []
                if images:
                    frames = []
                    for img in images:
                        try:
                            vr = requests.get(
                                f"{_host()}/view",
                                params={"filename": img.get("filename", ""),
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")},
                                timeout=30,
                            )
                            vr.raise_for_status()
                            frames.append(vr.content)
                        except Exception as e:
                            print(f"[svd] frame fetch failed: {e}")
                            return []
                    return frames
            if "status" in entry and entry["status"].get("status_str") == "error":
                print("[svd] ComfyUI reported a generation error")
                return []
        time.sleep(POLL_INTERVAL)

    print(f"[svd] timed out after {SVD_TIMEOUT}s waiting for frames")
    return []


_diffusers_pipe = None   # lazy singleton — ~9GB of weights, load once per process


def _load_diffusers_pipe():
    global _diffusers_pipe
    if _diffusers_pipe is not None:
        return _diffusers_pipe
    import torch
    from diffusers import StableVideoDiffusionPipeline

    model_id = os.environ.get("RUFUS_SVD_DIFFUSERS_MODEL",
                              "stabilityai/stable-video-diffusion-img2vid-xt")
    print(f"[svd] loading {model_id} via diffusers (first run downloads ~9GB)…")
    _diffusers_pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16")
    # Keeps peak VRAM well under the 3090's 24GB even with FLUX weights cached;
    # decode_chunk_size at call time bounds the VAE-decode spike.
    _diffusers_pipe.enable_model_cpu_offload()
    return _diffusers_pipe


def _run_diffusers(init_png: Path, frames_n: int, fps: int,
                   motion: int, steps: int, seed: int) -> list:
    """SVD via diffusers: 576×1024 init PNG → list of PIL frames. [] on failure.
    No text prompt — SVD conditions on the image alone."""
    try:
        import torch
        from PIL import Image

        pipe  = _load_diffusers_pipe()
        image = Image.open(init_png).convert("RGB")
        out   = pipe(
            image,
            num_frames=frames_n,
            fps=fps,
            motion_bucket_id=motion,
            noise_aug_strength=0.0,       # match the ComfyUI graph: max fidelity to the still
            num_inference_steps=steps,
            decode_chunk_size=8,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        return list(out.frames[0])
    except Exception as e:
        print(f"[svd] diffusers generation failed: {e}")
        return []


def _prep_init_image(src_png: Path, dst_png: Path) -> bool:
    """Cover-resize the 1080×1920 still to SVD's 576×1024 portrait size."""
    try:
        from PIL import Image
        img  = Image.open(src_png).convert("RGB")
        w, h = img.size
        scale = max(SVD_W / w, SVD_H / h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - SVD_W) // 2
        top  = (new_h - SVD_H) // 2
        img.crop((left, top, left + SVD_W, top + SVD_H)).save(str(dst_png), format="PNG")
        return dst_png.exists() and dst_png.stat().st_size > 5_000
    except Exception as e:
        print(f"[svd] init-image prep failed: {e}")
        return False


def _assemble(frames_dir: Path, fps: int, out_path: Path, duration: float) -> bool:
    """frames → 30fps ping-pong loop → 1080×1920 → looped to `duration`.

    Pass 1 interpolates SVD's low native fps up to FPS, appends the reversed
    stream (so the loop point is seamless by construction), and upscales.
    Pass 2 stream-loops that segment out to the requested clip duration."""
    inter = frames_dir / "pingpong.mp4"
    # mi_mode=blend, NOT mci: mci estimates per-pixel motion vectors and warps
    # along them — on fine, high-contrast detail (printed text, documents,
    # ornate patterns) that estimation frequently miscalculates and produces
    # a melting/warping artifact (confirmed live: a newspaper clipping frame
    # came out visibly warped, no face anywhere in that shot — this isn't a
    # face-specific bug, mci can warp ANY fine detail it mis-tracks). blend
    # cross-fades between frames instead of estimating motion, so it cannot
    # produce that class of artifact — worst case is mild ghosting on fast
    # motion, which SVD's own subtle drift rarely produces anyway. Costs a
    # touch of interpolation smoothness for a real reliability gain, matching
    # the channel owner's explicit preference for a slower/safer render.
    # unsharp after the Lanczos upscale: SVD generates at 576×1024 and gets
    # blown up 1.9× to 1080×1920, which softens everything — a mild luma-only
    # sharpen (0.4; halo-safe range) recovers perceived detail the upscale
    # smeared. Chroma untouched (last three args zeroed).
    fc = (
        f"[0:v]minterpolate=fps={FPS}:mi_mode=blend[m];"
        f"[m]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[pp];"
        f"[pp]scale={OUT_W}:{OUT_H}:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,"
        f"setsar=1,format=yuv420p[out]"
    )
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
         "-filter_complex", fc, "-map", "[out]",
         # crf 14, not 20: this is a TEMP intermediate that gets re-encoded
         # by the loop pass below and AGAIN by the final render — three
         # lossy generations compounding was a real source of "the still is
         # sharp but the video is mushy". Intermediates must be near-lossless
         # (their file size is irrelevant, they're deleted after the render);
         # only the final delivery encode should spend bits like a delivery.
         "-c:v", "libx264", "-preset", "fast", "-crf", "14",
         str(inter)],
        # mi_mode=mci (motion-compensated interpolation) is CPU-bound and the
        # heaviest step here — on a live run it timed out at 300s while the
        # GPU was concurrently busy with the next clip's FLUX/SVD generation.
        # Generous headroom so a slow-but-working pass doesn't get killed and
        # fall back to Ken Burns unnecessarily.
        capture_output=True, text=True, timeout=480,
    )
    if r.returncode != 0 or not inter.exists():
        print(f"[svd] ffmpeg assemble failed: {r.stderr[-300:]}")
        return False

    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-stream_loop", "-1", "-i", str(inter),
         "-t", f"{duration:.2f}",
         # Same rationale as pass 1: still an intermediate (the final render
         # re-encodes it once more) — keep it near-lossless.
         "-c:v", "libx264", "-preset", "fast", "-crf", "14",
         "-pix_fmt", "yuv420p", str(out_path)],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        print(f"[svd] ffmpeg loop failed: {r.stderr[-300:]}")
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50_000


def animate_image(png_path: Path, out_path: Path,
                  duration: float = 8.0, idx: int = 0,
                  engine: str = "comfy", prompt: str = "") -> bool:
    """FLUX still → SVD motion clip at out_path. False on ANY failure — the
    caller falls back to Ken Burns for this image, so a clip is never lost.
    `engine` comes from resolve_engine(): "comfy" or "diffusers". `prompt` is
    the image's own generation prompt — pass it so beats that likely show a
    face skip SVD (see _prompt_likely_shows_a_face) instead of risking the
    warped-face artifact Ken Burns can never produce."""
    if _prompt_likely_shows_a_face(prompt):
        print(f"[svd] clip {idx+1}: prompt shows a face — skipping SVD motion "
              f"(not face-aware, would warp features), Ken Burns instead")
        return False
    if _image_shows_a_face(png_path):
        print(f"[svd] clip {idx+1}: image contains a detected face (prompt text "
              f"didn't mention it, e.g. an engraved portrait) — skipping SVD "
              f"motion, Ken Burns instead")
        return False
    try:
        frames_n = int(os.environ.get("RUFUS_SVD_FRAMES", "25"))
        fps      = int(os.environ.get("RUFUS_SVD_FPS", "8"))
        # steps 30 (was 20): more denoising passes = cleaner frames; worth the
        # extra GPU seconds per the channel owner's explicit quality-over-speed
        # preference. motion 70 (was 100): motion_bucket_id is the single
        # biggest warping/deformation lever in SVD — high motion makes the
        # model invent larger pixel displacements, which is where structures
        # smear and melt. Subtle drift reads as "alive" without the damage.
        steps    = int(os.environ.get("RUFUS_SVD_STEPS", "30"))
        motion   = int(os.environ.get("RUFUS_SVD_MOTION", "70"))
        # Vary motion per clip so consecutive beats don't share one identical
        # drift energy — same idea as the Ken Burns 4-pattern rotation.
        motion   = max(1, min(255, motion + (idx % 3) * 15))
        seed     = random.randint(1, 2**31 - 1)

        with tempfile.TemporaryDirectory(prefix="rufus_svd_") as td:
            tmp = Path(td)
            init_png = tmp / f"init_{uuid.uuid4().hex[:8]}.png"
            if not _prep_init_image(png_path, init_png):
                return False

            t0 = time.time()
            if engine == "diffusers":
                pil_frames = _run_diffusers(init_png, frames_n, fps, motion, steps, seed)
                if len(pil_frames) < 2:
                    return False
                for j, im in enumerate(pil_frames):
                    im.save(str(tmp / f"frame_{j:04d}.png"))
                n_frames = len(pil_frames)
            else:
                image_name = _upload_image(init_png)
                if not image_name:
                    return False
                graph = _build_svd_graph(image_name, seed, _svd_model(),
                                         frames_n, fps, motion, steps)
                pid = _submit(graph, uuid.uuid4().hex)
                if not pid:
                    return False
                frames = _await_frames(pid)
                if len(frames) < 2:
                    return False
                for j, fb in enumerate(frames):
                    (tmp / f"frame_{j:04d}.png").write_bytes(fb)
                n_frames = len(frames)

            print(f"[svd] {n_frames} frames in {time.time() - t0:.0f}s "
                  f"(engine={engine}, motion={motion})")
            return _assemble(tmp, fps, out_path, duration)
    except Exception as e:
        print(f"[svd] animate failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not src or not src.exists():
        print("usage: python svd_client.py <still.png> [out.mp4]")
        sys.exit(1)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".svd.mp4")
    engine, why = resolve_engine()
    print(f"engine: {engine} ({why})")
    if engine and animate_image(src, out, engine=engine):
        print(f"CLIP={out}")
