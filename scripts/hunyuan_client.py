#!/usr/bin/env python3
"""
hunyuan_client.py — HunyuanVideo 1.5 image-to-video for Rufus (via ComfyUI).

THE FACE ENGINE. Wan 2.2 is the primary motion engine but blurs/glitches faces
under motion (live-verified), so face shots currently fall all the way through
to static Ken Burns. HunyuanVideo 1.5 (Tencent, 8.3B, Nov 2025) is reported to
have the best facial detail of any local video model — this client slots it
into the chain right after Wan: Wan skips a face shot → Hunyuan animates it.
Non-face shots still go to Wan first (proven quality). Any Hunyuan failure
falls through to SVD → Ken Burns as before; a clip is never lost.

TEMPLATE-DRIVEN, NOT BLIND-WIRED (the Wan lesson): this client does NOT
reconstruct the graph from documentation. It replays the channel owner's own
proven ComfyUI workflow, captured via Export (API). Setup, once:

  1. Update ComfyUI (HunyuanVideo 1.5 native support needs a Nov 2025+ build).
  2. In ComfyUI: Workflow → Browse Templates → "Hunyuan Video 1.5 720p Image
     to Video". ComfyUI offers to auto-download the models it needs
     (diffusion model, qwen 2.5 VL + byt5 text encoders, VAE, SigCLIP vision
     — ~15GB total; pick fp8/480p variants when offered — the 3090 doesn't
     need fp16 720p, and 480p roughly halves generation time).
  3. Load any test image and RUN IT ONCE — verify a clean video comes out.
  4. Set the POSITIVE prompt text to exactly:  RUFUS_PROMPT
     (Rufus substitutes the real motion prompt per clip at run time.)
  5. Export via Workflow → Export (API)  → save as:
        <Rufus>/config/hunyuan_i2v_api.json
  6. Done — the next run logs "[comfy] motion hunyuan 1.5: ON".

Rufus substitutes only: the RUFUS_PROMPT placeholder, the init image, the
seed, and width/height/length (portrait 480x832 by default). Everything else
(sampler stack, steps, cfg, shift, model files) stays exactly as the verified
export captured it.

Environment:
  RUFUS_HUNYUAN          1 (default) — 0 disables this engine
  RUFUS_HUNYUAN_W        480   (portrait width — 480p bucket, fast)
  RUFUS_HUNYUAN_H        832   (portrait height)
  RUFUS_HUNYUAN_FRAMES   121   (~5s at Hunyuan's native 24fps)
  RUFUS_HUNYUAN_FPS      24    (native frame rate, used for assembly)
  RUFUS_HUNYUAN_TIMEOUT  1800  (seconds to wait for one clip)
  RUFUS_HUNYUAN_TEMPLATE path override for the API-export JSON
"""

import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import requests

import comfy_template
from comfy_client import _host
from svd_client import _await_frames, _assemble, _upload_image, _stills_only

ROOT = Path(__file__).parent.parent

HY_FPS = int(os.environ.get("RUFUS_HUNYUAN_FPS", "24"))


def _template_path() -> Path:
    return Path(os.environ.get("RUFUS_HUNYUAN_TEMPLATE",
                               str(ROOT / "config" / "hunyuan_i2v_api.json")))


def enabled() -> bool:
    if _stills_only():
        return False
    return os.environ.get("RUFUS_HUNYUAN", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def ready() -> tuple[bool, str]:
    """Fail-closed preflight: template exported + placeholder present + every
    node class it uses exists in the running ComfyUI."""
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False, ("no API export at config/hunyuan_i2v_api.json — run the "
                       "ComfyUI 'Hunyuan Video 1.5 i2v' template once, set the "
                       "positive prompt to RUFUS_PROMPT, then Export (API) — "
                       "see hunyuan_client.py header")
    if not comfy_template.has_placeholder(tpl):
        return False, ("export found but no RUFUS_PROMPT placeholder — set the "
                       "positive prompt text to RUFUS_PROMPT and re-export")
    missing = comfy_template.missing_nodes(tpl, _host())
    if missing:
        return False, (f"ComfyUI is missing node(s): {', '.join(missing[:4])} "
                       f"(server down or ComfyUI needs an update)")
    return True, "HunyuanVideo 1.5 template loaded (face-capable motion)"


def _motion_prompt(beat_prompt: str) -> str:
    """Face-safe motion direction. Same one-way-action constraint as Wan (the
    clip is freeze-extended, so subtle sustained life beats a completed
    gesture), plus explicit facial-stability language — the whole point of
    this engine is faces that stay sharp while moving."""
    subject = " ".join((beat_prompt or "").split())[:220]
    return (f"{subject}. Subtle, lifelike motion: the person breathes and "
            f"blinks naturally, slight head or hand movement, hair and fabric "
            f"stir gently; the camera drifts slowly. Facial features stay "
            f"sharp, stable, and consistent — no morphing, no warping. "
            f"Restrained documentary realism; no sudden movements; no "
            f"completed one-way actions like page turns or full gestures.")


def _prep_init(src_png: Path, dst_png: Path, w: int, h: int) -> bool:
    """Cover-resize the 1080×1920 still to the Hunyuan generation bucket."""
    try:
        from PIL import Image
        img = Image.open(src_png).convert("RGB")
        sw, sh = img.size
        scale = max(w / sw, h / sh)
        nw, nh = round(sw * scale), round(sh * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        img.crop((left, top, left + w, top + h)).save(str(dst_png), format="PNG")
        return dst_png.exists() and dst_png.stat().st_size > 5_000
    except Exception as e:
        print(f"[hunyuan] init-image prep failed: {e}")
        return False


def _submit_verbose(graph: dict, client_id: str) -> str | None:
    """Submit with the rejection body printed — if the export drifts out of
    sync with the installed ComfyUI (update, renamed model file), the exact
    complaint lands in the run log."""
    try:
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": client_id},
                          timeout=30)
        if r.status_code != 200:
            print(f"[hunyuan] graph rejected (HTTP {r.status_code}): {r.text[:400]}")
            return None
        return r.json().get("prompt_id")
    except Exception as e:
        print(f"[hunyuan] submit failed: {e}")
        return None


def animate_image(png_path: Path, out_path: Path, duration: float = 8.0,
                  idx: int = 0, prompt: str = "") -> bool:
    """Wan-contract i2v: PNG in → 1080×1920 mp4 at out_path. False = the
    caller falls through to the next engine. Faces are WELCOME here — this
    engine exists to animate the shots Wan skips."""
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False

    try:
        w       = int(os.environ.get("RUFUS_HUNYUAN_W", "480"))
        h       = int(os.environ.get("RUFUS_HUNYUAN_H", "832"))
        frames  = int(os.environ.get("RUFUS_HUNYUAN_FRAMES", "121"))
        # 1800s (30 min), not 1200 — a live 16GB-RAM run showed ComfyUI's dynamic
        # VRAM streaming taking ~21-23 min per 480p/30-step clip (comfy-aimdo
        # async weight offloading trades speed for fitting in limited RAM), which
        # is LONGER than the old 1200s default — every clip was timing out and
        # silently falling through to SVD/Ken Burns, wasting the full 20 minutes
        # for zero motion output on every single beat.
        timeout = float(os.environ.get("RUFUS_HUNYUAN_TIMEOUT", "1800"))
        # Hunyuan's 3D causal VAE requires 4n+1 frames (121 ok, 120 not) —
        # snap a misconfigured count down to the nearest valid one instead of
        # letting the whole generation fail.
        if frames % 4 != 1:
            snapped = max(5, ((frames - 1) // 4) * 4 + 1)
            print(f"[hunyuan] frames={frames} invalid (must be 4n+1) — using {snapped}")
            frames = snapped

        with tempfile.TemporaryDirectory(prefix="rufus_hy_") as td:
            tmp = Path(td)
            init_png = tmp / f"init_{uuid.uuid4().hex[:8]}.png"
            if not _prep_init(png_path, init_png, w, h):
                return False
            image_name = _upload_image(init_png)
            if not image_name:
                return False

            graph = comfy_template.prepare(
                tpl,
                prompt=_motion_prompt(prompt),
                image_name=image_name,
                seed=random.randint(1, 2**31 - 1),
                dims=(w, h, frames),
                save_prefix="rufus_hunyuan",
            )
            pid = _submit_verbose(graph, uuid.uuid4().hex)
            if not pid:
                return False

            t0 = time.time()
            frame_bytes = _await_frames(pid, timeout=timeout)
            if len(frame_bytes) < 2:
                return False
            for j, fb in enumerate(frame_bytes):
                (tmp / f"frame_{j:04d}.png").write_bytes(fb)

            print(f"[hunyuan] {len(frame_bytes)} frames in "
                  f"{time.time() - t0:.0f}s ({w}x{h})")
            return _assemble(tmp, HY_FPS, out_path, duration, ping_pong=False)
    except Exception as e:
        print(f"[hunyuan] failed ({e}) — falling through")
        return False
