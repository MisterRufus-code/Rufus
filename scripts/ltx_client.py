#!/usr/bin/env python3
"""
ltx_client.py — LTX-2.3 image-to-video for Rufus (via ComfyUI).

NOT CONFIRMED FAST ON THIS HARDWARE. LTX-2.3 is built for speed in general,
but the checkpoint actually installed here is LTXAV (audio+video: node names
LTXAVTEModel_ / LTXAV, ~23.8GB staged — right at the 24GB VRAM ceiling). A
live test clip took 928s (00:15:28), i.e. in the same range as Hunyuan, not
faster. Two live suspects, unconfirmed: (1) audio generation is pure
overhead Rufus never uses (it has its own TTS/music/SFX) — a pure-video LTX
checkpoint, if one exists, would drop that whole branch; (2) at ~23.8GB the
model is close enough to the 24GB card that the same RAM-streaming slowdown
documented for Hunyuan on this 16GB-RAM box may apply here too. Until one of
those is ruled out, treat this as an alternative engine to compare, not a
speed win.

TEMPLATE-DRIVEN, like every other engine here: export your own verified
ComfyUI workflow rather than letting this file guess a node graph. Setup:

  1. Open the LTX-2.3 i2v template in ComfyUI and RUN IT ONCE on a test image.
  2. Set the prompt text to exactly:  RUFUS_PROMPT
  3. Workflow -> Export (API) -> save as:  <Rufus>/config/ltx_i2v_api.json

WHY THIS FILE EXISTS SEPARATELY FROM hunyuan_client
LTX's stock template is shaped differently from the Hunyuan/Wan ones in two
ways that silently break the shared path:

  * It sizes clips in SECONDS (width/height/duration + fps), not a frame
    count (width/height/length). comfy_template.prepare() handles both now,
    but without that the dims substitution no-ops and the export's own
    1280x720 LANDSCAPE survives into a 1080x1920 portrait pipeline.
  * Its all-in-one node emits a VIDEO directly, with no VAEDecode to hang a
    SaveImage off. The shared prepare() deletes video writers and re-wires to
    frames — on this graph that would delete the ONLY output and leave a graph
    that renders nothing. So this client keeps the video writer and downloads
    the finished mp4 instead of polling for frames.

Environment:
  RUFUS_LTX           1 (default) — 0 disables this engine
  RUFUS_LTX_W         832   (portrait width)
  RUFUS_LTX_H         1472  (portrait height)
  RUFUS_LTX_FRAMES    121   (converted to seconds using the template's fps)
  RUFUS_LTX_TIMEOUT   1800  (seconds to wait for one clip — a live LTXAV run
                            on a 16GB-RAM box took 928s; 900s cut it off
                            client-side 28s before ComfyUI actually finished)
  RUFUS_LTX_TEMPLATE  path override for the API-export JSON
"""

import os
import random
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests

import comfy_template
from comfy_client import _host
from svd_client import _upload_image, _stills_only, OUT_W, OUT_H

ROOT = Path(__file__).parent.parent

# Node classes that emit a finished video. Unlike the shared prepare() path we
# KEEP these — they are LTX's only output.
_VIDEO_WRITERS = {"SaveVideo", "SaveWEBM", "VHS_VideoCombine", "CreateVideo"}

LAST_CALL: dict = {}


def _template_path() -> Path:
    return Path(os.environ.get("RUFUS_LTX_TEMPLATE",
                               str(ROOT / "config" / "ltx_i2v_api.json")))


def enabled() -> bool:
    if _stills_only():
        return False
    return os.environ.get("RUFUS_LTX", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def settings() -> dict:
    """Resolved settings — env knobs plus what the template actually names, so
    the run report shows the real model rather than a documented guess."""
    out = {
        "width":   int(os.environ.get("RUFUS_LTX_W", str(OUT_W_DEFAULT))),
        "height":  int(os.environ.get("RUFUS_LTX_H", str(OUT_H_DEFAULT))),
        "frames":  int(os.environ.get("RUFUS_LTX_FRAMES", "121")),
        "timeout": float(os.environ.get("RUFUS_LTX_TIMEOUT", "1800")),
        "template": str(_template_path()),
    }
    tpl = comfy_template.load_template(_template_path()) or {}
    for node in tpl.values():
        ins = node.get("inputs") or {}
        for key in ("ckpt_name", "text_encoder", "fps", "duration"):
            if key in ins and key not in out:
                out[key] = ins[key]
    return out


OUT_W_DEFAULT, OUT_H_DEFAULT = 832, 1472


def ready() -> tuple[bool, str]:
    """Fail-closed preflight, same contract as the other engines: template
    exported, placeholder present, and every node class + model file the
    export names actually loadable in the running ComfyUI."""
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False, ("no API export at config/ltx_i2v_api.json — run the "
                       "ComfyUI LTX-2.3 i2v template once, set the prompt to "
                       "RUFUS_PROMPT, then Export (API) — see ltx_client.py header")
    if not comfy_template.has_placeholder(tpl):
        return False, ("export found but no RUFUS_PROMPT placeholder — set the "
                       "prompt text to RUFUS_PROMPT and re-export")
    missing = comfy_template.missing_nodes(tpl, _host())
    if missing:
        return False, (f"ComfyUI is missing node(s): {', '.join(missing[:4])} "
                       f"(server down, or the LTX nodes aren't installed)")
    missing_files = comfy_template.missing_models(tpl, _host())
    if missing_files:
        return False, (f"ComfyUI can't load model file(s): "
                       f"{'; '.join(missing_files[:3])}")
    return True, "LTX-2.3 template loaded (fast motion engine)"


def _motion_prompt(beat_prompt: str) -> str:
    """LTX responds to explicit camera + subject direction. Same one-way-action
    constraint as the other engines: the clip is freeze-extended to fill its
    slot, so a completed gesture visibly stalls, while sustained ambient motion
    reads as alive the whole way through."""
    subject = " ".join((beat_prompt or "").split())[:260]
    return (f"{subject}. The camera moves continuously and smoothly throughout — "
            f"a slow push-in or gentle drift, never static. Subject and "
            f"environment keep moving: fabric and hair stir, dust and light "
            f"shift, loose elements settle. Motion is sustained from first frame "
            f"to last and never completes or stops; no cuts, no scene change, "
            f"no sudden jumps. Cinematic live-action realism.")


def _prepare(tpl: dict, prompt: str, image_name: str, seed: int,
             dims: tuple[int, int, int]) -> dict:
    """Substitute Rufus' values but KEEP the video writer (see module docstring
    — deleting it on this graph shape removes the only output)."""
    g = comfy_template.prepare(tpl, prompt=prompt, image_name=image_name,
                               seed=seed, dims=dims, keep_video_writers=True)
    return g


def _await_video(prompt_id: str, timeout: float) -> bytes | None:
    """Poll history for the finished mp4 and download it. Returns raw bytes.

    The other engines poll for individual PNG frames; LTX hands back a
    container, so this fetches that instead of reconstructing it frame by
    frame."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{_host()}/history/{prompt_id}", timeout=15)
            if r.status_code == 200 and prompt_id in r.json():
                entry = r.json()[prompt_id]
                for node_out in (entry.get("outputs") or {}).values():
                    for key in ("videos", "gifs", "images"):
                        for item in node_out.get(key, []) or []:
                            fn = item.get("filename", "")
                            if not fn.lower().endswith((".mp4", ".webm")):
                                continue
                            v = requests.get(
                                f"{_host()}/view",
                                params={"filename": fn,
                                        "subfolder": item.get("subfolder", ""),
                                        "type": item.get("type", "output")},
                                timeout=120)
                            if v.status_code == 200 and len(v.content) > 10_000:
                                return v.content
                if entry.get("status", {}).get("status_str") == "error":
                    print("[ltx] ComfyUI reported a generation error")
                    return None
        except Exception:
            pass
        time.sleep(2.0)
    print(f"[ltx] timed out after {timeout:.0f}s waiting for the clip")
    return None


def _finish(src_mp4: Path, out_path: Path, duration: float) -> bool:
    """Scale/crop LTX's output to the pipeline's exact 1080x1920 and hold the
    last frame to fill the beat's slot — the same freeze-extend contract the
    other engines assemble to, so the renderer sees one uniform clip shape."""
    vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={OUT_W}:{OUT_H},tpad=stop_mode=clone:stop_duration=30,"
          f"trim=duration={duration:.3f},setpts=PTS-STARTPTS")
    cmd = ["ffmpeg", "-y", "-i", str(src_mp4), "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", str(out_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("[ltx] ffmpeg post-process timed out")
        return False
    if r.returncode != 0:
        print(f"[ltx] ffmpeg failed: {r.stderr[-300:]}")
        return False
    return out_path.exists() and out_path.stat().st_size > 30_000


def animate_image(png_path: Path, out_path: Path, duration: float = 8.0,
                  idx: int = 0, prompt: str = "") -> bool:
    """Same contract as every other motion engine: PNG in -> 1080x1920 mp4 at
    out_path. False means the caller walks down the chain, so a failure here
    can never cost a clip."""
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False
    try:
        w = int(os.environ.get("RUFUS_LTX_W", str(OUT_W_DEFAULT)))
        h = int(os.environ.get("RUFUS_LTX_H", str(OUT_H_DEFAULT)))
        frames = int(os.environ.get("RUFUS_LTX_FRAMES", "121"))
        timeout = float(os.environ.get("RUFUS_LTX_TIMEOUT", "1800"))

        with tempfile.TemporaryDirectory(prefix="rufus_ltx_") as td:
            tmp = Path(td)
            image_name = _upload_image(png_path)
            if not image_name:
                return False

            motion_prompt = _motion_prompt(prompt)
            LAST_CALL.clear()
            LAST_CALL.update(engine="ltx", motion_prompt=motion_prompt, **settings())

            graph = _prepare(tpl, motion_prompt, image_name,
                             random.randint(1, 2**31 - 1), (w, h, frames))
            try:
                r = requests.post(f"{_host()}/prompt",
                                  json={"prompt": graph, "client_id": uuid.uuid4().hex},
                                  timeout=30)
                if r.status_code != 200:
                    print(f"[ltx] graph rejected (HTTP {r.status_code}): {r.text[:300]}")
                    return False
                pid = r.json().get("prompt_id")
            except Exception as e:
                print(f"[ltx] submit failed: {e}")
                return False
            if not pid:
                return False

            t0 = time.time()
            data = _await_video(pid, timeout)
            if not data:
                return False
            raw = tmp / "ltx.mp4"
            raw.write_bytes(data)
            print(f"[ltx] clip in {time.time() - t0:.0f}s ({w}x{h})")
            return _finish(raw, out_path, duration)
    except Exception as e:
        print(f"[ltx] failed ({e}) — falling through")
        return False
