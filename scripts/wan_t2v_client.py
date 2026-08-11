#!/usr/bin/env python3
"""
wan_t2v_client.py — Wan 2.2 TEXT-to-video for Rufus (via ComfyUI).

Every other motion engine here takes a still and animates it
(`animate_image`). This one takes only words. That is a different contract and
a different set of problems, and the problems are the reason most of this file
is about consistency rather than about generation.

WHY CONSISTENCY IS HARD HERE, STATED PLAINLY. An image-to-video engine is
handed the world in its init frame: the coin, the cloak, the palette, the flat
2D style all arrive as pixels and the model only has to move them. A
text-to-video model builds every beat from noise having seen nothing. Ten beats
are ten unrelated worlds unless something forces them together, and no amount
of writing "the same coin" fixes that — "the same coin" is not a description of
a coin.

So this file gives three mechanisms, strongest last:

  1. WORLD LOCK (always on). One canonical block — style, palette, character,
     light — appended verbatim, byte for byte, to every beat. Identical text
     produces related images far more reliably than paraphrased text does,
     which is why this is built once per run and reused rather than rewritten
     per beat. Holds style and mood. Does NOT hold object identity.

  2. SEED LINEAGE (always on). Every beat's seed is derived deterministically
     from one run seed instead of being random per clip. Two effects: the run
     is reproducible from its base seed, and neighbouring beats share noise
     structure, which visibly steadies framing and palette. Cheap, and free.

  3. FRAME CHAINING (RUFUS_T2V_CHAIN=1, opt-in). Beat 1 is generated from
     text; its last frame is extracted and beats 2..N are generated from THAT
     by the image-to-video engine. This is the only mechanism here that
     carries actual objects forward, because it is the only one that hands the
     next beat a picture. It is also, honestly, no longer pure text-to-video —
     it is text-to-video seeded, then chained. If real continuity matters more
     than purity, this is the setting that provides it.

Mechanisms 1 and 2 are style consistency. Mechanism 3 is world consistency.
They are not substitutes for each other and this file does not pretend
otherwise.

A NOTE ON STYLE. This channel's prompts end in "Flat 2D vector illustration
style, not a photograph". Video models are trained overwhelmingly on real
footage, so text-to-video has to generate that style AND the motion from
noise, fighting its training the whole way, where image-to-video receives the
style as pixels and only has to move it. Expect drift toward photoreal, and
check the first render for it before committing a run.

TEMPLATE-DRIVEN, like every other engine here. Never hand-wired:

  1. In ComfyUI: Workflow -> Browse Templates -> "Wan 2.2 14B Text to Video".
     Run it once and verify a clean clip comes out. Note this needs the T2V
     model files, which are DIFFERENT files from the I2V ones — installing one
     does not give you the other.
  2. Set the positive prompt text to exactly:  RUFUS_PROMPT
  3. Workflow -> Export (API) -> save as:  <Rufus>/config/wan_t2v_api.json

CONTRACT: fail-open and inert until that file exists. No template, a rejected
graph, a timeout — every one returns False and the caller falls back to the
image path it uses today. Text-to-video is a different way to get a clip, never
a prerequisite for getting one.

Environment:
  RUFUS_T2V            0 (default) — 1 enables this engine
  RUFUS_T2V_CHAIN      0 (default) — 1 chains beats 2..N off beat 1's last frame
  RUFUS_T2V_W          832
  RUFUS_T2V_H          1472
  RUFUS_T2V_FRAMES     81
  RUFUS_T2V_TIMEOUT    1800
  RUFUS_T2V_SEED       base seed for the run (default: random, then reused)
  RUFUS_T2V_TEMPLATE   path override for the API-export JSON
"""

import hashlib
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
from svd_client import OUT_W, OUT_H, _stills_only

ROOT = Path(__file__).parent.parent

OUT_W_DEFAULT, OUT_H_DEFAULT = 832, 1472
FRAMES_DEFAULT = 81

LAST_CALL: dict = {}

# Set once per process so every beat in one video shares a lineage even though
# each is a separate call. Re-reading a random seed per beat would throw away
# mechanism 2 entirely.
_run_seed: int | None = None


def _template_path() -> Path:
    return Path(os.environ.get("RUFUS_T2V_TEMPLATE",
                               str(ROOT / "config" / "wan_t2v_api.json")))


def enabled() -> bool:
    """Off by default. Text-to-video is a deliberate choice with real
    trade-offs (see the module docstring), not something a run should fall
    into because a file happened to exist."""
    if _stills_only():
        return False
    return os.environ.get("RUFUS_T2V", "0").strip().lower() in ("1", "true", "yes", "on")


def chaining() -> bool:
    """RUFUS_T2V_CHAIN=1 — beats 2..N continue beat 1's last frame."""
    return os.environ.get("RUFUS_T2V_CHAIN", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def run_seed() -> int:
    """The run's base seed. Stable for the process, so a whole video is
    reproducible from one number and neighbouring beats share noise
    structure."""
    global _run_seed
    if _run_seed is None:
        explicit = os.environ.get("RUFUS_T2V_SEED", "").strip()
        try:
            _run_seed = int(explicit) if explicit else random.randint(1, 2**31 - 1)
        except ValueError:
            _run_seed = random.randint(1, 2**31 - 1)
    return _run_seed


def beat_seed(idx: int) -> int:
    """This beat's seed, derived from the run seed.

    Derived rather than random so the run is reproducible, and derived rather
    than identical because a truly identical seed across differing prompts
    produces near-duplicate compositions — steadiness, not repetition, is the
    goal.
    """
    digest = hashlib.sha256(f"{run_seed()}:{idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) or 1


def settings() -> dict:
    """Resolved settings, including what the template actually names — the run
    report should show the real model, not a documented guess."""
    out = {
        "width":    int(os.environ.get("RUFUS_T2V_W", str(OUT_W_DEFAULT))),
        "height":   int(os.environ.get("RUFUS_T2V_H", str(OUT_H_DEFAULT))),
        "frames":   int(os.environ.get("RUFUS_T2V_FRAMES", str(FRAMES_DEFAULT))),
        "timeout":  float(os.environ.get("RUFUS_T2V_TIMEOUT", "1800")),
        "chaining": chaining(),
        "run_seed": run_seed(),
        "template": str(_template_path()),
    }
    tpl = comfy_template.load_template(_template_path()) or {}
    for node in tpl.values():
        ins = node.get("inputs") or {}
        for key in ("unet_name", "ckpt_name", "clip_name", "vae_name"):
            if key in ins and key not in out:
                out[key] = ins[key]
    return out


def ready() -> tuple[bool, str]:
    """Fail-closed preflight, same contract as every other engine."""
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False, ("no API export at config/wan_t2v_api.json — run the "
                       "ComfyUI 'Wan 2.2 14B Text to Video' template once, set "
                       "the prompt to RUFUS_PROMPT, then Export (API) — see "
                       "wan_t2v_client.py header. Note T2V uses DIFFERENT model "
                       "files from I2V")
    if not comfy_template.has_placeholder(tpl):
        return False, ("export found but no RUFUS_PROMPT placeholder — set the "
                       "prompt text to RUFUS_PROMPT and re-export")
    missing = comfy_template.missing_nodes(tpl, _host())
    if missing:
        return False, (f"ComfyUI is missing node(s): {', '.join(missing[:4])} "
                       f"(server down, or the Wan nodes aren't installed)")
    missing_files = comfy_template.missing_models(tpl, _host())
    if missing_files:
        return False, (f"ComfyUI can't load model file(s): "
                       f"{'; '.join(missing_files[:3])}")
    return True, "Wan 2.2 T2V template loaded (text-to-video)"


# ------------------------------------------------------------- the world lock

def build_world(shots: list[str], character: str = "", style: str = "") -> str:
    """The canonical block appended to every beat of one video.

    Assembled once from what the run already knows — the niche's style suffix
    and the recurring character's description — and then reused byte for byte.
    Rewriting or paraphrasing it per beat is what breaks it: the model keys off
    the exact token sequence, so two beats that describe the same world in
    different words are two different worlds to it.

    `shots` is accepted so a caller can pass the storyboard; it is used only to
    keep the signature honest about what a richer world lock could draw on, and
    ignored today rather than guessed at.
    """
    parts = []
    if character:
        parts.append(" ".join(character.split()))
    if style:
        parts.append(" ".join(style.split()))
    parts.append(
        "The same world throughout: identical palette, identical light "
        "direction, identical rendering style in every shot.")
    return " ".join(p.rstrip(".") + "." for p in parts if p)


def _motion_prompt(beat_prompt: str, world: str = "") -> str:
    """Beat text + sustained-motion direction + the world lock, in that order.

    Motion direction carries the same one-way-action constraint as every other
    engine here: a clip is freeze-extended to fill its slot, so a completed
    gesture visibly stalls while sustained ambient motion reads as alive the
    whole way through.
    """
    subject = " ".join((beat_prompt or "").split())[:280]
    core = (f"{subject}. The camera moves continuously and smoothly throughout "
            f"— a slow push-in or gentle drift, never static. Motion is "
            f"sustained from first frame to last and never completes or stops; "
            f"no cuts, no scene change, no sudden jumps.")
    return f"{core} {world}".strip() if world else core


# ------------------------------------------------------------------ chaining

def last_frame(mp4_path: Path, png_path: Path) -> bool:
    """Extract the final frame of a clip, for chaining the next beat off it.

    `-sseof -0.1` seeks relative to the end, which is the only reliable way to
    land on the last frame without decoding the whole file or knowing its exact
    duration.
    """
    cmd = ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(mp4_path),
           "-update", "1", "-q:v", "2", "-frames:v", "1", str(png_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("[t2v] last-frame extraction timed out")
        return False
    if r.returncode != 0:
        print(f"[t2v] last-frame extraction failed: {r.stderr[-200:]}")
        return False
    return png_path.exists() and png_path.stat().st_size > 1_000


# ------------------------------------------------------------------ generation

def _await_video(prompt_id: str, timeout: float) -> bytes | None:
    """Poll history for the finished container and download it."""
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
                    print("[t2v] ComfyUI reported a generation error")
                    return None
        except Exception:
            pass
        time.sleep(2.0)
    print(f"[t2v] timed out after {timeout:.0f}s waiting for the clip")
    return None


def _finish(src_mp4: Path, out_path: Path, duration: float) -> bool:
    """Scale/crop to the pipeline's exact 1080x1920 and freeze-extend to fill
    the beat's slot — the same clip shape every other engine assembles to."""
    vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={OUT_W}:{OUT_H},tpad=stop_mode=clone:stop_duration=30,"
          f"trim=duration={duration:.3f},setpts=PTS-STARTPTS")
    cmd = ["ffmpeg", "-y", "-i", str(src_mp4), "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", str(out_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("[t2v] ffmpeg post-process timed out")
        return False
    if r.returncode != 0:
        print(f"[t2v] ffmpeg failed: {r.stderr[-300:]}")
        return False
    return out_path.exists() and out_path.stat().st_size > 30_000


def generate_clip(prompt: str, out_path: Path, duration: float = 5.0,
                  idx: int = 0, world: str = "") -> bool:
    """Text in -> 1080x1920 mp4 at out_path. False means the caller falls back.

    Deliberately NOT named animate_image: it takes no image, and giving it that
    name would let it be dropped into the motion chain, where every other entry
    is contracted to receive a still.
    """
    tpl = comfy_template.load_template(_template_path())
    if tpl is None:
        return False
    try:
        cfg = settings()
        w, h = int(cfg["width"]), int(cfg["height"])
        frames, timeout = int(cfg["frames"]), float(cfg["timeout"])

        text = _motion_prompt(prompt, world)
        seed = beat_seed(idx)
        LAST_CALL.clear()
        LAST_CALL.update(engine="wan_t2v", motion_prompt=text, seed=seed, **cfg)

        graph = comfy_template.prepare(tpl, prompt=text, seed=seed,
                                       dims=(w, h, frames),
                                       keep_video_writers=True)
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": uuid.uuid4().hex},
                          timeout=30)
        if r.status_code != 200:
            print(f"[t2v] graph rejected (HTTP {r.status_code}): {r.text[:300]}")
            return False
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            print("[t2v] ComfyUI returned no prompt_id")
            return False

        blob = _await_video(prompt_id, timeout)
        if not blob:
            return False

        with tempfile.TemporaryDirectory(prefix="rufus_t2v_") as td:
            raw = Path(td) / "raw.mp4"
            raw.write_bytes(blob)
            return _finish(raw, Path(out_path), duration)
    except Exception as e:
        print(f"[t2v] clip {idx + 1} failed ({e}) — falling back")
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Wan 2.2 text-to-video smoke test")
    p.add_argument("prompt")
    p.add_argument("--out", default="t2v_test.mp4")
    p.add_argument("--duration", type=float, default=5.0)
    args = p.parse_args()

    ok, why = ready()
    print(f"[t2v] ready: {ok} — {why}")
    if ok:
        print(f"[t2v] wrote {args.out}"
              if generate_clip(args.prompt, Path(args.out), args.duration)
              else "[t2v] generation failed")
