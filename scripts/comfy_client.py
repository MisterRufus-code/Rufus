#!/usr/bin/env python3
"""
comfy_client.py — ComfyUI stills generator for Rufus.

Generates photoreal portrait images via a local ComfyUI server, then reuses
sd_client's PIL/FFmpeg helpers to upscale → crop 1080×1920 → Ken Burns mp4.
Produces the same deliverable as sd_client / hyperframes — a list of .mp4
paths — so the rest of the pipeline is untouched.

MODEL: exclusively via config/stills_api.json — a user-exported ComfyUI API
workflow (see comfy_template.py). This is Rufus's ONLY stills engine here;
there is deliberately no built-in fallback model, because the built-in engine
this file used to fall back to (FLUX.1-dev) is non-commercial-licensed and
this pipeline is monetized. The default/documented model is Z-Image-Turbo
(Alibaba Tongyi, Apache 2.0, fully commercial-safe) — see README for the
exact ComfyUI setup. Any other Apache/MIT/commercial-safe image model works
the same way: run its workflow once, set the positive prompt to RUFUS_PROMPT,
Export (API) → config/stills_api.json.

Requirements: ComfyUI running with --listen, and config/stills_api.json in
place (see README's "Swappable stills model" section). With no template
exported, this backend has nothing to render and generate_clips() returns []
so main.py falls through to sd/diffusers/pexels.

Environment:
  COMFY_HOST    (default: http://localhost:8188)
  SD_CLIPS      (clip count — shared with sd_client)
  RUFUS_DEBUG=1 (also print verbose per-clip progress; keyframes + prompts
                are ALWAYS kept under media_library/debug/<stamp>/, on every
                run, regardless of this flag — see _housekeeping's retention)

Usage:
  RUFUS_VIDEO_SOURCE=comfy python scripts/main.py --skip-upload
  python scripts/comfy_client.py "modern luxury kitchen, golden hour, wide angle"
"""

import json
import os
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import paths

import requests

# Reuse the proven, dependency-free (PIL/FFmpeg-only) helpers from sd_client so
# animation and perceptual dedup behave identically across backends. One-way
# import — sd_client never imports comfy_client.
from sd_client import (
    _animate_to_clip,
    _avg_hash,
    _hamming,
    DUP_THRESHOLD,
    MAX_DUP_RETRIES,
    OUT_W,
    OUT_H,
)


def _fit_to_portrait(img_bytes: bytes, out_path: Path) -> bool:
    """Cover-resize a stills-model frame to exactly 1080×1920, preserving composition.

    832×1472 (0.5652) vs 1080×1920 (0.5625) are near-identical aspect ratios, so
    we Lanczos-scale to just cover the target and trim the ~0.5% sliver. This
    keeps ~99% of the frame as composed — unlike sd_client's fixed 2× upscale
    + center-crop, which at this generation size would discard 35% of the image.
    """
    from PIL import Image
    import io as _io

    img  = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    scale = max(OUT_W / w, OUT_H / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - OUT_W) // 2
    top  = (new_h - OUT_H) // 2
    img  = img.crop((left, top, left + OUT_W, top + OUT_H))

    img.save(str(out_path), format="PNG", optimize=False)
    return out_path.exists() and out_path.stat().st_size > 20_000

POLL_INTERVAL = 1.5    # seconds between /history polls
GEN_TIMEOUT   = 300    # max seconds to wait for one image (~20-30s on a 3090)
GEN_ERROR_BACKOFF = 3.0  # pause before resubmitting after a submit/generation failure

# Cross-run visual freshness: perceptual hashes of images accepted in RECENT
# runs are persisted and pre-seeded into the dup check, so a new image that
# merely LOOKS like one from a previous video triggers a regen retry — the
# prompt-level DO-NOT-REPEAT list (main._freshness_block) catches repeated
# ideas, this catches repeated pixels. Disable with RUFUS_FRESH_IMAGES=0.
FRESH_HASH_FILE = Path(__file__).parent.parent / "config" / "recent_image_hashes.json"
FRESH_HASH_CAP  = 120   # ~12 runs of history — enough to stop déjà vu, small file


def _fresh_images_enabled() -> bool:
    return os.environ.get("RUFUS_FRESH_IMAGES", "1").strip().lower() \
        not in ("0", "false", "no", "off")


# ── Frames-per-beat: animate by cutting between stills, not by a motion model ──
# A motion model (Hunyuan/Wan/LTX) costs ~10 min/video on a 3090. Z-Image-Turbo
# renders a still in ~2-4s, so in that same budget you can render 100+ images.
# Generating SEVERAL stills per beat that advance the same action, then hard-
# cutting between them, buys an animated feel for a fraction of the time — the
# limited-animation approach, rather than interpolated motion.
#
# 1 (default) = unchanged single-still-per-beat behaviour. >1 switches the run
# to this mode and BYPASSES the motion engines entirely: they are two different
# answers to the same question, and running both would animate each sub-frame
# separately, which is not what this is for.
def _frames_per_beat() -> int:
    try:
        return max(1, int(os.environ.get("RUFUS_FRAMES_PER_BEAT", "1")))
    except ValueError:
        return 1


# How a beat moves. One selector instead of four interacting flags, because
# these are alternatives, not layers:
#   i2v      motion model per still (Wan/Hunyuan/LTX/SVD). Best-looking real
#            motion; measured 600-1800s PER CLIP on a 3090, i.e. hours a video.
#   i2i      each frame img2img'd from the previous one at low denoise, so the
#            frames genuinely continue each other, then crossfaded. ~1-2s a
#            frame. Needs config/stills_i2i_api.json.
#   cut      several independent stills on one seed, hard cut. No extra setup.
#   kenburns one still, zoom only.
# Unset keeps the historical behaviour exactly: RUFUS_FRAMES_PER_BEAT>1 means
# `cut`, otherwise the motion chain.
BEAT_MOTION_MODES = ("i2v", "i2i", "cut", "kenburns")
I2I_DEFAULT_FRAMES = 5


def _beat_motion() -> str:
    mode = os.environ.get("RUFUS_BEAT_MOTION", "").strip().lower()
    return mode if mode in BEAT_MOTION_MODES else ""


I2I_TEMPLATE = Path(__file__).parent.parent / "config" / "stills_i2i_api.json"


def _i2i_template() -> dict | None:
    """The exported img2img workflow used to chain frames, or None.

    Same proven-template contract as every other engine: the channel owner
    builds LoadImage → VAEEncode → KSampler(denoise ~0.4, 10-12 steps) in
    ComfyUI, verifies it, sets the positive prompt to RUFUS_PROMPT and exports.
    Steps matter here — Z-Image-Turbo's 8-step default leaves only ~3 effective
    steps at denoise 0.4, too few to move the picture."""
    if os.environ.get("RUFUS_I2I_TEMPLATE", "1").strip().lower() in \
            ("0", "false", "no", "off"):
        return None
    import comfy_template
    tpl = comfy_template.load_template(I2I_TEMPLATE)
    if tpl is not None and comfy_template.has_placeholder(tpl):
        return tpl
    return None


def _render_image_i2i(prompt: str, seed: int, client_id: str,
                      init_png: Path) -> bytes | None:
    """One img2img step from `init_png`. Returns raw PNG bytes or None.

    Deliberately takes the PREVIOUS RAW model output rather than the finished
    1080×1920 frame: the pipeline's _fit_to_portrait upscales and crops, and
    feeding that back in would re-resample on every link of the chain, so the
    degradation compounds down the beat."""
    tpl = _i2i_template()
    if tpl is None:
        return None
    from svd_client import _upload_image
    image_name = _upload_image(init_png)
    if not image_name:
        return None
    import comfy_template
    g = comfy_template.prepare(tpl, prompt=prompt, image_name=image_name,
                               seed=seed, save_prefix="rufus_i2i",
                               negative=_stills_negative())
    pid = _submit(g, client_id)
    if not pid:
        return None
    return _await_image(pid)


# Forward nudges for the i2i chain. Unlike the `cut` arc (before/peak/after),
# each step here is relative to the image it starts FROM, so the sequence only
# ever moves forward — which is also what keeps drift bounded per beat.
_I2I_STEPS = [
    "The same scene an instant later: the action has advanced a little.",
    "The same scene a moment later: the action continues.",
    "The same scene slightly later still: the movement carries on.",
    "The same scene moments later: the action is nearly complete.",
    "The same scene just after: the action has finished, everything settling.",
]


def _i2i_step_prompt(base: str, k: int) -> str:
    """Prompt for the k-th chained frame (k>=1)."""
    step = _I2I_STEPS[min(k - 1, len(_I2I_STEPS) - 1)]
    return f"{base.rstrip().rstrip('.')}. {step}"


def _build_i2i_chain(*, base_png: Path, base_raw: bytes, prompt: str, seed: int,
                     client_id: str, n: int, tmp_dir: Path, stamp: str,
                     beat: int) -> list[Path]:
    """Chain `n` frames for one beat, each img2img'd from the previous.

    Returns the finished 1080×1920 frames in play order. Stops early rather
    than failing the beat: a broken link just makes this beat shorter, and the
    frames already produced stay usable.

    Each link runs on a DIFFERENT seed. Reusing one seed across an img2img
    chain pulls every step back toward the same result, which defeats the
    point — the previous image is already supplying the continuity."""
    frames = [base_png]
    raws: list[Path] = []
    prev_raw = tmp_dir / f"{stamp}_{beat}_raw0.png"
    try:
        prev_raw.write_bytes(base_raw)
        raws.append(prev_raw)
    except OSError as e:
        print(f"[comfy] i2i chain could not stage the first frame: {e}")
        return frames

    for k in range(1, n):
        step_seed = (seed + k * 101) % (2**31 - 1)
        nxt = _render_image_i2i(_i2i_step_prompt(prompt, k), step_seed,
                                client_id, prev_raw)
        if not nxt:
            print(f"[comfy] i2i chain stopped at frame {k+1} of clip {beat+1} "
                  f"— beat keeps {len(frames)} frame(s)")
            break
        raw_path = tmp_dir / f"{stamp}_{beat}_raw{k}.png"
        fitted = tmp_dir / f"{stamp}_{beat}_{k}.png"
        try:
            raw_path.write_bytes(nxt)
        except OSError:
            break
        raws.append(raw_path)
        if not _fit_to_portrait(nxt, fitted):
            break
        frames.append(fitted)
        prev_raw = raw_path

    for r in raws:
        r.unlink(missing_ok=True)
    return frames


# A micro-arc within one beat. Index 1 is empty on purpose: that is the prompt
# exactly as written, i.e. the peak moment the prompt-builder actually composed.
# The others nudge the SAME scene a moment before/after, so the seed keeps the
# composition and only the action advances.
_PROGRESSION_STEPS = [
    "Captured a moment earlier: the action is only just beginning.",
    "",
    "Captured a moment later: the action has just completed.",
    "Captured a beat afterwards: the aftermath, everything settling.",
]


def _progression_modifiers(n: int) -> list[str]:
    """`n` prompt modifiers spanning one beat's micro-arc, peak included."""
    if n <= 1:
        return [""]
    return _PROGRESSION_STEPS[:min(n, len(_PROGRESSION_STEPS))]


# Target playback fps for the interpolated i2i beat. Matches sd_client.FPS /
# svd_client's assembly so every clip entering the renderer is 30fps.
SMOOTH_FPS = 30


def _assemble_smooth_beat(frames: list[Path], out_path: Path,
                          duration: float) -> bool:
    """Turn a beat's i2i keyframes into ONE smooth clip.

    Crossfading stills still reads as a slideshow; what actually produces
    motion is interpolating BETWEEN the keyframes. That only works because the
    i2i chain makes consecutive frames genuine continuations of each other —
    motion estimation has something real to track. Run over independent images
    the same filter produces warping mush.

    mi_mode=mci is motion-compensated (true in-between frames). svd_client's
    assembly uses the cheaper mi_mode=blend, which is a cross-dissolve — right
    there, because SVD already outputs 25 real frames and only needs topping
    up to 30fps. Here the source is a handful of keyframes, so blend would put
    the slideshow back. Falls back to blend if mci errors."""
    if not frames:
        return False
    if len(frames) == 1:
        return _animate_to_clip(frames[0], out_path, duration=duration,
                                idx=0, min_bytes=10_000)

    seq_dir = out_path.parent / f"{out_path.stem}_seq"
    try:
        seq_dir.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(frames):
            shutil.copyfile(frame, seq_dir / f"frame_{k:04d}.png")
        # Stage the LAST keyframe twice. Measured: minterpolate emits only
        # (N-2) intervals, not (N-1) — it needs a frame to interpolate TOWARD,
        # so the final input frame is never emitted. Verified at n=5 and n=7:
        # without this the beat came out 3.63s instead of 4.80s and the last
        # keyframe of every beat was invisible. The duplicate is the frame that
        # gets dropped, so all the real keyframes survive.
        shutil.copyfile(frames[-1], seq_dir / f"frame_{len(frames):04d}.png")
    except OSError as e:
        print(f"[comfy] could not stage frames for interpolation: {e}")
        return False

    n_staged = len(frames) + 1
    src_fps = max(0.05, (n_staged - 2) / max(duration, 0.1))

    # Interpolate at HALF linear resolution and upscale after. Measured on a
    # 4.8s beat: 117.9s at full 1080x1920 vs 27.6s here — 4.3x, and the whole
    # point of this mode is that it beats a motion model on time. Flat 2D
    # illustration upscales cleanly (no fine texture to lose) and interpolated
    # in-between frames are approximate regardless.
    try:
        small_w = max(2, int(os.environ.get("RUFUS_SMOOTH_SCALE", OUT_W // 2)))
    except ValueError:
        small_w = OUT_W // 2
    small_h = max(2, round(small_w * OUT_H / OUT_W / 2) * 2)

    for mi_mode, extra in (("mci", ":me_mode=bidir:mc_mode=aobmc"), ("blend", "")):
        vf = (f"scale={small_w}:{small_h}:flags=bilinear,"
              f"minterpolate=fps={SMOOTH_FPS}:mi_mode={mi_mode}{extra},"
              f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,format=yuv420p")
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", f"{src_fps:.4f}",
                 "-i", str(seq_dir / "frame_%04d.png"),
                 "-vf", vf, "-t", f"{duration:.2f}",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "14",
                 "-pix_fmt", "yuv420p", str(out_path)],
                capture_output=True, text=True,
                timeout=max(300, int(duration * 120)))
        except subprocess.SubprocessError as e:
            print(f"[comfy] interpolation ({mi_mode}) failed: {e}")
            continue
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000:
            if mi_mode == "blend":
                print("[comfy] motion interpolation fell back to blend")
            shutil.rmtree(seq_dir, ignore_errors=True)
            return True
        print(f"[comfy] interpolation ({mi_mode}) failed: {r.stderr[-200:]}")

    shutil.rmtree(seq_dir, ignore_errors=True)
    return False


def _concat_clips(parts: list[Path], out_path: Path) -> bool:
    """Join same-codec clips into one via the ffmpeg concat demuxer.

    Stream-copy, so this is near-instant and adds no generation loss: every
    part comes from _animate_to_clip, which emits identical libx264/yuv420p at
    one resolution and fps. as_posix() because the demuxer's list file wants
    forward slashes even on Windows."""
    if not parts:
        return False
    if len(parts) == 1:
        parts[0].replace(out_path)
        return out_path.exists()

    listfile = out_path.with_suffix(".concat.txt")
    try:
        listfile.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(out_path)],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"[comfy] concat failed: {r.stderr[-300:]}")
            return False
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[comfy] concat failed: {e}")
        return False
    finally:
        listfile.unlink(missing_ok=True)
    return out_path.exists() and out_path.stat().st_size > 20_000


def _load_prior_hashes() -> list[int]:
    try:
        data = json.loads(FRESH_HASH_FILE.read_text())
        return [int(h) for h in data.get("hashes", [])][-FRESH_HASH_CAP:]
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def _save_hashes(hashes: list[int]) -> None:
    try:
        FRESH_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        FRESH_HASH_FILE.write_text(
            json.dumps({"hashes": hashes[-FRESH_HASH_CAP:]}))
    except OSError as e:
        print(f"[comfy] couldn't save image-hash history: {e}")


def _host() -> str:
    return os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")


def is_available() -> bool:
    """Return True if a ComfyUI server is up and responsive."""
    try:
        r = requests.get(f"{_host()}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _parse_checkpoint_list(obj_info: dict) -> list[str]:
    """Extract loadable checkpoint names from /object_info/CheckpointLoaderSimple.

    Shape: {"CheckpointLoaderSimple": {"input": {"required":
            {"ckpt_name": [["a.safetensors", ...], ...]}}}}
    Returns [] on any shape mismatch — callers treat that as "can't verify".
    """
    try:
        names = obj_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
    except (KeyError, IndexError, TypeError):
        return []


def list_checkpoints() -> list[str]:
    """Checkpoint names the running ComfyUI can actually load. [] on any failure."""
    try:
        r = requests.get(f"{_host()}/object_info/CheckpointLoaderSimple", timeout=10)
        r.raise_for_status()
        return _parse_checkpoint_list(r.json())
    except Exception:
        return []


# THE STILLS MODEL — swap it without touching code. Same template pattern as
# hunyuan_client: replay a user-exported, verified ComfyUI graph instead of
# blind-wiring each new model's node stack. Drop in ANY commercial-safe image
# model this way — Z-Image-Turbo (recommended, Apache 2.0), Qwen-Image, etc.:
# run its ComfyUI workflow once at portrait ~832×1472, set the positive prompt
# text to RUFUS_PROMPT, Export (API) → config/stills_api.json.
#
# Deliberately NO built-in fallback model here. This used to fall back to a
# hardcoded FLUX.1-dev graph on any failure — removed because FLUX.1-dev is
# non-commercial-licensed and this pipeline is monetized; a "safety net" that
# silently renders a non-commercial model into the final video isn't safe at
# all. If the template is missing or a render fails, generate_clips() returns
# [] and main.py falls through to sd/diffusers/pexels instead. Opt the
# template out with RUFUS_STILLS_TEMPLATE=0.
#   config/stills_api.json  — the current/primary image model (model-agnostic)
#   config/flux2_api.json   — kept as a fallback FILENAME for back-compat
#                             (historical name from FLUX.2 testing; holds
#                             whatever commercial-safe model you export to it)
STILLS_TEMPLATE = Path(__file__).parent.parent / "config" / "stills_api.json"
FLUX2_TEMPLATE  = Path(__file__).parent.parent / "config" / "flux2_api.json"


def _stills_template() -> dict | None:
    """The exported image workflow to use for every still, or None. Honors the
    new RUFUS_STILLS_TEMPLATE kill-switch and the legacy RUFUS_FLUX2 one."""
    off = ("0", "false", "no", "off")
    if os.environ.get("RUFUS_STILLS_TEMPLATE", "1").strip().lower() in off:
        return None
    if os.environ.get("RUFUS_FLUX2", "1").strip().lower() in off:
        return None
    import comfy_template
    for path in (STILLS_TEMPLATE, FLUX2_TEMPLATE):
        tpl = comfy_template.load_template(path)
        if tpl is not None and comfy_template.has_placeholder(tpl):
            return tpl
    return None


# Back-compat alias — older references / tests still call _flux2_template().
_flux2_template = _stills_template


# ── Detail / realism direction appended to every stills prompt ────────────────
# NOT the SD1.5 idiom. sd_client's QUALITY_SUFFIX stacks booru-style tokens
# ("8k, masterpiece, ultra-detailed") because Realistic Vision v5.1 was trained
# on caption text full of exactly those tags, so they land as real signal there.
#
# The stills model here (Z-Image-Turbo by default) encodes prompts with
# Qwen3-4B — a modern LLM text encoder trained on long, descriptive natural-
# language captions. Keyword spam is out-of-distribution for it: "8k,
# masterpiece" contributes little and can crowd out the actual subject tokens.
# What DOES drive fine detail in an LLM-encoder model is concrete physical
# description — real optics, a named light behaviour, and micro-surface facts
# the model can actually render. So this reads like a photographer's note, not
# a tag dump.
#
# Tune or replace wholesale with RUFUS_STILLS_DETAIL; set it empty to disable.
#
# Flat 2D illustration, not photorealism — changed together with
# main.py's _FLUX_INSTRUCTION (its "PHOTOREALISM, NOT ILLUSTRATION" section
# became "FLAT 2D ILLUSTRATION, NOT A PHOTOGRAPH") and money_history's
# style_suffix in config/niches.json, so all three agree instead of fighting
# each other in the same prompt. SCOPE NOTE: this constant is genuinely
# global — every ComfyUI stills render uses it, both the video pipeline
# (comfy_client.generate_clips, currently only reached by the money_history
# niche — it's the only one with video_source=comfy) AND the standalone
# thumbnail tool (image_gen.py, used from the dashboard for any prompt,
# regardless of niche). If a second niche adopts video_source=comfy wanting
# a PHOTOREALISTIC look, this default needs to become niche-aware rather
# than edited in place again — RUFUS_STILLS_DETAIL is the per-run escape
# hatch until then.
DEFAULT_DETAIL_SUFFIX = (
    "Flat 2D vector illustration style, not a photograph: clean confident "
    "outlines of consistent stroke weight, simplified geometric shapes "
    "rendered in flat, unshaded color fills. No gradients, no photographic "
    "lighting, no film grain, no lens blur or depth of field, no skin pores "
    "or fabric-weave texture. Figures and objects are graphic and stylized "
    "rather than anatomically photographic — bold silhouettes, minimal "
    "internal linework, an expressive pose read through shape and posture. "
    "Lighting is rendered as bold graphic contrast — a hard-edged light "
    "shape or color-block shadow — never a soft photographic gradient. "
    "Backgrounds simplify into clean shapes and generous negative space "
    "rather than photographic clutter. Reads like modern explainer-video or "
    "storybook illustration: crisp, deliberate, and stylized, never a "
    "photograph or a photo-real render."
)


# ── Negative conditioning ────────────────────────────────────────────────────
# Suppression belongs HERE, not in the positive prompt. A live money_history
# batch of 40 stills came back with invented lettering on a coin ("national"),
# a newspaper ("NEVLES / NAOTRO"), a ledger, a bank facade and two documents —
# and every one of those prompts already carried main.py's de-text clause.
# That clause is a negation inside the POSITIVE prompt, where CLIP reads its
# tokens (text, numbers, lettering, readable) as things to paint. Garbled
# words are the single most obvious "AI slop" tell in a finished Short, so
# this list leads with them. Overridable per-run; RUFUS_STILLS_NEGATIVE="" or
# "0" turns it off entirely for a template whose own negative is already tuned.
DEFAULT_STILLS_NEGATIVE = (
    "text, letters, words, writing, lettering, typography, caption, subtitle, "
    "watermark, signature, logo, brand name, numbers, digits, gibberish text, "
    "garbled writing, fake language, misspelled words, distorted letterforms, "
    "photorealistic, photograph, 3d render, gradient shading, film grain, "
    "extra fingers, deformed hands, extra limbs, mutated face, blurry, "
    "lowres, jpeg artifacts"
)


def _stills_negative() -> str:
    """Negative conditioning for stills, or "" when disabled."""
    raw = os.environ.get("RUFUS_STILLS_NEGATIVE")
    if raw is None:
        return DEFAULT_STILLS_NEGATIVE
    raw = raw.strip()
    return "" if raw.lower() in ("0", "false", "no", "off") else raw


def _detail_suffix() -> str:
    return os.environ.get("RUFUS_STILLS_DETAIL", DEFAULT_DETAIL_SUFFIX).strip()


def _with_detail(prompt: str) -> str:
    """Append the detail/realism direction, unless it's disabled or the prompt
    already carries its own photographic direction (a niche style_suffix, or a
    hand-written --topic prompt, shouldn't get a second contradictory one)."""
    tail = _detail_suffix()
    if not tail:
        return prompt
    low = prompt.lower()
    if "f/1.4" in low or "depth of field" in low:
        return prompt
    return f"{prompt.rstrip().rstrip('.')}. {tail}"


def _render_image(prompt: str, seed: int, client_id: str,
                  niche: str | None = None) -> bytes | None:
    """Render one still → raw PNG bytes, or None.

    Tries the recurring-character path first when the niche has one
    configured and enabled (see character_engine.py); falls through to the
    plain stills template on any character-path miss (no template exported,
    reference bootstrap failed, render failed) so character mode can never
    turn a working pipeline into a broken one — same fail-open contract as
    every other optional feature in this codebase.

    No built-in fallback MODEL though — see the module docstring for why
    (licensing). Returns None if no plain template is exported either, or if
    the render itself fails; the caller's own retry loop (generate_clips'
    MAX_DUP_RETRIES) and beat-alignment reuse already handle a transient
    failure, same as any other clip-generation error."""
    if niche:
        try:
            import character_engine
            if character_engine.enabled(niche):
                out = _render_character_image(prompt, seed, client_id, niche)
                if out is not None:
                    return out
        except Exception as e:
            print(f"[comfy] character path skipped (non-fatal): {e}")

    tpl = _stills_template()
    if tpl is None:
        return None
    import comfy_template
    g = comfy_template.prepare(tpl, prompt=prompt, seed=seed,
                               save_prefix="rufus_stills",
                               negative=_stills_negative())
    pid = _submit(g, client_id)
    if not pid:
        return None
    return _await_image(pid)


CHARACTER_TEMPLATE = Path(__file__).parent.parent / "config" / "character_stills_api.json"


def _character_template() -> dict | None:
    """The exported image-conditioning workflow (IPAdapter/PuLID/etc.) for
    recurring-character stills, or None if it hasn't been exported yet.
    Honors RUFUS_CHARACTER_TEMPLATE=0 as an explicit opt-out even when the
    file exists. Same load/placeholder contract as _stills_template()."""
    if os.environ.get("RUFUS_CHARACTER_TEMPLATE", "1").strip().lower() in \
            ("0", "false", "no", "off"):
        return None
    import comfy_template
    tpl = comfy_template.load_template(CHARACTER_TEMPLATE)
    if tpl is not None and comfy_template.has_placeholder(tpl):
        return tpl
    return None


def _ensure_character_reference(niche: str, client_id: str) -> Path | None:
    """The persistent reference portrait for this niche's character, rendering
    it once (via the PLAIN stills template) if it doesn't exist yet. Returns
    None on any failure — the caller then falls back to the ordinary
    pipeline, it never blocks a run."""
    import character_engine
    ref_path = character_engine.reference_image_path(niche)
    if ref_path is None:
        return None
    if ref_path.exists() and ref_path.stat().st_size > 0:
        return ref_path

    sheet_prompt = character_engine.character_sheet_prompt(niche)
    tpl = _stills_template()
    if sheet_prompt is None or tpl is None:
        return None
    import comfy_template
    g = comfy_template.prepare(tpl, prompt=_with_detail(sheet_prompt),
                               seed=random.randint(1, 2_000_000_000),
                               save_prefix="rufus_character_ref",
                               negative=_stills_negative())
    pid = _submit(g, client_id)
    if not pid:
        return None
    img_bytes = _await_image(pid)
    if not img_bytes:
        return None
    try:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_bytes(img_bytes)
        print(f"[comfy] bootstrapped character reference → {ref_path}")
    except OSError as e:
        print(f"[comfy] couldn't save character reference: {e}")
        return None
    return ref_path


def _render_character_image(prompt: str, seed: int, client_id: str,
                            niche: str) -> bytes | None:
    """Render one recurring-character still via config/character_stills_api.json
    (an IPAdapter/PuLID-style image-conditioning template), or None if the
    template isn't exported yet, the reference portrait can't be produced, or
    the render fails — any of which sends the caller back to the plain
    stills path."""
    tpl = _character_template()
    if tpl is None:
        return None
    ref_path = _ensure_character_reference(niche, client_id)
    if ref_path is None:
        return None
    from svd_client import _upload_image
    image_name = _upload_image(ref_path)
    if not image_name:
        return None
    import comfy_template
    g = comfy_template.prepare(tpl, prompt=prompt, image_name=image_name,
                               seed=seed, save_prefix="rufus_character",
                               negative=_stills_negative())
    pid = _submit(g, client_id)
    if not pid:
        return None
    return _await_image(pid)


def _free_comfy_memory() -> None:
    """Ask ComfyUI to unload models + free VRAM/RAM (its /free endpoint).
    Used at the stills→motion phase boundary: a 24GB card can't hold the
    stills model and a 14B/8B video model together, and letting ComfyUI
    evict lazily under pressure is exactly the documented RAM-leak/
    degradation pattern. Best effort — an older ComfyUI without /free just
    ignores this."""
    try:
        r = requests.post(f"{_host()}/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=30)
        if r.status_code == 200:
            print("[comfy] freed ComfyUI model memory (stills done — loading motion model)")
    except Exception:
        pass


def _submit(graph: dict, client_id: str) -> str | None:
    """POST a workflow to /prompt. Returns prompt_id or None."""
    try:
        r = requests.post(f"{_host()}/prompt",
                          json={"prompt": graph, "client_id": client_id},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("prompt_id")
    except Exception as e:
        print(f"[comfy] submit failed: {e}")
        return None


def _await_image(prompt_id: str, timeout: float | None = None) -> bytes | None:
    """Poll /history/<id> until the SaveImage node reports an output, then fetch
    the PNG via /view. Returns raw bytes or None on timeout/failure.

    `timeout` overrides GEN_TIMEOUT for callers that cannot afford the full
    wait — the dashboard renders a thumbnail inline on a threaded=False Flask
    app, so a request that blocks for the default 300s freezes the page for
    every other user. A pipeline run has nobody waiting on it and keeps the
    generous default."""
    deadline = time.time() + (GEN_TIMEOUT if timeout is None else timeout)
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
            outputs = entry.get("outputs", {})
            for node in outputs.values():
                images = node.get("images") or []
                if images:
                    img = images[0]
                    try:
                        vr = requests.get(
                            f"{_host()}/view",
                            params={"filename": img.get("filename", ""),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")},
                            timeout=30,
                        )
                        vr.raise_for_status()
                        return vr.content
                    except Exception as e:
                        print(f"[comfy] image fetch failed: {e}")
                        return None
            # entry exists but no images → generation errored in ComfyUI
            if "status" in entry and entry["status"].get("status_str") == "error":
                print("[comfy] ComfyUI reported a generation error for this prompt")
                return None
        time.sleep(POLL_INTERVAL)

    waited = GEN_TIMEOUT if timeout is None else timeout
    print(f"[comfy] timed out after {waited}s waiting for image")
    return None


def generate_clips(queries: list[str], n: int = 4,
                   clip_duration: float = 8.0, niche: str | None = None) -> list[Path]:
    """Generate one Ken Burns clip per query via ComfyUI, in order.

    Pipeline per clip:
      query → stills model (config/stills_api.json) 832×1472 → Lanczos 2× →
      crop 1080×1920 → Ken Burns mp4

    `niche` is optional and only used to route into the recurring-character
    path (character_engine.py) when that niche has one configured — omitting
    it (or passing one with no character configured) is identical to the
    behavior before character mode existed.

    Matches sd_client.generate_clips' contract: returns a list of 1080×1920 mp4
    Paths (one per query), or [] if ComfyUI is not running / no stills template
    is exported / all images fail, so main.py can fall through to the next
    backend.
    """
    if not is_available():
        print(f"[comfy] ComfyUI not running at {_host()} — start it with --listen, "
              f"or set COMFY_HOST. Falling back.")
        return []

    if _stills_template() is None:
        print("[comfy] no stills model configured — export a ComfyUI image "
              "workflow (Z-Image-Turbo recommended, Apache 2.0/commercial-safe) "
              "to config/stills_api.json. See README's 'Swappable stills model' "
              "section. Falling back.")
        return []

    beat_mode = _beat_motion()
    frames_per_beat = _frames_per_beat()
    if beat_mode == "i2i":
        if _i2i_template() is None:
            print("[comfy] RUFUS_BEAT_MOTION=i2i but no img2img workflow at "
                  "config/stills_i2i_api.json — export one from ComfyUI "
                  "(LoadImage → VAEEncode → KSampler denoise ~0.4, 10-12 steps, "
                  "prompt = RUFUS_PROMPT). Falling back to single stills.")
            beat_mode = ""
        else:
            if frames_per_beat == 1:
                frames_per_beat = I2I_DEFAULT_FRAMES
            print(f"[comfy] beat motion: i2i chain, {frames_per_beat} frames/beat "
                  f"→ motion-interpolated to {SMOOTH_FPS}fps")
    elif beat_mode == "kenburns":
        frames_per_beat = 1
    elif beat_mode == "cut" and frames_per_beat == 1:
        frames_per_beat = 3

    # Image-to-video: animate each still into real motion instead of the
    # Ken Burns zoom, via an ORDERED engine chain resolved once per run —
    # Wan 2.2 (best temporal consistency, takes a motion prompt) → SVD →
    # Ken Burns. Any per-image failure walks down the chain, so a clip is
    # never lost to a fancier engine.
    motion_engines: list[tuple[str, object]] = []
    if beat_mode == "i2v":
        # Explicitly asked for the motion chain — a stale RUFUS_FRAMES_PER_BEAT
        # must not quietly bypass it.
        frames_per_beat = 1
    elif frames_per_beat > 1 and beat_mode != "i2i":
        # Mutually exclusive with the motion chain by design — both answer
        # "how does this beat move", and running both would animate each
        # sub-frame separately, which is not what cutting between stills is.
        print(f"[comfy] frames-per-beat: {frames_per_beat} stills per beat, "
              f"hard-cut — motion engines bypassed (RUFUS_FRAMES_PER_BEAT)")
    try:
        import svd_client
        _stills_only_reason = ("RUFUS_STILLS_ONLY=1 forces images-only"
                               if svd_client._stills_only() else None)
    except Exception:
        _stills_only_reason = None
    if frames_per_beat > 1:
        # Reuse the existing "why is motion off" plumbing so each engine
        # reports the real reason instead of silently vanishing from the log.
        _stills_only_reason = (
            f"RUFUS_BEAT_MOTION=i2i chains stills instead" if beat_mode == "i2i"
            else f"RUFUS_FRAMES_PER_BEAT={frames_per_beat} cuts between stills instead")
    try:
        import wan_client
        if frames_per_beat == 1 and wan_client.enabled():
            wan_ok, wan_why = wan_client.ready()
            print(f"[comfy] motion wan 2.2: {'ON' if wan_ok else 'off'} — {wan_why}")
            if wan_ok:
                motion_engines.append(("wan", wan_client.animate_image))
        else:
            print(f"[comfy] motion wan 2.2: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_WAN=0'})")
    except Exception as e:
        print(f"[comfy] wan unavailable ({e})")
    try:
        import hunyuan_client
        if frames_per_beat == 1 and hunyuan_client.enabled():
            hy_ok, hy_why = hunyuan_client.ready()
            print(f"[comfy] motion hunyuan 1.5: {'ON' if hy_ok else 'off'} — {hy_why}")
            if hy_ok:
                # After Wan on purpose: Wan keeps non-face shots (proven
                # quality), and its face-skip returns False fast — so face
                # shots land here instead of falling to static Ken Burns.
                motion_engines.append(("hunyuan", hunyuan_client.animate_image))
        else:
            print(f"[comfy] motion hunyuan 1.5: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_HUNYUAN=0'})")
    except Exception as e:
        print(f"[comfy] hunyuan unavailable ({e})")
    try:
        import ltx_client
        if frames_per_beat == 1 and ltx_client.enabled():
            lx_ok, lx_why = ltx_client.ready()
            print(f"[comfy] motion ltx 2.3: {'ON' if lx_ok else 'off'} — {lx_why}")
            if lx_ok:
                motion_engines.append(("ltx", ltx_client.animate_image))
        else:
            print(f"[comfy] motion ltx 2.3: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_LTX=0'})")
    except Exception as e:
        print(f"[comfy] ltx unavailable ({e})")
    try:
        import svd_client
        if frames_per_beat == 1 and svd_client.img2vid_enabled():
            svd_engine, svd_why = svd_client.resolve_engine()
            print(f"[comfy] motion svd: "
                  f"{'ON via ' + svd_engine if svd_engine else 'off'} — {svd_why}")
            if svd_engine:
                def _svd_animate(png, clip, duration=8.0, idx=0, prompt="",
                                 _engine=svd_engine):
                    return svd_client.animate_image(png, clip, duration=duration,
                                                    idx=idx, engine=_engine,
                                                    prompt=prompt)
                motion_engines.append(("svd", _svd_animate))
    except Exception as e:
        print(f"[comfy] img2vid unavailable ({e}) — Ken Burns only")

    tmp_dir = paths.media_root() / "temp" / "comfy"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(queries or ["cinematic establishing shot"])
    if len(prompts) < n:
        base = prompts[:]
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + ", different angle, wider shot")
    prompts = prompts[:n]
    prompts = [_with_detail(p) for p in prompts]

    # pid in the stamp: with per-channel locks two channels may run
    # concurrently, and two runs starting the same second would otherwise
    # collide on identical {stamp}_{i}.png/.mp4 temp names in the shared dir.
    stamp        = f"{int(time.time())}_{os.getpid()}"
    client_id    = uuid.uuid4().hex
    master_seed  = random.randint(1, 2_000_000_000)
    # Seed the dup check with prior runs' hashes so cross-run look-alikes
    # regen too, not just within-run ones. n_prior marks where history ends.
    accepted_hashes: list[int] = _load_prior_hashes() if _fresh_images_enabled() else []
    n_prior = len(accepted_hashes)
    if n_prior:
        print(f"[comfy] freshness: {n_prior} image hash(es) from recent runs loaded")
    clips: list[Path] = []
    try:
        import character_engine
        _char_on = character_engine.enabled(niche) and _character_template() is not None
    except Exception:
        _char_on = False
    print(f"[comfy] stills: config/stills_api.json  base_seed={master_seed}"
          + ("  [recurring character ON]" if _char_on else ""))

    # Every run keeps its keyframes + prompts, not just RUFUS_DEBUG=1 runs —
    # the quality-review workflow needs every image logged, not a sampled
    # subset a reviewer has to remember to opt into. Prefer the shared run id
    # main.py sets (RUFUS_DEBUG_RUN_ID) so this run's images land in the SAME
    # folder as its script/voiceover instead of a folder named after this
    # stage's own timestamp. Falls back to the temp-file stamp when
    # comfy_client runs standalone (its __main__).
    debug_name = os.environ.get("RUFUS_DEBUG_RUN_ID") or str(stamp)
    debug_dir = paths.debug_root() / debug_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"[comfy] keeping keyframes in {debug_dir}")

    # ── Phase 1: generate every still (the stills model stays loaded the whole time) ────
    # `stills` holds a LIST of frames per beat — length 1 in normal mode, and
    # frames_per_beat when animating by cutting between stills.
    stills: list[tuple[int, list[Path], str]] = []   # (beat index, png paths, prompt)
    for i, prompt in enumerate(prompts):
        print(f"[comfy] {i+1}/{len(prompts)}: {prompt}")
        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False

        for retry in range(MAX_DUP_RETRIES + 1):
            # %(2**31) keeps the seed in range for any backend; offset per clip/retry.
            seed  = (master_seed + i + 1000 * retry) % (2**31 - 1)
            img_bytes = _render_image(prompt, seed, client_id, niche=niche)
            if not img_bytes:
                # A hard generation error (vs. a plain duplicate) is often a
                # transient GPU/model-loading hiccup on the ComfyUI side —
                # a short pause before resubmitting gives it a chance to clear
                # instead of hammering the same broken state 3x back-to-back.
                time.sleep(GEN_ERROR_BACKOFF)
                continue

            if not _fit_to_portrait(img_bytes, png_path):  # → exactly 1080×1920
                continue

            h = _avg_hash(png_path)
            is_dup = (h is not None and accepted_hashes
                      and min(_hamming(h, p) for p in accepted_hashes) < DUP_THRESHOLD)
            if is_dup and retry < MAX_DUP_RETRIES:
                print(f"[comfy] dup on clip {i+1} → regen (retry {retry+1})")
                continue
            if is_dup:
                print(f"[comfy] clip {i+1} still near-dup after retries — keeping")
            if h is not None:
                accepted_hashes.append(h)
            accepted = True
            break

        if not accepted:
            # BEAT ALIGNMENT: skipping would shift every LATER clip one beat
            # earlier than its narration (clip lists are positional — the
            # renderer cuts assume clip[i] ↔ beat[i]). Reuse the previous
            # accepted still instead: a repeated image with a different
            # Ken Burns/motion treatment is far less damaging than every
            # subsequent image narrating the wrong sentence.
            if stills:
                print(f"[comfy] no usable image for clip {i+1} — reusing "
                      f"previous still to keep images aligned with narration")
                png_path.write_bytes(stills[-1][1][0].read_bytes())
            else:
                print(f"[comfy] no usable image for clip {i+1} — skipping")
                continue

        # Extra frames for this beat: SAME seed (so the seed holds the
        # composition) plus a progression modifier that advances the action.
        # Deliberately NOT hash-checked: these are meant to resemble the base
        # frame closely, which is exactly what the dedup gate exists to reject,
        # and hashing them into accepted_hashes would make every later beat
        # look like a duplicate. A failed sub-frame just shortens the beat.
        #
        # The already-rendered png_path IS the peak (the prompt as written), so
        # it is slotted at the peak's position in the arc rather than simply
        # placed first — otherwise the sequence plays peak → peak → after, with
        # the "moment earlier" frame never rendered at all.
        if beat_mode == "i2i":
            beat_frames = _build_i2i_chain(
                base_png=png_path, base_raw=img_bytes, prompt=prompt, seed=seed,
                client_id=client_id, n=frames_per_beat, tmp_dir=tmp_dir,
                stamp=stamp, beat=i)
            if debug_dir is not None:
                try:
                    for f_idx, fp in enumerate(beat_frames):
                        sfx = "" if f_idx == 0 else chr(ord("a") + f_idx)
                        (debug_dir / f"{i+1:02d}{sfx}.png").write_bytes(fp.read_bytes())
                    (debug_dir / f"{i+1:02d}.txt").write_text(
                        f"FLUX PROMPT:\n{prompt}\n", encoding="utf-8")
                except Exception as e:
                    print(f"[comfy] debug-save failed for clip {i+1}: {e}")
            stills.append((i, beat_frames, prompt))
            continue

        modifiers = _progression_modifiers(frames_per_beat)
        peak_pos = modifiers.index("") if "" in modifiers else 0
        slots: list[Path | None] = [None] * len(modifiers)
        slots[peak_pos] = png_path
        for pos, modifier in enumerate(modifiers):
            if pos == peak_pos:
                continue
            sub_path = tmp_dir / f"{stamp}_{i}_{pos}.png"
            sub_prompt = f"{prompt.rstrip().rstrip('.')}. {modifier}"
            sub_bytes = _render_image(sub_prompt, seed, client_id, niche=niche)
            if sub_bytes and _fit_to_portrait(sub_bytes, sub_path):
                slots[pos] = sub_path
            else:
                print(f"[comfy] sub-frame {pos+1} of clip {i+1} failed — "
                      f"beat will be shorter by one frame")
        beat_frames = [p for p in slots if p is not None]

        if debug_dir is not None:
            try:
                for f_idx, fp in enumerate(beat_frames):
                    suffix = "" if f_idx == 0 else chr(ord("a") + f_idx)
                    (debug_dir / f"{i+1:02d}{suffix}.png").write_bytes(fp.read_bytes())
                (debug_dir / f"{i+1:02d}.txt").write_text(
                    f"FLUX PROMPT:\n{prompt}\n", encoding="utf-8")
            except Exception as e:
                print(f"[comfy] debug-save failed for clip {i+1}: {e}")

        stills.append((i, beat_frames, prompt))

    # ── Phase 2: animate every still ────────────────────────────────────────
    # Two phases instead of image→animate per clip: interleaving forced
    # ComfyUI to swap FLUX in and out for EVERY clip (10 model swaps per
    # video on a 24GB card that can't hold both), thrashing RAM/VRAM and —
    # per a ComfyUI-reliability audit — degrading until every clip silently
    # fell through to Ken Burns. Batching stills first means exactly ONE
    # switch, with an explicit /free between so the motion model loads into
    # a clean card.
    if motion_engines and stills:
        _free_comfy_memory()

    motion_log: list[dict] = []
    for i, beat_frames, prompt in stills:
        png_path = beat_frames[0]
        clip_path = tmp_dir / f"{stamp}_{i}.mp4"

        # Frames-per-beat mode: Ken Burns each frame for its share of the beat,
        # then hard-cut them together. The SAME `idx` is used for every
        # sub-frame on purpose — _animate_to_clip picks its zoom/pan pattern
        # from it, so reusing it keeps the camera moving in one direction
        # across the cuts and reads as a single continuous shot with the
        # action advancing, instead of three unrelated shots.
        # i2i: the frames genuinely continue each other, so interpolate BETWEEN
        # them into real motion rather than cutting. Cutting here would throw
        # away the continuity the chain just paid for.
        if beat_mode == "i2i" and len(beat_frames) > 1:
            made = _assemble_smooth_beat(beat_frames, clip_path, clip_duration)
            if made:
                clips.append(clip_path)
                print(f"[comfy] clip {i+1} ready "
                      f"({len(beat_frames)} i2i frames → interpolated)")
            else:
                print(f"[comfy] smooth assembly failed for clip {i+1} — later "
                      f"images may drift ahead of narration")
            for frame in beat_frames:
                frame.unlink(missing_ok=True)
            continue

        if len(beat_frames) > 1:
            share = clip_duration / len(beat_frames)
            # The 50KB default floor is calibrated for a full-length beat clip
            # and rejects valid short ones (a 1.0s flat-illustration clip
            # measures ~47KB), so scale it to this sub-clip's duration. Kept
            # well UNDER the measured encode rate (~43KB/s on detailed flat
            # art, less on simple art) on purpose — this gate only has to
            # catch "ffmpeg wrote nothing usable", not judge quality, and a
            # tight floor silently drops valid frames and shortens the beat.
            floor = max(10_000, int(share * 8_000))
            parts: list[Path] = []
            for f_idx, frame in enumerate(beat_frames):
                part = tmp_dir / f"{stamp}_{i}_{f_idx}.mp4"
                if _animate_to_clip(frame, part, duration=share, idx=i,
                                    min_bytes=floor):
                    parts.append(part)
            made = _concat_clips(parts, clip_path)
            for part in parts:
                part.unlink(missing_ok=True)
            if made:
                clips.append(clip_path)
                print(f"[comfy] clip {i+1} ready ({len(parts)} cut frames)")
            else:
                print(f"[comfy] animation failed for clip {i+1} — later images "
                      f"may drift ahead of narration")
            for frame in beat_frames:
                frame.unlink(missing_ok=True)
            continue

        made_via = None
        tried: list[str] = []
        for eng_name, animate in motion_engines:
            t0 = time.time()
            okd = animate(png_path, clip_path, duration=clip_duration, idx=i,
                          prompt=prompt)
            secs = time.time() - t0
            # Record what the engine ACTUALLY used, not what we assume it did:
            # the motion prompt is derived inside the engine and the settings
            # come from env + the exported template, so neither is knowable
            # here. Engines publish both via LAST_CALL/settings().
            rec = {"beat": i + 1, "engine": eng_name, "ok": bool(okd),
                   "seconds": round(secs, 1)}
            try:
                mod = {"hunyuan": "hunyuan_client", "wan": "wan_client",
                       "ltx": "ltx_client"}.get(eng_name)
                if mod:
                    rec.update(__import__(mod).LAST_CALL)
            except Exception:
                pass
            motion_log.append(rec)
            tried.append(f"{eng_name} {'ok' if okd else 'failed'} in {secs:.0f}s")
            if okd:
                made_via = eng_name
                break
            print(f"[comfy] {eng_name} failed for clip {i+1} — trying next engine")
        made = made_via is not None or _animate_to_clip(png_path, clip_path,
                                                        duration=clip_duration, idx=i)
        if made_via is None and motion_engines:
            motion_log.append({"beat": i + 1, "engine": "kenburns", "ok": bool(made),
                               "note": "all motion engines declined/failed"})
        if made:
            clips.append(clip_path)
            print(f"[comfy] clip {i+1} ready"
                  + (f" ({made_via} motion)" if made_via else " (Ken Burns)"))
        else:
            print(f"[comfy] animation failed for clip {i+1} — later images may "
                  f"drift ahead of narration")
        png_path.unlink(missing_ok=True)

    if motion_log:
        try:
            paths.write_run_report(debug_name, motion=motion_log)
        except Exception as e:
            print(f"[comfy] motion-report write skipped (non-fatal): {e}")

    if _fresh_images_enabled() and len(accepted_hashes) > n_prior:
        _save_hashes(accepted_hashes)
    print(f"[comfy] {len(clips)}/{len(prompts)} clips ready")
    return clips


if __name__ == "__main__":
    import sys
    qs = sys.argv[1:] or ["modern luxury kitchen interior, golden hour light, wide angle",
                          "sunlit living room, floor to ceiling windows, city view"]
    for p in generate_clips(qs, n=len(qs)):
        print(f"CLIP={p}")
