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
import copy
import random
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import frame_gate
import paths
import video_format as _vf

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


# How much of the model's picture may be thrown away by the cover-crop before
# it stops being a fit and starts being a different composition. The matched
# case discards ~0.5%; a portrait template rendered for a landscape frame
# discards about two thirds, taking the head off every shot the storyboard
# framed.
_MAX_CROP_LOSS = 0.15
_crop_warned = False


def _warn_if_mostly_cropped(w: int, h: int) -> float:
    """Fraction of the model's frame the cover-crop will discard. Warns once.

    THE TEMPLATE IS THE OWNER'S, NOT THE PROFILE'S. config/stills_api.json is
    exported from ComfyUI and this module never rewrites its size — that is the
    template-only contract, and it is the right one: the workflow's latent size
    is tuned to the checkpoint it was exported with. But it means the format
    switch cannot reach it. Ask for long-form with a portrait stills workflow
    and everything still "works": the render succeeds, QC passes, the file is
    1920×1080, and every picture in it is the middle third of a portrait image.

    Nothing downstream can see that, because by the time anything looks, the
    frame is the right shape. So this says it at the only moment it is visible
    — once per run, naming the fix — rather than resizing the latent behind the
    owner's back or failing a render that is otherwise fine.
    """
    global _crop_warned
    if w <= 0 or h <= 0:
        return 0.0
    scale = max(OUT_W / w, OUT_H / h)
    kept = (OUT_W * OUT_H) / float(round(w * scale) * round(h * scale))
    loss = max(0.0, 1.0 - kept)
    if loss > _MAX_CROP_LOSS and not _crop_warned:
        _crop_warned = True
        sw, sh = _vf.still_dimensions()
        print(f"[comfy] ⚠ the stills workflow renders {w}×{h} but this run's "
              f"frame is {OUT_W}×{OUT_H} — {loss * 100:.0f}% of every picture "
              f"is cropped away to fit. Re-export config/stills_api.json with "
              f"the latent at {sw}×{sh} for {_vf.name()}.")
    return loss


def _fit_to_frame(img_bytes: bytes, out_path: Path) -> bool:
    """Cover-resize a stills-model frame to exactly OUT_W×OUT_H, preserving
    composition.

    832×1472 (0.5652) vs 1080×1920 (0.5625) are near-identical aspect ratios, so
    we Lanczos-scale to just cover the target and trim the ~0.5% sliver. This
    keeps ~99% of the frame as composed — unlike sd_client's fixed 2× upscale
    + center-crop, which at this generation size would discard 35% of the image.

    The same holds landscape: the long-form profile asks the model for
    1472×832 against a 1920×1080 target, so the sliver stays ~0.5% there too.
    Nothing here was ever portrait-specific except the name, and a function
    called _fit_to_portrait is one nobody would think to call for a landscape
    render.
    """
    from PIL import Image
    import io as _io

    img  = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    _warn_if_mostly_cropped(w, h)
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

# Extra attempts a clip gets when frame_gate rejects it, ON TOP of the
# duplicate budget. Two, for the same reason MAX_DUP_RETRIES is two: a third
# re-roll of a prompt the model keeps answering the same way is GPU time spent
# on a disagreement, and the run has a hundred and fifty other pictures to
# draw. With the gate off this is not spent at all.
GATE_RETRIES = 2

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


def _frames_per_beat_was_asked_for() -> bool:
    """Did someone actually SET the number, or is 1 just the default?

    These are the same integer and they are not the same instruction. The
    stills-only branch below auto-selects `cut` when nothing was asked for,
    and `cut` rewrites 1 to 3 — so a run that reads RUFUS_FRAMES_PER_BEAT=1
    and one that reads nothing at all both ended up at three stills a beat.
    The owner set the dashboard's "Stills per beat" to 1 to stop the
    near-identical triplets and got triplets, with the log still saying it
    was cutting between stills. A setting that reports as honoured and is not
    is worse than one that is missing, because the missing one gets added.
    """
    return os.environ.get("RUFUS_FRAMES_PER_BEAT", "").strip() != ""


# How a beat moves. One selector instead of four interacting flags, because
# these are alternatives, not layers:
#   i2v      motion model per still (Wan/Hunyuan/LTX/SVD). Best-looking real
#            motion; measured 600-1800s PER CLIP on a 3090, i.e. hours a video.
#   i2i      each frame img2img'd from the previous one at low denoise, so the
#            frames genuinely continue each other, then crossfaded. ~1-2s a
#            frame. Needs config/stills_i2i_api.json.
#   hero     ONE beat gets a real motion clip; every other beat is a cut still.
#            The beat chosen is the one carrying the story architect's THE
#            SCENE — the single line in the script that is already a motion
#            prompt, because it names a date, a place and a person doing
#            something. Every other beat is evidence (a total, a share, a
#            consequence), and a video model handed an abstraction produces a
#            slow drift over generic scenery. i2v costs 600-1800s PER CLIP on
#            this hardware, so nine of them is hours; one of them is minutes,
#            and one moving shot among stills reads as a deliberate accent
#            rather than wallpaper the viewer stops noticing by beat three.
#   cut      several independent stills on one seed, hard cut. No extra setup.
#   kenburns one still, zoom only.
# Unset keeps the historical behaviour exactly: RUFUS_FRAMES_PER_BEAT>1 means
# `cut`, otherwise the motion chain.
BEAT_MOTION_MODES = ("i2v", "i2i", "cut", "hero", "kenburns")
I2I_DEFAULT_FRAMES = 5
# How many stills a non-hero beat gets in hero mode. Same as `cut`'s default —
# the point of hero mode is that the OTHER beats stay cheap.
#
# TUNABLE BECAUSE IT TURNED OUT TO BE THE BIGGER HALF. Once the hero beat is
# the only motion clip, the stills phase is what the run is made of: the
# owner's ComfyUI queue shows 12-14 seconds per still, so 9 beats x 3 = 27
# stills is about six minutes before any motion starts. Dropping to 1 makes it
# two. The cost is real and worth stating: at 3 the beat hard-cuts between
# three related images and the shot advances inside the narration line; at 1 it
# is a single Ken Burns move. Speed, paid for in visual interest on the eight
# beats nobody was going to look at twice.
HERO_OTHER_FRAMES = 3


def _hero_other_frames() -> int:
    """Stills per non-hero beat. RUFUS_HERO_OTHER_FRAMES overrides."""
    raw = os.environ.get("RUFUS_HERO_OTHER_FRAMES", "").strip()
    try:
        return max(1, int(raw)) if raw else HERO_OTHER_FRAMES
    except ValueError:
        print(f"[comfy] RUFUS_HERO_OTHER_FRAMES={raw!r} is not a number — "
              f"using {HERO_OTHER_FRAMES}")
        return HERO_OTHER_FRAMES


def _beat_motion() -> str:
    mode = os.environ.get("RUFUS_BEAT_MOTION", "").strip().lower()
    return mode if mode in BEAT_MOTION_MODES else ""


def _say_if_ready_but_switched_off(label: str, client) -> None:
    """Report an engine that is fully installed and only turned off by a flag.

    "off — disabled (RUFUS_STILLS_ONLY=1 forces images-only)" reads the same
    whether the engine has no template and no models, or is one environment
    variable away from producing motion. On this box the Hunyuan template is
    exported and committed and the models are installed, and that line has been
    printed on every run for weeks while the owner believed motion required a
    35GB download. Checking readiness costs one HTTP call to a server that is
    already up.
    """
    try:
        ok, _why = client.ready()
    except Exception:
        return
    if ok:
        print(f"[comfy]   ...but {label}'s template IS exported and its models "
              f"ARE loadable. Set RUFUS_STILLS_ONLY=0 to use it.")


def _hero_beat(prompts: list[str], scene: str) -> int | None:
    """Which beat carries THE SCENE, or None if nothing matches.

    Scored by content-word overlap between the architect's filmable moment and
    each beat's image prompt — both describe pictures, so they share vocabulary
    when they are about the same moment. Reuses storyboard's stopword and
    abstract-word filtering rather than adding a third tokenizer: "power" and
    "history" appearing in both is not a match, it is two abstractions.

    None on no scene, no overlap, or any import failure. The caller then
    animates nothing, which is exactly the run the pipeline produces today —
    a hero shot is a better way to spend motion time, never a prerequisite.
    """
    if not scene or not prompts:
        return None
    try:
        import storyboard
        stop = storyboard._STOPWORDS
        abstract = storyboard._ABSTRACT
    except Exception:
        return None

    def _words(text: str) -> set:
        return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
                if w not in stop and w not in abstract}

    scene_words = _words(scene)
    if not scene_words:
        return None

    overlaps = [len(scene_words & _words(p)) for p in prompts]
    best_n = max(overlaps)
    if best_n < 1:
        return None

    # A CLEAR winner, not a fixed threshold. A scene and the shot built on it
    # describe one moment in different words — "sent Charles V a letter" against
    # "seals a letter with wax, then slides it across" shares exactly one word.
    # Demanding two would decline the correct beat. What separates signal from
    # coincidence is not the count but whether one beat leads: a word that
    # appears in every shot is not discriminating, and a word that appears in
    # exactly one is the beat about that moment.
    if overlaps.count(best_n) > 1:
        return None
    return overlaps.index(best_n)


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
    1080×1920 frame: the pipeline's _fit_to_frame upscales and crops, and
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
        if not _fit_to_frame(nxt, fitted):
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
    "Captured a moment before that: nothing has started yet, the stillness "
    "just before.",
    "Captured a moment earlier: the action is only just beginning.",
    "",
    "Captured a moment later: the action has just completed.",
    "Captured a beat afterwards: the aftermath, everything settling.",
    "Captured last of all: the place after everyone has gone, only what was "
    "left behind.",
]


def _progression_modifiers(n: int) -> list[str]:
    """`n` prompt modifiers spanning one beat's micro-arc, peak included.

    THE CAP IS ANNOUNCED, not silent. This list is what a beat can be split
    into and asking for more frames than there are steps used to return the
    steps and say nothing — RUFUS_FRAMES_PER_BEAT=6 quietly rendered four, and
    the owner had no way to know why the picture count did not move. Every
    other clamp in this pipeline that stayed quiet has cost a debugging
    session (AGENTS.md), so this one talks.
    """
    if n <= 1:
        return [""]
    if n > len(_PROGRESSION_STEPS):
        print(f"[comfy] RUFUS_FRAMES_PER_BEAT={n} is more than the "
              f"{len(_PROGRESSION_STEPS)} steps a beat can be split into — "
              f"rendering {len(_PROGRESSION_STEPS)} per beat. For more "
              f"pictures raise SD_CLIPS (more beats) instead.")
    # Centred on the peak: with 3 steps you want earlier/peak/later, not the
    # two frames before the peak and no peak at all.
    peak = _PROGRESSION_STEPS.index("")
    take = min(n, len(_PROGRESSION_STEPS))
    start = max(0, min(peak - (take - 1) // 2, len(_PROGRESSION_STEPS) - take))
    return _PROGRESSION_STEPS[start:start + take]


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


# Two different questions, so two different thresholds.
#
# WITHIN a run: "are two beats of THIS video the same picture?" A viewer sees
# them 4 seconds apart, so even a loose resemblance is a real defect and
# DUP_THRESHOLD (6 of 64 bits) is right.
#
# ACROSS runs: "does this look like a video I published last week?" Nobody
# watches two of them back to back, so only a near-identical frame matters.
# Judging that at the same threshold is what broke: a single-image test run
# with NO in-run predecessors was flagged as a duplicate twice in a row, purely
# against the 120-hash history. aHash reduces an image to an 8x8 grayscale
# grid, and this channel's flat-2D style — cream ground, few flat shapes,
# generous negative space — occupies a tiny corner of that space. With 120
# accumulated hashes almost any new flat image lands within 6 bits of
# something, so the pool became a ratchet: the longer the channel ran, the more
# often a perfectly good image was rejected and re-rendered for nothing.
FRESH_DUP_THRESHOLD = 3

# A CHAINED shot (shot_chain.py) is generated from the previous beat's image on
# purpose, so resembling it is the goal, not the defect — the ordinary dup gate
# would reject every one of them. What must still be caught is the opposite
# failure: an edit template that behaves like img2img and hands back the source
# picture unchanged. Below this many differing bits the "new scene" isn't one.
CHAIN_COPY_THRESHOLD = 2


def _is_duplicate(h: int, accepted: list[int], n_prior: int) -> bool:
    """Whether hash `h` is too close to an already-accepted image.

    `accepted[:n_prior]` came from previous runs and is judged strictly (a
    near-identical frame only); the rest is this run and uses the normal
    threshold."""
    prior, current = accepted[:n_prior], accepted[n_prior:]
    if current and min(_hamming(h, p) for p in current) < DUP_THRESHOLD:
        return True
    return bool(prior) and min(_hamming(h, p) for p in prior) < FRESH_DUP_THRESHOLD


def _load_prior_hashes() -> list[int]:
    try:
        data = json.loads(FRESH_HASH_FILE.read_text(encoding="utf-8"))
        return [int(h) for h in data.get("hashes", [])][-FRESH_HASH_CAP:]
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def _save_hashes(hashes: list[int]) -> None:
    try:
        FRESH_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        FRESH_HASH_FILE.write_text(
            json.dumps({"hashes": hashes[-FRESH_HASH_CAP:]}), encoding="utf-8")
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
    # 1. Lettering. Leads the list because garbled words are the most obvious
    #    AI tell in a finished Short, and early terms carry more weight.
    "text, letters, words, writing, lettering, typography, caption, subtitle, "
    "watermark, signature, logo, brand name, numbers, digits, gibberish text, "
    "garbled writing, fake language, misspelled words, distorted letterforms, "
    # 2. Style contamination. A flat-2D look drifts toward whatever medium the
    #    subject usually appears in — a 1923 street becomes sepia photography,
    #    a coin becomes a 3D product render — and one drifted beat inside nine
    #    flat ones reads worse than either look on its own. Naming the mediums
    #    to stay out of holds the style far better than asking for it once in
    #    the positive prompt.
    "photorealistic, photograph, 3d render, cgi, octane render, film grain, "
    "watercolor, oil painting, acrylic, pencil sketch, charcoal, engraving, "
    "airbrush, gradient shading, soft shading, ambient occlusion, bloom, "
    "lens flare, depth of field, bokeh, noise, rough texture, canvas texture, "
    # 3. LAYOUT. A 832x1472 canvas is very tall, and a model asked for a
    #    "cartoon illustration" fills tall canvases by stacking panels — a
    #    gallery under Z-Image-Base came back with three- and four-band
    #    contact sheets in a third of its frames, each band a different
    #    moment. One beat is ONE picture; a strip of four is four beats the
    #    edit never asked for and cannot cut between.
    #
    #    THESE TERMS ARE NEW BECAUSE THE NEGATIVE IS NEW. Every earlier
    #    gallery ran on z_image_turbo at CFG 1, where the negative prompt is
    #    mathematically inert — adding words to it then would have been
    #    cargo cult. At CFG 4 it is live, and this is the first list written
    #    against a defect it can actually reach.
    "comic strip, comic panels, multiple panels, split panel, split screen, "
    "storyboard sheet, contact sheet, photo grid, collage, diptych, triptych, "
    "tiled layout, film strip, borders between scenes, gutter lines, "
    "picture frame, letterbox bars, "
    # 4. Anatomy and encoding faults, last: cheapest to fix, least distinctive.
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


STYLES_FILE = Path(__file__).parent.parent / "config" / "styles.json"


def style_presets() -> dict:
    """Named looks from config/styles.json, or {} if it is absent or broken.

    Fail-open like every other config read here: a missing or malformed file
    leaves the pipeline on its built-in default rather than stopping a run over
    a look.
    """
    try:
        raw = json.loads(STYLES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in raw.items()
            if isinstance(v, str) and not k.startswith("_")}


def _detail_suffix() -> str:
    """The look every still is rendered in.

    THREE SOURCES, MOST SPECIFIC FIRST. A literal RUFUS_STILLS_DETAIL wins, so a
    one-off experiment never has to be added to a config file first. Then
    RUFUS_STYLE names a preset from config/styles.json — the whole point being
    that a look is ONE block of text appended byte for byte to every prompt,
    because an image model renders each beat from noise with no memory of the
    others and two paraphrases of one style are two styles to it. Then the
    built-in default.

    An unknown RUFUS_STYLE is loud rather than silent: a typo'd style name that
    quietly rendered the default look would be indistinguishable from the
    preset not working.
    """
    literal = os.environ.get("RUFUS_STILLS_DETAIL")
    if literal is not None and literal.strip():
        _say_look("RUFUS_STILLS_DETAIL (a literal override)")
        return literal.strip()

    name = os.environ.get("RUFUS_STYLE", "").strip()
    if name:
        presets = style_presets()
        if name in presets:
            _say_look(f"{name} (config/styles.json)")
            return presets[name].strip()
        known = ", ".join(sorted(presets)) or "none loaded"
        print(f"[comfy] RUFUS_STYLE={name!r} is not a known style — using the "
              f"default look. Known: {known}")

    # SAY WHICH LOOK IS IN FORCE, ESPECIALLY THIS ONE. The fall-through was
    # silent, and a run came back rendered in flat vector when the owner
    # expected stickman — because RUFUS_STYLE was set in a terminal for weeks
    # and then wasn't. Nothing in twenty-five minutes of log said which look
    # was chosen; the style text was there, buried inside all twenty-eight
    # prompts, where nobody reads it.
    _say_look("the built-in default — set RUFUS_STYLE for a named look "
              f"({', '.join(sorted(style_presets())) or 'none loaded'})")
    return os.environ.get("RUFUS_STILLS_DETAIL", DEFAULT_DETAIL_SUFFIX).strip()


_LOOK_SAID = ""


def _say_look(source: str) -> None:
    """Announce the look once per run, not once per prompt."""
    global _LOOK_SAID
    if _LOOK_SAID != source:
        _LOOK_SAID = source
        print(f"[comfy] look: {source}")


# Photographic direction that CONTRADICTS the flat-2D style: camera bodies,
# lens/aperture specs, depth-of-field and film-stock language. The prompt-writer
# emits these out of habit — a live batch carried "Shot on a Canon EOS 5D Mark
# IV, 50mm f/1.8 lens, with warm sepia tones and fine film grain" on beat 01 —
# and the old guard treated their presence as "this prompt has its own style,
# leave it alone", which silently rendered that ONE beat photoreal while the
# other nine were flat vector. Mixed looks inside a single Short are more
# obvious than either look on its own.
_PHOTO_SPEC_RE = re.compile(
    r"(?i)\b(shot on|captured on|"
    r"canon|nikon|sony|fujifilm|leica|pentax|panasonic|olympus|contax|"
    r"hasselblad|rolleiflex|"
    r"\d{2,3}\s?mm|f/\d|depth of field|bokeh|film grain|"
    r"photorealistic|hyperrealistic|photojournalism|full-frame)\b")


def _strip_photo_direction(prompt: str) -> str:
    """Remove camera/lens/film language from a prompt. Used only when the style
    suffix is a non-photographic one, where such language is a contradiction
    rather than a second opinion.

    Works at CLAUSE level, not token level. These specs are always written as
    whole comma-separated clauses ("Shot on a Canon EOS 5D Mark IV", "50mm
    f/1.8 lens", "with warm sepia tones and fine film grain"), and deleting the
    matched token alone leaves debris — ".8 lens, with warm sepia tones and
    fine ." — which is worse than the contradiction it was fixing. A sentence
    left empty by the removal is dropped entirely."""
    kept_sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", prompt):
        clauses = [c for c in sentence.split(",")
                   if not _PHOTO_SPEC_RE.search(c)]
        rebuilt = ",".join(clauses).strip(" ,.")
        if rebuilt:
            kept_sentences.append(rebuilt + ".")
    return re.sub(r"\s{2,}", " ", " ".join(kept_sentences)).strip()


def _is_photographic(style: str) -> bool:
    """Whether a style suffix asks for a photograph. Decides how a prompt's own
    camera language is treated: a second opinion (leave the prompt alone) or a
    contradiction (strip it)."""
    low = style.lower()
    return "not a photograph" not in low and (
        "photorealistic" in low or "photojournalism" in low
        or "shot on" in low or "real camera" in low)


# The marker inside a style preset that separates "true of any shot" from
# "true only when a person is in frame". A preset without it is one block and
# behaves exactly as it always has.
STYLE_FIGURE_MARKER = "--- FIGURE ONLY ---"

# THE DEFAULT DISTANCE, AND WHY IT IS DELIMITED RATHER THAN QUALIFIED.
#
# SCALE says the figures stand between half and three quarters of the frame's
# height. That stopped a real bug — a vast green field with one figure in the
# corner — and it is the right answer for a beat that names no distance.
#
# It is the WRONG answer for a beat that names one, and storyboard names one on
# every beat it plans: _apply_framing puts "Close shot: head and shoulders,
# filling most of the frame" at the very front of the prompt, deliberately,
# because the opening of a prompt is weighted most heavily.
#
# The first attempt at reconciling them was a sentence — "unless the shot names
# its own distance ... the shot's distance always wins". It did not work, and
# could not have: a text encoder has no meta level. "half and three quarters of
# the frame's height" is concrete and drawable; "unless the shot names its own
# distance" is abstract and is not. Both concepts reach the latent together and
# the drawable one wins. Six probe runs came back full-body against a prompt
# that opened with "Close on one figure's face".
#
# So the default is DELIMITED and DELETED instead. This is the same move
# _detail_for_shot already makes at FIGURE ONLY, for the same stated reason:
# the only reliable way not to get a thing is not to mention it.
STYLE_SCALE_OPEN = "--- DEFAULT DISTANCE ---"
STYLE_SCALE_CLOSE = "--- /DEFAULT DISTANCE ---"

# The distances a shot can name. storyboard._FRAMINGS supplies the first four
# phrasings; the rest are what a hand-typed prompt and the bench probes use for
# the same thing. tests/test_styles.py asserts every _FRAMINGS phrase matches,
# so the two files cannot drift apart again without a test failing.
_NAMES_DISTANCE_RE = re.compile(
    r"\b(wide shot|wide angle|establishing shot|medium shot|mid shot|"
    r"close shot|close detail|close ?-?up|close on|extreme close)\b", re.I)


def names_own_distance(prompt: str) -> bool:
    """Whether this shot already says how far away the camera is."""
    return bool(_NAMES_DISTANCE_RE.search(prompt or ""))


# AND DELETING THE DEFAULT DISTANCE WAS NOT ENOUGH.
#
# It was necessary — a competing instruction cannot stay — but the bench of
# 2026-08-23 ran with it gone and `face` came back full-body anyway, head to
# feet, against a prompt opening "Close on one figure's face".
#
# Because the absence of a contradiction is not the presence of an
# instruction, and thirteen other phrases in the block still required a whole
# body and a landscape:
#
#     a ground plane across the bottom third, a horizon line, open sky
#     two legs; each leg bends once at the knee
#     both arms are visible in every figure
#     PROPORTION IS FIXED: the head is one third of the whole figure's height
#
# None of those can be drawn inside a head-and-shoulders frame. The FIGURE
# half was written, every sentence of it, for a full-body mid-shot — so the
# style did not need a better sentence about distance, it needed the rules
# that assume distance to stop shipping when there is none.
#
# Same mechanism, second pair of markers.
STYLE_FARSHOT_OPEN = "--- WIDE OR MID ONLY ---"
STYLE_FARSHOT_CLOSE = "--- /WIDE OR MID ONLY ---"

_CLOSE_SHOT_RE = re.compile(
    r"\b(close shot|close detail|close ?-?up|close on|extreme close)\b", re.I)


def is_close_shot(prompt: str) -> bool:
    """Whether this shot comes in close enough that legs leave the frame."""
    return bool(_CLOSE_SHOT_RE.search(prompt or ""))


def _strip_region(text: str, opener: str, closer: str, drop: bool) -> str:
    """Remove every opener..closer region, or just the markers themselves.

    Every occurrence, not the first: the far-shot rules sit in two places —
    the place geometry in the shared half and the body geometry in the figure
    half — and a resolver that handled one would leave a raw marker line in
    the prompt, which is a defect this file has already shipped once.
    """
    if opener not in text:
        return text
    out: list[str] = []
    rest = text
    while opener in rest:
        head, _, tail = rest.partition(opener)
        inner, _, rest = tail.partition(closer)
        out.append(head.strip())
        if not drop:
            out.append(inner.strip())
    out.append(rest.strip())
    return " ".join(p for p in out if p)


def _resolve_framing(tail: str, framed: bool, close: bool) -> str:
    """The style text left once this shot's own distance has had its say.

    A style with no markers is returned untouched — every preset that predates
    this, and every literal RUFUS_STILLS_DETAIL, keeps its old behaviour.
    """
    tail = _strip_region(tail, STYLE_SCALE_OPEN, STYLE_SCALE_CLOSE, drop=framed)
    return _strip_region(tail, STYLE_FARSHOT_OPEN, STYLE_FARSHOT_CLOSE, drop=close)

# GPT tags each prompt with the kind of shot it wrote. The tag never reaches
# the image model — it is stripped here, after it has chosen the style.
_SHOT_TAG_RE = re.compile(r"^\s*\[SHOT\s*=\s*(figure|object)\s*\]\s*", re.I)


def shot_kind(prompt: str) -> str:
    """"figure" or "object" for a tagged prompt; "figure" when untagged.

    UNTAGGED MEANS FIGURE, deliberately. Every prompt written before this
    existed, every hand-typed prompt in the dashboard's regen box, and every
    prompt from a model that ignored the instruction all arrive without a tag
    — and for all of them the old behaviour (the whole style block) is the
    right answer. Defaulting to "object" would silently strip the figure rules
    from a run that never asked for that.
    """
    m = _SHOT_TAG_RE.match(prompt or "")
    return m.group(1).lower() if m else "figure"


def strip_shot_tag(prompt: str) -> str:
    return _SHOT_TAG_RE.sub("", prompt or "", count=1).strip()


def _detail_for_shot(kind: str, framed: bool = False,
                     close: bool = False) -> str:
    """The style text this shot should actually receive.

    WHY THE STYLE HAS TO KNOW. The block is appended byte for byte to every
    prompt, and roughly half of stickman describes how to draw a body — five
    separate parts, the oval head, where the arms leave the torso. On a beat
    whose subject is a banana, that is a long paragraph about limbs sitting
    directly after "a macro shot of a banana", and the model draws a figure.
    A style block has no meta level: every word in it is a word in the prompt,
    and the only reliable way not to get a figure is not to mention one.
    """
    tail = _resolve_framing(_detail_suffix(), framed, close)
    if STYLE_FIGURE_MARKER not in tail:
        return tail
    shared, _, figure = tail.partition(STYLE_FIGURE_MARKER)
    if kind == "object":
        return shared.strip()
    return f"{shared.strip()} {figure.strip()}"


def shot_last() -> bool:
    """Whether the shot's own words go LAST in the prompt instead of first.

    THE MEASUREMENT THAT PROMPTED THIS. Two probes, same seeds, same nine
    steps at CFG 1, the only difference being the style block:

        with stickman     "the animal drawn in full with its spots and its
                          real proportions" → a rock, and a tail.
                          "Close on one figure's face … brows raised high,
                          mouth a small open oval" → a full-body wide shot
                          with the brows angled DOWN.
        --plain           both drawn exactly as asked, first try.

    So the checkpoint follows those instructions perfectly well. What stops it
    is that the shot is 129 characters and the style block is 4,816 — the part
    of the prompt that says what THIS picture is comes to 2.6% of it, and then
    97.4% of constant text follows and buries it.

    Shortening the block is one lever (see stickman_lean). This is the other,
    and it is independent: put the constant first and let the variable have
    the last word. If position is what matters, this costs nothing and needs
    no rules dropped; if length is what matters, it will change nothing and
    should be turned back off rather than left on as folklore.

    AND IT FIGHTS A DECISION ANOTHER MODULE MADE ON PURPOSE. storyboard's
    _apply_framing puts the shot's distance — "Close shot: head and shoulders
    ... filling most of the frame" — at the OPENING of the prompt, and says
    why: it is "the one instruction that decides what the picture IS rather
    than what is in it". This flag moves the whole shot, that phrase included,
    behind three thousand characters of style. Whatever it buys at the end of
    the prompt, it costs at the start.

    OFF BY DEFAULT, because it reorders every image prompt on the channel and
    the evidence for it is a hypothesis, not a gallery. Turn it on for a probe
    first:  $env:RUFUS_SHOT_LAST = "1"
    """
    return os.environ.get("RUFUS_SHOT_LAST", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def _with_detail(prompt: str) -> str:
    """Append the style direction, reconciling it with any photographic
    direction the prompt already carries.

    Photographic style → the prompt's own camera spec is a legitimate,
    more specific choice; leave the prompt untouched.
    Non-photographic style (the flat-2D default) → the prompt's camera spec
    directly contradicts the look; strip it and apply the style anyway. The
    style is a channel-wide decision and must never lose to one stray line."""
    kind = shot_kind(prompt)
    prompt = strip_shot_tag(prompt)
    tail = _detail_for_shot(kind, names_own_distance(prompt),
                            is_close_shot(prompt))
    if not tail:
        return prompt
    low = prompt.lower()
    has_photo_spec = bool(_PHOTO_SPEC_RE.search(prompt)) or "depth of field" in low
    if has_photo_spec and _is_photographic(tail):
        return prompt
    if has_photo_spec:
        prompt = _strip_photo_direction(prompt)
    body = prompt.rstrip().rstrip('.')
    if shot_last():
        # The shot ONCE, at the end — not bookended. Repeating it would
        # change two things at a time and produce a result that cannot be
        # attributed to either.
        return f"{tail.rstrip().rstrip('.')}. {body}."
    return f"{body}. {tail}"


def _shrink(graph: dict, px: int) -> dict:
    """Set every latent's width and height to `px`, in place on a copy.

    WHY NOT comfy_template.prepare's dims. That branch needs width, height AND
    a length-or-duration together, because it exists for VIDEO nodes. A stills
    graph has width and height on an EmptyLatentImage and no third field, so
    the branch never fires and a size argument would be silently ignored —
    which is exactly the kind of substitution-that-does-nothing this codebase
    keeps getting caught by.

    Only touches nodes that carry BOTH dimensions as plain numbers; a linked
    input ([node, slot]) is a wire and must not be overwritten with an integer.
    """
    out = copy.deepcopy(graph)
    for node in out.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        if isinstance(ins.get("width"), (int, float)) and \
                isinstance(ins.get("height"), (int, float)):
            ins["width"], ins["height"] = px, px
    return out


def _render_image(prompt: str, seed: int, client_id: str,
                  niche: str | None = None, px: int | None = None) -> bytes | None:
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
    if px:
        g = _shrink(g, px)
    pid = _submit(g, client_id)
    if not pid:
        return None
    return _await_image(pid)


CHARACTER_TEMPLATE = Path(__file__).parent.parent / "config" / "character_stills_api.json"


def _character_template() -> dict | None:
    """The exported image-conditioning workflow (IPAdapter/PuLID/etc.) for
    recurring-character stills, or None if it hasn't been exported yet.
    Honors RUFUS_CHARACTER_TEMPLATE=0 as an explicit opt-out even when the
    file exists. Same load/placeholder contract as _stills_template().

    REJECTS a plain img2img graph, which is a different thing wearing the same
    filename. Image conditioning carries the character's IDENTITY while the
    latent still starts from noise, so the scene is whatever the prompt says.
    Img2img makes the reference the STARTING LATENT, so the sampler can only
    redraw it. Live (run #59) this file had the img2img shape at denoise 0.55,
    and all ten beats came back as the same hooded figure standing centred on a
    plain background — prompts asking for miners with pickaxes, a newspaper
    office, a mining camp and a classroom produced none of those. The
    near-duplicate detector fired on all ten and was correct.

    Falling back to the plain stills path costs only the image-level identity
    lock; the text-level character clause still works there, which is how the
    Chronicler appeared in varied scenes before this file existed. Rendering
    the same portrait ten times is strictly worse than that."""
    if os.environ.get("RUFUS_CHARACTER_TEMPLATE", "1").strip().lower() in \
            ("0", "false", "no", "off"):
        return None
    import comfy_template
    tpl = comfy_template.load_template(CHARACTER_TEMPLATE)
    if tpl is None or not comfy_template.has_placeholder(tpl):
        return None
    if not comfy_template.is_image_conditioned(tpl) and \
            comfy_template.starts_from_loaded_image(tpl):
        print(f"[comfy] {CHARACTER_TEMPLATE.name} is a plain img2img graph, not "
              f"an image-conditioning one — the reference is the start latent, "
              f"so every beat would render the reference portrait instead of "
              f"its scene. Ignoring it and using the plain stills path "
              f"(text-level character consistency still applies). To enable "
              f"the identity lock, build a workflow with an IPAdapter/PuLID/"
              f"InstantID node in ComfyUI, verify it, and Export (API) over "
              f"this file — see character_engine.py's header.")
        return None
    return tpl


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


def _insert_px() -> int:
    """Square edge for an insert render. RUFUS_INSERT_PX overrides.

    512 rather than the full still size because an insert occupies 0.42 of
    frame width for under a second — roughly an eighth of the pixels of a beat
    still, which is most of what makes twenty-eight of them affordable. Rounded
    to a multiple of 64: latent dimensions that are not break most samplers.
    """
    raw = os.environ.get("RUFUS_INSERT_PX", "").strip()
    try:
        px = int(raw) if raw else 512
    except ValueError:
        print(f"[inserts] RUFUS_INSERT_PX={raw!r} is not a number — using 512")
        px = 512
    return max(256, (px // 64) * 64)


# ── one beat, on its own ─────────────────────────────────────────────────────

PROMPT_SIDECAR_PREFIX = "FLUX PROMPT:"


def read_beat_prompt(txt_path) -> str:
    """The prompt out of a run's NN.txt sidecar, or "".

    The file is written as "FLUX PROMPT:\n<prompt>\n" beside every still, and
    what it holds is the FINAL prompt — the shot description with the style
    block already appended (see the _with_detail pass before the render loop).
    That is the property the regenerate button rests on: rendering this text
    verbatim reproduces the same request, and an owner who edits it gets
    exactly what they typed rather than their words plus a second helping of
    the style.
    """
    try:
        raw = Path(txt_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    body = raw.split(PROMPT_SIDECAR_PREFIX, 1)[-1] if PROMPT_SIDECAR_PREFIX in raw else raw
    return body.strip()


def write_beat_prompt(txt_path, prompt: str) -> bool:
    """Put an edited prompt back beside its still, in the format the run wrote.

    The sidecar is the run's record of what produced the picture. Leaving it
    stale after a hand-edited regenerate would make the debug folder lie about
    its own contents, which is worse than not recording it at all.
    """
    try:
        Path(txt_path).write_text(
            f"{PROMPT_SIDECAR_PREFIX}\n{prompt.strip()}\n", encoding="utf-8")
        return True
    except OSError as e:
        print(f"[comfy] could not update {txt_path} ({e})")
        return False


def render_one_beat(prompt: str, out_png, *, seed: int | None = None,
                    niche: str | None = None) -> bool:
    """Render exactly one still to `out_png`, at this format's frame size.

    For the review page's per-beat regenerate: one picture, a fresh seed, and
    the file overwritten in place so `review_proxy.contact_sheet` — which
    rebuilds whenever a still is newer than the sheet — refreshes the grid on
    its own.

    Deliberately NOT part of the beat loop. That loop owns beat alignment, the
    i2i chain, the frame gate's retries and the insert budget; reaching into it
    to redo one index would mean teaching all of that to run for a single
    picture. This is the same two calls the loop makes for a plain still, and
    nothing else.

    Returns False rather than raising: a failed regenerate must leave the
    existing still exactly where it was.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        print("[comfy] no prompt to render")
        return False
    if not is_available():
        print(f"[comfy] ComfyUI is not reachable at {_host()}")
        return False
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    raw = _render_image(prompt, seed, uuid.uuid4().hex, niche=niche)
    if not raw:
        print("[comfy] no image came back — the existing still is untouched")
        return False
    # Written to a temp first: _fit_to_frame failing halfway through the real
    # path would leave a truncated png where a good one used to be.
    out_png = Path(out_png)
    staged = out_png.with_suffix(".regen.png")
    if not _fit_to_frame(raw, staged):
        print("[comfy] could not fit the frame — the existing still is untouched")
        return False
    try:
        staged.replace(out_png)
    except OSError as e:
        print(f"[comfy] could not replace {out_png.name} ({e})")
        return False
    print(f"[comfy] regenerated {out_png.name} (seed {seed})")
    return True


def render_inserts(inserts: list[dict], out_dir: Path,
                   niche: str | None = None, base_seed: int | None = None
                   ) -> list[dict]:
    """Draw one small image per planned insert. Returns the plan, annotated.

    SEPARATE FROM THE BEATS ON PURPOSE, and cheap by design. An insert is on
    screen for well under a second at a fraction of frame width, so it is
    rendered small and simple: what has to read is the silhouette, and detail
    at that size is GPU spent on pixels nobody resolves.

    Called while the stills model is ALREADY LOADED, between the beats and the
    /free that precedes any motion engine — twenty-eight extra renders on a
    warm model, not twenty-eight model loads. That ordering is the whole reason
    this format is affordable on a box where loading dominates.

    Fail-open per insert: one that does not render is dropped from the returned
    plan rather than failing the run. The renderer simply shows fewer pictures,
    which is a quieter video and not a broken one.
    """
    if not inserts:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base_seed if base_seed is not None else random.randint(1, 2**31 - 1)
    done: list[dict] = []
    # A dead engine fails every one of these the same way. Three in a row is
    # not bad luck, it is the engine, and forty identical error lines bury the
    # one sentence that explains the whole run.
    misses = 0
    for i, item in enumerate(inserts):
        if misses >= 3:
            print(f"[inserts] 3 failures in a row — stopping after "
                  f"{len(done)}/{len(inserts)}. The image engine is not "
                  f"answering; start ComfyUI (or set RUFUS_INSERTS=0).")
            break
        prompt = str(item.get("prompt") or item.get("word") or "").strip()
        if not prompt:
            continue
        name = f"insert_{i:02d}_{re.sub(r'[^a-z0-9]+', '', str(item.get('word', '')))[:16]}.png"
        path = out_dir / name
        try:
            raw = _render_image(prompt, base + i * 7919, uuid.uuid4().hex,
                                niche=niche, px=_insert_px())
        except Exception as e:
            print(f"[inserts] {item.get('word')!r} failed ({e}) — skipping")
            misses += 1
            continue
        if not raw:
            print(f"[inserts] {item.get('word')!r} produced nothing — skipping")
            misses += 1
            continue
        try:
            path.write_bytes(raw)
        except OSError as e:
            print(f"[inserts] could not write {name} ({e}) — skipping")
            continue
        misses = 0
        done.append({**item, "file": name})
    print(f"[inserts] {len(done)}/{len(inserts)} rendered into {out_dir.name}")
    return done


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

    # STILLS-ONLY MEANS CUT, NOT HOLD. With every motion engine switched off, a
    # beat with one still in it is a photograph held for four to six seconds —
    # QC says so on its own ("2 stretches over 5s without a cut") and a viewer
    # answers it by swiping. The owner ran exactly this and asked why ten
    # prompts reached ComfyUI when they wanted twenty or thirty.
    #
    # `cut` is the answer that was already in this file: three stills per beat
    # on ONE seed, prompted a moment earlier / the moment / a moment later, so
    # the shot advances inside the narration line instead of freezing on it.
    # Ten beats become thirty pictures, each still matched to the sentence it
    # illustrates, and the continuity is stronger rather than weaker because
    # the three frames are the same scene a second apart.
    #
    # Only when nothing else was asked for: an explicit RUFUS_BEAT_MOTION or
    # RUFUS_FRAMES_PER_BEAT still wins, and RUFUS_BEAT_MOTION=kenburns is how
    # to ask for the old one-still-per-beat behaviour back.
    if (not beat_mode and frames_per_beat == 1
            and not _frames_per_beat_was_asked_for()):
        try:
            import svd_client
            if svd_client._stills_only():
                beat_mode = "cut"
                print("[comfy] stills-only: beats will HARD-CUT between "
                      "stills rather than hold one (RUFUS_BEAT_MOTION=kenburns "
                      "for one still per beat)")
        except Exception:
            pass

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
    elif beat_mode == "hero":
        # The hero beat needs ONE still to animate; the rest stay cheap cuts.
        # Resolved per beat further down — this only sets the default for the
        # non-hero majority.
        if frames_per_beat == 1:
            frames_per_beat = _hero_other_frames()
        # Halve the generated clip length unless the owner asked otherwise.
        # The one measured figure for this hardware is ~21-23 min per 480p clip
        # at 30 steps / 121 frames; the exported template is already 12 steps
        # (x0.40) and 61 frames halves the rest (x0.50), which is what puts one
        # clip near the 5-minute target instead of 20. Not a quality cut: every
        # engine freeze-extends its clip to fill the beat's slot
        # (tpad=stop_mode=clone), so 2.5s of real motion then holds, rather
        # than the beat becoming shorter. 61 is a valid 4n+1 count, which
        # Hunyuan's 3D causal VAE requires.
        os.environ.setdefault("RUFUS_HUNYUAN_FRAMES", "61")

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
    elif beat_mode == "hero":
        # Hero mode wants the chain resolved (one beat will use it) while every
        # other beat still cuts between stills, so it must not take the
        # "frames_per_beat > 1 bypasses motion" branch below.
        pass
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
    # Hero mode uses >1 frame on the NON-hero beats while still needing the
    # motion chain resolved for the one beat that gets it, so it is the one
    # mode where frames_per_beat > 1 does not mean "motion off".
    want_motion = frames_per_beat == 1 or beat_mode == "hero"
    if frames_per_beat > 1 and not want_motion:
        # Reuse the existing "why is motion off" plumbing so each engine
        # reports the real reason instead of silently vanishing from the log.
        _stills_only_reason = (
            f"RUFUS_BEAT_MOTION=i2i chains stills instead" if beat_mode == "i2i"
            else f"RUFUS_FRAMES_PER_BEAT={frames_per_beat} cuts between stills instead")
    try:
        import wan_client
        if want_motion and wan_client.enabled():
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
        if want_motion and hunyuan_client.enabled():
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
            # An engine whose template IS exported and whose models ARE loadable,
            # switched off by a flag, is the fail-silent shape this pipeline
            # keeps hitting: "off — disabled" reads identically whether the
            # engine is unusable or one env var away from working. Say which.
            _say_if_ready_but_switched_off("hunyuan 1.5", hunyuan_client)
    except Exception as e:
        print(f"[comfy] hunyuan unavailable ({e})")
    try:
        import ltx_client
        if want_motion and ltx_client.enabled():
            lx_ok, lx_why = ltx_client.ready()
            print(f"[comfy] motion ltx 2.3: {'ON' if lx_ok else 'off'} — {lx_why}")
            if lx_ok:
                motion_engines.append(("ltx", ltx_client.animate_image))
        else:
            print(f"[comfy] motion ltx 2.3: off — disabled "
                  f"({_stills_only_reason or 'RUFUS_LTX=0'})")
    except Exception as e:
        print(f"[comfy] ltx unavailable ({e})")
    # Text-to-video is never added to the motion chain: every entry in that
    # chain is contracted to receive a still, and this engine takes only words.
    # It is used for exactly one beat — the hero — and held here as a handle
    # rather than a chain entry so that contract stays true.
    #
    # WHY ONE BEAT AND NOT ALL OF THEM. Wan 2.2 14B is a mixture-of-experts:
    # two expert models per clip, swapped sequentially, on a 24GB card backed
    # by 16GB of system RAM — so the weights stream from disk every time. That
    # is minutes per clip, and ten of them is an hour of GPU for a 42-second
    # video. The hero beat is the one the architect already identified as a
    # filmable moment, and THE SCENE is a sentence — which is exactly the input
    # a text-to-video model wants. One clip of real generated motion where the
    # video actually turns, stills everywhere else.
    t2v = None
    try:
        import wan_t2v_client
        if wan_t2v_client.enabled():
            t2v_ok, t2v_why = wan_t2v_client.ready()
            print(f"[comfy] text-to-video wan 2.2: "
                  f"{'ON' if t2v_ok else 'off'} — {t2v_why}")
            if t2v_ok:
                t2v = wan_t2v_client
                print(f"[comfy]   seed lineage {wan_t2v_client.run_seed()}, "
                      f"chaining {'on' if wan_t2v_client.chaining() else 'off'} "
                      f"(RUFUS_T2V_CHAIN=1 carries objects between beats)")
                if beat_mode != "hero":
                    print(f"[comfy]   ⚠ RUFUS_BEAT_MOTION={beat_mode} — "
                          f"text-to-video only renders the hero beat. Set "
                          f"RUFUS_BEAT_MOTION=hero to use it.")
        elif want_motion:
            _say_if_ready_but_switched_off("text-to-video wan 2.2", wan_t2v_client)
    except Exception as e:
        print(f"[comfy] text-to-video unavailable ({e})")
    try:
        import svd_client
        if want_motion and svd_client.img2vid_enabled():
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
    # Shot chaining: when the storyboard says a beat continues the last one,
    # generate it FROM the last one's picture instead of from fresh noise —
    # "the same coin, now thinner" cannot be produced by a model that has never
    # seen the coin. Inert unless an edit template is exported (shot_chain.py).
    import shot_chain
    chain_ready, chain_why = shot_chain.ready()
    print(f"[comfy] {chain_why}")
    anchor_png: Path | None = None    # previous beat's RAW model output
    anchor_hash: int | None = None

    # Every rejection this run, so the gate can be judged by a number instead
    # of by opening the folder again — which is the loop it exists to close.
    gate_rejects: list[dict] = []

    stills: list[tuple[int, list[Path], str]] = []   # (beat index, png paths, prompt)
    for i, prompt in enumerate(prompts):
        print(f"[comfy] {i+1}/{len(prompts)}: {prompt}")
        png_path = tmp_dir / f"{stamp}_{i}.png"
        accepted = False
        accepted_raw: bytes | None = None
        accepted_hash: int | None = None
        # The gate's own budget, so a frame rejected twice on quality and once
        # as a duplicate is not silently starved of attempts. With the gate off
        # the range below is exactly what it has always been.
        gate_tries = 0
        gate_hints: list[str] = []
        attempts = MAX_DUP_RETRIES + 1 + (GATE_RETRIES
                                          if frame_gate.enabled() else 0)

        for retry in range(attempts):
            # %(2**31) keeps the seed in range for any backend; offset per clip/retry.
            seed  = (master_seed + i + 1000 * retry) % (2**31 - 1)
            # Only the first attempt chains. A retry means the chained result
            # was unusable, so falling back to a fresh render is the point.
            chained = False
            img_bytes = None
            if chain_ready and retry == 0 and anchor_png is not None:
                img_bytes = shot_chain.continue_shot(
                    anchor_png, prompt, seed, client_id,
                    negative=_stills_negative())
                if img_bytes:
                    chained = True
                    print(f"[comfy] clip {i+1} continued from clip {i}: "
                          f"{shot_chain.carried(prompt)}")
            if img_bytes is None:
                # The gate's notes ride on the prompt for the attempts that
                # follow a rejection, and only those — the first attempt of
                # every clip is the prompt the storyboard wrote.
                attempt_prompt = prompt
                if gate_hints:
                    attempt_prompt = f"{prompt} {' '.join(gate_hints)}"
                img_bytes = _render_image(attempt_prompt, seed, client_id,
                                          niche=niche)
            if not img_bytes:
                # A hard generation error (vs. a plain duplicate) is often a
                # transient GPU/model-loading hiccup on the ComfyUI side —
                # a short pause before resubmitting gives it a chance to clear
                # instead of hammering the same broken state 3x back-to-back.
                time.sleep(GEN_ERROR_BACKOFF)
                continue

            if not _fit_to_frame(img_bytes, png_path):  # → exactly 1080×1920
                continue

            # THE RE-ROLL A PERSON DOES BY HAND. Until this, the only thing
            # that could reject a frame was the duplicate check below — so a
            # six-panel contact sheet, or a figure on blank paper, or a picture
            # that does not show what its prompt asked for, was accepted on the
            # first attempt. Rejecting here reuses the retry this loop already
            # runs, with the seed it already offsets.
            gate_failed = ""
            if frame_gate.enabled():
                gate_ok, gate_why, gate_detail = frame_gate.check(
                    png_path, prompt=prompt)
                # CHECKED EVEN ON THE LAST ATTEMPT, when there is no re-roll
                # left to spend. Otherwise the final frame of every exhausted
                # clip is unexamined, and the run cannot tell "it failed three
                # times" from "the third one was fine".
                if not gate_ok:
                    gate_failed = gate_why
                if not gate_ok and gate_tries < GATE_RETRIES:
                    gate_tries += 1
                    # SAY WHAT WAS WRONG. A re-roll with the same prompt and a
                    # new seed is the same prompt; the hint is what makes the
                    # next attempt different in the way that matters.
                    hint = frame_gate.retry_hint(gate_why, gate_detail)
                    if hint:
                        gate_hints.append(hint)
                    print(f"[gate] clip {i+1}: {gate_why} "
                          f"({gate_detail}) → re-rolling")
                    gate_rejects.append({"clip": i + 1, "reason": gate_why,
                                         "detail": gate_detail})
                    continue

            h = _avg_hash(png_path)
            if chained and h is not None and anchor_hash is not None \
                    and _hamming(h, anchor_hash) < CHAIN_COPY_THRESHOLD:
                # The edit template redrew its input instead of editing it —
                # the img2img failure mode, caught here rather than shipped.
                print(f"[comfy] chained clip {i+1} came back as a copy of clip "
                      f"{i} — rendering it fresh instead")
                continue
            # A chained shot is SUPPOSED to resemble its predecessor; the copy
            # check above is its gate, so the near-dup gate would only undo it.
            is_dup = (h is not None and not chained
                      and _is_duplicate(h, accepted_hashes, n_prior))
            # AGAINST THE ATTEMPTS LEFT, not against MAX_DUP_RETRIES. `retry`
            # is the loop index, and the gate's rejections have already spent
            # some of it — comparing to the duplicate constant would let two
            # gate re-rolls starve the duplicate check of its own budget, which
            # is the starvation the gate's separate counter exists to prevent.
            # With the gate off, attempts - 1 IS MAX_DUP_RETRIES.
            if is_dup and retry < attempts - 1:
                print(f"[comfy] dup on clip {i+1} → regen (retry {retry+1})")
                continue
            if is_dup:
                print(f"[comfy] clip {i+1} still near-dup after retries — keeping")
            if gate_failed:
                # KEPT, NOT SKIPPED, for the same reason the duplicate path
                # keeps one: clip lists are positional, and dropping a frame
                # would make every later picture narrate the wrong sentence.
                # But a frame that is still failing after its whole budget is
                # the one to go and look at, so it says so.
                print(f"[comfy] ⚠ clip {i+1} is still {gate_failed} after "
                      f"{gate_tries} re-roll(s) — keeping it anyway")
                gate_rejects.append({"clip": i + 1, "reason": gate_failed,
                                     "detail": "kept after the budget ran out"})
            if h is not None:
                accepted_hashes.append(h)
            accepted = True
            accepted_raw, accepted_hash = img_bytes, h
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

        # What the NEXT beat continues from, when it says it continues something.
        # The raw model output, not png_path: _fit_to_portrait upscales and
        # crops, and re-feeding that would re-resample on every link, so the
        # degradation would compound down the whole video instead of stopping
        # at one beat. On a reused still there is no new raw output, so the last
        # real one stays the anchor rather than the chain breaking.
        if chain_ready and accepted_raw:
            anchor_png = tmp_dir / f"{stamp}_{i}_raw.png"
            anchor_png.write_bytes(accepted_raw)
            anchor_hash = accepted_hash

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
            if sub_bytes and _fit_to_frame(sub_bytes, sub_path):
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

    # THE GATE'S OWN RECORD, beside the frames it judged. Written whenever the
    # gate ran, including with nothing to report — an empty list is the useful
    # answer ("it ran and found nothing") and a missing file is not.
    if frame_gate.enabled():
        try:
            (debug_dir / "gate.json").write_text(
                json.dumps({"rejects": gate_rejects,
                            "frames": len(prompts)}, indent=2),
                encoding="utf-8")
            if gate_rejects:
                print(f"[gate] {len(gate_rejects)} rejection(s) across "
                      f"{len(prompts)} frame(s) — see {debug_dir.name}/gate.json")
        except OSError as e:
            print(f"[gate] could not write gate.json ({e})")

    # ── Phase 2: animate every still ────────────────────────────────────────
    # Two phases instead of image→animate per clip: interleaving forced
    # ComfyUI to swap FLUX in and out for EVERY clip (10 model swaps per
    # video on a 24GB card that can't hold both), thrashing RAM/VRAM and —
    # per a ComfyUI-reliability audit — degrading until every clip silently
    # fell through to Ken Burns. Batching stills first means exactly ONE
    # switch, with an explicit /free between so the motion model loads into
    # a clean card.
    # Hero mode: resolve WHICH beat gets the motion clip, once, before the loop.
    # None means no beat does — the architect found no filmable moment, or none
    # of the shots is about it — and the run is then exactly the stills run the
    # pipeline produces today.
    hero_i: int | None = None
    hero_scene = ""
    t2v_world = ""
    # `or t2v` because text-to-video needs no still to animate, so a run with a
    # T2V template and no image-to-video engine still has a hero beat to make.
    if beat_mode == "hero" and (motion_engines or t2v):
        try:
            import script_writer
            hero_scene = getattr(script_writer, "LAST_SCENE", "") or ""
        except Exception:
            hero_scene = ""
        hero_i = _hero_beat([p for _, _, p in stills], hero_scene)
        if hero_i is None:
            print("[comfy] hero: no beat matches the scene — all beats stay "
                  "stills (no filmable moment in this source)")
        else:
            via = "text-to-video" if t2v else "motion"
            print(f"[comfy] hero: beat {hero_i + 1} gets the {via} clip")
            print(f"[comfy]   scene: {hero_scene[:100]}")
        if t2v and hero_i is not None:
            # Built once and reused byte for byte — see build_world. The style
            # and character text is the SAME text the stills were rendered
            # with, so the one moving beat belongs to the same world as the
            # nine still ones instead of being a visitor from another render.
            try:
                import character_engine
                char = (character_engine.short_ref(niche)
                        if character_engine.enabled(niche) else "")
            except Exception:
                char = ""
            t2v_world = t2v.build_world([p for _, _, p in stills], char,
                                        _detail_suffix())

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

        # Hero mode animates exactly one beat. Skipping the chain here rather
        # than earlier keeps every other beat on the identical stills path it
        # already takes, so nothing about the other eight changes.
        beat_engines = motion_engines
        if beat_mode == "hero":
            beat_engines = motion_engines if i == hero_i else []

        made_via = None
        motion_secs = 0.0
        tried: list[str] = []

        # Text-to-video gets first refusal on the hero beat, ahead of the
        # image-to-video chain. It is generating this beat from THE SCENE
        # rather than from the still, so it can stage an action the still
        # never contained — the still froze one instant of the moment, the
        # sentence describes the whole of it. When it declines or fails, the
        # beat drops into the ordinary chain below with its still intact and
        # nothing about the run changes.
        if t2v is not None and beat_mode == "hero" and i == hero_i:
            t0 = time.time()
            ok_t2v = t2v.generate_clip(hero_scene or prompt, clip_path,
                                       duration=clip_duration, idx=i,
                                       world=t2v_world)
            secs = time.time() - t0
            rec = {"beat": i + 1, "engine": "wan_t2v", "ok": bool(ok_t2v),
                   "seconds": round(secs, 1)}
            try:
                rec.update(t2v.LAST_CALL)
            except Exception:
                pass
            motion_log.append(rec)
            tried.append(f"wan_t2v {'ok' if ok_t2v else 'failed'} in {secs:.0f}s")
            if ok_t2v:
                made_via, motion_secs = "wan_t2v", secs
                beat_engines = []
            else:
                print(f"[comfy] text-to-video declined beat {i+1} after "
                      f"{secs:.0f}s — falling back to the still")

        for eng_name, animate in beat_engines:
            # THE SCENE names the ACTION; the image prompt names the
            # composition. For the one beat built on that moment the action is
            # the better motion prompt — it is why this beat was chosen.
            motion_text = hero_scene if (beat_mode == "hero" and hero_scene) else prompt
            t0 = time.time()
            okd = animate(png_path, clip_path, duration=clip_duration, idx=i,
                          prompt=motion_text)
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
                motion_secs = secs
                break
            print(f"[comfy] {eng_name} failed for clip {i+1} — trying next engine")
        made = made_via is not None or _animate_to_clip(png_path, clip_path,
                                                        duration=clip_duration, idx=i)
        if made_via is None and beat_engines:
            motion_log.append({"beat": i + 1, "engine": "kenburns", "ok": bool(made),
                               "note": "all motion engines declined/failed"})
        if made:
            clips.append(clip_path)
            # The seconds were always measured into motion_log but never shown,
            # so nobody could tell a 5-minute clip from a 20-minute one without
            # opening the run report. That number is the one that decides the
            # next tuning move (frames, steps, or accepting the fixed
            # weight-streaming cost on a 16GB box).
            took = f", {motion_secs:.0f}s" if made_via and motion_secs else ""
            print(f"[comfy] clip {i+1} ready"
                  + (f" ({made_via} motion{took})" if made_via else " (Ken Burns)"))
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


def _regen_beat_cli(argv: list[str]) -> int:
    """`--regen-beat`: redraw one still, as its own process.

    THE DASHBOARD CANNOT DO THIS ON A REQUEST THREAD. regen_beat called
    render_one_beat() inline, and a GPU render is tens of seconds to minutes —
    which under the old threaded=False server froze the whole dashboard, and
    even now means a browser tab holding an open connection while /healthz
    cannot answer. The watchdog reads that as death, tries to start a second
    dashboard, and hits the port the first one is still holding: the exact
    ten-hour outage, triggered by pressing Regen.

    THE PROMPT ARRIVES IN A FILE, not in argv. Beat prompts run to several
    hundred characters, Windows caps a command line at 32,767, and quoting a
    multi-line prompt through cmd is a source of bugs nobody needs. The file
    is deleted once it has been read.

    THE SIDECAR IS WRITTEN ONLY ON SUCCESS, which is the semantic the inline
    version had: a failed regenerate must leave both the picture and the run's
    record of it exactly as they were.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="comfy_client --regen-beat")
    ap.add_argument("--regen-beat", action="store_true")
    ap.add_argument("--out", required=True, help="the PNG to overwrite")
    ap.add_argument("--prompt-file", required=True,
                    help="file holding the prompt (deleted after reading)")
    ap.add_argument("--sidecar", default="",
                    help="write the prompt here on success")
    ap.add_argument("--niche", default="")
    args = ap.parse_args(argv)

    src = Path(args.prompt_file)
    try:
        prompt = src.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"[comfy] could not read the prompt: {e}")
        return 1
    finally:
        src.unlink(missing_ok=True)

    if not render_one_beat(prompt, Path(args.out), niche=args.niche or None):
        return 1
    if args.sidecar:
        try:
            write_beat_prompt(Path(args.sidecar), prompt)
        except Exception as e:
            print(f"[comfy] redrew the frame but could not save the prompt: {e}")
    print(f"OUTPUT={args.out}")
    return 0


if __name__ == "__main__":
    import sys
    if "--regen-beat" in sys.argv[1:]:
        raise SystemExit(_regen_beat_cli(sys.argv[1:]))
    # STILLS-ONLY BY DEFAULT when run by hand. RUFUS_STILLS_ONLY=1 is set by
    # run.bat, not by this module, so a one-prompt check from the CLI used to
    # fall straight through to the motion chain — on a 16GB-RAM box that turned
    # "does this prompt look right?" into a four-minute sample plus a VAE decode
    # that can run for over an hour. Nobody testing a prompt wants a video.
    # Ask for one explicitly with RUFUS_STILLS_ONLY=0.
    os.environ.setdefault("RUFUS_STILLS_ONLY", "1")
    qs = sys.argv[1:] or ["modern luxury kitchen interior, golden hour light, wide angle",
                          "sunlit living room, floor to ceiling windows, city view"]
    for p in generate_clips(qs, n=len(qs)):
        print(f"CLIP={p}")
