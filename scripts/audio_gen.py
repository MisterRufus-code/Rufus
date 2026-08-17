#!/usr/bin/env python3
"""Rufus – Autonomous Shorts Renderer (v4.0)

Changes from v3.0 — "cinematic edit" upgrade:
- Cuts snap to SENTENCE BOUNDARIES from Whisper word timestamps (editor-grade
  pacing) with a short punchy first cut (~2-4s) for the hook pattern-interrupt.
- Sound design: synthesized SFX layer (sub-bass hit on the hook, bubble on
  every cut, riser into the final beat) — see sfx_gen.py, zero APIs.
- Music is DUCKED DYNAMICALLY under the voice via sidechaincompress (breathes
  back up in speech gaps) instead of a fixed low volume.
- Voice chain: highpass + compressor + presence EQ — Edge TTS sounds studio.
- Final mix mastered to -14 LUFS (loudnorm), YouTube's reference loudness.
- Retention progress bar along the bottom edge, in the niche accent color.
- Caption highlights use the per-niche accent color and also fire on opinion
  words (worst/never/secret…), not just digits.

Fallback ladder preserved: xfade+full-mix → hard-concat+simple-mix, so a
render always completes even on minimal FFmpeg builds.
"""

import argparse
import functools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import paths

from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).parent))

# music_fetcher / sfx_gen are optional layers — renderer degrades gracefully.
try:
    from music_fetcher import fetch_music as _fetch_music
except Exception:
    def _fetch_music(niche: str):           # type: ignore[misc]
        return None

try:
    from sfx_gen import ensure_sfx as _ensure_sfx
except Exception:
    def _ensure_sfx():                       # type: ignore[misc]
        return {}

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
FONTS_DIR  = ROOT / "assets" / "fonts"

# The render's shape, from the active format profile — 1080×1920 for a
# Short, 1920×1080 for long-form. Read at import: a run does not change
# format halfway through, and every consumer below wants plain ints.
import video_format as _vf
W, H         = _vf.dimensions()
FPS          = 30
# THE TIMELINE CEILING, AND IT TRUNCATES. Line ~1225 clamps the transcribed
# duration to MAX_DUR, so this is not a warning threshold — everything past it
# is cut off mid-sentence, after the voice has been generated and paid for.
# Fixed at 60.0 it was the Shorts cap, and a nine-minute narration would have
# come out sixty seconds long: QC would then have called the file broken for
# being 60s when the format wanted 240-1500, which is the truthful complaint
# about entirely the wrong thing.
MAX_DUR      = float(_vf.get("render_max_s", 60.0))
MIN_DUR      = float(_vf.get("render_min_s", 30.0))
# Words per caption, and whether it shouts. 1 and uppercase is the Hormozi
# Shorts style and stays exactly that for Shorts; see video_format for why a
# nine-minute explainer takes phrases in natural case instead.
CLUSTER_SIZE = max(1, int(_vf.get("caption_words", 1)))
CAPTION_UPPER = bool(_vf.get("caption_upper", True))
# The colour sweep along the bottom edge — a retention device that competes
# with YouTube's own scrubber on anything long enough to have one.
RETENTION_BAR = bool(_vf.get("retention_bar", True))

XFADE_DUR    = 0.30        # crossfade duration between clips (seconds)
FADE_IN      = 0.0         # fade-in duration — 0 = hard cut (top Shorts open cold)
FADE_EDGE    = 0.40        # fade-to-black at end duration (seconds)
MUSIC_VOL    = 0.14        # static music volume (simple-mix fallback path)
MUSIC_BED    = 0.30        # music bed volume BEFORE sidechain ducking (full mix)
BAR_HEIGHT   = 14          # retention progress bar thickness (px)

# WORD-SYNCED INSERT LAYER (see insert_director.py). Small pictures that pop in
# on the second their phrase is spoken, over the beat clip and UNDER the
# captions. Geometry only — which pictures and when is planned elsewhere.
INSERT_W      = _vf.get("insert_w", 460)   # per-format; see video_format
INSERT_MARGIN = 70         # gap from the frame edge
# Cycled so consecutive inserts don't stack. Kept as FRACTIONS of the frame
# height rather than the pixels they used to be: 300, 560 and 430 put every
# insert in the upper third of a 1920-tall Short, safely above the caption
# band — and on a 1080-tall landscape frame the same three numbers put one at
# mid-picture. Unlike the caption size, this one really is proportional: the
# rule is "upper third, three staggered rows", and that rule is the same
# whatever the frame. The Shorts pixels come back out exactly.
INSERT_Y_FRACTIONS = (300 / 1920, 560 / 1920, 430 / 1920)
INSERT_YS     = tuple(round(f * H) for f in INSERT_Y_FRACTIONS)

# RUFUS_SFX=0 drops the whole synthesized layer (hit/bubble/riser) — the
# lever for "these effects don't belong on this channel" without touching
# three separate gain values. Default stays on for anyone who hasn't
# formed an opinion either way.
SFX_ENABLED = os.environ.get("RUFUS_SFX", "1").strip().lower() not in ("0", "false", "no", "off")

# SFX layer gains (relative, 0-1). The cut sound plays on EVERY cut (up to 23x per
# video) so it went through five rounds of channel-owner feedback pushing it
# down to near-inaudible — hit and riser each play only ONCE per video and
# were never tuned the same way, on the (wrong) assumption that "once" meant
# "unlikely to bother anyone." Live feedback: the sub-bass hit at full volume
# 0.03s into every single video and the riser's second-long swell before the
# payoff read as "inappropriate background noise" on a history/finance
# channel — a jump-scare boom doesn't suit the tone. Halved both and made
# all three env-tunable the same way, so any future adjustment (including
# "0" on an individual layer) needs no code change.
SFX_HIT_GAIN    = float(os.environ.get("RUFUS_HIT_GAIN", "0.45"))     # sub-bass hit on the hook (once, 0.03s in)


def _bubble_gain() -> float:
    """Gain for the cut sound, honouring the old whoosh variable out loud.

    The whoosh it replaces sat at 0.02 — five rounds of owner feedback pushed
    it to near-inaudible, which was right for a sheet of filtered noise playing
    on every cut. A bubble is a different animal: one short rounded tone, the
    same drawing style as the pictures, and at 0.02 it would simply not be
    there. 0.05 is felt without stepping on the narration.

    RUFUS_WHOOSH_GAIN still works, because someone who tuned that number is
    tuning THIS layer, and silently ignoring their setting would be the worst
    of both. It says so when it does.
    """
    raw = os.environ.get("RUFUS_BUBBLE_GAIN", "").strip()
    if not raw:
        legacy = os.environ.get("RUFUS_WHOOSH_GAIN", "").strip()
        if legacy:
            print(f"[sfx] RUFUS_WHOOSH_GAIN={legacy} — the whoosh is gone; "
                  f"using it for the bubble (RUFUS_BUBBLE_GAIN renames it)")
            raw = legacy
    try:
        return float(raw) if raw else 0.05
    except ValueError:
        print(f"[sfx] bubble gain {raw!r} is not a number — using 0.05")
        return 0.05


SFX_BUBBLE_GAIN = _bubble_gain()   # bubble into each cut
SFX_RISER_GAIN  = float(os.environ.get("RUFUS_RISER_GAIN", "0.28"))   # riser leading into the final beat (once)

# Cut planning
FIRST_CUT_MIN = 2.0        # hook cut window — research: pattern interrupt by ~3s
FIRST_CUT_MAX = 4.2
SNAP_WINDOW   = 2.0        # max distance a cut may move to land on a sentence end
# MINIMUM SHOT LENGTH. Raised from 1.2s after a real 24-picture run came out
# machine-gun: thirteen of its twenty-four shots sat EXACTLY on this floor
# (6.3s, 7.5s, 8.7s — 1.2 apart to the frame), which is not an edit, it is a
# clamp. The planner had more pictures than the narration had pauses and spent
# the remainder at the minimum. Below about 1.5s a picture reads as a flash
# rather than a shot, so the floor now sits where a shot starts being one.
# The floor is per-FORMAT: an explainer holding ~3.5s a picture wants a calmer
# 2.5s minimum, and one number cannot be right for both.
MIN_SEG       = _vf.get("min_seg_s", 1.6)

WHITE = "&H00FFFFFF"
GREEN = "&H0000FF00"

_HIGHLIGHT_RE = re.compile(r'[\d$%]')
_SENT_END_RE  = re.compile(r'[.!?…]["\')\]]*$')
# Words whose trailing period is an abbreviation, not a sentence end — without
# this guard, "the U.S. dollar" put a scene cut + cut SFX mid-sentence.
_ABBREV_RE    = re.compile(
    r'^(mr|mrs|ms|dr|st|vs|etc|inc|co|jr|sr|prof|gen|col|sgt|no'
    r'|[a-z](\.[a-z])+)\.$', re.IGNORECASE)

FONT_NAME = "Anton"        # downloaded to assets/fonts/; Arial fallback if missing
FONT_FILE = FONTS_DIR / "Anton-Regular.ttf"
# CAPTIONS ARE PER-FORMAT, not per-pipeline. 140px and MarginV 600 are right
# for a phone held at arm's length with the Shorts UI covering the bottom
# fifth; on a 1080-tall landscape frame the same numbers are 13% of the height
# with the words sitting halfway up the picture. See video_format.PROFILES.
FONTSIZE  = _vf.get("caption_size", 140)
MARGIN_V  = _vf.get("caption_margin_v", 600)

# Hard ceiling for a single ffmpeg render pass — a hung/looping ffmpeg must
# fail the attempt (and fall through to the simple pipeline), never freeze an
# autonomous cron run forever. Override with RENDER_TIMEOUT (seconds).
RENDER_TIMEOUT = int(os.environ.get("RENDER_TIMEOUT", "600"))

DEFAULT_ACCENT = "#FFD23F"   # warm gold — used when a niche has no accent_color

# Cut timestamps of the most recent render, for QC's pacing check. A module
# global rather than a return value because render()'s signature is shared with
# the Remotion path and every caller in the tree; this is diagnostic output,
# not a result anyone renders from.
LAST_CUTS: list[float] = []

# The word stream of the voice that actually shipped, as (start_seconds, word).
# Same reasoning as LAST_CUTS, and the same shape of consumer: chapters.py
# finds each section's real start in here rather than dividing the runtime by
# the number of sections. It has to be the RENDERED audio, not the script — a
# TTS engine that drops or merges a word makes every estimate after it late,
# and a chapter mark thirty seconds off is a promise the video breaks.
LAST_WORDS: list[tuple[float, str]] = []


# ── Font bootstrap ───────────────────────────────────────────────────────────────

def _ensure_font() -> str:
    """Download Anton-Regular.ttf if not present. Return display name to use in ASS."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    if FONT_FILE.exists() and FONT_FILE.stat().st_size > 50_000:
        return FONT_NAME

    url = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
    try:
        import requests
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        FONT_FILE.write_bytes(r.content)
        if FONT_FILE.stat().st_size > 50_000:
            print(f"[font] Anton downloaded → {FONT_FILE}")
            return FONT_NAME
    except Exception as e:
        print(f"[font] Anton download failed ({e}) — using Arial")
        FONT_FILE.unlink(missing_ok=True)
    return "Arial"


# ── FFmpeg capability detection ──────────────────────────────────────────────────

# GPU mode: set RUFUS_GPU=1 on a cloud instance so Whisper uses CUDA and FFmpeg uses
# NVENC. Defaults to CPU so the same code runs unchanged on Daniel's home machine.
_GPU = os.environ.get("RUFUS_GPU", "").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=8)
def _ffmpeg_has_filter(name: str) -> bool:
    """Return True if the installed FFmpeg includes the given filter."""
    try:
        r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        return name in r.stdout
    except Exception:
        return False


def _ffmpeg_has_xfade() -> bool:
    return _ffmpeg_has_filter("xfade")


@functools.lru_cache(maxsize=1)
def _ffmpeg_has_nvenc() -> bool:
    """Return True if the installed FFmpeg includes the h264_nvenc encoder."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=10)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def _video_encoder_args() -> list[str]:
    """Pick the H.264 encoder: NVENC on GPU instances, libx264 on CPU.

    This is the DELIVERY encode — the one YouTube ingests and re-compresses.
    Feeding YouTube's transcoder a higher-quality master measurably improves
    what viewers see after its re-encode, so spend time here: p7 is NVENC's
    slowest/highest-quality preset (still fast on a 3090), cq/crf ~18-19
    instead of 23/20. Intermediates upstream are near-lossless (crf 14) so
    this is the only real compression the pixels go through on our side."""
    if _GPU and _ffmpeg_has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "19"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]


# ── Whisper singleton ────────────────────────────────────────────────────────────

# The DLLs ctranslate2 needs for GPU transcribe, by the exact names it asks for.
# Checked by name because "the directory registered" and "the file is there" are
# different facts, and only the second one predicts whether CUDA will work.
_REQUIRED_CUDA_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")


def _is_windows() -> bool:
    """Platform check behind a function so tests can flip it without touching
    the stdlib `os` module — patching os.name globally also changes what
    pathlib.Path() constructs, which breaks pytest's own tmp-dir cleanup."""
    return os.name == "nt"


def _register_dll_dir(path: str) -> None:
    """Make `path` searchable for DLLs, by BOTH mechanisms Windows uses.

    os.add_dll_directory alone is not enough here, and that is why the previous
    fix looked correct and changed nothing. It only affects DLLs loaded through
    LoadLibraryEx with the LOAD_LIBRARY_SEARCH_* flags. cublas64_12.dll is not
    loaded that way — it is an IMPLICIT dependency of ctranslate2's own DLL, and
    Windows resolves those through the standard search order, which consults
    PATH and never consults the add_dll_directory list.

    The live signature of exactly that: every diagnostic in _add_nvidia_dll_dirs
    stayed silent (so the directories registered fine), WhisperModel(device=
    "cuda") constructed fine and printed "CUDA / float16 (GPU mode)", and then
    the first transcribe failed with "Library cublas64_12.dll is not found or
    cannot be loaded". Registration succeeded; the loader was never going to
    look there.

    So do both: add_dll_directory for anything loaded explicitly, and prepend to
    PATH for the implicit dependency resolution that actually matters here.
    """
    os.add_dll_directory(path)
    current = os.environ.get("PATH", "")
    if path not in current.split(os.pathsep):
        os.environ["PATH"] = path + os.pathsep + current


def _add_nvidia_dll_dirs() -> list[str]:
    """Windows: make pip-installed CUDA runtime DLLs visible to ctranslate2.

    GPU Whisper needs cuBLAS/cuDNN. Instead of the multi-GB CUDA Toolkit, the
    runtime DLLs install via `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
    into site-packages/nvidia/<lib>/bin — but Windows won't find them there
    unless we register the directories. torch registers its own bundled copies
    on import; ctranslate2 does not. No-op on Linux and when not installed.

    WHY THIS SILENTLY DID NOTHING. `nvidia` is a NAMESPACE package, so
    nvidia.__file__ is None and Path(None) raises TypeError — which the old
    bare `except Exception: pass` swallowed whole. The live result: pip
    reported nvidia-cublas-cu12 12.9.2.10 and nvidia-cudnn-cu12 9.24.0.43
    "already satisfied" while whisper kept printing "Library cublas64_12.dll
    is not found or cannot be loaded" and transcribed on CPU for weeks, and
    the advice that followed was to install what was already installed.
    Namespace packages expose __path__, never __file__.

    Returns the directories registered, so a caller (and a test) can tell
    "nothing to do" apart from "tried and failed".
    """
    if not _is_windows():
        return []

    roots: list[Path] = []
    try:
        import nvidia
        roots += [Path(p) for p in getattr(nvidia, "__path__", [])]
    except ImportError:
        pass

    # torch ships its OWN copy of the same CUDA runtime in torch/lib, and torch
    # is already installed here for Kokoro. When the nvidia wheels are absent,
    # misnamed, or a version ctranslate2 does not want, this is a second real
    # source of cublas64_12.dll rather than a suggestion to install something.
    try:
        import torch
        roots += [Path(p) / "lib" for p in getattr(torch, "__path__", [])]
    except ImportError:
        pass

    if not roots:
        print("[whisper] no pip CUDA runtime installed — "
              "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 for GPU transcribe")
        return []

    added: list[str] = []
    for root in roots:
        # Layout varies by wheel: <lib>/bin on most, <lib>/bin/<arch> on some
        # Windows builds, and torch/lib is flat. Register any directory that
        # actually holds DLLs.
        candidates = [root] + sorted(root.glob("*/bin")) + sorted(root.glob("*/bin/*"))
        for d in candidates:
            if not d.is_dir() or not any(d.glob("*.dll")):
                continue
            if str(d) in added:
                continue
            try:
                _register_dll_dir(str(d))
                added.append(str(d))
            except OSError as e:
                print(f"[whisper] could not register {d} ({e})")

    if not added:
        print("[whisper] CUDA packages present but no DLL directory found — "
              "GPU transcribe will fall back to CPU")
        return []

    # Say whether the ONE DLL that keeps failing is actually reachable now.
    # Registering directories and then failing anyway is what wasted weeks
    # here: the fix reported success while the loader still could not find the
    # file, and nothing in the log distinguished "registered the wrong place"
    # from "the file is not installed at all".
    missing = [dll for dll in _REQUIRED_CUDA_DLLS
               if not any((Path(d) / dll).exists() for d in added)]
    if missing:
        print(f"[whisper] {', '.join(missing)} not present in any registered "
              f"directory — GPU transcribe will fail. Searched: "
              f"{'; '.join(added[:4])}")
    return added


_whisper_model  = None
_whisper_device = None   # "cuda" or "cpu" — tracks what the singleton actually is

def _whisper(force_cpu: bool = False) -> WhisperModel:
    global _whisper_model, _whisper_device
    if _whisper_model is None or (force_cpu and _whisper_device == "cuda"):
        # "small" (~244M params) vs "base" (~74M): measurably better word accuracy
        # and sentence boundaries at ~2x CPU time. RUFUS_WHISPER_MODEL=base is the
        # low-RAM escape hatch (halves CPU-mode memory) for constrained machines.
        model_name = os.environ.get("RUFUS_WHISPER_MODEL", "small").strip() or "small"
        if _GPU and not force_cpu:
            _add_nvidia_dll_dirs()
            try:
                _whisper_model  = WhisperModel(model_name, device="cuda", compute_type="float16")
                _whisper_device = "cuda"
                print(f"[whisper] CUDA / float16 (GPU mode) — {model_name} model")
                return _whisper_model
            except Exception as e:
                print(f"[whisper] CUDA init failed ({e}) — falling back to CPU")
        _whisper_model  = _load_whisper_cpu(model_name)
        _whisper_device = "cpu"
    return _whisper_model


def _load_whisper_cpu(model_name: str) -> WhisperModel:
    """CPU model with an offline retry: constructing WhisperModel re-checks
    HuggingFace for the model revision, and a transient network blip there
    ('Server disconnected without sending a response', seen live) killed a
    whole render — even though the model files were already cached on disk
    (the CUDA attempt had loaded them seconds earlier). On any load failure,
    retry from the local cache only; if the model genuinely isn't cached,
    that retry raises the real error."""
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"[whisper] CPU model load failed ({e}) — retrying from local cache")
        return WhisperModel(model_name, device="cpu", compute_type="int8",
                            local_files_only=True)


def _transcribe(mp3: Path):
    """Transcribe with automatic CPU fallback.

    ctranslate2 lazy-loads its CUDA backend (cuBLAS/cuDNN) — a missing DLL
    (e.g. cublas64_12.dll when only the CUDA runtime bundled with another app
    like ComfyUI is present, not the system-wide CUDA Toolkit) only surfaces
    on the FIRST actual transcribe() call, not at WhisperModel() construction.
    So the GPU→CPU fallback has to wrap this call too, not just model init.
    """
    global _whisper_model
    try:
        return _whisper().transcribe(str(mp3), word_timestamps=True)
    except Exception as e:
        if _whisper_device == "cuda":
            print(f"[whisper] CUDA transcribe failed ({e}) — retrying on CPU")
            if "cublas" in str(e).lower() or "cudnn" in str(e).lower():
                # The runtime DLLs are a 2-minute pip install away — say so,
                # instead of silently eating the ~4x CPU transcribe penalty
                # forever.
                print("[whisper]   to enable GPU transcribe: "
                      "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12  (then rerun)")
            _whisper_model = None   # discard the broken CUDA instance
            return _whisper(force_cpu=True).transcribe(str(mp3), word_timestamps=True)
        raise


# ── Config ───────────────────────────────────────────────────────────────────────

def _load_niche() -> dict:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text(encoding="utf-8"))
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active]


def _active_niche_name() -> str:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text(encoding="utf-8"))
    return os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]


def _hex_to_ass(hex_color: str) -> str:
    """'#RRGGBB' → ASS '&H00BBGGRR' (ASS is BGR-ordered)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return GREEN
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


@functools.lru_cache(maxsize=1)
def _opinion_words() -> frozenset:
    """Opinion words from script_standards.json, uppercased for caption matching."""
    try:
        std = json.loads((CONFIG_DIR / "script_standards.json").read_text(encoding="utf-8"))
        return frozenset(w.upper() for w in std.get("opinion_pool", []))
    except Exception:
        return frozenset()


# ── ASS subtitle builder ─────────────────────────────────────────────────────────

def _ts(sec: float) -> str:
    """ASS timestamp. Integer centisecond math, NOT float %-formatting:
    `59.998 % 60` formatted with %05.2f produced '0:00:60.00' — an invalid
    timestamp libass silently mishandles — and start/end could round to the
    same string (a zero-length event). Carrying centiseconds through divmod
    rolls 59.998s over to 0:01:00.00 correctly."""
    cs = max(0, round(sec * 100))
    h,  rem = divmod(cs, 360000)
    m,  rem = divmod(rem, 6000)
    s,  cs  = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# Caption display tuning. Whisper word timestamps end each word the instant
# the sound stops — showing captions for exactly that window makes every
# inter-word pause a blank screen (strobing, the #1 "dirty captions" look).
CAPTION_GAP_FILL_S = 1.5    # max seconds a caption may linger past its word to
                            # bridge the silence to the next word
CAPTION_MIN_S      = 0.12   # minimum readable display time for any caption


def _cluster_words(segments, audio_dur: float):
    words = [w for seg in segments for w in seg.words]
    n = len(words)
    for i in range(0, n, CLUSTER_SIZE):
        group = words[i:i + CLUSTER_SIZE]
        start = group[0].start
        end   = group[-1].end
        if start >= audio_dur:
            break
        nxt = words[i + CLUSTER_SIZE].start if i + CLUSTER_SIZE < n else audio_dur
        # Bridge the gap to the next caption (kills strobing) without ever
        # overlapping it, and without lingering forever across a long pause.
        end = max(end, min(nxt, end + CAPTION_GAP_FILL_S))
        # Enforce a readable minimum, still capped at the next caption's start.
        if end - start < CAPTION_MIN_S:
            end = min(start + CAPTION_MIN_S, nxt) if nxt > start else start + CAPTION_MIN_S
        end = min(end, audio_dur)
        if end <= start:
            continue
        text = " ".join(w.word.strip() for w in group)
        yield start, end, text.upper() if CAPTION_UPPER else text


def _is_highlight(text: str) -> bool:
    """Accent-color a caption if it carries a number/$/% or an opinion word."""
    if _HIGHLIGHT_RE.search(text):
        return True
    stripped = re.sub(r"[^A-Z']", "", text.upper())
    return stripped in _opinion_words()


def build_ass(segments, ass_path: Path, audio_dur: float,
              font_name: str = "Arial", accent: str = GREEN) -> None:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nCollisions: Normal\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{FONTSIZE},"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,4,2,2,60,60,{MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for start, end, text in _cluster_words(segments, audio_dur):
        c = accent if _is_highlight(text) else WHITE
        # Staggered pop: highlights (numbers/$/%/opinion words) get the biggest
        # fastest pop to punch emphasis; regular words get a subtler scale.
        if _is_highlight(text):
            scale_start, pop_ms = 138, 45   # biggest pop, fastest — maximum emphasis
        elif text[:1].upper() in "TKPBDGFVS":  # strong consonant onset = punch
            scale_start, pop_ms = 122, 62
        else:
            scale_start, pop_ms = 112, 88   # subtle scale, slower — background words
        styled = (f"{{\\c{c}\\shad2"
                  f"\\fscx{scale_start}\\fscy{scale_start}"
                  f"\\t(0,{pop_ms},\\fscx100\\fscy100)}}{text}")
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{styled}")
    ass_path.write_text(header + "\n".join(lines), encoding="utf-8")


# ── TTS ──────────────────────────────────────────────────────────────────────────

def _tts(script: str, mp3_path: Path, tones: list[str] | None = None) -> None:
    """Generate voice via tts_engine (Edge TTS default, XTTS v2 if RUFUS_TTS=xtts).

    `tones` lets the local Kokoro backend size the silence after each beat by
    what that beat is doing, not only by its trailing punctuation. Optional
    everywhere — without it the voice is exactly today's.
    """
    import tts_engine
    tts_engine.synthesize(script, mp3_path, tones)


# ── Cut planning (sentence-aligned, editor-grade pacing) ─────────────────────────

def _sentence_ends(segments) -> list[float]:
    """Timestamps where a spoken sentence ends (word text ends with . ! ? …).
    Abbreviations ('U.S.', 'Mr.', 'vs.') are excluded — their periods were
    counted as sentence ends, landing scene cuts and cut SFX mid-sentence."""
    ends = []
    for seg in segments:
        for w in seg.words:
            token = w.word.strip()
            if _SENT_END_RE.search(token) and not _ABBREV_RE.match(token.strip('"\')]')):
                ends.append(round(w.end, 3))
    return ends


# Punctuation that ends a clause without ending a sentence. Whisper keeps it
# attached to the word, so these are free pause markers.
_CLAUSE_END_RE = re.compile(r'[,;:—–]["\')\]]*$')


def _clause_ends(segments) -> list[float]:
    """Timestamps where a spoken CLAUSE ends — commas, semicolons, dashes.

    WHY THE CUT PLANNER NEEDS THESE NOW. Cuts used to snap to sentence ends
    only, which was right when a 40-second video had ten of them and ten
    pictures. With a picture roughly every five spoken words there are twice
    as many cuts as there are sentences, so most of them fell back to the even
    grid — landing mid-phrase, which is exactly the "image doesn't match what
    he's saying" the owner reported. A comma is a real pause in the narration
    and a free place to change picture.
    """
    ends = []
    for seg in segments:
        for w in seg.words:
            token = w.word.strip()
            if _CLAUSE_END_RE.search(token):
                ends.append(round(w.end, 3))
    return ends


def _max_shots(audio_dur: float) -> int:
    """The most shots this audio can hold without any of them being a flash.

    ASKING FOR MORE PICTURES THAN THE NARRATION CAN CARRY is what produced the
    machine-gun run: the beat count is decided from the script's word count
    before the voice exists, and if the finished audio is shorter than that
    assumed, every surplus cut lands on the minimum. Better to render fewer
    pictures than to show them all too briefly to register.
    """
    if audio_dur <= 0:
        return 1
    return max(1, int(audio_dur // MIN_SEG))


def _tone_grid(first: float, total: float, n: int,
               tones: list[str] | None) -> list[float]:
    """Where each remaining cut wants to be, before snapping to a pause.

    An EVEN grid gives every beat the same length whatever it carries, so the
    number, the turn and the line that lets it sit all pass at the rate of "and
    then this happened" — which a viewer reads as a slideshow however good the
    pictures are. Each beat's share of the remaining time is instead weighted
    by its tone (see emotional_map.hold_weight), so the reveal breathes and the
    connective tissue does not.

    No tones, or the wrong number of them, gives exactly the even grid this
    replaced — the weighting is an improvement on the rhythm, never a
    requirement for having one.
    """
    span = total - first
    if n <= 1 or span <= 0:
        return []
    weights = [1.0] * (n - 1)
    if tones and len(tones) >= n:
        try:
            import emotional_map
            # Beat 0 ends at `first`, so the weights that matter here are the
            # ones for beats 1..n-1.
            weights = [emotional_map.hold_weight(t) for t in tones[1:n]]
        except Exception:
            weights = [1.0] * (n - 1)
    scale = span / sum(weights)
    grid, at = [], first
    for w in weights[:-1]:
        at += w * scale
        grid.append(at)
    return grid


def _plan_cuts(sentence_ends: list[float], audio_dur: float, n: int,
               tones: list[str] | None = None) -> list[float]:
    """Choose n-1 cut timestamps that land on sentence boundaries.

    - Cut 1 lands in [FIRST_CUT_MIN, FIRST_CUT_MAX]s: a quick scene change right
      after the hook (the pattern interrupt that resets swipe-away attention).
    - Remaining cuts snap to the nearest sentence end within SNAP_WINDOW of a
      TONE-WEIGHTED grid, so scene changes happen where the narration breathes
      and the beats that carry the story get longer to do it in.
    - Monotonic with MIN_SEG spacing; falls back to the grid where no sentence
      end is close enough.
    """
    if n <= 1 or audio_dur <= 0:
        return []
    n = min(n, _max_shots(audio_dur))
    if n <= 1:
        return []

    usable = sorted(e for e in sentence_ends if 1.0 < e < audio_dur - 1.0)

    # Hook cut first: earliest sentence end inside the window, else grid clamp.
    window = [e for e in usable
              if FIRST_CUT_MIN <= e <= min(FIRST_CUT_MAX, audio_dur - MIN_SEG * (n - 1))]
    first = window[0] if window else min(max(audio_dur / n, FIRST_CUT_MIN + 0.4), FIRST_CUT_MAX)

    # Remaining cuts: re-spread evenly across [first, audio_dur] (NOT the original
    # n-grid — the hook cut moved, so the rest must rebalance) and snap each to
    # the nearest unused sentence end within SNAP_WINDOW.
    cuts: list[float] = [first]
    for target in _tone_grid(first, audio_dur, n, tones):
        near = [e for e in usable if abs(e - target) <= SNAP_WINDOW and e not in cuts]
        cuts.append(min(near, key=lambda e: abs(e - target)) if near else target)

    # Sanitize: strictly increasing, MIN_SEG apart, inside the timeline
    clean: list[float] = []
    prev = 0.0
    for i, c in enumerate(sorted(cuts)):
        c = max(c, prev + MIN_SEG)
        remaining = len(cuts) - i
        c = min(c, audio_dur - MIN_SEG * remaining)
        if c <= prev + 0.05:
            continue
        clean.append(round(c, 3))
        prev = c
    return clean


def _xfade_input_lengths(boundaries: list[float], total: float) -> list[float]:
    """Per-clip -t values for the xfade chain. Fade k ends exactly at boundary k.

    Clip 1 runs to b1; clip k covers (b_{k-1}-XFADE) → b_k; last covers to total.
    Chained output length telescopes to exactly `total`.
    """
    if not boundaries:
        return [total]
    lengths = [boundaries[0]]
    pts = boundaries + [total]
    for k in range(1, len(pts)):
        lengths.append(round(pts[k] - pts[k - 1] + XFADE_DUR, 3))
    return lengths


def _concat_input_lengths(boundaries: list[float], total: float) -> list[float]:
    """Per-clip -t values for the hard-concat fallback (no overlap)."""
    if not boundaries:
        return [total]
    pts = [0.0] + boundaries + [total]
    return [round(pts[k + 1] - pts[k], 3) for k in range(len(pts) - 1)]


# ── FFmpeg filter_complex builders ───────────────────────────────────────────────

def _ken_burns_part(i: int, dur: float, over_w: int, over_h: int, pad_y: int,
                    grade: str = "") -> str:
    """Scale-up + animated crop = Ken Burns pan for clip i over its own duration.

    `grade` is this beat's tone grade from emotional_map, applied per clip
    instead of the single global `ffmpeg_eq` the whole video used to share.
    Empty string keeps the old behaviour exactly — the global grade still runs
    later in _finish_video either way, so this only ever bends the niche's look,
    never replaces it.
    """
    x_exprs = [
        f"({over_w}-{W})*t/{dur:.3f}",
        f"({over_w}-{W})*(1-t/{dur:.3f})",
    ]
    y_exprs = [
        str(pad_y),
        f"({over_h}-{H})*t/{dur:.3f}",
        f"({over_h}-{H})*(1-t/{dur:.3f})",
    ]
    pan_x = x_exprs[i % len(x_exprs)]
    pan_y = y_exprs[i % len(y_exprs)]
    grade_str = f"{grade}," if grade else ""
    return (
        f"[{i}:v]setpts=PTS-STARTPTS,scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H}:{pan_x}:{pan_y},"
        f"{grade_str}"
        f"setsar=1,fps={FPS},format=yuv420p,settb=AVTB[v{i}]"
    )


def _parse_base_eq(eq_filter: str) -> tuple[float, float]:
    """(contrast, saturation) out of a niche's `ffmpeg_eq` string.

    The per-beat grade multiplies onto these so a channel keeps its own look
    and the tone only bends it. A niche that ships an eq this can't read — or
    ships something other than eq= entirely — falls back to neutral 1.0/1.0,
    which grades relative to nothing and is still a valid picture.
    """
    def _field(name: str) -> float:
        m = re.search(rf"\b{name}=(-?\d+(?:\.\d+)?)", eq_filter or "")
        return float(m.group(1)) if m else 1.0

    return _field("contrast"), _field("saturation")


def _ffmpeg_filter_path_escape(path) -> str:
    """Escape a filesystem path for safe use inside an ffmpeg filtergraph string
    (e.g. ass='...':fontsdir='...').

    Two independent escapes are needed, in this order:
    1. Backslash → forward slash (Windows paths use \\, filtergraphs treat \\ as
       an escape character).
    2. Colon → \\: — the ass/subtitles filter has its OWN internal option parser
       that splits on ':' to separate filename from fontsdir=/charenc=/etc. A
       Windows drive letter (C:/Users/...) has a colon right after the drive
       letter, which that parser misreads as an option separator, corrupting
       the whole filter chain. This never surfaces on Linux (no drive-letter
       colon in paths), which is why it only bit on the user's first Windows
       render — the fix must be here, not conditional on platform.
    Single-quote escaping (for the outer ass='...' wrapping) is handled by the
    caller, since not every use of this helper is wrapped in single quotes.
    """
    return str(path).replace("\\", "/").replace(":", "\\:")


def _insert_overlay_parts(inserts: list[dict], base: int,
                          src: str = "vb") -> tuple[list[str], str]:
    """Overlay chain for the word-synced inserts. Returns (parts, last label).

    WHY THIS LIVES IN THE FFMPEG PATH AT ALL. The layer was built into the
    Remotion renderer first, and RUFUS_RENDERER defaults to `ffmpeg` — so the
    owner set RUFUS_INSERT_MODE, RUFUS_INSERT_MAX=40 and got exactly the ten
    beat pictures they had before, twice, with nothing in the log to say why.
    A feature that only exists on the path nobody runs is not shipped, and
    silence about it is the fail-silent this repo keeps paying for.

    Each insert is one still image input, scaled and hard-cut in and out with
    `enable=between(t,...)` — hard cuts on purpose, since the fast-cut style
    this serves reads better without an ease. eof_action=pass so a short input
    can never truncate the video: the worst case is a picture that stops
    early, and the video still finishes.
    """
    parts: list[str] = []
    prev = src
    for k, ins in enumerate(inserts):
        try:
            at = float(ins.get("at", 0.0))
            hold = float(ins.get("hold", 0.7))
        except (TypeError, ValueError):
            continue
        end = at + max(0.2, hold)
        x = INSERT_MARGIN if k % 2 == 0 else W - INSERT_W - INSERT_MARGIN
        y = INSERT_YS[k % len(INSERT_YS)]
        parts.append(
            f"[{base + k}:v]scale={INSERT_W}:-1,setsar=1,fps={FPS},"
            f"format=rgba[ins{k}]"
        )
        parts.append(
            f"[{prev}][ins{k}]overlay=x={x}:y={y}:"
            f"enable='between(t,{at:.3f},{end:.3f})':eof_action=pass[ov{k}]"
        )
        prev = f"ov{k}"
    return parts, prev


def _finish_video(parts: list[str], total: float, eq_filter: str,
                  ass_esc: str, fonts_dir_esc: str, accent_hex: str,
                  inserts: list[dict] | None = None,
                  insert_base: int = 0) -> str:
    """Shared tail: edge fades → grade → bar → inserts → captions → [vout]."""
    fade_out_st = max(0.0, total - FADE_EDGE)
    fade_in_str = f"fade=type=in:st=0:d={FADE_IN:.3f}," if FADE_IN > 0 else ""
    parts.append(
        f"[vcat]{fade_in_str}"
        f"fade=type=out:st={fade_out_st:.3f}:d={FADE_EDGE:.3f},"
        f"{eq_filter},vignette=PI/4[vg]"
    )
    if RETENTION_BAR:
        bar_color = accent_hex.lstrip("#")
        parts.append(
            f"color=c=0x{bar_color}:size={W}x{BAR_HEIGHT}:rate={FPS}:duration={total:.3f}[bar];"
            f"[vg][bar]overlay=x='-{W}+{W}*t/{total:.3f}':y={H - BAR_HEIGHT}:"
            f"eof_action=pass,format=yuv420p[vb]"
        )
        last = "vb"
    else:
        # The chain still has to hand a labelled stream to what follows —
        # dropping the overlay must not drop the link.
        parts.append("[vg]format=yuv420p[vb]")
        last = "vb"
    if inserts:
        # Under the captions, exactly as the Remotion path stacks them: a word
        # is illustrated by the picture, never covered by it.
        ins_parts, last = _insert_overlay_parts(inserts, insert_base)
        parts.extend(ins_parts)
        parts.append(f"[{last}]format=yuv420p[vins]")
        last = "vins"
    parts.append(f"[{last}]ass='{ass_esc}':fontsdir='{fonts_dir_esc}'[vout]")
    return ";\n".join(parts)


def _video_filter_complex(input_lengths: list[float], boundaries: list[float],
                          total: float, over_w: int, over_h: int, pad_y: int,
                          eq_filter: str, ass_esc: str, fonts_dir_esc: str,
                          accent_hex: str, grades: list[str] | None = None,
                          inserts: list[dict] | None = None,
                          insert_base: int = 0) -> str:
    """Ken Burns per clip → sentence-aligned xfades → grade/bar/captions."""
    n = len(input_lengths)
    grades = grades or []
    parts = [_ken_burns_part(i, input_lengths[i], over_w, over_h, pad_y,
                             grades[i] if i < len(grades) else "")
             for i in range(n)]

    if n == 1:
        parts.append("[v0]null[vcat]")
    else:
        current = "v0"
        for k, b in enumerate(boundaries, start=1):
            offset   = max(0.0, b - XFADE_DUR)
            out_name = f"xf{k}"
            parts.append(
                f"[{current}][v{k}]xfade=transition=dissolve:"
                f"duration={XFADE_DUR:.3f}:offset={offset:.3f}[{out_name}]"
            )
            current = out_name
        parts.append(f"[{current}]null[vcat]")

    return _finish_video(parts, total, eq_filter, ass_esc, fonts_dir_esc,
                         accent_hex, inserts, insert_base)


def _video_filter_complex_concat(input_lengths: list[float], total: float,
                                 over_w: int, over_h: int, pad_y: int,
                                 eq_filter: str, ass_esc: str, fonts_dir_esc: str,
                                 accent_hex: str, grades: list[str] | None = None,
                                 inserts: list[dict] | None = None,
                                 insert_base: int = 0) -> str:
    """Hard-concat fallback (no transitions). Used when xfade errors."""
    n = len(input_lengths)
    grades = grades or []
    parts = [_ken_burns_part(i, input_lengths[i], over_w, over_h, pad_y,
                             grades[i] if i < len(grades) else "")
             for i in range(n)]
    if n == 1:
        parts.append("[v0]null[vcat]")
    else:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vcat]")
    return _finish_video(parts, total, eq_filter, ass_esc, fonts_dir_esc,
                         accent_hex, inserts, insert_base)


def _silence_trim_cmd(src: Path, dst: Path) -> list[str]:
    """ffmpeg command that strips leading AND trailing silence from a voice track.

    Leading silence from TTS delays the hook and desyncs the 0.03s SFX hit and
    the first caption — on Shorts, dead air at 0:00 is a swipe-away. Head trim
    via silenceremove; tail trim via the areverse sandwich (reverse → head-trim
    → reverse back).
    """
    af = (
        "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.05,"
        "areverse,"
        "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.10,"
        "areverse"
    )
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-af", af, "-c:a", "libmp3lame", "-b:a", "192k", str(dst)]


def _trim_silence(mp3: Path) -> None:
    """Trim silence off the TTS mp3 in place. Fail-open: any problem keeps the
    original file untouched. MUST run before Whisper so word timestamps (which
    drive cuts, SFX, and captions) describe the trimmed audio."""
    trimmed = mp3.with_suffix(".trim.mp3")
    try:
        r = subprocess.run(_silence_trim_cmd(mp3, trimmed),
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and trimmed.exists() and trimmed.stat().st_size > 5_000:
            trimmed.replace(mp3)
        else:
            trimmed.unlink(missing_ok=True)
            print("[audio] silence trim skipped (ffmpeg filter unavailable?)")
    except Exception as e:
        trimmed.unlink(missing_ok=True)
        print(f"[audio] silence trim skipped ({e})")


# ── Two-pass loudness normalization ──────────────────────────────────────────────

def _parse_loudnorm_json(stderr: str) -> dict | None:
    """Extract the loudnorm measurement JSON that ffmpeg prints to stderr.

    ffmpeg emits it as the LAST {...} block; other log lines may precede it.
    Returns the parsed dict, or None if no valid block is found."""
    depth, start = 0, -1
    last = None
    for i, ch in enumerate(stderr):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    cand = json.loads(stderr[start:i + 1])
                    if "input_i" in cand:
                        last = cand
                except json.JSONDecodeError:
                    pass
    return last


def _normalize_loudness(video: Path) -> None:
    """Two-pass loudnorm to exactly -14 LUFS on the finished mp4.

    The in-graph loudnorm is single-pass (can miss the target by 1-3 LU and
    pump). Pass 1 measures the final mix; pass 2 re-encodes ONLY the audio
    stream with the measured values in linear mode (-c:v copy — video bits are
    untouched, so qc_check's resolution/duration checks stay valid).
    Fail-open: any problem leaves the render as-is."""
    tuned = video.with_suffix(".ln.mp4")
    try:
        measure = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(video),
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
             "-vn", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
        m = _parse_loudnorm_json(measure.stderr)
        if not m:
            print("[audio] loudnorm pass-2 skipped (no measurement)")
            return
        af = (
            f"loudnorm=I=-14:TP=-1.5:LRA=11:"
            f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m.get('target_offset', 0)}:linear=true"
        )
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-af", af, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
             str(tuned)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0 and tuned.exists() and tuned.stat().st_size > 500_000:
            tuned.replace(video)
            print(f"[audio] loudness locked to -14 LUFS (measured {m['input_i']} LUFS)")
        else:
            tuned.unlink(missing_ok=True)
            print("[audio] loudnorm pass-2 skipped (re-encode failed)")
    except Exception as e:
        tuned.unlink(missing_ok=True)
        print(f"[audio] loudnorm pass-2 skipped ({e})")


def _audio_filter_complex(n: int, audio_dur: float, has_music: bool,
                          sfx_events: list[tuple[float, float]],
                          with_deesser: bool = False) -> str:
    """Full broadcast mix → [aout].

    Voice: highpass → compressor → presence EQ (studio-izes Edge TTS).
    Music: bed volume → DUCKED under voice via sidechaincompress (breathes back
           up in speech gaps — the pro alternative to a fixed low volume).
    SFX:   each event is its own (tiny) input, delayed into place.
    Master: loudnorm to -14 LUFS / -1.5 dBTP — YouTube reference loudness.

    Input layout: [0..n-1]=clips, [n]=voice, [n+1]=music (if any), then SFX.
    """
    # aformat pins 48 kHz stereo on every branch — sidechaincompress and amix
    # require matching rate/layout (Edge TTS is mono 24 kHz, music is stereo).
    fmt    = "aformat=sample_rates=48000:channel_layouts=stereo"
    parts  = []
    # De-esser tames TTS sibilance that the 3 kHz presence boost would otherwise
    # sharpen. Gated on filter availability (older ffmpeg builds lack it).
    deess  = "deesser=i=0.4," if with_deesser else ""
    voice  = (
        f"[{n}:a]highpass=f=70,"
        f"acompressor=threshold=0.1:ratio=3:attack=12:release=180:makeup=2,"
        f"equalizer=f=3000:t=q:w=1.2:g=2,{deess}{fmt}"
    )
    mix_in = []

    if has_music:
        parts.append(voice + "[vc]")
        parts.append("[vc]asplit=2[vmain][vkey]")
        fade_out_st = max(0.0, audio_dur - 1.8)
        parts.append(
            f"[{n + 1}:a]volume={MUSIC_BED},"
            f"afade=type=in:st=0:d=1.2,"
            f"afade=type=out:st={fade_out_st:.3f}:d=1.8,{fmt}[mbed]"
        )
        # ratio 4, not 10: a Short's narration is nearly CONTINUOUS, so the
        # voice keys this compressor ~100% of the runtime — at ratio 10 the
        # bed was crushed to inaudible for the whole video ("there's no
        # music"). ratio 4 keeps music clearly present UNDER the voice (like
        # every professionally-mixed Short) while still yielding to speech;
        # the gap-breathing behavior at pauses is unchanged.
        parts.append(
            "[mbed][vkey]sidechaincompress="
            "threshold=0.02:ratio=4:attack=35:release=600[mduck]"
        )
        mix_in = ["[vmain]", "[mduck]"]
    else:
        parts.append(voice + "[vmain]")
        mix_in = ["[vmain]"]

    sfx_base = n + (2 if has_music else 1)
    if sfx_events:
        sfx_labels = []
        for j, (delay_s, gain) in enumerate(sfx_events):
            ms = max(1, round(delay_s * 1000))
            parts.append(f"[{sfx_base + j}:a]volume={gain:.2f},{fmt},adelay={ms}|{ms}[s{j}]")
            sfx_labels.append(f"[s{j}]")
        if len(sfx_labels) == 1:
            parts.append(f"{sfx_labels[0]}anull[sx]")
        else:
            parts.append(
                f"{''.join(sfx_labels)}amix=inputs={len(sfx_labels)}:"
                f"duration=longest:normalize=0[sx]"
            )
        mix_in.append("[sx]")

    if len(mix_in) == 1:
        parts.append(f"{mix_in[0]}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    else:
        parts.append(
            f"{''.join(mix_in)}amix=inputs={len(mix_in)}:duration=first:"
            f"dropout_transition=0:normalize=0,"
            f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]"
        )
    return ";\n".join(parts)


def _audio_filter_simple(n: int, audio_dur: float, has_music: bool,
                         with_loudnorm: bool = False) -> str:
    """Minimal proven mix for the fallback attempt: voice + static-volume music.

    with_loudnorm keeps fallback renders at the same -14 LUFS as the full mix —
    pass it only when the installed FFmpeg has the loudnorm filter, so the
    fallback stays viable on minimal builds.
    """
    master = ",loudnorm=I=-14:TP=-1.5:LRA=11" if with_loudnorm else ""
    if not has_music:
        if with_loudnorm:
            return f"[{n}:a]loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
        return f"[{n}:a]anull[aout]"
    fade_out_st = max(0.0, audio_dur - 1.5)
    return (
        f"[{n + 1}:a]volume={MUSIC_VOL},"
        f"afade=type=in:st=0:d=1.5,"
        f"afade=type=out:st={fade_out_st:.3f}:d=1.5[music];"
        f"[{n}:a][music]amix=inputs=2:duration=first:dropout_transition=1"
        f"{master}[aout]"
    )


def _save_debug_artifacts(script: str, voiceover_mp3: Path) -> None:
    """Save the script text and the raw (pre-mix) voiceover into the same
    media_library/debug/<run_id>/ folder comfy_client uses for
    keyframes+prompts — one place to review everything from EVERY run before
    it ever reaches YouTube, not just RUFUS_DEBUG=1 runs (the quality-review
    workflow needs every run's script logged, not an opt-in subset).
    Complements the automated post-publish feedback loop
    (analytics_fetcher/feedback_analyzer) with a pre-publish, human one.
    Non-fatal: a debug-save failure must never break the actual render."""
    try:
        run_id = os.environ.get("RUFUS_DEBUG_RUN_ID") or f"audio_{int(time.time())}"
        debug_dir = paths.debug_root() / run_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "script.txt").write_text(script, encoding="utf-8")
        shutil.copy2(voiceover_mp3, debug_dir / "voiceover.mp3")
        print(f"[audio] saved script + voiceover to {debug_dir}")
    except Exception as e:
        print(f"[audio] debug-save failed (non-fatal): {e}")


# ── Renderer ─────────────────────────────────────────────────────────────────────

def render(script: str, bg_paths: "Path | list[Path]", out_dir: Path,
           music_path: Path | None = None) -> Path:
    if isinstance(bg_paths, Path):
        bg_paths = [bg_paths]
    bg_paths = [Path(p) for p in bg_paths if Path(p).exists()]
    if not bg_paths:
        raise FileNotFoundError("No valid background video files found")

    n = len(bg_paths)

    niche_cfg  = _load_niche()
    niche_name = _active_niche_name()
    raw_eq     = niche_cfg.get("ffmpeg_eq", "eq=contrast=1.1:saturation=1.0")
    # Strip shell-dangerous characters to prevent filter injection via niches.json.
    eq_filter  = re.sub(r"[;\|`$\\]", "", raw_eq)
    accent_hex = niche_cfg.get("accent_color", DEFAULT_ACCENT)
    accent_ass = _hex_to_ass(accent_hex)

    # Auto-fetch music if not provided
    if music_path is None:
        try:
            music_path = _fetch_music(niche_name)
        except Exception as e:
            print(f"[music] {niche_name} mood fetch failed: {e}")
            music_path = None
    if music_path is None:
        print("[audio] no music track — rendering voice-only")

    # The emotional map is built ONCE here, before the voice, because the
    # tone-sized pauses have to be baked into the audio and everything
    # downstream (cut count, grade, SFX weight) is derived from the same plan.
    # Computing it later would mean re-splitting beats against a different
    # clip count, missing edit_director's memo, and grading the video against a
    # second plan the narration never heard.
    plan_tones: list[str] = []
    try:
        import emotional_map
        import edit_director
        import main as _main
        _beats = _main._split_beats(script, max_scenes=n, grow=True)
        _plan  = edit_director.direct(_beats) if _beats else None
        if _plan is not None:
            plan_tones = emotional_map.tones_from_plan(_plan, len(_beats))
    except Exception as e:
        print(f"[grade] emotional map unavailable (non-fatal): {e}")

    font_name = _ensure_font()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = paths.media_root() / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time())
    mp3   = tmp_dir / f"{stamp}.mp3"
    ass   = tmp_dir / f"{stamp}.ass"
    out   = out_dir / f"short_{stamp}.mp4"

    try:
        print("[1/4] Generating voice…")
        _tts(script, mp3, plan_tones)
        # Strip leading/trailing TTS silence BEFORE transcription — Whisper's
        # word timestamps then describe the trimmed audio, so cuts, the 0.03s
        # SFX hit, and the first caption all land on the actual first word.
        _trim_silence(mp3)
        _save_debug_artifacts(script, mp3)

        print("[2/4] Transcribing…")
        segs, _ = _transcribe(mp3)
        segments = list(segs)
        if not segments:
            raise RuntimeError("Whisper produced no segments")

        audio_dur = max((w.end for seg in segments for w in seg.words), default=0)
        audio_dur = min(max(audio_dur + 0.3, 1.0), MAX_DUR)
        if audio_dur <= 0:
            raise RuntimeError(f"Invalid audio duration: {audio_dur}")
        if audio_dur < MIN_DUR:
            print(f"      ⚠ audio is only {audio_dur:.1f}s (target ≥{MIN_DUR:.0f}s)")

        print("[3/4] Building subtitles…")
        build_ass(segments, ass, audio_dur, font_name=font_name, accent=accent_ass)

        # Sentence-aligned cut plan — scene changes land where narration breathes
        n_supplied = n
        # Sentence ends first, then clause ends — both are real pauses, and at
        # this cut density there are not enough sentences to go round.
        _snap_points = _sentence_ends(segments)
        if n > len(_snap_points) + 1:
            _snap_points = sorted(set(_snap_points) | set(_clause_ends(segments)))
        boundaries = _plan_cuts(_snap_points, audio_dur, n, plan_tones)
        n = len(boundaries) + 1 if boundaries else 1
        bg_paths = bg_paths[:n]
        if n < n_supplied:
            print(f"      ⚠ using {n} of {n_supplied} clips (not enough room for more cuts)")
        if boundaries:
            print(f"      cuts at: {', '.join(f'{b:.1f}s' for b in boundaries)}")
        # Published for QC's pacing check: cut timestamps cannot be recovered
        # from the finished mp4, and a run that quietly held one picture for
        # nine seconds is exactly what nobody notices until a viewer swipes.
        globals()["LAST_CUTS"] = list(boundaries)
        globals()["LAST_WORDS"] = [
            (float(w.start), w.word.strip())
            for seg in segments for w in seg.words if w.word.strip()
        ]

        lens_xfade  = _xfade_input_lengths(boundaries, audio_dur)
        lens_concat = _concat_input_lengths(boundaries, audio_dur)
        over_w      = int(W * 1.10)
        over_h      = int(H * 1.10)
        pad_y       = (over_h - H) // 2

        # Per-beat grading. The FFmpeg path is the one that actually renders
        # today (Remotion has its own fallback into here), so the emotional map
        # has to land in BOTH or it repeats the mistake that left the edit plan
        # computed-and-discarded on every run. edit_director memoises per
        # process, so asking here after Remotion already asked costs nothing
        # and — more importantly — returns the SAME plan rather than a second
        # one at temperature 0.7.
        grades: list[str] = []
        tones:  list[str] = []
        try:
            import emotional_map
            import edit_director
            import main as _main
            beats = _main._split_beats(script, max_scenes=n, grow=True)
            plan  = edit_director.direct(beats) if len(beats) == n else None
            tones = emotional_map.tones_from_plan(plan, n)
            base_c, base_s = _parse_base_eq(eq_filter)
            grades = [emotional_map.grade_filter(t, base_c, base_s) for t in tones]
            if plan is not None:
                print(f"      grade: {emotional_map.describe(tones)}")
        except Exception as e:
            # Never a prerequisite for a render — an ungraded video is the
            # video this pipeline shipped yesterday.
            print(f"[grade] skipped (non-fatal): {e}")
            grades, tones = [], []
        ass_esc     = _ffmpeg_filter_path_escape(ass).replace("'", "\\'")
        fonts_esc   = _ffmpeg_filter_path_escape(FONTS_DIR).replace("'", "\\'")
        has_music   = music_path is not None and Path(music_path).exists()

        # SFX layer: hit on the hook, bubble leading into every cut, riser into
        # the final beat. Synthesized locally — skipped cleanly if unavailable,
        # or entirely opted out of via RUFUS_SFX=0.
        sfx = {}
        if SFX_ENABLED:
            try:
                sfx = _ensure_sfx()
            except Exception:
                sfx = {}
        sfx_files:  list[Path] = []
        sfx_events: list[tuple[float, float]] = []
        # Each effect is weighted by the tone of the beat it introduces, so the
        # riser into a revelation is audible and a bubble does not compete with
        # a resolution beat's closing line. Without tones every weight is 1.0
        # and these are the exact gains the mix used before.
        def _w(beat_index: int) -> float:
            if not tones:
                return 1.0
            import emotional_map
            return emotional_map.sfx_weight(tones[min(beat_index, len(tones) - 1)])

        if sfx:
            sfx_files.append(sfx["hit"])
            sfx_events.append((0.03, SFX_HIT_GAIN * _w(0)))
            for k, b in enumerate(boundaries):
                sfx_files.append(sfx["bubble"])
                sfx_events.append((max(0.0, b - 0.18), SFX_BUBBLE_GAIN * _w(k + 1)))
            if boundaries:
                riser_at = max(0.5, boundaries[-1] - 1.25)
                sfx_files.append(sfx["riser"])
                sfx_events.append((riser_at, SFX_RISER_GAIN * _w(len(boundaries))))

        # WORD-SYNCED INSERTS. Planned here because this is the first moment
        # the FINISHED voiceover has been transcribed — an insert is pinned to
        # the second its phrase is actually spoken, and only Whisper knows
        # that. Fail-open: any failure leaves `inserts` empty and this renders
        # exactly the video it rendered before the layer existed.
        inserts: list[dict] = []
        # Alongside the keyframes of the same run, not in temp: these are the
        # only record of what the insert layer actually drew, the owner
        # reviews them in the dashboard gallery, and the debug root is already
        # the thing the maintenance job prunes.
        try:
            _run_id = os.environ.get("RUFUS_DEBUG_RUN_ID") or f"audio_{stamp}"
            insert_dir = paths.debug_root() / _run_id / "inserts"
        except Exception:
            insert_dir = tmp_dir / f"{stamp}_inserts"
        try:
            import insert_director
            if insert_director.enabled():
                spoken = [{"text": w.word.strip(),
                           "start": float(w.start), "end": float(w.end)}
                          for seg in segments for w in seg.words]
                planned = insert_director.plan_for(
                    script, spoken, insert_director.style_suffix())
                if planned:
                    print(insert_director.describe(planned))
                    import comfy_client
                    inserts = comfy_client.render_inserts(planned, insert_dir)
        except Exception as e:
            print(f"[inserts] unavailable ({e}) — rendering without them")
        insert_paths = [insert_dir / str(i["file"]) for i in inserts
                        if (insert_dir / str(i.get("file", ""))).exists()]
        inserts = inserts[:len(insert_paths)]

        use_xfade = n > 1 and _ffmpeg_has_xfade()
        print(f"[4/4] Rendering {n} clip{'s' if n > 1 else ''} → {audio_dur:.1f}s"
              f"{' + music' if has_music else ''}"
              f"{f' + {len(sfx_events)} sfx' if sfx_events else ''}"
              f"{f' + {len(inserts)} insert(s)' if inserts else ''}"
              f"{' [xfade]' if use_xfade else ''}…")

        def _insert_base(with_sfx: bool) -> int:
            """Input index of the first insert image.

            Inserts are appended LAST so that every audio index the mix
            filters already compute from `n` keeps its meaning — the audio
            graph must not have to know this layer exists.
            """
            return n + 1 + (1 if has_music else 0) + \
                (len(sfx_files) if with_sfx else 0)

        def _build_cmd(fc: str, input_lengths: list[float], with_sfx: bool,
                       with_inserts: bool = True) -> list:
            c = ["ffmpeg", "-y", "-loglevel", "warning"]
            for bg, seg_t in zip(bg_paths, input_lengths):
                c += ["-stream_loop", "-1", "-t", f"{seg_t + 0.05:.3f}", "-i", str(bg)]
            c += ["-i", str(mp3)]
            if has_music:
                c += ["-stream_loop", "-1", "-t", f"{audio_dur + 1.0:.3f}", "-i", str(music_path)]
            if with_sfx:
                for sf in sfx_files:
                    c += ["-i", str(sf)]
            if with_inserts:
                for pic in insert_paths:
                    # Held for the whole video and switched on by the overlay's
                    # `enable` window: a still costs nothing to keep available,
                    # and an input that ENDS mid-video is how an overlay
                    # silently stops appearing.
                    c += ["-loop", "1", "-framerate", str(FPS),
                          "-t", f"{audio_dur + 0.5:.3f}", "-i", str(pic)]
            c += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
            c += [
                "-t", f"{audio_dur:.3f}",
                *_video_encoder_args(),
                "-c:a", "aac", "-b:a", "160k",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                str(out),
            ]
            return c

        def _xfade_fc(with_inserts: bool) -> str:
            return (
                _video_filter_complex(lens_xfade, boundaries, audio_dur, over_w, over_h,
                                      pad_y, eq_filter, ass_esc, fonts_esc, accent_hex,
                                      grades,
                                      inserts if with_inserts else None,
                                      _insert_base(bool(sfx_events)))
                + ";\n"
                + _audio_filter_complex(n, audio_dur, has_music, sfx_events,
                                        with_deesser=_ffmpeg_has_filter("deesser"))
            )

        rendered = False
        # THE LADDER, with the insert layer as its own rung. The layer adds
        # forty inputs and forty overlays to the most fragile string in this
        # repo, so a failure there must cost the PICTURES and not the video —
        # and it must say so, because a video that quietly lost its inserts
        # looks exactly like a video that was never planning to have any.
        attempts: list[tuple[str, str, list[float], bool, bool]] = []
        if use_xfade:
            if inserts:
                attempts.append(("full mix", _xfade_fc(True), lens_xfade,
                                 bool(sfx_events), True))
                attempts.append(("full mix without inserts", _xfade_fc(False),
                                 lens_xfade, bool(sfx_events), False))
            else:
                attempts.append(("full mix", _xfade_fc(False), lens_xfade,
                                 bool(sfx_events), False))

        for label, fc, lens, with_sfx, with_ins in attempts:
            # A timeout must degrade exactly like a nonzero return code — fall
            # through to the next rung, never freeze an autonomous run.
            try:
                r = subprocess.run(_build_cmd(fc, lens, with_sfx=with_sfx,
                                              with_inserts=with_ins),
                                   capture_output=True, text=True, timeout=RENDER_TIMEOUT)
                if r.returncode == 0:
                    rendered = True
                    break
                print(f"[render] {label} failed (rc={r.returncode}: "
                      f"{r.stderr[-200:].strip()}) — trying the next rung…")
            except subprocess.TimeoutExpired:
                print(f"[render] {label} timed out after {RENDER_TIMEOUT}s — "
                      f"trying the next rung…")
            if with_ins:
                print(f"[inserts] ⚠ {len(inserts)} insert(s) dropped — the "
                      f"filtergraph carrying them would not render")

        if not rendered:
            fc = (
                _video_filter_complex_concat(lens_concat, audio_dur, over_w, over_h,
                                             pad_y, eq_filter, ass_esc, fonts_esc,
                                             accent_hex, grades)
                + ";\n"
                + _audio_filter_simple(n, audio_dur, has_music,
                                       with_loudnorm=_ffmpeg_has_filter("loudnorm"))
            )
            if inserts and not attempts:
                print(f"[inserts] ⚠ {len(inserts)} insert(s) dropped — the "
                      f"simple pipeline renders without them")
            try:
                r2 = subprocess.run(_build_cmd(fc, lens_concat, with_sfx=False,
                                               with_inserts=False),
                                    capture_output=True, text=True, timeout=RENDER_TIMEOUT)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"FFmpeg fallback render timed out after {RENDER_TIMEOUT}s")
            if r2.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg fallback render failed (rc={r2.returncode}):\n{r2.stderr[-600:]}"
                )

        # Two-pass loudness lock: measure the finished mix, re-encode audio only
        # (video bits untouched). Fail-open — a skipped pass leaves single-pass
        # loudnorm output, which is still within ~1-3 LU of target.
        _normalize_loudness(out)

    finally:
        for f in (mp3, ass):
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Done → {out}")
    return out


# ── Entry point ──────────────────────────────────────────────────────────────────

def main():
    if not sys.stdin.isatty():
        payload    = json.load(sys.stdin)
        script     = payload["script"]
        bg         = Path(payload["bg"])
        out        = Path(payload["out"])
        music_path = Path(payload["music"]) if payload.get("music") else None
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--script", required=True)
        p.add_argument("--bg",     required=True)
        p.add_argument("--out",    required=True)
        p.add_argument("--music",  default=None)
        a = p.parse_args()
        script     = a.script
        bg         = Path(a.bg)
        out        = Path(a.out)
        music_path = Path(a.music) if a.music else None

    result = render(script, bg, out, music_path=music_path)
    print(f"OUTPUT_PATH={result}")


if __name__ == "__main__":
    main()
