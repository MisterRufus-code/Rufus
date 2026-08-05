#!/usr/bin/env python3
"""Rufus – Autonomous Shorts Renderer (v4.0)

Changes from v3.0 — "cinematic edit" upgrade:
- Cuts snap to SENTENCE BOUNDARIES from Whisper word timestamps (editor-grade
  pacing) with a short punchy first cut (~2-4s) for the hook pattern-interrupt.
- Sound design: synthesized SFX layer (sub-bass hit on the hook, whoosh on
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

W, H         = 1080, 1920
FPS          = 30
MAX_DUR      = 60.0
MIN_DUR      = 30.0
CLUSTER_SIZE = 1           # 1 word at a time — Hormozi style

XFADE_DUR    = 0.30        # crossfade duration between clips (seconds)
FADE_IN      = 0.0         # fade-in duration — 0 = hard cut (top Shorts open cold)
FADE_EDGE    = 0.40        # fade-to-black at end duration (seconds)
MUSIC_VOL    = 0.14        # static music volume (simple-mix fallback path)
MUSIC_BED    = 0.30        # music bed volume BEFORE sidechain ducking (full mix)
BAR_HEIGHT   = 14          # retention progress bar thickness (px)

# RUFUS_SFX=0 drops the whole synthesized layer (hit/whoosh/riser) — the
# lever for "these effects don't belong on this channel" without touching
# three separate gain values. Default stays on for anyone who hasn't
# formed an opinion either way.
SFX_ENABLED = os.environ.get("RUFUS_SFX", "1").strip().lower() not in ("0", "false", "no", "off")

# SFX layer gains (relative, 0-1). whoosh plays on EVERY cut (up to 9x per
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
SFX_WHOOSH_GAIN = float(os.environ.get("RUFUS_WHOOSH_GAIN", "0.02"))  # transition swoosh into each cut
SFX_RISER_GAIN  = float(os.environ.get("RUFUS_RISER_GAIN", "0.28"))   # riser leading into the final beat (once)

# Cut planning
FIRST_CUT_MIN = 2.0        # hook cut window — research: pattern interrupt by ~3s
FIRST_CUT_MAX = 4.2
SNAP_WINDOW   = 2.0        # max distance a cut may move to land on a sentence end
MIN_SEG       = 1.2        # minimum clip duration after planning

WHITE = "&H00FFFFFF"
GREEN = "&H0000FF00"

_HIGHLIGHT_RE = re.compile(r'[\d$%]')
_SENT_END_RE  = re.compile(r'[.!?…]["\')\]]*$')
# Words whose trailing period is an abbreviation, not a sentence end — without
# this guard, "the U.S. dollar" put a scene cut + whoosh SFX mid-sentence.
_ABBREV_RE    = re.compile(
    r'^(mr|mrs|ms|dr|st|vs|etc|inc|co|jr|sr|prof|gen|col|sgt|no'
    r'|[a-z](\.[a-z])+)\.$', re.IGNORECASE)

FONT_NAME = "Anton"        # downloaded to assets/fonts/; Arial fallback if missing
FONT_FILE = FONTS_DIR / "Anton-Regular.ttf"
FONTSIZE  = 140            # larger = better mobile readability
MARGIN_V  = 600            # ~31% from bottom: below the portrait face zone, above Shorts UI

# Hard ceiling for a single ffmpeg render pass — a hung/looping ffmpeg must
# fail the attempt (and fall through to the simple pipeline), never freeze an
# autonomous cron run forever. Override with RENDER_TIMEOUT (seconds).
RENDER_TIMEOUT = int(os.environ.get("RENDER_TIMEOUT", "600"))

DEFAULT_ACCENT = "#FFD23F"   # warm gold — used when a niche has no accent_color


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

def _add_nvidia_dll_dirs() -> None:
    """Windows: make pip-installed CUDA runtime DLLs visible to ctranslate2.

    GPU Whisper needs cuBLAS/cuDNN. Instead of the multi-GB CUDA Toolkit, the
    runtime DLLs install via `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
    into site-packages/nvidia/<lib>/bin — but Windows won't find them there
    unless we register the directories. No-op on Linux and when not installed.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
        base = Path(nvidia.__file__).parent
        for sub in ("cublas", "cudnn"):
            d = base / sub / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
    except Exception:
        pass


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
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active]


def _active_niche_name() -> str:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
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
        std = json.loads((CONFIG_DIR / "script_standards.json").read_text())
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
        yield start, end, " ".join(w.word.strip().upper() for w in group)


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

def _tts(script: str, mp3_path: Path) -> None:
    """Generate voice via tts_engine (Edge TTS default, XTTS v2 if RUFUS_TTS=xtts)."""
    import tts_engine
    tts_engine.synthesize(script, mp3_path)


# ── Cut planning (sentence-aligned, editor-grade pacing) ─────────────────────────

def _sentence_ends(segments) -> list[float]:
    """Timestamps where a spoken sentence ends (word text ends with . ! ? …).
    Abbreviations ('U.S.', 'Mr.', 'vs.') are excluded — their periods were
    counted as sentence ends, landing scene cuts and whoosh SFX mid-sentence."""
    ends = []
    for seg in segments:
        for w in seg.words:
            token = w.word.strip()
            if _SENT_END_RE.search(token) and not _ABBREV_RE.match(token.strip('"\')]')):
                ends.append(round(w.end, 3))
    return ends


def _plan_cuts(sentence_ends: list[float], audio_dur: float, n: int) -> list[float]:
    """Choose n-1 cut timestamps that land on sentence boundaries.

    - Cut 1 lands in [FIRST_CUT_MIN, FIRST_CUT_MAX]s: a quick scene change right
      after the hook (the pattern interrupt that resets swipe-away attention).
    - Remaining cuts snap to the nearest sentence end within SNAP_WINDOW of an
      equal-spacing grid, so scene changes happen where the narration breathes.
    - Monotonic with MIN_SEG spacing; falls back to the grid where no sentence
      end is close enough.
    """
    if n <= 1 or audio_dur <= 0:
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
    for j in range(1, n - 1):
        target = first + (audio_dur - first) * j / (n - 1)
        near   = [e for e in usable if abs(e - target) <= SNAP_WINDOW and e not in cuts]
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

def _ken_burns_part(i: int, dur: float, over_w: int, over_h: int, pad_y: int) -> str:
    """Scale-up + animated crop = Ken Burns pan for clip i over its own duration."""
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
    return (
        f"[{i}:v]setpts=PTS-STARTPTS,scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H}:{pan_x}:{pan_y},"
        f"setsar=1,fps={FPS},format=yuv420p,settb=AVTB[v{i}]"
    )


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


def _finish_video(parts: list[str], total: float, eq_filter: str,
                  ass_esc: str, fonts_dir_esc: str, accent_hex: str) -> str:
    """Shared tail: edge fades → grade → progress bar → captions → [vout]."""
    fade_out_st = max(0.0, total - FADE_EDGE)
    fade_in_str = f"fade=type=in:st=0:d={FADE_IN:.3f}," if FADE_IN > 0 else ""
    parts.append(
        f"[vcat]{fade_in_str}"
        f"fade=type=out:st={fade_out_st:.3f}:d={FADE_EDGE:.3f},"
        f"{eq_filter},vignette=PI/4[vg]"
    )
    bar_color = accent_hex.lstrip("#")
    parts.append(
        f"color=c=0x{bar_color}:size={W}x{BAR_HEIGHT}:rate={FPS}:duration={total:.3f}[bar];"
        f"[vg][bar]overlay=x='-{W}+{W}*t/{total:.3f}':y={H - BAR_HEIGHT}:"
        f"eof_action=pass,format=yuv420p[vb]"
    )
    parts.append(f"[vb]ass='{ass_esc}':fontsdir='{fonts_dir_esc}'[vout]")
    return ";\n".join(parts)


def _video_filter_complex(input_lengths: list[float], boundaries: list[float],
                          total: float, over_w: int, over_h: int, pad_y: int,
                          eq_filter: str, ass_esc: str, fonts_dir_esc: str,
                          accent_hex: str) -> str:
    """Ken Burns per clip → sentence-aligned xfades → grade/bar/captions."""
    n = len(input_lengths)
    parts = [_ken_burns_part(i, input_lengths[i], over_w, over_h, pad_y) for i in range(n)]

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

    return _finish_video(parts, total, eq_filter, ass_esc, fonts_dir_esc, accent_hex)


def _video_filter_complex_concat(input_lengths: list[float], total: float,
                                 over_w: int, over_h: int, pad_y: int,
                                 eq_filter: str, ass_esc: str, fonts_dir_esc: str,
                                 accent_hex: str) -> str:
    """Hard-concat fallback (no transitions). Used when xfade errors."""
    n = len(input_lengths)
    parts = [_ken_burns_part(i, input_lengths[i], over_w, over_h, pad_y) for i in range(n)]
    if n == 1:
        parts.append("[v0]null[vcat]")
    else:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vcat]")
    return _finish_video(parts, total, eq_filter, ass_esc, fonts_dir_esc, accent_hex)


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
        _tts(script, mp3)
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
        boundaries = _plan_cuts(_sentence_ends(segments), audio_dur, n)
        n = len(boundaries) + 1 if boundaries else 1
        bg_paths = bg_paths[:n]
        if n < n_supplied:
            print(f"      ⚠ using {n} of {n_supplied} clips (not enough room for more cuts)")
        if boundaries:
            print(f"      cuts at: {', '.join(f'{b:.1f}s' for b in boundaries)}")

        lens_xfade  = _xfade_input_lengths(boundaries, audio_dur)
        lens_concat = _concat_input_lengths(boundaries, audio_dur)
        over_w      = int(W * 1.10)
        over_h      = int(H * 1.10)
        pad_y       = (over_h - H) // 2
        ass_esc     = _ffmpeg_filter_path_escape(ass).replace("'", "\\'")
        fonts_esc   = _ffmpeg_filter_path_escape(FONTS_DIR).replace("'", "\\'")
        has_music   = music_path is not None and Path(music_path).exists()

        # SFX layer: hit on the hook, whoosh leading into every cut, riser into
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
        if sfx:
            sfx_files.append(sfx["hit"]);  sfx_events.append((0.03, SFX_HIT_GAIN))
            for b in boundaries:
                sfx_files.append(sfx["whoosh"])
                sfx_events.append((max(0.0, b - 0.18), SFX_WHOOSH_GAIN))
            if boundaries:
                riser_at = max(0.5, boundaries[-1] - 1.25)
                sfx_files.append(sfx["riser"]); sfx_events.append((riser_at, SFX_RISER_GAIN))

        use_xfade = n > 1 and _ffmpeg_has_xfade()
        print(f"[4/4] Rendering {n} clip{'s' if n > 1 else ''} → {audio_dur:.1f}s"
              f"{' + music' if has_music else ''}"
              f"{f' + {len(sfx_events)} sfx' if sfx_events else ''}"
              f"{' [xfade]' if use_xfade else ''}…")

        def _build_cmd(fc: str, input_lengths: list[float], with_sfx: bool) -> list:
            c = ["ffmpeg", "-y", "-loglevel", "warning"]
            for bg, seg_t in zip(bg_paths, input_lengths):
                c += ["-stream_loop", "-1", "-t", f"{seg_t + 0.05:.3f}", "-i", str(bg)]
            c += ["-i", str(mp3)]
            if has_music:
                c += ["-stream_loop", "-1", "-t", f"{audio_dur + 1.0:.3f}", "-i", str(music_path)]
            if with_sfx:
                for sf in sfx_files:
                    c += ["-i", str(sf)]
            c += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"]
            c += [
                "-t", f"{audio_dur:.3f}",
                *_video_encoder_args(),
                "-c:a", "aac", "-b:a", "160k",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                str(out),
            ]
            return c

        rendered = False
        if use_xfade:
            fc = (
                _video_filter_complex(lens_xfade, boundaries, audio_dur, over_w, over_h,
                                      pad_y, eq_filter, ass_esc, fonts_esc, accent_hex)
                + ";\n"
                + _audio_filter_complex(n, audio_dur, has_music, sfx_events,
                                        with_deesser=_ffmpeg_has_filter("deesser"))
            )
            # A timeout must degrade exactly like a nonzero return code — fall
            # through to the simple pipeline, never freeze an autonomous run.
            try:
                r = subprocess.run(_build_cmd(fc, lens_xfade, with_sfx=bool(sfx_events)),
                                   capture_output=True, text=True, timeout=RENDER_TIMEOUT)
                if r.returncode == 0:
                    rendered = True
                else:
                    print(f"[render] full mix failed (rc={r.returncode}: "
                          f"{r.stderr[-200:].strip()}), retrying with simple pipeline…")
            except subprocess.TimeoutExpired:
                print(f"[render] full mix timed out after {RENDER_TIMEOUT}s — "
                      f"retrying with simple pipeline…")

        if not rendered:
            fc = (
                _video_filter_complex_concat(lens_concat, audio_dur, over_w, over_h,
                                             pad_y, eq_filter, ass_esc, fonts_esc, accent_hex)
                + ";\n"
                + _audio_filter_simple(n, audio_dur, has_music,
                                       with_loudnorm=_ffmpeg_has_filter("loudnorm"))
            )
            try:
                r2 = subprocess.run(_build_cmd(fc, lens_concat, with_sfx=False),
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
