#!/usr/bin/env python3
"""Rufus – Autonomous Shorts Renderer (v2.2)

Changes from v2.1:
- No riser SFX (sounded like a siren)
- 1 word per subtitle cluster (Hormozi style)
- Multi-clip rendering: all candidate videos cut together inside one Short
- MAX_DUR raised to 60.0 (YouTube Shorts hard cap) — value content needs the room
"""

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import edge_tts
from faster_whisper import WhisperModel

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"

VOICE        = "en-US-ChristopherNeural"
VOICE_RATE   = "+20%"    # tighter pacing, reduces inter-sentence gaps
W, H         = 1080, 1920
FPS          = 30
MAX_DUR      = 60.0     # YouTube Shorts hard cap
MIN_DUR      = 30.0     # below this, value content feels rushed
CLUSTER_SIZE = 1        # 1 word at a time — Hormozi style

# Subtitle palette: white for normal words, green for numbers/amounts/percentages
WHITE = "&H00FFFFFF"
GREEN = "&H0000FF00"

_HIGHLIGHT_RE = re.compile(r'[\d$%]')

FONT     = "Arial"
FONTSIZE = 90
MARGIN_V = 734          # golden ratio from bottom: 1920/1.618 ≈ 734


# ── Whisper singleton ───────────────────────────────────────────────────────────

_whisper_model = None

def _whisper() -> WhisperModel:
    """Lazy-init Whisper once per process — saves 5-10s per render."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_niche() -> dict:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active]


# ── ASS subtitle builder ────────────────────────────────────────────────────────

def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _cluster_words(segments, audio_dur: float):
    """Yield (start, end, text) clusters, clipped to audio_dur."""
    words = [w for seg in segments for w in seg.words]
    for i in range(0, len(words), CLUSTER_SIZE):
        group = words[i:i + CLUSTER_SIZE]
        start = group[0].start
        end   = group[-1].end
        if start >= audio_dur:
            break
        end = min(end, audio_dur)
        yield (
            start,
            end,
            " ".join(w.word.strip().upper() for w in group),
        )


def build_ass(segments, ass_path: Path, audio_dur: float) -> None:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nCollisions: Normal\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT},{FONTSIZE},"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for start, end, text in _cluster_words(segments, audio_dur):
        # Green for numbers, dollar amounts, percentages — white for everything else
        c      = GREEN if _HIGHLIGHT_RE.search(text) else WHITE
        styled = f"{{\\c{c}\\fscx120\\fscy120\\t(0,80,\\fscx100\\fscy100)}}{text}"
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{styled}")

    ass_path.write_text(header + "\n".join(lines), encoding="utf-8")


# ── TTS ─────────────────────────────────────────────────────────────────────────

async def _tts(script: str, mp3_path: Path) -> None:
    comm = edge_tts.Communicate(script, VOICE, rate=VOICE_RATE)
    await comm.save(str(mp3_path))


# ── Renderer ────────────────────────────────────────────────────────────────────

def _video_filter_complex(
    n: int, over_w: int, over_h: int, pad_y: int,
    seg_dur: float, eq_filter: str, ass_esc: str,
) -> str:
    """Build FFmpeg filter_complex for N clips: Ken Burns per clip → concat → grade → subtitles."""
    # Ken Burns directions: alternate between 4 motion types so consecutive clips look different.
    # x_exprs: left→right, right→left
    # y_exprs: fixed center, slow tilt down, slow tilt up
    x_exprs = [f"({over_w}-{W})*t/{seg_dur:.3f}", f"({over_w}-{W})*(1-t/{seg_dur:.3f})"]
    y_exprs = [str(pad_y), f"({over_h}-{H})*t/{seg_dur:.3f}", f"({over_h}-{H})*(1-t/{seg_dur:.3f})"]
    parts = []
    for i in range(n):
        pan_x = x_exprs[i % len(x_exprs)]           # alternates L→R, R→L per clip
        pan_y = y_exprs[i % len(y_exprs)]            # cycles center, down, up per clip
        parts.append(
            f"[{i}:v]scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:{pan_x}:{pan_y},"
            f"setsar=1[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vraw]")
    parts.append(f"[vraw]{eq_filter},vignette=PI/4,ass='{ass_esc}'[vout]")
    return ";\n".join(parts)


def render(script: str, bg_paths: "Path | list[Path]", out_dir: Path) -> Path:
    # Accept a single path or a list
    if isinstance(bg_paths, Path):
        bg_paths = [bg_paths]
    bg_paths = [Path(p) for p in bg_paths if Path(p).exists()]
    if not bg_paths:
        raise FileNotFoundError("No valid background video files found")

    niche_cfg = _load_niche()
    eq_filter = niche_cfg.get("ffmpeg_eq", "eq=contrast=1.1:saturation=1.0")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "media_library" / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp     = int(time.time())
    mp3       = tmp_dir / f"{stamp}.mp3"
    ass       = tmp_dir / f"{stamp}.ass"
    out       = out_dir / f"short_{stamp}.mp4"
    audio_src = mp3

    try:
        # 1. TTS
        print("[1/4] Generating voice…")
        asyncio.run(_tts(script, mp3))

        # 2. Whisper transcription (cached model)
        print("[2/4] Transcribing…")
        segs, _  = _whisper().transcribe(str(mp3), word_timestamps=True)
        segments = list(segs)
        if not segments:
            raise RuntimeError("Whisper produced no segments")

        audio_dur = max((w.end for seg in segments for w in seg.words), default=0)
        audio_dur = min(max(audio_dur + 0.3, 1.0), MAX_DUR)
        if audio_dur < MIN_DUR:
            print(f"      ⚠ audio is only {audio_dur:.1f}s — value content target is ≥{MIN_DUR:.0f}s")

        # 3. Subtitles (clipped to audio_dur)
        print("[3/4] Building subtitles…")
        build_ass(segments, ass, audio_dur)

        # 4. FFmpeg render — multi-clip concat
        n        = len(bg_paths)
        seg_dur  = audio_dur / n
        over_w   = int(W * 1.10)
        over_h   = int(H * 1.10)
        pad_y    = (over_h - H) // 2
        ass_esc  = str(ass).replace("\\", "/")
        print(f"[4/4] Rendering {n} clip{'s' if n > 1 else ''} → {audio_dur:.1f}s ({seg_dur:.1f}s each)…")

        fc = _video_filter_complex(n, over_w, over_h, pad_y, seg_dur, eq_filter, ass_esc)

        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for bg in bg_paths:
            cmd += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(bg)]
        cmd += ["-i", str(audio_src)]
        cmd += [
            "-filter_complex", fc,
            "-map", "[vout]",
            "-map", f"{n}:a",
            "-t", f"{audio_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-r", str(FPS), "-pix_fmt", "yuv420p",
            str(out),
        ]
        subprocess.run(cmd, check=True)

    finally:
        for f in (mp3, ass):
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Done → {out}")
    return out


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    if not sys.stdin.isatty():
        payload = json.load(sys.stdin)
        script  = payload["script"]
        bg      = Path(payload["bg"])
        out     = Path(payload["out"])
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--script", required=True)
        p.add_argument("--bg",     required=True)
        p.add_argument("--out",    required=True)
        a = p.parse_args()
        script, bg, out = a.script, Path(a.bg), Path(a.out)

    result = render(script, bg, out)
    print(f"OUTPUT_PATH={result}")


if __name__ == "__main__":
    main()
