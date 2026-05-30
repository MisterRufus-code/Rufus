#!/usr/bin/env python3
"""Rufus – Autonomous Shorts Renderer (v3.0)

Changes from v2.2:
- Background music: auto-fetched by mood (Jamendo → archive.org), ducked under voice at vol=0.16
- Smooth transitions: xfade dissolve between Ken-Burns clips (~0.3s), fade-in/out at edges
- Display font: Anton (downloaded to assets/fonts/ at first run), falls back to Arial
- Voice pacing: rate dropped from +20% to +6% for deliberate, authoritative read
- Guard: audio_dur=0 / n=0 raise early instead of ZeroDivision
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
FONTS_DIR  = ROOT / "assets" / "fonts"

VOICE        = "en-US-ChristopherNeural"
VOICE_RATE   = "+6%"       # deliberate, authoritative read
W, H         = 1080, 1920
FPS          = 30
MAX_DUR      = 60.0
MIN_DUR      = 30.0
CLUSTER_SIZE = 1           # 1 word at a time — Hormozi style

XFADE_DUR    = 0.30        # crossfade duration between clips (seconds)
FADE_EDGE    = 0.40        # fade-in from black / fade-to-black duration (seconds)
MUSIC_VOL    = 0.14        # music ducked well under voice

WHITE = "&H00FFFFFF"
GREEN = "&H0000FF00"

_HIGHLIGHT_RE = re.compile(r'[\d$%]')

FONT_NAME = "Anton"        # downloaded to assets/fonts/; Arial fallback if missing
FONT_FILE = FONTS_DIR / "Anton-Regular.ttf"
FONTSIZE  = 88
MARGIN_V  = 734


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


# ── Whisper singleton ────────────────────────────────────────────────────────────

_whisper_model = None

def _whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ── Config ───────────────────────────────────────────────────────────────────────

def _load_niche() -> dict:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active]


def _active_niche_name() -> str:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
    return os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]


# ── ASS subtitle builder ─────────────────────────────────────────────────────────

def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _cluster_words(segments, audio_dur: float):
    words = [w for seg in segments for w in seg.words]
    for i in range(0, len(words), CLUSTER_SIZE):
        group = words[i:i + CLUSTER_SIZE]
        start = group[0].start
        end   = group[-1].end
        if start >= audio_dur:
            break
        end = min(end, audio_dur)
        yield start, end, " ".join(w.word.strip().upper() for w in group)


def build_ass(segments, ass_path: Path, audio_dur: float, font_name: str = "Arial") -> None:
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
        c      = GREEN if _HIGHLIGHT_RE.search(text) else WHITE
        styled = f"{{\\c{c}\\fscx120\\fscy120\\t(0,80,\\fscx100\\fscy100)}}{text}"
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{styled}")
    ass_path.write_text(header + "\n".join(lines), encoding="utf-8")


# ── TTS ──────────────────────────────────────────────────────────────────────────

async def _tts(script: str, mp3_path: Path) -> None:
    comm = edge_tts.Communicate(script, VOICE, rate=VOICE_RATE)
    await comm.save(str(mp3_path))


# ── FFmpeg filter_complex builders ───────────────────────────────────────────────

def _video_filter_complex(
    n: int, over_w: int, over_h: int, pad_y: int,
    seg_dur: float, eq_filter: str, ass_esc: str, fonts_dir_esc: str,
) -> str:
    """Build filter_complex: Ken Burns per clip → xfade transitions → grade → subs."""
    x_exprs = [
        f"({over_w}-{W})*t/{seg_dur:.3f}",
        f"({over_w}-{W})*(1-t/{seg_dur:.3f})",
    ]
    y_exprs = [
        str(pad_y),
        f"({over_h}-{H})*t/{seg_dur:.3f}",
        f"({over_h}-{H})*(1-t/{seg_dur:.3f})",
    ]
    parts = []
    for i in range(n):
        pan_x = x_exprs[i % len(x_exprs)]
        pan_y = y_exprs[i % len(y_exprs)]
        parts.append(
            f"[{i}:v]scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:{pan_x}:{pan_y},"
            f"setsar=1[v{i}]"
        )

    if n == 1:
        # Single clip — fade in/out only
        total = seg_dur
        fade_out_st = max(0.0, total - FADE_EDGE)
        parts.append(
            f"[v0]fade=t=in:st=0:d={FADE_EDGE:.3f},"
            f"fade=t=out:st={fade_out_st:.3f}:d={FADE_EDGE:.3f}[vraw]"
        )
    else:
        # Chain xfade between clips
        # Offset for i-th xfade (1-indexed) = i * (seg_dur - XFADE_DUR)
        current = "v0"
        for i in range(1, n):
            offset   = i * (seg_dur - XFADE_DUR)
            out_name = f"xf{i}"
            parts.append(
                f"[{current}][v{i}]xfade=transition=dissolve:"
                f"duration={XFADE_DUR:.3f}:offset={offset:.3f}[{out_name}]"
            )
            current = out_name

        total = n * seg_dur - (n - 1) * XFADE_DUR
        fade_out_st = max(0.0, total - FADE_EDGE)
        parts.append(
            f"[{current}]fade=t=in:st=0:d={FADE_EDGE:.3f},"
            f"fade=t=out:st={fade_out_st:.3f}:d={FADE_EDGE:.3f}[vraw]"
        )

    # fontsdir path for custom font; ASS filter uses single quotes — escape internal ones
    parts.append(
        f"[vraw]{eq_filter},vignette=PI/4,"
        f"ass='{ass_esc}':fontsdir='{fonts_dir_esc}'[vout]"
    )
    return ";\n".join(parts)


def _audio_filter_complex(n: int, audio_dur: float, music_path: Path | None) -> str:
    """Build audio filter: voice only, or voice + ducked music."""
    if not music_path:
        return ""
    fade_out_st = max(0.0, audio_dur - 1.5)
    return (
        f"[{n+1}:a]volume={MUSIC_VOL},"
        f"afade=t=in:st=0:d=1.5,"
        f"afade=t=out:st={fade_out_st:.3f}:d=1.5[music];"
        f"[{n}:a][music]amix=inputs=2:duration=first:dropout_transition=1[aout]"
    )


# ── Renderer ─────────────────────────────────────────────────────────────────────

def render(script: str, bg_paths: "Path | list[Path]", out_dir: Path,
           music_path: Path | None = None) -> Path:
    if isinstance(bg_paths, Path):
        bg_paths = [bg_paths]
    bg_paths = [Path(p) for p in bg_paths if Path(p).exists()]
    if not bg_paths:
        raise FileNotFoundError("No valid background video files found")

    n = len(bg_paths)
    if n == 0:
        raise FileNotFoundError("No valid background video files found")

    niche_cfg = _load_niche()
    niche_name = _active_niche_name()
    eq_filter  = niche_cfg.get("ffmpeg_eq", "eq=contrast=1.1:saturation=1.0")

    # Auto-fetch music if not provided
    if music_path is None:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from music_fetcher import fetch_music
            music_path = fetch_music(niche_name)
        except Exception as e:
            print(f"[music] fetch skipped: {e}")
            music_path = None

    font_name = _ensure_font()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "media_library" / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time())
    mp3   = tmp_dir / f"{stamp}.mp3"
    ass   = tmp_dir / f"{stamp}.ass"
    out   = out_dir / f"short_{stamp}.mp4"

    try:
        print("[1/4] Generating voice…")
        asyncio.run(_tts(script, mp3))

        print("[2/4] Transcribing…")
        segs, _ = _whisper().transcribe(str(mp3), word_timestamps=True)
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
        build_ass(segments, ass, audio_dur, font_name=font_name)

        seg_dur     = audio_dur / n
        over_w      = int(W * 1.10)
        over_h      = int(H * 1.10)
        pad_y       = (over_h - H) // 2
        ass_esc     = str(ass).replace("\\", "/").replace("'", "\\'")
        fonts_esc   = str(FONTS_DIR).replace("\\", "/").replace("'", "\\'")
        has_music   = music_path is not None and Path(music_path).exists()

        print(f"[4/4] Rendering {n} clip{'s' if n > 1 else ''} → {audio_dur:.1f}s"
              f"{' + music' if has_music else ''}…")

        fc_video = _video_filter_complex(n, over_w, over_h, pad_y, seg_dur,
                                         eq_filter, ass_esc, fonts_esc)

        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for bg in bg_paths:
            cmd += ["-stream_loop", "-1", "-t", f"{seg_dur + XFADE_DUR:.3f}", "-i", str(bg)]
        cmd += ["-i", str(mp3)]
        if has_music:
            cmd += ["-stream_loop", "-1", "-t", f"{audio_dur:.3f}", "-i", str(music_path)]

        if has_music:
            afc = _audio_filter_complex(n, audio_dur, music_path)
            cmd += [
                "-filter_complex", fc_video + ";\n" + afc,
                "-map", "[vout]",
                "-map", "[aout]",
            ]
        else:
            cmd += [
                "-filter_complex", fc_video,
                "-map", "[vout]",
                "-map", f"{n}:a",
            ]

        cmd += [
            "-t", f"{audio_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-r", str(FPS), "-pix_fmt", "yuv420p",
            str(out),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # xfade can fail on very short clips — retry with hard concat fallback
            print(f"[render] xfade failed ({result.stderr[-200:]}), retrying with hard concat…")
            fc_fallback = _video_filter_complex_concat(n, over_w, over_h, pad_y, seg_dur,
                                                        eq_filter, ass_esc, fonts_esc)
            cmd2 = ["ffmpeg", "-y", "-loglevel", "error"]
            for bg in bg_paths:
                cmd2 += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(bg)]
            cmd2 += ["-i", str(mp3)]
            if has_music:
                cmd2 += ["-stream_loop", "-1", "-t", f"{audio_dur:.3f}", "-i", str(music_path)]
            if has_music:
                afc2 = _audio_filter_complex(n, audio_dur, music_path)
                cmd2 += ["-filter_complex", fc_fallback + ";\n" + afc2,
                         "-map", "[vout]", "-map", "[aout]"]
            else:
                cmd2 += ["-filter_complex", fc_fallback,
                         "-map", "[vout]", "-map", f"{n}:a"]
            cmd2 += [
                "-t", f"{audio_dur:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                str(out),
            ]
            subprocess.run(cmd2, check=True)

    finally:
        for f in (mp3, ass):
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Done → {out}")
    return out


def _video_filter_complex_concat(
    n: int, over_w: int, over_h: int, pad_y: int,
    seg_dur: float, eq_filter: str, ass_esc: str, fonts_dir_esc: str,
) -> str:
    """Hard-concat fallback (no transitions). Used when xfade errors."""
    x_exprs = [
        f"({over_w}-{W})*t/{seg_dur:.3f}",
        f"({over_w}-{W})*(1-t/{seg_dur:.3f})",
    ]
    y_exprs = [
        str(pad_y),
        f"({over_h}-{H})*t/{seg_dur:.3f}",
        f"({over_h}-{H})*(1-t/{seg_dur:.3f})",
    ]
    parts = []
    for i in range(n):
        pan_x = x_exprs[i % len(x_exprs)]
        pan_y = y_exprs[i % len(y_exprs)]
        parts.append(
            f"[{i}:v]scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:{pan_x}:{pan_y},"
            f"setsar=1[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vraw]")
    parts.append(
        f"[vraw]{eq_filter},vignette=PI/4,"
        f"ass='{ass_esc}':fontsdir='{fonts_dir_esc}'[vout]"
    )
    return ";\n".join(parts)


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
