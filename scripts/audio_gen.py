#!/usr/bin/env python3
"""
Rufus – Autonomous Shorts Renderer
Accepts a JSON payload via stdin or CLI args, produces a 1080x1920 MP4.

Usage:
    python audio_gen.py --script "Your script here" \
                        --bg /path/to/background.mp4 \
                        --out /path/to/output/

    or pipe JSON:
    echo '{"script":"...", "bg":"...", "out":"..."}' | python audio_gen.py
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import edge_tts
from faster_whisper import WhisperModel

# ── Config ─────────────────────────────────────────────────────────────────────
VOICE = "en-US-ChristopherNeural"
FONT_NAME = "Arial"
FONT_SIZE = 100                     # ASS points at PlayResY=1920 → big centred text
W, H = 1080, 1920
FPS = 30
MAX_DUR = 57.0                      # YouTube Shorts hard limit

# Word-level colour cycling (Hormozi palette)
WORD_COLOURS = ["&H00FFFFFF", "&H0000FFFF", "&H0000FF00"]  # white / yellow / green
PRIMARY_COLOUR = "&H00FFFFFF"
OUTLINE_COLOUR = "&H00000000"
BACK_COLOUR = "&H00000000"


# ── ASS subtitle builder ────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    """Convert float seconds → ASS timestamp  H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(segments, ass_path: Path) -> None:
    """
    Build an ASS file with one event per word, cycling colours.
    Each word appears centred, stays on screen alone.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOUR},&H000000FF,{OUTLINE_COLOUR},{BACK_COLOUR},-1,0,0,0,100,100,0,0,1,5,3,5,60,60,960,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    colour_idx = 0
    for seg in segments:
        for word in seg.words:
            colour = WORD_COLOURS[colour_idx % len(WORD_COLOURS)]
            colour_idx += 1
            text = word.word.strip().upper()
            # Override primary colour inline
            styled = f"{{\\c{colour}}}{text}"
            events.append(
                f"Dialogue: 0,{_ts(word.start)},{_ts(word.end)},Default,,0,0,0,,{styled}"
            )

    ass_path.write_text(header + "\n".join(events), encoding="utf-8")


# ── Core pipeline ───────────────────────────────────────────────────────────────

async def _tts(script: str, mp3_path: Path) -> None:
    comm = edge_tts.Communicate(script, VOICE)
    await comm.save(str(mp3_path))


def render(script: str, bg_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time())
    mp3 = tmp_dir / f"{stamp}.mp3"
    ass = tmp_dir / f"{stamp}.ass"
    out = out_dir / f"short_{stamp}.mp4"

    # 1. TTS
    print("[1/4] Generating voice…")
    asyncio.run(_tts(script, mp3))

    # 2. Transcribe with word timestamps
    print("[2/4] Transcribing…")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(mp3), word_timestamps=True)
    segments = list(segments)

    # 3. Build ASS subtitles
    print("[3/4] Building subtitles…")
    build_ass(segments, ass)

    # 4. FFmpeg render
    print("[4/4] Rendering video…")
    audio_dur = sum(w.end for seg in segments for w in seg.words)
    audio_dur = min(max(audio_dur + 0.3, 1.0), MAX_DUR)

    # Escape backslashes and colons in path for libass on Linux
    ass_escaped = str(ass).replace("\\", "/").replace(":", "\\:")

    ffmpeg_cmd = (
        f'ffmpeg -y -loglevel error '
        f'-stream_loop -1 -i "{bg_path}" '
        f'-i "{mp3}" '
        f'-t {audio_dur:.3f} '
        f'-vf "scale={W}:{H}:flags=lanczos,setsar=1,'
        f"ass='{ass_escaped}'\""
        f' -c:v libx264 -preset fast -crf 20 '
        f'-c:a aac -b:a 128k '
        f'-r {FPS} -pix_fmt yuv420p '
        f'"{out}"'
    )
    ret = os.system(ffmpeg_cmd)
    if ret != 0:
        raise RuntimeError(f"FFmpeg exited with code {ret}")

    # Cleanup temp
    mp3.unlink(missing_ok=True)
    ass.unlink(missing_ok=True)

    print(f"Done → {out}")
    return out


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    # Support piped JSON (from n8n SSH node)
    if not sys.stdin.isatty():
        payload = json.load(sys.stdin)
        script  = payload["script"]
        bg      = Path(payload["bg"])
        out     = Path(payload["out"])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--script", required=True)
        parser.add_argument("--bg",     required=True)
        parser.add_argument("--out",    required=True)
        args   = parser.parse_args()
        script = args.script
        bg     = Path(args.bg)
        out    = Path(args.out)

    result = render(script, bg, out)
    # Print path so n8n can capture it
    print(f"OUTPUT_PATH={result}")


if __name__ == "__main__":
    main()
