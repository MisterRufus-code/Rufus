#!/usr/bin/env python3
"""Rufus – Autonomous Shorts Renderer (v2)"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import edge_tts
from faster_whisper import WhisperModel

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"

VOICE        = "en-US-ChristopherNeural"
W, H         = 1080, 1920
FPS          = 30
MAX_DUR      = 57.0
CLUSTER_SIZE = 2        # words per subtitle cluster

# Hormozi palette: white → yellow → green
COLOURS  = ["&H00FFFFFF", "&H0000FFFF", "&H0000FF00"]
FONT     = "Arial"
FONTSIZE = 90           # PlayResY=1920 → large, readable
MARGIN_V = 734          # golden ratio from bottom: 1920/1.618 ≈ 734


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_niche() -> dict:
    data   = json.loads((CONFIG_DIR / "niches.json").read_text())
    active = data["active"]
    return data["niches"][active]


# ── ASS subtitle builder ────────────────────────────────────────────────────────

def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _cluster_words(segments):
    words = [w for seg in segments for w in seg.words]
    for i in range(0, len(words), CLUSTER_SIZE):
        group = words[i:i + CLUSTER_SIZE]
        yield (
            group[0].start,
            group[-1].end,
            " ".join(w.word.strip().upper() for w in group),
        )


def build_ass(segments, ass_path: Path) -> None:
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
    lines      = []
    colour_idx = 0
    for start, end, text in _cluster_words(segments):
        c       = COLOURS[colour_idx % len(COLOURS)]
        colour_idx += 1
        # Pop-in animation: scale 120→100 over 80ms + colour
        styled  = f"{{\\c{c}\\fscx120\\fscy120\\t(0,80,\\fscx100\\fscy100)}}{text}"
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{styled}")

    ass_path.write_text(header + "\n".join(lines), encoding="utf-8")


# ── TTS ─────────────────────────────────────────────────────────────────────────

async def _tts(script: str, mp3_path: Path) -> None:
    comm = edge_tts.Communicate(script, VOICE)
    await comm.save(str(mp3_path))


# ── Riser SFX ───────────────────────────────────────────────────────────────────

def _add_riser(mp3_path: Path, duration: float) -> Path:
    out = mp3_path.with_name(mp3_path.stem + "_mix.mp3")
    # Sine wave rising 180 Hz → 500 Hz, amplitude builds 0→20%
    riser = (
        f"aevalsrc=sin(2*PI*(180+320*t/{duration:.3f})*t)"
        f"*0.20*(t/{duration:.3f}):s=44100:d={duration:.3f}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3_path),
            "-f", "lavfi", "-i", riser,
            "-filter_complex", "[0][1]amix=inputs=2:duration=first:weights=1 0.3",
            str(out),
        ],
        check=True,
    )
    return out


# ── Renderer ────────────────────────────────────────────────────────────────────

def render(script: str, bg_path: Path, out_dir: Path) -> Path:
    if not Path(bg_path).exists():
        raise FileNotFoundError(f"Background video not found: {bg_path}")

    niche_cfg = _load_niche()
    eq_filter = niche_cfg.get("ffmpeg_eq", "eq=contrast=1.1:saturation=1.0")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "media_library" / "temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp     = int(time.time())
    mp3       = tmp_dir / f"{stamp}.mp3"
    mix       = tmp_dir / f"{stamp}_mix.mp3"
    ass       = tmp_dir / f"{stamp}.ass"
    out       = out_dir / f"short_{stamp}.mp4"
    audio_src = mp3     # may be replaced by riser mix

    try:
        # 1. TTS
        print("[1/4] Generating voice…")
        asyncio.run(_tts(script, mp3))

        # 2. Whisper transcription
        print("[2/4] Transcribing…")
        model    = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _  = model.transcribe(str(mp3), word_timestamps=True)
        segments = list(segs)
        if not segments:
            raise RuntimeError("Whisper produced no segments")

        audio_dur = max((w.end for seg in segments for w in seg.words), default=0)
        audio_dur = min(max(audio_dur + 0.3, 1.0), MAX_DUR)

        # 3. Subtitles
        print("[3/4] Building subtitles…")
        build_ass(segments, ass)

        # Riser (non-fatal if FFmpeg lavfi unavailable)
        try:
            mix       = _add_riser(mp3, audio_dur)
            audio_src = mix
            print("      Riser SFX added ✓")
        except Exception as e:
            print(f"      Riser skipped ({e})")

        # 4. FFmpeg render
        print("[4/4] Rendering video…")

        # Ken Burns: scale 10% larger, slow left-to-right pan
        over_w    = int(W * 1.10)
        over_h    = int(H * 1.10)
        pad_y     = (over_h - H) // 2
        ass_esc   = str(ass).replace("\\", "/")

        vf = (
            f"scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:({over_w}-{W})*t/{audio_dur:.3f}:{pad_y},"
            f"setsar=1,"
            f"{eq_filter},"
            f"vignette=PI/4,"
            f"ass='{ass_esc}'"
        )

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-stream_loop", "-1", "-i", str(bg_path),
                "-i", str(audio_src),
                "-t", f"{audio_dur:.3f}",
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=True,
        )

    finally:
        for f in (mp3, mix, ass):
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
