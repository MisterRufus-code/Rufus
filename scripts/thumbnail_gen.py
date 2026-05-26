#!/usr/bin/env python3
"""
thumbnail_gen.py
Generates a custom thumbnail from a rendered Short.
1. Probes video duration
2. Extracts a striking frame ~30% in (past intro, before close)
3. Overlays a 2-3 word punchy text from the script's hook
4. Returns the thumbnail JPG path

Pure FFmpeg + drawtext – no Pillow dependency.
"""

import json
import subprocess
import sys
from pathlib import Path

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _find_font() -> str:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise RuntimeError(
        "No bold TTF font found. Install DejaVu: apt install fonts-dejavu-core"
    )


def _probe_duration(video_path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                d = float(stream.get("duration", 0))
                if d > 0:
                    return d
    except Exception:
        pass
    return 20.0


def _hook_text(script: str, max_words: int = 3) -> str:
    """First line of the script, up to max_words, uppercased."""
    first = script.strip().split("\n")[0]
    # Drop trailing punctuation, keep readable
    words = first.replace(",", "").replace(".", "").split()[:max_words]
    text  = " ".join(words).upper().strip()
    # ffmpeg drawtext: escape single quotes and colons by removing them
    return text.replace("'", "").replace(":", "").replace("\\", "")


def make_thumbnail(video_path: Path, script: str, out_path: Path = None) -> Path:
    """Render a JPG thumbnail with a hook-text overlay over a striking frame."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if out_path is None:
        out_path = video_path.with_suffix(".thumb.jpg")
    out_path = Path(out_path)

    duration = _probe_duration(video_path)
    ts       = max(0.5, duration * 0.30)
    text     = _hook_text(script)
    font     = _find_font()

    if not text:
        text = "WATCH THIS"

    drawtext = (
        f"drawtext=text='{text}':"
        f"fontfile='{font}':"
        f"fontsize=180:"
        f"fontcolor=yellow:"
        f"bordercolor=black:borderw=10:"
        f"shadowcolor=black@0.7:shadowx=4:shadowy=4:"
        f"x=(w-text_w)/2:y=h*0.18"
    )

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts:.2f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", drawtext,
            "-q:v", "2",
            str(out_path),
        ],
        check=True,
    )

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python thumbnail_gen.py <video.mp4> '<script text>'")
        sys.exit(1)

    video  = Path(sys.argv[1])
    script = sys.argv[2]
    out    = make_thumbnail(video, script)
    print(f"THUMBNAIL={out}")
