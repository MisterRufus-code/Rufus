"""
Shorts renderer — produces 1080x1920 vertical MP4 for YouTube Shorts.

Takes landscape footage, center-crops to 9:16, burns large centered
captions, and trims to ≤60 seconds.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import re

from rich.console import Console

from src.database.models import TimelineClip

console = Console()

_CLIP_REF_RE = re.compile(r"(\[?(?:CLIP|footage|clip)\s*[\d_]+\]?:?\s*)", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

MAX_SHORTS_DURATION = 58.0  # leave 2s buffer under YouTube's 60s limit

FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]


def _find_font() -> str:
    for p in FONT_PATH_CANDIDATES:
        if Path(p).exists():
            return p
    return ""


def _ffmpeg(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + list(args)
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _trim_vertical(clip: TimelineClip, output_path: Path) -> Path:
    """Trim clip and center-crop landscape → 9:16 vertical (1080x1920)."""
    duration = min(clip.out_point - clip.in_point, 8.0)
    _ffmpeg(
        "-ss", str(clip.in_point),
        "-i", clip.asset.path,
        "-t", str(duration),
        "-vf",
        # Scale so height = 1920, then center crop width to 1080
        "scale=-2:1920,crop=1080:1920",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(output_path),
    )
    return output_path


def _concat_clips(clip_paths: list[Path], output_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_file = Path(f.name)
    _ffmpeg(
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    )
    list_file.unlink(missing_ok=True)
    return output_path


def _add_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    _ffmpeg(
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    )
    return output_path


def _trim_to_60s(video_path: Path, output_path: Path) -> Path:
    _ffmpeg(
        "-i", str(video_path),
        "-t", str(MAX_SHORTS_DURATION),
        "-c", "copy",
        str(output_path),
    )
    return output_path


def _clean_caption_text(text: str) -> str:
    """Strip clip references and markdown from caption text."""
    text = _CLIP_REF_RE.sub("", text)
    text = re.sub(r'[*_#`~]', '', text)
    return text.replace("\n", " ").strip()


def _burn_shorts_captions(
    video_path: Path,
    script_sections: list[dict],
    hook: str,
    output_path: Path,
    total_duration: float,
) -> Path:
    """Burn large centered bold captions — the signature Shorts look."""
    font = _find_font()
    font_arg = f":fontfile='{font}'" if font else ""

    # Build sentence-level segments for tighter caption timing
    raw_segs: list[str] = []
    if hook:
        raw_segs.append(_clean_caption_text(hook))
    for s in script_sections:
        text = _clean_caption_text(s.get("script", s.get("heading", "")) or "")
        if text:
            raw_segs.append(text)

    if not raw_segs or total_duration <= 0:
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

    # Expand to sentence level
    segments: list[str] = []
    for seg in raw_segs:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(seg) if s.strip()]
        segments.extend(sents or [seg])

    seg_dur = total_duration / len(segments)

    # Build a drawtext filter chain for each sentence
    filters = []
    for i, text in enumerate(segments):
        t_start = i * seg_dur
        t_end = t_start + seg_dur - 0.15
        # Truncate to ~45 chars for readability on mobile
        display = text[:45].replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
        filters.append(
            f"drawtext=text='{display}'"
            f"{font_arg}"
            f":fontsize=58:fontcolor=white"
            f":borderw=4:bordercolor=black"
            f":x=(w-text_w)/2:y=h*0.72"
            f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
        )

    vf = ",".join(filters)
    _ffmpeg(
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    )
    return output_path


def _get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def render_shorts(
    clips: list[TimelineClip],
    audio_path: Optional[Path],
    output_path: Path,
    script_sections: list[dict] | None = None,
    hook: str = "",
    tmp_dir: Optional[Path] = None,
) -> Path:
    """
    Full Shorts render:
      1. Vertical crop each clip (landscape → 9:16)
      2. Concatenate
      3. Add voiceover
      4. Trim to ≤58s
      5. Burn centered captions
    """
    tmp = Path(tmp_dir or output_path.parent / "_tmp_shorts")
    tmp.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Rendering Shorts — {len(clips)} clips → 9:16[/cyan]")

    # 1. Vertical crop each clip
    trimmed: list[Path] = []
    for i, clip in enumerate(clips):
        out = tmp / f"short_clip_{i:03d}.mp4"
        if clip.asset.asset_type == "image":
            duration = min(clip.out_point - clip.in_point or 3.0, 5.0)
            _ffmpeg(
                "-loop", "1", "-i", clip.asset.path,
                "-t", str(duration),
                "-vf", "scale=-2:1920,crop=1080:1920",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(out),
            )
        else:
            _trim_vertical(clip, out)
        trimmed.append(out)

    # 2. Concat
    concat_path = tmp / "shorts_concat.mp4"
    _concat_clips(trimmed, concat_path)

    # 3. Add audio
    if audio_path and audio_path.exists():
        voiced_path = tmp / "shorts_voiced.mp4"
        _add_audio(concat_path, audio_path, voiced_path)
    else:
        voiced_path = concat_path

    # 4. Trim to 58s
    trimmed_path = tmp / "shorts_trimmed.mp4"
    _trim_to_60s(voiced_path, trimmed_path)

    # 5. Burn captions
    if script_sections:
        duration = _get_duration(trimmed_path)
        captioned_path = tmp / "shorts_captioned.mp4"
        _burn_shorts_captions(trimmed_path, script_sections, hook, captioned_path, duration)
        current = captioned_path
    else:
        current = trimmed_path

    import shutil
    shutil.copy2(current, output_path)
    console.print(f"[bold green]✓ Shorts rendered:[/bold green] {output_path}")
    return output_path
