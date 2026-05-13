"""
Subtitle and text overlay utilities.

Generates an SRT file from script sections and burns subtitles +
title card overlays into the video using FFmpeg drawtext / subtitles filter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

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


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + list(args)
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(
    sections: list[dict],
    total_duration: float,
    hook: str = "",
    call_to_action: str = "",
) -> str:
    """
    Build an SRT subtitle string from script sections.
    Distributes sections evenly across the video duration.
    """
    all_segments = []
    if hook:
        all_segments.append(hook)
    for s in sections:
        text = s.get("script", s.get("heading", ""))
        if text:
            # Take first 100 chars — subtitles should be short
            all_segments.append(text[:100].replace("\n", " "))
    if call_to_action:
        all_segments.append(call_to_action[:80])

    if not all_segments:
        return ""

    seg_duration = total_duration / len(all_segments)
    lines = []
    for i, text in enumerate(all_segments):
        start = i * seg_duration
        end = start + seg_duration - 0.3
        lines.append(str(i + 1))
        lines.append(f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}")
        # Wrap at ~42 chars per line
        words = text.split()
        subtitle_lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 <= 42:
                current = f"{current} {word}".strip()
            else:
                subtitle_lines.append(current)
                current = word
        if current:
            subtitle_lines.append(current)
        lines.extend(subtitle_lines[:2])
        lines.append("")

    return "\n".join(lines)


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    font_size: int = 22,
    font_color: str = "white",
    outline_color: str = "black",
) -> Path:
    """Burn SRT subtitles into video using FFmpeg subtitles filter."""
    font = _find_font()
    style = (
        f"FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,Outline=2,Shadow=1,"
        f"Alignment=2,MarginV=30"
    )
    if font:
        style += f",FontName={Path(font).stem}"

    # Escape path for ffmpeg filter
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

    _ffmpeg(
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    )
    return output_path


def add_title_card(
    video_path: Path,
    title: str,
    output_path: Path,
    duration: float = 2.5,
    font_size: int = 52,
) -> Path:
    """
    Overlay a semi-transparent title card for the first `duration` seconds.
    Uses FFmpeg drawtext filter.
    """
    font = _find_font()
    safe_title = title[:60].replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

    font_arg = f":fontfile='{font}'" if font else ""
    fade_out = max(0, duration - 0.5)

    vf = (
        # Dark rectangle behind text
        f"drawbox=x=0:y=ih*0.35:w=iw:h=ih*0.30:color=black@0.55:t=fill"
        f",drawtext=text='{safe_title}'"
        f"{font_arg}"
        f":fontsize={font_size}:fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
        f":shadowcolor=black:shadowx=2:shadowy=2"
        f":enable='between(t,0,{duration})'"
        f":alpha='if(lt(t,{fade_out}),1,1-(t-{fade_out})/{duration-fade_out})'"
    )

    _ffmpeg(
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    )
    return output_path


def apply_overlays(
    video_path: Path,
    output_path: Path,
    script_sections: list[dict],
    hook: str = "",
    call_to_action: str = "",
    title: str = "",
    tmp_dir: Optional[Path] = None,
) -> Path:
    """
    Full overlay pipeline: title card → burned subtitles.
    Returns path to the finished video.
    """
    tmp = Path(tmp_dir or output_path.parent / "_tmp_overlays")
    tmp.mkdir(parents=True, exist_ok=True)

    current = video_path

    # Step 1: title card
    if title:
        try:
            titled = tmp / "titled.mp4"
            add_title_card(current, title, titled)
            current = titled
            console.print("[dim]title card added[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Title card skipped ({exc})[/yellow]")

    # Step 2: subtitles
    try:
        duration = _get_video_duration(current)
        if duration > 0 and script_sections:
            srt_content = build_srt(script_sections, duration, hook, call_to_action)
            if srt_content.strip():
                srt_path = tmp / "subtitles.srt"
                srt_path.write_text(srt_content, encoding="utf-8")
                subtitled = tmp / "subtitled.mp4"
                burn_subtitles(current, srt_path, subtitled)
                current = subtitled
                console.print("[dim]subtitles burned in[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Subtitles skipped ({exc})[/yellow]")

    # Copy to final output
    if current != output_path:
        import shutil
        shutil.copy2(current, output_path)

    return output_path
