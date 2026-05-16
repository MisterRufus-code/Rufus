"""
Subtitle and text overlay utilities.

Generates an SRT file from script sections and burns subtitles +
title card overlays into the video using FFmpeg drawtext / subtitles filter.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

_CLIP_REF_RE = re.compile(
    r"(\[?(?:CLIP|footage|clip)\s*[\d_]+\]?:?\s*)", re.IGNORECASE
)


def _strip_clip_refs(text: str) -> str:
    """Remove LLM-echoed clip labels like '[CLIP 1]:', 'footage_2:' from text."""
    return _CLIP_REF_RE.sub("", text).strip()


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str, max_chars: int = 90) -> list[str]:
    """Split text into sentence-level chunks suitable for subtitle display."""
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    result: list[str] = []
    for sent in sents:
        if len(sent) <= max_chars:
            result.append(sent)
        else:
            # Break long sentences at commas
            parts = [p.strip() for p in sent.split(',') if p.strip()]
            buf = ""
            for part in parts:
                candidate = f"{buf}, {part}".lstrip(", ") if buf else part
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    if buf:
                        result.append(buf)
                    buf = part
            if buf:
                result.append(buf)
    return result or [text[:max_chars]]


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
    section_durations: list[float] | None = None,
) -> str:
    """
    Build an SRT subtitle string with sentence-level granularity.

    If section_durations is provided (one float per section in the same order
    as hook + sections + call_to_action), subtitle timing is proportional to
    actual audio duration rather than evenly split.
    """
    raw_segs: list[str] = []
    if hook:
        raw_segs.append(_strip_clip_refs(hook.strip()))
    for s in sections:
        text = _strip_clip_refs(s.get("script", s.get("heading", "")).strip())
        if text:
            raw_segs.append(text.replace("\n", " "))
    if call_to_action:
        raw_segs.append(_strip_clip_refs(call_to_action.strip()))

    raw_segs = [seg for seg in raw_segs if seg.strip()]
    if not raw_segs or total_duration <= 0:
        return ""

    # Per-segment time budget — use audio durations when available
    if section_durations and len(section_durations) >= len(raw_segs):
        raw_times = list(section_durations[:len(raw_segs)])
        total_audio = sum(raw_times)
        if total_audio > 0:
            times = [t * total_duration / total_audio for t in raw_times]
        else:
            times = [total_duration / len(raw_segs)] * len(raw_segs)
    else:
        seg_dur = total_duration / len(raw_segs)
        times = [seg_dur] * len(raw_segs)

    lines: list[str] = []
    entry_num = 1
    cursor = 0.0

    for seg_text, seg_time in zip(raw_segs, times):
        sentences = _split_sentences(seg_text)
        per_sent = seg_time / len(sentences)
        for sentence in sentences:
            start = cursor
            end = cursor + per_sent - 0.1
            cursor += per_sent

            # Word-wrap at 50 chars per line, max 2 lines
            words = sentence.split()
            sub_lines: list[str] = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 50:
                    current = f"{current} {word}".strip()
                else:
                    sub_lines.append(current)
                    current = word
            if current:
                sub_lines.append(current)

            lines.append(str(entry_num))
            lines.append(f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}")
            lines.extend(sub_lines[:2])
            lines.append("")
            entry_num += 1

    return "\n".join(lines)


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    font_size: int = 30,
    font_color: str = "white",
    outline_color: str = "black",
) -> Path:
    """Burn SRT subtitles into video using FFmpeg subtitles filter."""
    font = _find_font()
    # Bold white text, thick black outline, 50px bottom margin
    style = (
        f"FontSize={font_size},Bold=1,PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,Outline=3,Shadow=2,"
        f"Alignment=2,MarginV=50"
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
    fade_out = max(0.0, duration - 0.5)
    fade_range = max(duration - fade_out, 0.01)  # avoid division by zero

    vf = (
        # Dark rectangle behind text — enable matches drawtext so it disappears with the title
        f"drawbox=x=0:y=ih*0.35:w=iw:h=ih*0.30:color=black@0.55:t=fill"
        f":enable='between(t,0,{duration})'"
        f",drawtext=text='{safe_title}'"
        f"{font_arg}"
        f":fontsize={font_size}:fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
        f":shadowcolor=black:shadowx=2:shadowy=2"
        f":enable='between(t,0,{duration})'"
        f":alpha='if(lt(t,{fade_out}),1,1-(t-{fade_out})/{fade_range})'"
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
    section_durations: list[float] | None = None,
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
            srt_content = build_srt(script_sections, duration, hook, call_to_action, section_durations)
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
