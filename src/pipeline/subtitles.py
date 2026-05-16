"""
Subtitle and text overlay utilities.

Generates word-level karaoke ASS subtitles from TTS audio via faster-whisper.
Falls back to proportional SRT when faster-whisper is not installed.

Karaoke style: words shown in groups of 5, unspoken = white, currently spoken
word sweeps to yellow (ASS \\kf fill karaoke).
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
    return _CLIP_REF_RE.sub("", text).strip()


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

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


# ---------------------------------------------------------------------------
# Word-level timestamp extraction (faster-whisper)
# ---------------------------------------------------------------------------

def get_word_timestamps(audio_path: Path) -> list[dict]:
    """
    Transcribe *audio_path* with faster-whisper and return per-word timestamps.
    Each dict: {"word": str, "start": float, "end": float}
    Returns [] if faster-whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []

    console.print("[dim]Karaoke: transcribing audio for word timestamps (tiny model)…[/dim]")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    words: list[dict] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                text = w.word.strip()
                if text:
                    words.append({"word": text, "start": w.start, "end": w.end})
    console.print(f"[dim]Karaoke: {len(words)} words timed[/dim]")
    return words


# ---------------------------------------------------------------------------
# ASS karaoke builder
# ---------------------------------------------------------------------------

def _ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass_karaoke(
    words: list[dict],
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_size: int = 38,
    words_per_group: int = 5,
    margin_v: int = 60,
) -> str:
    """
    Build an ASS subtitle string with word-level fill-karaoke highlighting.

    Words are grouped into short phrases. Each phrase is one dialogue event.
    Within the phrase, \\kf tags make the active word sweep from white → yellow.

    ASS colour format: &HAABBGGRR
      PrimaryColour  (active word)  = yellow  &H0000FFFF
      SecondaryColour (not yet spoken) = white  &H00FFFFFF
    """
    if not words:
        return ""

    found_font = _find_font()
    fname = Path(found_font).stem if found_font else "DejaVu Sans"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fname},{font_size},"
        "&H0000FFFF,"   # PrimaryColour   = yellow (active word)
        "&H00FFFFFF,"   # SecondaryColour = white  (unspoken)
        "&H00000000,"   # OutlineColour   = black
        "&H80000000,"   # BackColour      = semi-transparent black
        "-1,0,0,0,100,100,0,0,1,3,1,"  # Bold=-1 (on), no italic/underline, border+shadow
        f"2,20,20,{margin_v},1\n"       # Alignment=2 (bottom center), margins
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    groups = [words[i:i + words_per_group] for i in range(0, len(words), words_per_group)]

    events: list[str] = []
    for group in groups:
        if not group:
            continue
        t_start = group[0]["start"]
        t_end = group[-1]["end"]

        parts: list[str] = []
        for w in group:
            dur_cs = max(1, int((w["end"] - w["start"]) * 100))
            # Escape ASS special chars
            safe = (
                w["word"]
                .replace("\\", "\\\\ ")
                .replace("{", "\\{")
                .replace("}", "\\}")
            )
            parts.append(f"{{\\kf{dur_cs}}}{safe}")

        text = " ".join(parts)
        events.append(
            f"Dialogue: 0,{_ass_time(t_start)},{_ass_time(t_end)},"
            f"Default,,0,0,0,,{text}"
        )

    return header + "\n".join(events) + "\n"


def burn_ass_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> Path:
    """Burn ASS karaoke subtitles using FFmpeg ass filter (requires libass)."""
    # FFmpeg ass filter path: forward slashes, colon-escaped on Windows
    ass_str = str(ass_path.resolve()).replace("\\", "/")
    _ffmpeg(
        "-i", str(video_path),
        "-vf", f"ass={ass_str}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    )
    return output_path


# ---------------------------------------------------------------------------
# SRT fallback (used when faster-whisper is unavailable)
# ---------------------------------------------------------------------------

def _split_sentences(text: str, max_chars: int = 90) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    result: list[str] = []
    for sent in sents:
        if len(sent) <= max_chars:
            result.append(sent)
        else:
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
    Used as fallback when word timestamps are unavailable.
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
    """Burn SRT subtitles into video using FFmpeg subtitles filter (SRT fallback)."""
    font = _find_font()
    style = (
        f"FontSize={font_size},Bold=1,PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,Outline=3,Shadow=2,"
        f"Alignment=2,MarginV=50"
    )
    if font:
        style += f",FontName={Path(font).stem}"

    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    _ffmpeg(
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    )
    return output_path


# ---------------------------------------------------------------------------
# Title card overlay
# ---------------------------------------------------------------------------

def add_title_card(
    video_path: Path,
    title: str,
    output_path: Path,
    duration: float = 2.5,
    font_size: int = 52,
) -> Path:
    """Overlay a semi-transparent title card for the first `duration` seconds."""
    font = _find_font()
    safe_title = title[:60].replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

    font_arg = f":fontfile='{font}'" if font else ""
    fade_out = max(0.0, duration - 0.5)
    fade_range = max(duration - fade_out, 0.01)

    vf = (
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_overlays(
    video_path: Path,
    output_path: Path,
    script_sections: list[dict],
    hook: str = "",
    call_to_action: str = "",
    title: str = "",
    tmp_dir: Optional[Path] = None,
    section_durations: list[float] | None = None,
    audio_path: Optional[Path] = None,
    shorts: bool = False,
) -> Path:
    """
    Full overlay pipeline: title card → karaoke subtitles (or SRT fallback).

    When *audio_path* is provided and faster-whisper is installed, word-level
    timestamps are extracted and burned as ASS karaoke (yellow highlight on
    the currently spoken word).  Otherwise falls back to proportional SRT.
    """
    tmp = Path(tmp_dir or output_path.parent / "_tmp_overlays")
    tmp.mkdir(parents=True, exist_ok=True)

    current = video_path

    # Step 1: title card
    if title and not shorts:
        try:
            titled = tmp / "titled.mp4"
            add_title_card(current, title, titled)
            current = titled
            console.print("[dim]title card added[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Title card skipped ({exc})[/yellow]")

    # Step 2: karaoke subtitles
    try:
        word_timestamps: list[dict] = []
        if audio_path and audio_path.exists():
            word_timestamps = get_word_timestamps(audio_path)

        if word_timestamps:
            # ASS karaoke path — perfect word-level timing
            font_size = 58 if shorts else 38
            words_per_group = 3 if shorts else 5
            margin_v = 120 if shorts else 60
            play_res_x, play_res_y = (1080, 1920) if shorts else (1920, 1080)

            ass_content = build_ass_karaoke(
                word_timestamps,
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                font_size=font_size,
                words_per_group=words_per_group,
                margin_v=margin_v,
            )
            ass_path = tmp / "karaoke.ass"
            ass_path.write_text(ass_content, encoding="utf-8")
            subtitled = tmp / "subtitled.mp4"
            burn_ass_subtitles(current, ass_path, subtitled)
            current = subtitled
            console.print("[dim]karaoke subtitles burned in[/dim]")
        else:
            # SRT fallback — proportional sentence-level timing
            if not word_timestamps and audio_path:
                console.print("[yellow]faster-whisper not installed — using proportional SRT subtitles[/yellow]")
            duration = _get_video_duration(current)
            if duration > 0 and script_sections:
                srt_content = build_srt(
                    script_sections, duration, hook, call_to_action, section_durations
                )
                if srt_content.strip():
                    srt_path = tmp / "subtitles.srt"
                    srt_path.write_text(srt_content, encoding="utf-8")
                    subtitled = tmp / "subtitled.mp4"
                    burn_subtitles(current, srt_path, subtitled)
                    current = subtitled
                    console.print("[dim]subtitles burned in (SRT fallback)[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Subtitles skipped ({exc})[/yellow]")

    if current != output_path:
        import shutil
        shutil.copy2(current, output_path)

    return output_path
