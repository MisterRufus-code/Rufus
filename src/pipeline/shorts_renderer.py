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
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/impact.ttf",
    "/usr/share/fonts/Impact.ttf",
    "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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


def _has_nvenc() -> bool:
    """Return True if NVIDIA NVENC h264 encoder is available."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


_NVENC_AVAILABLE: bool | None = None


def _video_codec_args() -> list[str]:
    """Return the best available video encoder args (NVENC if possible)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _has_nvenc()
    if _NVENC_AVAILABLE:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]


def _trim_vertical(clip: TimelineClip, output_path: Path, ken_burns: bool = True) -> Path:
    """
    Trim clip and convert landscape → 9:16 vertical (1080×1920).
    Uses subject-biased crop (upper-third) so faces stay in frame.
    Adds subtle Ken Burns zoom for motion when enabled.
    """
    duration = min(clip.out_point - clip.in_point, 8.0)
    # Upper-third crop: y offset = H*0.1 keeps heads in frame for talking-head footage.
    # The crop starts at 10% from top instead of center (50%).
    crop_vf = "scale=-2:1920,crop=1080:1920:x=(iw-1080)/2:y=ih*0.10"
    if ken_burns and clip.asset.asset_type != "image":
        # Very subtle zoom: 1.00 → 1.04 over the clip duration (barely perceptible, adds life)
        zoom_vf = (
            f"scale=2160:3840,zoompan=z='min(1+({0.04}/({duration}*25))*on,1.04)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*25)}:s=1080x1920"
        )
        vf = zoom_vf
    else:
        vf = crop_vf
    _ffmpeg(
        "-ss", str(clip.in_point),
        "-i", clip.asset.path,
        "-t", str(duration),
        "-vf", vf,
        *_video_codec_args(),
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


def _loop_video_to_duration(video_path: Path, target_duration: float, output_path: Path) -> Path:
    """Loop video until it is at least target_duration long."""
    vid_dur = _get_duration(video_path)
    if vid_dur <= 0:
        return video_path
    loops = int(target_duration / vid_dur) + 1
    # Re-encode instead of -c copy to prevent B-frame freeze at loop boundary
    _ffmpeg(
        "-stream_loop", str(loops),
        "-i", str(video_path),
        "-t", str(target_duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(output_path),
    )
    return output_path


def _speed_audio_to_fit(audio_path: Path, target_dur: float, output_path: Path) -> Path:
    """
    Apply atempo FFmpeg filter to compress audio duration to target_dur.
    atempo is capped at 1.25× (any faster sounds unnatural for a voice).
    If multiple passes are needed they are chained (atempo max = 2.0 per pass).
    """
    current_dur = _get_duration(audio_path)
    if current_dur <= 0 or current_dur <= target_dur:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path

    factor = min(1.25, current_dur / target_dur)
    if factor <= 1.02:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path

    console.print(
        f"[dim]Smart render: audio {current_dur:.1f}s > {target_dur:.0f}s"
        f" — speeding up ×{factor:.2f} (atempo)[/dim]"
    )
    # Chain two atempo passes if factor > 2.0 (atempo limit)
    if factor > 2.0:
        af = f"atempo=2.0,atempo={factor/2.0:.3f}"
    else:
        af = f"atempo={factor:.3f}"
    _ffmpeg("-i", str(audio_path), "-filter:a", af, str(output_path))
    return output_path


def _add_audio_with_ducking(
    video_path: Path,
    audio_path: Path,
    music_path: Path | None,
    output_path: Path,
    music_volume: float = 0.10,
) -> Path:
    """
    Merge TTS voiceover + optional background music with sidechain ducking.
    Music automatically drops when TTS is speaking and rises in gaps.
    Falls back to simple merge if music_path is None.
    """
    if music_path is None or not music_path.exists():
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

    # Sidechain compress: music ducked when TTS amplitude exceeds threshold
    _ffmpeg(
        "-i", str(video_path),
        "-i", str(audio_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex",
        (
            f"[1:a]asplit=2[tts_main][tts_sc];"
            f"[2:a]volume={music_volume}[bg];"
            f"[bg][tts_sc]sidechaincompress="
            f"threshold=0.025:ratio=8:attack=5:release=200:makeup=1[bg_duck];"
            f"[tts_main][bg_duck]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        ),
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    )
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
            f":fontsize=64:fontcolor=yellow"
            f":borderw=6:bordercolor=black"
            f":shadowx=3:shadowy=3:shadowcolor=black@0.8"
            f":x=(w-text_w)/2:y=h*0.70"
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
    music_path: Optional[Path] = None,
) -> Path:
    """
    Full Shorts render:
      1. Vertical crop each clip (landscape → 9:16)
      2. Concatenate
      3. Add voiceover
      4. Trim to ≤58s
      5. Burn karaoke subtitles (word-level highlight) or fallback captions
    """
    tmp = Path(tmp_dir or output_path.parent / "_tmp_shorts")
    tmp.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Rendering Shorts — {len(clips)} clips → 9:16[/cyan]")

    # 1. Vertical crop each clip (Ken Burns enabled by default)
    trimmed: list[Path] = []
    for i, clip in enumerate(clips):
        out = tmp / f"short_clip_{i:03d}.mp4"
        if clip.asset.asset_type == "image":
            duration = min(clip.out_point - clip.in_point or 3.0, 5.0)
            # Ken Burns on images: gentle zoom 1.00→1.08 over the still
            frames = int(duration * 25)
            _ffmpeg(
                "-loop", "1", "-i", clip.asset.path,
                "-t", str(duration),
                "-vf",
                f"scale=2160:3840,"
                f"zoompan=z='min(1+0.003*on,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s=1080x1920",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(out),
            )
        else:
            _trim_vertical(clip, out, ken_burns=True)
        trimmed.append(out)

    # 2. Concat
    concat_path = tmp / "shorts_concat.mp4"
    _concat_clips(trimmed, concat_path)

    # 3. Smart rendering loop: if audio > 58s, speed it up with atempo
    working_audio = audio_path
    if audio_path and audio_path.exists():
        audio_dur = _get_duration(audio_path)
        if audio_dur > MAX_SHORTS_DURATION:
            sped_audio = tmp / "shorts_sped.wav"
            working_audio = _speed_audio_to_fit(audio_path, MAX_SHORTS_DURATION, sped_audio)
            audio_dur = _get_duration(working_audio)

        # Loop video if shorter than (possibly sped-up) audio
        if audio_dur > 0 and _get_duration(concat_path) < audio_dur:
            looped = tmp / "shorts_looped.mp4"
            _loop_video_to_duration(concat_path, audio_dur, looped)
            concat_path = looped

        voiced_path = tmp / "shorts_voiced.mp4"
        _add_audio_with_ducking(concat_path, working_audio, music_path, voiced_path)
    else:
        voiced_path = concat_path

    # 4. Trim to 58s
    trimmed_path = tmp / "shorts_trimmed.mp4"
    _trim_to_60s(voiced_path, trimmed_path)

    # 5. Burn karaoke subtitles (word-level) or fall back to drawtext captions
    from src.pipeline.subtitles import get_word_timestamps, build_ass_karaoke, burn_ass_subtitles

    captioned_path = tmp / "shorts_captioned.mp4"
    word_timestamps: list[dict] = []
    if audio_path and audio_path.exists():
        word_timestamps = get_word_timestamps(audio_path)

    if word_timestamps:
        ass_content = build_ass_karaoke(
            word_timestamps,
            play_res_x=1080, play_res_y=1920,
            font_size=58, words_per_group=3, margin_v=120,
        )
        ass_path = tmp / "shorts_karaoke.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
        burn_ass_subtitles(trimmed_path, ass_path, captioned_path)
        console.print("[dim]karaoke subtitles burned in (Shorts)[/dim]")
        current = captioned_path
    elif script_sections:
        duration = _get_duration(trimmed_path)
        _burn_shorts_captions(trimmed_path, script_sections, hook, captioned_path, duration)
        current = captioned_path
    else:
        current = trimmed_path

    import shutil
    shutil.copy2(current, output_path)
    console.print(f"[bold green]✓ Shorts rendered:[/bold green] {output_path}")
    return output_path
