"""
FFmpeg video renderer.

PATCHED — fix 1.3 from REVIEW.md:
  apply_visual_mutation no longer requests atempo on silent video (was failing
  silently, defeating anti-duplicate protection on every render).
  Speed variation moved to setpts on the video filter chain.

Assembles a list of TimelineClip objects + an audio file into a final MP4.
100% free — FFmpeg must be installed on the system.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.database.models import TimelineClip

console = Console()

_NVENC_AVAILABLE: bool | None = None


def _has_nvenc() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in r.stdout
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def _video_codec_args() -> list[str]:
    if _has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]


def _ffmpeg(*args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + list(args)
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out after {timeout}s. Command: {' '.join(cmd[:6])}...")


def _check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _has_motion(clip: TimelineClip) -> bool:
    return clip.asset.asset_type == "image" or (clip.out_point - clip.in_point) < 3.0


def trim_clip(clip: TimelineClip, output_path: Path, ken_burns: bool = True) -> Path:
    asset_path = clip.asset.path
    duration = clip.out_point - clip.in_point

    base_vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"

    if ken_burns and _has_motion(clip):
        frames = max(25, int(duration * 25))
        vf = (
            f"scale=3840:2160,"
            f"zoompan=z='min(1.01+({0.05}/{frames})*on,1.06)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s=1920x1080:fps=25"
        )
    else:
        vf = base_vf

    _ffmpeg(
        "-ss", str(clip.in_point),
        "-i", asset_path,
        "-t", str(duration),
        *_video_codec_args(),
        "-an",
        "-vf", vf,
        str(output_path),
    )
    return output_path


def apply_visual_mutation(input_path: Path, output_path: Path, seed: int = 0) -> Path:
    """
    Apply subtle randomised visual variations so the same stock clip
    looks different across videos (defeats hash-based duplicate detection).

    FIXED: input is silent (trimmed with -an), so atempo is not applicable.
    Speed micro-ramp is now done with setpts on the video stream.
    """
    import random as _rnd
    _rnd.seed(seed)
    crop_pct = _rnd.uniform(0.01, 0.03)
    brightness = _rnd.uniform(-0.03, 0.03)
    saturation = _rnd.uniform(0.90, 1.10)
    speed = _rnd.uniform(0.985, 1.015)
    # setpts ratio: 1/speed inverts (smaller PTS = faster playback)
    speed_filter = f"setpts={1/speed:.4f}*PTS"
    eq_filter = f"eq=brightness={brightness:.3f}:saturation={saturation:.2f}"
    _ffmpeg(
        "-i", str(input_path),
        "-vf",
        f"crop=iw*{1 - crop_pct*2:.4f}:ih*{1 - crop_pct*2:.4f}:"
        f"iw*{crop_pct:.4f}:ih*{crop_pct:.4f},"
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        f"{eq_filter},{speed_filter}",
        *_video_codec_args(),
        "-an",
        str(output_path),
    )
    return output_path


def concat_clips(clip_paths: list[Path], output_path: Path) -> Path:
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


def _get_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError, OSError):
        return 0.0


def add_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Merge voiceover with video. Loops footage smoothly if shorter than audio."""
    audio_dur = _get_duration(audio_path)
    video_dur = _get_duration(video_path)

    working_video = video_path
    if audio_dur > 0 and video_dur < audio_dur - 0.5:
        console.print(
            f"[dim]Video ({video_dur:.1f}s) shorter than audio ({audio_dur:.1f}s)"
            f" — looping footage[/dim]"
        )
        looped = video_path.parent / ("_looped_" + video_path.name)
        loops = int(audio_dur / max(video_dur, 1)) + 2
        _ffmpeg(
            "-stream_loop", str(loops),
            "-i", str(video_path),
            "-t", str(audio_dur + 1.0),
            *_video_codec_args(),
            "-an",
            str(looped),
        )
        working_video = looped

    _ffmpeg(
        "-i", str(working_video),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    )
    return output_path


def add_background_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    music_volume: float = 0.12,
) -> Path:
    """Mix background music with sidechain ducking."""
    _ffmpeg(
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex",
        (
            f"[0:a]asplit=2[tts_main][tts_sc];"
            f"[1:a]volume={music_volume}[bg];"
            f"[bg][tts_sc]sidechaincompress="
            f"threshold=0.025:ratio=8:attack=5:release=200:makeup=1[bg_duck];"
            f"[tts_main][bg_duck]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        ),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    )
    return output_path


def render_video(
    clips: list[TimelineClip],
    audio_path: Optional[Path],
    output_path: Path,
    music_path: Optional[Path] = None,
    tmp_dir: Optional[Path] = None,
) -> Path:
    """Full render pipeline."""
    if not _check_ffmpeg():
        raise RuntimeError(
            "FFmpeg not found.\n"
            "Install with: sudo apt install ffmpeg"
        )

    tmp = Path(tmp_dir or output_path.parent / "_tmp_render")
    tmp.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Rendering {len(clips)} clips...[/cyan]")

    trimmed: list[Path] = []
    for i, clip in enumerate(clips):
        asset_path = Path(clip.asset.path)
        if not asset_path.exists():
            console.print(f"[yellow]  skipping missing asset: {asset_path}[/yellow]")
            continue
        out = tmp / f"clip_{i:03d}.mp4"
        if clip.asset.asset_type == "image":
            duration = clip.out_point - clip.in_point or 4.0
            frames = max(25, int(duration * 25))
            _ffmpeg(
                "-loop", "1",
                "-i", clip.asset.path,
                "-t", str(duration),
                *_video_codec_args(),
                "-vf",
                f"scale=3840:2160,"
                f"zoompan=z='min(1.01+({0.05}/{frames})*on,1.06)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps=25",
                "-pix_fmt", "yuv420p",
                str(out),
            )
        else:
            trim_clip(clip, out, ken_burns=True)

        # Apply visual mutation — NOW ACTUALLY WORKS (was silently failing before)
        try:
            mutated = tmp / f"clip_{i:03d}_mut.mp4"
            apply_visual_mutation(out, mutated, seed=i)
            out = mutated
        except Exception as exc:
            console.print(f"[yellow]  mutation failed on clip {i}: {exc}[/yellow]")

        trimmed.append(out)
        console.print(f"  [dim]trimmed clip {i + 1}/{len(clips)}[/dim]")

    silent_video = tmp / "concat.mp4"
    concat_clips(trimmed, silent_video)
    console.print("[dim]clips concatenated[/dim]")

    if audio_path and audio_path.exists():
        voiced_video = tmp / "voiced.mp4"
        add_audio(silent_video, audio_path, voiced_video)
        console.print("[dim]voiceover added[/dim]")
    else:
        voiced_video = silent_video

    if music_path and music_path.exists():
        add_background_music(voiced_video, music_path, output_path)
        console.print("[dim]background music mixed[/dim]")
    else:
        import shutil
        shutil.copy2(voiced_video, output_path)

    console.print(f"[bold green]✓ Video rendered:[/bold green] {output_path}")
    return output_path
