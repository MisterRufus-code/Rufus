"""
FFmpeg video renderer.

Assembles a list of TimelineClip objects + an audio file into a final MP4.
100% free — FFmpeg must be installed on the system.

Install: sudo apt install ffmpeg
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.database.models import TimelineClip

console = Console()


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


def trim_clip(clip: TimelineClip, output_path: Path) -> Path:
    """Trim a single asset to [in_point, out_point] and re-encode to H.264."""
    asset_path = clip.asset.path
    duration = clip.out_point - clip.in_point
    _ffmpeg(
        "-ss", str(clip.in_point),
        "-i", asset_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",                    # drop original audio — will add voiceover
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        str(output_path),
    )
    return output_path


def concat_clips(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate trimmed clips using FFmpeg concat demuxer."""
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


def add_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Merge voiceover audio with muted video. Audio ends when shortest track ends."""
    _ffmpeg(
        "-i", str(video_path),
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
    """Mix background music at low volume under the voiceover."""
    _ffmpeg(
        "-i", str(video_path),
        "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
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
    """
    Full render pipeline:
      1. Trim each clip to required duration
      2. Concatenate into one video
      3. Add voiceover audio
      4. Optionally mix in background music
    """
    if not _check_ffmpeg():
        raise RuntimeError(
            "FFmpeg not found.\n"
            "Install with: sudo apt install ffmpeg"
        )

    tmp = Path(tmp_dir or output_path.parent / "_tmp_render")
    tmp.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Rendering {len(clips)} clips...[/cyan]")

    # Step 1: trim clips
    trimmed: list[Path] = []
    for i, clip in enumerate(clips):
        asset_path = Path(clip.asset.path)
        if not asset_path.exists():
            console.print(f"[yellow]  skipping missing asset: {asset_path}[/yellow]")
            continue
        out = tmp / f"clip_{i:03d}.mp4"
        if clip.asset.asset_type == "image":
            # Convert image to short video clip
            duration = clip.out_point - clip.in_point or 4.0
            _ffmpeg(
                "-loop", "1",
                "-i", clip.asset.path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-pix_fmt", "yuv420p",
                str(out),
            )
        else:
            trim_clip(clip, out)
        trimmed.append(out)
        console.print(f"  [dim]trimmed clip {i + 1}/{len(clips)}[/dim]")

    # Step 2: concatenate
    silent_video = tmp / "concat.mp4"
    concat_clips(trimmed, silent_video)
    console.print("[dim]clips concatenated[/dim]")

    # Step 3: add voiceover
    if audio_path and audio_path.exists():
        voiced_video = tmp / "voiced.mp4"
        add_audio(silent_video, audio_path, voiced_video)
        console.print("[dim]voiceover added[/dim]")
    else:
        voiced_video = silent_video

    # Step 4: background music (optional)
    if music_path and music_path.exists():
        add_background_music(voiced_video, music_path, output_path)
        console.print("[dim]background music mixed[/dim]")
    else:
        import shutil
        shutil.copy2(voiced_video, output_path)

    console.print(f"[bold green]✓ Video rendered:[/bold green] {output_path}")
    return output_path
