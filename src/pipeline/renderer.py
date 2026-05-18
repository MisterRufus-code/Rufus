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
    """Return the best available H.264 encoder args: NVENC if available, else libx264."""
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
    """Heuristic: images and very short clips get Ken Burns; longer video clips don't need it."""
    return clip.asset.asset_type == "image" or (clip.out_point - clip.in_point) < 3.0


def trim_clip(clip: TimelineClip, output_path: Path, ken_burns: bool = True) -> Path:
    """
    Trim a single asset to [in_point, out_point] and re-encode to H.264.
    Applies a subtle Ken Burns zoom-pan to static / short clips so there's
    always some visual motion (increases retention for talking-head-style edits).
    """
    asset_path = clip.asset.path
    duration = clip.out_point - clip.in_point

    base_vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"

    if ken_burns and _has_motion(clip):
        frames = max(25, int(duration * 25))
        # Start at z=1.01 (not 1.00) to stay safely inside the scaled frame and
        # avoid the gray edge artifact caused by sub-pixel overflow at z=1.0 exactly.
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
    looks different across videos (prevents YouTube duplicate-content detection).
    Variations: micro crop offset, brightness/saturation nudge, speed micro-ramp.
    All changes are imperceptible to humans but defeat hash-based duplicate checks.
    """
    import random as _rnd
    _rnd.seed(seed)
    # Crop: randomly trim 1-3% from each edge (maintains aspect ratio)
    crop_pct = _rnd.uniform(0.01, 0.03)
    # Brightness ±3%, saturation ±10%
    brightness = _rnd.uniform(-0.03, 0.03)
    saturation = _rnd.uniform(0.90, 1.10)
    eq_filter = f"eq=brightness={brightness:.3f}:saturation={saturation:.2f}"
    # Speed micro-ramp: ±1.5% so duration varies slightly
    speed = _rnd.uniform(0.985, 1.015)
    _ffmpeg(
        "-i", str(input_path),
        "-vf",
        f"crop=iw*{1 - crop_pct*2:.4f}:ih*{1 - crop_pct*2:.4f}:"
        f"iw*{crop_pct:.4f}:ih*{crop_pct:.4f},"
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        f"{eq_filter}",
        "-af", f"atempo={speed:.3f}",
        *_video_codec_args(),
        "-c:a", "copy",
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


def _get_duration(path: Path) -> float:
    """Get video/audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError, OSError):
        return 0.0


def _loop_video_to_duration(video_path: Path, target_duration: float, output_path: Path) -> Path:
    """Loop the video track until it covers target_duration seconds."""
    vid_dur = _get_duration(video_path)
    if vid_dur <= 0:
        return video_path
    loops = int(target_duration / vid_dur) + 2
    _ffmpeg(
        "-stream_loop", str(loops),
        "-i", str(video_path),
        "-t", str(target_duration + 0.5),   # tiny extra buffer
        *_video_codec_args(),
        "-an",
        str(output_path),
    )
    return output_path


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
        # Re-encode the loop so there's no freeze at the seam
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
    """
    Mix background music with sidechain ducking.
    Music auto-lowers when the TTS voiceover is loud, rises in pauses.
    This gives a professional "audio ducking" effect matching human editors.
    """
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
            # Convert image to short video clip with Ken Burns zoom
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

        # Apply subtle visual mutation (unique per clip index) to prevent
        # YouTube detecting reused stock footage via perceptual hashing
        try:
            mutated = tmp / f"clip_{i:03d}_mut.mp4"
            apply_visual_mutation(out, mutated, seed=i)
            out = mutated
        except Exception:
            pass  # mutation is optional — fall back to un-mutated clip

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
