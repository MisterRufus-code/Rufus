"""
Thumbnail generator — produces YouTube-ready 1280x720 JPG thumbnails.

Strategy:
  1. Extract the sharpest/most-colourful frame from the rendered video
  2. Apply a dark gradient overlay for text contrast
  3. Render bold title text (top) and a short hook line (bottom)
  4. Save as high-quality JPG
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

THUMB_W, THUMB_H = 1280, 720
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


def _best_frame(video_path: Path, tmp_dir: Path) -> Optional[Path]:
    """Extract the frame with highest visual activity (seek to 20% of duration)."""
    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 5.0

    seek = max(0.5, duration * 0.20)
    frame_path = tmp_dir / "thumb_frame.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", str(seek), "-i", str(video_path),
         "-vframes", "1", "-q:v", "2",
         "-vf", f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=increase,crop={THUMB_W}:{THUMB_H}",
         str(frame_path)],
        check=True,
    )
    return frame_path if frame_path.exists() else None


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_thumbnail(
    video_path: Path,
    title: str,
    hook: str,
    output_path: Path,
    accent_color: str = "#FF4500",
) -> Optional[Path]:
    """
    Generate a YouTube thumbnail from the video.
    Returns the thumbnail path on success, None on failure.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except ImportError:
        console.print("[yellow]Pillow not installed — thumbnail skipped.[/yellow]")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 1. Extract frame
            frame_path = _best_frame(video_path, tmp_dir)
            if not frame_path:
                console.print("[yellow]Could not extract frame for thumbnail.[/yellow]")
                return None

            img = Image.open(frame_path).convert("RGB").resize((THUMB_W, THUMB_H))

            # 2. Dark gradient overlay (bottom 55% of image)
            overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
            draw_ov = ImageDraw.Draw(overlay)
            gradient_start = int(THUMB_H * 0.30)
            for y in range(gradient_start, THUMB_H):
                alpha = int(210 * (y - gradient_start) / (THUMB_H - gradient_start))
                draw_ov.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))
            # Top dark bar for title
            for y in range(0, int(THUMB_H * 0.38)):
                alpha = int(170 * (1 - y / (THUMB_H * 0.38)))
                draw_ov.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))

            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            font_path = _find_font()

            # 3. Accent bar on left edge
            accent_rgb = tuple(int(accent_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            draw.rectangle([(0, 0), (12, THUMB_H)], fill=accent_rgb)

            # 4. Title text (top area)
            title_clean = title[:80].upper()
            title_lines = _wrap_text(title_clean, 28)[:3]

            try:
                font_title = ImageFont.truetype(font_path, 72) if font_path else ImageFont.load_default()
                font_hook = ImageFont.truetype(font_path, 38) if font_path else ImageFont.load_default()
            except Exception:
                font_title = ImageFont.load_default()
                font_hook = ImageFont.load_default()

            y_title = 28
            for line in title_lines:
                # Shadow
                draw.text((32, y_title + 3), line, font=font_title, fill=(0, 0, 0, 200))
                draw.text((30, y_title), line, font=font_title, fill=(255, 255, 255))
                bbox = draw.textbbox((0, 0), line, font=font_title)
                y_title += (bbox[3] - bbox[1]) + 8

            # 5. Hook text (bottom)
            hook_clean = hook[:100]
            hook_lines = _wrap_text(hook_clean, 48)[:2]
            y_hook = THUMB_H - 30 - len(hook_lines) * 52
            for line in hook_lines:
                draw.text((32, y_hook + 2), line, font=font_hook, fill=(0, 0, 0, 180))
                draw.text((30, y_hook), line, font=font_hook, fill=(*accent_rgb, 255))
                bbox = draw.textbbox((0, 0), line, font=font_hook)
                y_hook += (bbox[3] - bbox[1]) + 6

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path), "JPEG", quality=92)
            console.print(f"[green]Thumbnail saved:[/green] {output_path.name}")
            return output_path

    except Exception as exc:
        console.print(f"[yellow]Thumbnail generation failed ({exc})[/yellow]")
        return None
