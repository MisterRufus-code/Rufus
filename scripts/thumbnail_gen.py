#!/usr/bin/env python3
"""
thumbnail_gen.py
Generates a channel-branded thumbnail from a rendered Short.

Pipeline:
  1. Extract a striking frame at ~30% video duration (FFmpeg)
  2. Open with Pillow, draw dark gradient overlay at bottom 40%
  3. Draw niche accent bar (left edge brand mark)
  4. Render full hook sentence (wrapped, 2 lines max) in Anton font
  5. Export as high-quality JPG

The hook text uses the niche's accent color for numbers/key words and white
for the rest — matching the in-video caption style. Result looks like a real
channel thumbnail, not a screenshot with 3 yellow words.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
FONTS_DIR  = ROOT / "assets" / "fonts"
FONT_FILE  = FONTS_DIR / "Anton-Regular.ttf"

# Cross-platform fallback fonts if the bundled Anton TTF is missing.
# Windows first (the user's rig), then macOS, then Linux.
FONT_CANDIDATES = [
    str(FONT_FILE),
    # Windows 11
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariblk.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

THUMB_W, THUMB_H = 1080, 1920
FONT_SIZE        = 110    # readable at 300px preview width on YouTube
MAX_LINE_CHARS   = 22     # wrap at ~22 chars for 2-line readability


def _find_font() -> str:
    for path in FONT_CANDIDATES:
        if path and Path(path).exists():
            return path
    return ""   # PIL will fall back to default


def _load_niche_accent() -> str:
    """Read active niche's accent_color from niches.json. Returns hex string."""
    try:
        data   = json.loads((CONFIG_DIR / "niches.json").read_text())
        active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data.get("active", "finance")
        return data["niches"][active].get("accent_color", "#FFC53D")
    except Exception:
        return "#FFC53D"   # warm gold default


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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


def _extract_frame(video_path: Path, ts: float, tmp_png: str) -> bool:
    """Extract one frame from video at timestamp ts into tmp_png. Returns success."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{ts:.2f}", "-i", str(video_path),
         "-frames:v", "1", tmp_png],
        capture_output=True,
    )
    return r.returncode == 0 and Path(tmp_png).exists()


def _score_frame(img: Image.Image) -> float:
    """Score a frame by visual interestingness (edge density). Higher = more striking."""
    small = img.convert("L").resize((108, 192), Image.BILINEAR)
    edges = small.filter(ImageFilter.FIND_EDGES)
    return sum(edges.getdata()) / (108 * 192)


def _best_frame(video_path: Path, duration: float, tmp_png: str) -> Image.Image | None:
    """Try 5 timestamps across the video; return the most visually dynamic frame."""
    candidates = [duration * p for p in (0.20, 0.30, 0.45, 0.60, 0.75)]
    best_img, best_score = None, -1.0
    for ts in candidates:
        if not _extract_frame(video_path, ts, tmp_png):
            continue
        try:
            img = Image.open(tmp_png).convert("RGBA")
            score = _score_frame(img)
            if score > best_score:
                best_score = score
                best_img = img.copy()
        except Exception:
            continue
    return best_img


def _wrap_hook(text: str, max_chars: int = MAX_LINE_CHARS) -> list[str]:
    """Word-wrap hook text to at most 2 lines for thumbnail readability."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip() if current else w
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
        if len(lines) == 2:
            break
    if current and len(lines) < 2:
        lines.append(current)
    return [l.upper() for l in lines[:2]]


def _draw_gradient_overlay(draw: ImageDraw.Draw, height_pct: float = 0.45) -> None:
    """Draw a dark-to-transparent gradient at the bottom of the thumbnail."""
    band_h = int(THUMB_H * height_pct)
    for y_off in range(band_h):
        # alpha 0 at top of band → 200 at bottom
        alpha = int(200 * (y_off / band_h) ** 1.5)
        y     = THUMB_H - band_h + y_off
        draw.rectangle([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))


def _draw_accent_bar(draw: ImageDraw.Draw, accent_rgb: tuple[int, int, int]) -> None:
    """Draw a thin niche-colored bar on the left edge — brand mark."""
    draw.rectangle([(0, 0), (12, THUMB_H)], fill=accent_rgb + (220,))


def make_thumbnail(video_path: Path, script: str, out_path: Path = None) -> Path:
    """Render a branded thumbnail JPG with hook text over a strong video frame."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if out_path is None:
        out_path = video_path.with_suffix(".thumb.jpg")
    out_path = Path(out_path)

    duration    = _probe_duration(video_path)
    accent_hex  = _load_niche_accent()
    accent_rgb  = _hex_to_rgb(accent_hex)
    font_path   = _find_font()

    # Extract best frame from 5 candidate timestamps
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = tf.name
    try:
        img = _best_frame(video_path, duration, tmp_png)
        if img is None:
            raise RuntimeError("Frame extraction failed at all candidate timestamps")
        # Ensure correct dimensions
        if img.size != (THUMB_W, THUMB_H):
            img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        # Dark gradient at bottom (behind text)
        _draw_gradient_overlay(draw)

        # Niche accent bar on left edge
        _draw_accent_bar(draw, accent_rgb)

        # Merge overlay onto frame
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype(font_path, FONT_SIZE) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Get and wrap the hook (first line of script)
        hook_raw = script.strip().split("\n")[0].rstrip(".!?,;")
        lines    = _wrap_hook(hook_raw)

        # Position text in the lower quarter of the frame
        total_text_h = len(lines) * (FONT_SIZE + 12)
        y_start      = THUMB_H - total_text_h - 80   # 80px from bottom

        for i, line in enumerate(lines):
            y = y_start + i * (FONT_SIZE + 12)
            # Measure text width for centering
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = len(line) * FONT_SIZE * 0.6
            x = (THUMB_W - text_w) / 2

            # Shadow pass (deep black, offset 4px)
            draw.text((x + 4, y + 4), line, font=font, fill=(0, 0, 0, 210))
            # Main text — accent color for first line (hook), white for second
            color = accent_rgb + (255,) if i == 0 else (255, 255, 255, 255)
            draw.text((x, y), line, font=font, fill=color)

        # Convert back to RGB and save as high-quality JPG
        final = img.convert("RGB")
        final.save(str(out_path), "JPEG", quality=92)

    finally:
        Path(tmp_png).unlink(missing_ok=True)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python thumbnail_gen.py <video.mp4> '<script text>'")
        sys.exit(1)

    video  = Path(sys.argv[1])
    script = sys.argv[2]
    out    = make_thumbnail(video, script)
    print(f"THUMBNAIL={out}")
