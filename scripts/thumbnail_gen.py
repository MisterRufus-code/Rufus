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

# The thumbnail is the video's own frame with a title band on it, and line 234
# RESIZES the extracted frame to this size without preserving aspect. Hard-coded
# portrait, a long-form thumbnail would have been a 1920×1080 frame squeezed
# into 1080×1920 — every face in it a third as wide as it was drawn, on the one
# image that decides whether anybody clicks. Taking the frame size means the
# ratio always matches the source, and 1920×1080 clears YouTube's 1280×720
# minimum with room to spare.
import video_format as _vf
THUMB_W, THUMB_H = _vf.dimensions()
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
        data   = json.loads((CONFIG_DIR / "niches.json").read_text(encoding="utf-8"))
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
        capture_output=True, text=True, timeout=30,
    )
    try:
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                d = float(stream.get("duration", 0))
                if d > 0:
                    return d
    except Exception as e:
        print(f"[thumb] duration probe failed ({e}) — assuming 20s for frame picks")
    return 20.0


def _extract_frame(video_path: Path, ts: float, tmp_png: str) -> bool:
    """Extract one frame from video at timestamp ts into tmp_png. Returns success."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{ts:.2f}", "-i", str(video_path),
         "-frames:v", "1", tmp_png],
        capture_output=True, timeout=60,
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


def _wrap_hook(text: str, max_chars: int = MAX_LINE_CHARS,
               max_lines: int = 2) -> list[str]:
    """Word-wrap hook text to at most `max_lines` lines.

    THE DEFAULT STILL DROPS WORDS, AND THAT IS STILL RIGHT FOR ITS CALLER. A
    video's hook is a spoken sentence pulled off the front of a script; two
    lines of it is a thumbnail, and the rest is narration nobody was going to
    read at 168x94 anyway.

    It is NOT right for a headline somebody typed, which is why max_lines
    exists. compose() searches for a wrap that keeps every word — quietly
    deleting the last word of what a person wrote is the kind of thing that is
    only noticed after it has gone out.
    """
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
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return [l.upper() for l in lines[:max_lines]]


def _keeps_every_word(text: str, lines: list[str]) -> bool:
    return len(" ".join(lines).split()) == len(text.split())


# EVERY DRAWING HELPER TAKES THE SIZE IT IS DRAWING ON.
#
# They used to read the module-level THUMB_W/THUMB_H, which come from
# video_format.dimensions() — the VIDEO's shape, portrait 1080x1920 for Shorts.
# That is correct for the one thing this file could do (composite over an
# extracted video frame) and wrong for everything else. A 1280x720 YouTube
# thumbnail composed with portrait constants puts the gradient band off the
# bottom of the image, draws an accent bar two and a half times too long, and
# lands the character badge outside the frame entirely.
#
# The globals stay as the defaults, so make_thumbnail is unchanged.

def _draw_gradient_overlay(draw: ImageDraw.Draw, w: int, h: int,
                           height_pct: float = 0.45) -> None:
    """Draw a dark-to-transparent gradient at the bottom of the image."""
    band_h = max(1, int(h * height_pct))
    for y_off in range(band_h):
        # alpha 0 at top of band → 200 at bottom
        alpha = int(200 * (y_off / band_h) ** 1.5)
        y     = h - band_h + y_off
        draw.rectangle([(0, y), (w, y)], fill=(0, 0, 0, alpha))


def _draw_accent_bar(draw: ImageDraw.Draw, w: int, h: int,
                     accent_rgb: tuple[int, int, int]) -> None:
    """Draw a thin niche-colored bar on the left edge — brand mark.

    Width scales with the image rather than being a flat 12px: 12px is a
    confident stripe on a 1080-wide portrait frame and a hairline on a 1280
    landscape one, and a brand mark that disappears at one of the two sizes
    it is used at is not a brand mark."""
    bar = max(8, round(w * 0.011))
    draw.rectangle([(0, 0), (bar, h)], fill=accent_rgb + (220,))


def _composite_character_badge(img: Image.Image, ref_path: Path,
                               accent_rgb: tuple[int, int, int]) -> Image.Image:
    """Paste a circular 'brand badge' of the niche's recurring character
    (character_engine.py) in the top-right corner, ringed in the niche's
    accent color — the same channel-recognition trick real branded channels
    use: whatever the video's own scene is, the SAME face in the SAME corner
    every time makes the channel instantly recognizable in a crowded feed,
    which is a real lever on click-through and subscriber recall, not just
    a visual nicety. Crops the top square of the reference portrait (a
    head-to-waist character sheet — the head is what a small badge needs).
    Silently no-ops on any image error; a thumbnail must never fail because
    the badge couldn't be drawn."""
    try:
        ref = Image.open(ref_path).convert("RGBA")
    except Exception as e:
        print(f"[thumb] character badge skipped (non-fatal): {e}")
        return img

    # Sized against the SHORTER edge, not the width. 30% of the width is a
    # reasonable badge on a portrait frame and swallows a third of a landscape
    # thumbnail; the short edge gives the same visual weight on both.
    img_w, img_h = img.size
    badge_d = max(48, int(min(img_w, img_h) * 0.26))
    w, h = ref.size
    side = min(w, h)
    left = (w - side) // 2
    ref = ref.crop((left, 0, left + side, side)).resize((badge_d, badge_d), Image.LANCZOS)

    mask = Image.new("L", (badge_d, badge_d), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, badge_d, badge_d), fill=255)

    ring_pad = max(4, round(badge_d * 0.022))
    ring_d   = badge_d + ring_pad * 2
    ring     = Image.new("RGBA", (ring_d, ring_d), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, ring_d, ring_d), fill=accent_rgb + (255,))

    margin = max(12, round(min(img_w, img_h) * 0.045))
    x = img_w - ring_d - margin
    y = margin
    img.paste(ring, (x, y), ring)
    img.paste(ref, (x + ring_pad, y + ring_pad), mask)
    return img


def _fit_font(draw: ImageDraw.Draw, lines: list[str], font_path: str,
              box_w: int, box_h: int, start_px: int) -> "ImageFont.FreeTypeFont":
    """The largest font size at which these lines fit the box.

    A FIXED SIZE CANNOT BE RIGHT FOR BOTH. FONT_SIZE was 110 for every
    headline: correct for one short word, and for five it simply ran off both
    edges of the image with nothing to notice. Since the words are typed by a
    person on the thumbnails page, "however long they felt like" is the actual
    input, and the renderer has to cope with it rather than hope.

    GROWS AS WELL AS SHRINKS. This shrank only, on the theory that inflating
    "GOLD" to fill the frame would look like an accident. Rendered side by side
    at feed size that theory was simply wrong: one short word at 110px is a
    caption, and the thumbnail's whole job is to be readable as a postage
    stamp. The cap is a fraction of the image height, so it gets big without
    getting silly.
    """
    size = start_px
    while size > 14:
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
        widest = 0
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                widest = max(widest, bbox[2] - bbox[0])
            except Exception:
                widest = max(widest, int(len(line) * size * 0.6))
        total_h = len(lines) * int(size * 1.18)
        if widest <= box_w and total_h <= box_h:
            return font
        size -= 4
    try:
        return ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# How many characters a line may hold, tried widest-first. A headline is
# wrapped by the layout that produces the LARGEST type while still containing
# every word — which is not the same as the fewest lines, because three short
# lines can carry a bigger font than two long ones.
_WRAP_CANDIDATES = (12, 15, 18, 22, 26, 30, 34)


def _wrap_score(lines: list[str], box_ratio: float = 2.6) -> float:
    """Roughly how big the type can be at this wrap. Bigger is better.

    FEWEST LINES IS NOT BIGGEST TYPE, which is what the first version of this
    assumed. "THE BANK THAT PRINTED ITSELF" fits on one line and is therefore
    tiny, because a single line 28 characters long is limited by the box WIDTH
    long before it is limited by its height; broken over two it can be half
    again as large. Rendered side by side that was obvious and it was not
    obvious at all from the code.

    So score both constraints the way _fit_font will actually apply them —
    width across the longest line, height shared between the lines — and take
    whichever binds. box_ratio is the text box's width over its height, which
    for the layout compose() uses is about 2.6.
    """
    if not lines:
        return 0.0
    longest = max(len(l) for l in lines)
    by_width  = box_ratio / (longest * 0.55)   # 0.55em is Anton's rough advance
    by_height = 1.0 / (len(lines) * 1.18)
    return min(by_width, by_height)


def _best_wrap(text: str, max_lines: int = 3) -> list[str]:
    """The wrap that keeps every word and renders largest.

    Falls back to the lossy default only when nothing fits — a headline of one
    forty-letter word has no wrap that works, and half of it on screen beats a
    blank thumbnail. The caller can tell the two apart with
    _keeps_every_word().
    """
    if not text.strip():
        return []
    best_lines, best_score = None, -1.0
    for n_lines in range(1, max_lines + 1):
        for max_chars in _WRAP_CANDIDATES:
            lines = _wrap_hook(text, max_chars, n_lines)
            if not lines or not _keeps_every_word(text, lines):
                continue
            score = _wrap_score(lines)
            if score > best_score:
                best_lines, best_score = lines, score
    if best_lines is not None:
        return best_lines
    return _wrap_hook(text, MAX_LINE_CHARS, max_lines)


def compose(background, headline: str, out_path: Path,
            niche: str | None = None) -> Path:
    """Put the branding and the headline on a finished background image.

    THE ONLY PLACE COMPOSITION HAPPENS. make_thumbnail extracts a frame and
    then calls this; the dashboard generates a background and then calls this.
    Two implementations of "what a Rufus thumbnail looks like" would drift, and
    the one that drifts is the one nobody notices is wrong — the same reason
    _when_cell in dashboard.py is one function serving two pages.

    `background` is a path or an already-open PIL image. Whatever size it is,
    is the size everything is drawn against.
    """
    img = (background if isinstance(background, Image.Image)
           else Image.open(Path(background)))
    img = img.convert("RGBA")
    w, h = img.size
    accent_rgb = _hex_to_rgb(_load_niche_accent())
    font_path  = _find_font()

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    _draw_gradient_overlay(draw, w, h)
    _draw_accent_bar(draw, w, h, accent_rgb)
    img = Image.alpha_composite(img, overlay)

    # Recurring-character brand badge (top-right), if this niche has one
    # enabled AND a reference portrait has actually been bootstrapped
    # (character_engine.py / comfy_client.py). A niche with no character block
    # — which is every niche today — silently no-ops here.
    try:
        import character_engine
        if character_engine.enabled(niche):
            ref_path = character_engine.reference_image_path(niche)
            if ref_path and ref_path.exists():
                img = _composite_character_badge(img, ref_path, accent_rgb)
    except Exception as e:
        print(f"[thumb] character badge skipped (non-fatal): {e}")

    draw = ImageDraw.Draw(img)
    text = (headline or "").strip().split("\n")[0].rstrip(".!?,;")
    lines = _best_wrap(text)

    if lines:
        # KEEP CLEAR OF WHAT YOUTUBE DRAWS ON TOP. The duration badge sits in
        # the bottom-right corner and the progress bar runs across the bottom
        # on hover — and the text was centred at 80px from the bottom, i.e.
        # underneath both. The box stops short of the bottom and of the right
        # edge, so the words survive contact with the player.
        side_pad   = max(24, round(w * 0.05))
        bottom_pad = max(28, round(h * 0.12))
        box_w = w - side_pad * 2 - round(w * 0.13)    # room for the duration badge
        box_h = round(h * 0.42)
        # Start from a size proportional to the image rather than a flat 110px,
        # so the same headline has the same visual weight on a 1280-wide
        # thumbnail and a 1080-wide portrait frame.
        font = _fit_font(draw, lines, font_path, box_w, box_h,
                         max(FONT_SIZE, round(h * 0.26)))

        try:
            line_h = int(font.size * 1.18)
        except AttributeError:                        # the PIL default font
            line_h = int(FONT_SIZE * 1.18)
        y_start = h - bottom_pad - len(lines) * line_h

        # A STROKE, NOT A DROP SHADOW. An offset shadow only darkens one side,
        # so a light background swallows the other three; an outline holds the
        # letters apart from whatever is behind them at any size — which is
        # the entire job at 168x94 in a phone feed.
        stroke = max(2, round((getattr(font, "size", FONT_SIZE)) * 0.075))
        for i, line in enumerate(lines):
            y = y_start + i * line_h
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = len(line) * getattr(font, "size", FONT_SIZE) * 0.6
            x = side_pad + max(0, (box_w - text_w) / 2)
            color = accent_rgb + (255,) if i == 0 else (255, 255, 255, 255)
            draw.text((x, y), line, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 235))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_path), "JPEG", quality=92)
    return out_path


def make_thumbnail(video_path: Path, script: str, out_path: Path = None,
                   niche: str | None = None) -> Path:
    """Render a branded thumbnail JPG with hook text over a strong video frame.

    `niche` is optional — when it has an enabled recurring character with a
    bootstrapped reference portrait (character_engine.py), that character's
    face is badged into the corner for cross-video brand recognition.
    Omitting it (every pre-existing caller) is identical to before this.

    Extract, then compose. The composition used to live here, inline; it now
    lives in compose() so the dashboard's thumbnails page produces the same
    thing rather than its own approximation of it.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if out_path is None:
        out_path = video_path.with_suffix(".thumb.jpg")

    duration = _probe_duration(video_path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = tf.name
    try:
        img = _best_frame(video_path, duration, tmp_png)
        if img is None:
            raise RuntimeError("Frame extraction failed at all candidate timestamps")
        if img.size != (THUMB_W, THUMB_H):
            img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        hook = (script or "").strip().split("\n")[0]
        return compose(img, hook, Path(out_path), niche=niche)
    finally:
        Path(tmp_png).unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python thumbnail_gen.py <video.mp4> '<script text>'")
        sys.exit(1)

    video  = Path(sys.argv[1])
    script = sys.argv[2]
    out    = make_thumbnail(video, script)
    print(f"THUMBNAIL={out}")
