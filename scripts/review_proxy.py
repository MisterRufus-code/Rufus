#!/usr/bin/env python3
"""
review_proxy.py — a small, watchable copy of a finished Short, for review.

Why this exists: reviewing a run from a phone had exactly one option, and it
was a bad one. The dashboard serves the master mp4 with as_attachment=True, so
the phone must download 15-25MB before showing a single frame — no preview, no
seek. And notify.send_file can post a file into Discord, but Discord's attach
limit is 8MB, so the master never fits: it falls through to "File too large to
attach" and posts a link, which is the same dead end.

So the master is the wrong artifact to move. What review needs is small and
watchable, not archival:

  - half-height (540x960 from 1080x1920), which is already more than a phone
    shows in a review pass
  - a bitrate ceiling that keeps a 40-second Short comfortably under the
    Discord limit
  - AAC audio at a real bitrate, because the voice IS the thing being judged —
    the whole point of the review is whether the script sounds good aloud, and
    that survives compression far worse than the picture does

The master is never touched, never replaced, and never uploaded from here.
This produces a throwaway alongside it.

CONTRACT: nothing in here may raise into the pipeline or the dashboard. A
missing ffmpeg, a corrupt input or a full disk returns None; the caller keeps
whatever it was already going to do. Review tooling must never be able to turn
a good render into a failure.
"""

import os
import subprocess
from pathlib import Path

import paths

# Discord's attach limit for a free server, with headroom for the multipart
# envelope. notify.py holds the same number for its own pre-check; this one is
# the encoder's target, so it is deliberately lower than the wall.
TARGET_BYTES = 7 * 1024 * 1024

# Encoder settings. Chosen to land a 30-60s vertical Short in 3-6MB, which is
# the range where Discord accepts it and a phone on cellular still loads it in
# a couple of seconds.
PROXY_HEIGHT = 960          # half of the 1920 master
VIDEO_BITRATE = "900k"
AUDIO_BITRATE = "128k"      # the voice is what's being reviewed — don't starve it
ENCODE_TIMEOUT = 300


def proxy_path(source: Path) -> Path:
    """Where the proxy for `source` lives: alongside the masters, suffixed.

    Same directory rather than a new root, because these are per-run artifacts
    that should be swept by the same housekeeping that sweeps the masters —
    a review copy outliving the video it reviews is just litter."""
    source = Path(source)
    return source.parent / f"{source.stem}.review.mp4"


def enabled() -> bool:
    return os.environ.get("RUFUS_REVIEW_PROXY", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def build(source, *, force: bool = False) -> Path | None:
    """Encode (or reuse) a small review copy of `source`. None on any failure.

    Reuses an existing proxy that is NEWER than its source, so a re-review or a
    second notification doesn't re-encode. `force` re-encodes regardless."""
    if not enabled():
        return None
    src = Path(source)
    if not src.exists() or src.stat().st_size == 0:
        return None

    out = proxy_path(src)
    if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime \
            and out.stat().st_size > 0:
        return out

    # A master already small enough is its own proxy — encoding it again would
    # only lose quality for no gain.
    if src.stat().st_size <= TARGET_BYTES:
        return src

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        # -2 keeps the width even (libx264 requires it) while preserving aspect.
        "-vf", f"scale=-2:{PROXY_HEIGHT}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", VIDEO_BITRATE, "-maxrate", VIDEO_BITRATE, "-bufsize", "1800k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        # Puts the index at the front so a phone can start playing before the
        # whole file has arrived — the difference between "loads" and "spins".
        "-movflags", "+faststart",
        str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=ENCODE_TIMEOUT)
    except FileNotFoundError:
        print("[proxy] ffmpeg not found — skipping review copy")
        return None
    except subprocess.TimeoutExpired:
        print(f"[proxy] encode timed out after {ENCODE_TIMEOUT}s — skipping")
        _unlink(out)
        return None
    except Exception as e:
        print(f"[proxy] encode failed ({e}) — skipping")
        _unlink(out)
        return None

    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        print(f"[proxy] ffmpeg failed: {err[-1] if err else 'no output'}")
        _unlink(out)
        return None

    # Still over the wall after encoding (a very long video) — report it rather
    # than handing the caller a file Discord will reject with a 413.
    if out.stat().st_size > TARGET_BYTES:
        print(f"[proxy] review copy is still "
              f"{out.stat().st_size // 1024 // 1024}MB — too big to attach")
        return None

    print(f"[proxy] review copy: {out.name} "
          f"({out.stat().st_size // 1024}KB from "
          f"{src.stat().st_size // 1024 // 1024}MB)")
    return out


# ── Contact sheet ────────────────────────────────────────────────────────────
# The other half of the same problem. A run's debug folder holds 8-10 stills at
# 1.1-2.7MB EACH, so putting them on a page as-is means a 15-25MB load — worse
# than the mp4 it was meant to replace. Reviewing the visuals from a phone
# needs one small image, not ten large ones.

SHEET_COLUMNS = 5
SHEET_CELL_WIDTH = 210      # ~1050px wide overall: sharp on a phone, still small
SHEET_QUALITY = 78
SHEET_NAME = "contact_sheet.jpg"


def contact_sheet(run_dir, *, force: bool = False) -> Path | None:
    """One JPEG of a run's beat stills in order, or None.

    Cached next to the stills and rebuilt only when a still is newer than the
    sheet. Beat order is the review's whole point — the stills are named
    01.png..10.png, so lexical order IS narration order, and it is preserved
    exactly rather than sorted by mtime (which the parallel render scrambles)."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        print(f"[proxy] Pillow unavailable ({e}) — no contact sheet")
        return None

    d = Path(run_dir)
    if not d.is_dir():
        return None
    stills = sorted(p for p in d.glob("*.png") if p.stem.isdigit())
    if not stills:
        return None

    out = d / SHEET_NAME
    if not force and out.exists():
        newest = max(p.stat().st_mtime for p in stills)
        if out.stat().st_mtime >= newest and out.stat().st_size > 0:
            return out

    try:
        cols = min(SHEET_COLUMNS, len(stills))
        rows = (len(stills) + cols - 1) // cols
        with Image.open(stills[0]) as probe:
            ratio = probe.height / probe.width
        cw = SHEET_CELL_WIDTH
        ch = max(1, round(cw * ratio))
        sheet = Image.new("RGB", (cols * cw, rows * ch), (17, 17, 20))
        draw = ImageDraw.Draw(sheet)
        for i, still in enumerate(stills):
            with Image.open(still) as im:
                im = im.convert("RGB").resize((cw, ch), Image.LANCZOS)
                sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
            # The beat number, so a fault can be reported as "beat 7" without
            # counting cells — which is how every review of these runs has
            # actually been written.
            x, y = (i % cols) * cw + 6, (i // cols) * ch + 6
            draw.rectangle([x - 2, y - 2, x + 20, y + 16], fill=(0, 0, 0))
            draw.text((x, y), still.stem, fill=(255, 255, 255))
        sheet.save(out, "JPEG", quality=SHEET_QUALITY, optimize=True)
    except Exception as e:
        print(f"[proxy] contact sheet failed ({e})")
        _unlink(out)
        return None

    print(f"[proxy] contact sheet: {out.name} ({out.stat().st_size // 1024}KB "
          f"for {len(stills)} stills)")
    return out


def _unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        latest = sorted(paths.output_dir().glob("*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        latest = [p for p in latest if not p.name.endswith(".review.mp4")]
        if not latest:
            print("usage: python scripts/review_proxy.py <video.mp4>")
            raise SystemExit(1)
        target = latest[0]
        print(f"[proxy] no path given — using the newest render: {target.name}")
    else:
        target = Path(sys.argv[1])
    result = build(target, force=True)
    print(result or "failed")
