#!/usr/bin/env python3
"""Does this installation actually work — proved by doing, not by checking.

WHY THIS IS DIFFERENT FROM health_check AND preflight. Those ask whether the
PIECES are present: is there a key, is ffmpeg on PATH, is ComfyUI answering.
Present is not the same as working. An ffmpeg on PATH built without libx264
encodes nothing; a Python with a broken Pillow imports fine and fails on the
first thumbnail; an sqlite on a filesystem that will not honour a write-ahead
log looks perfect until two processes touch it at once. Every one of those
passes a presence check and fails a video.

So this one RUNS things. Small things — a one-second clip, a caption file, a
database round trip, a page render — but real ones, using the same code paths a
video uses, and it takes seconds rather than the hour a full render costs.

WHAT IT DELIBERATELY DOES NOT DO. It does not call a language model, draw a
picture or synthesise a voice. Those cost money, GPU hours and somebody else's
rate limit, and a check nobody can afford to run is a check nobody runs. It
proves the machinery around them: the parts that are the same for every video
and that fail the same way for everybody.

WHAT A PASS MEANS AND WHAT IT DOES NOT. A pass means this installation can
encode, caption, store and serve. It does not mean the videos will be any good
— that is measured on the channel, not here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


class Check:
    """One thing that was actually done, and what happened."""

    def __init__(self, name: str, what: str):
        self.name = name
        self.what = what
        self.ok: bool | None = None
        self.detail = ""
        self.seconds = 0.0
        self.skipped_because = ""

    def run(self, fn) -> "Check":
        started = time.time()
        try:
            self.detail = fn() or ""
            self.ok = True
        except _Skip as e:
            self.ok = None
            self.skipped_because = str(e)
        except Exception as e:
            self.ok = False
            self.detail = f"{type(e).__name__}: {e}"
        self.seconds = time.time() - started
        return self


class _Skip(Exception):
    """Not applicable to this installation — different from a failure.

    Collapsing the two is how a smoke test starts lying. A machine with no
    ffmpeg cannot encode and that is a FAILURE, because every video needs one.
    A machine with no ComfyUI has simply not chosen that engine, and reporting
    it as broken teaches people the test is noise.
    """


def _encode_a_real_clip() -> str:
    """Encode one second of video with the same encoder a render uses.

    THE HARDEST DEPENDENCY, PROVED RATHER THAN LOCATED. `shutil.which` finding
    ffmpeg says a file exists with that name. This says the file can take a
    stream and produce a playable H.264 mp4 — which is what every single video
    ends with, and which a build compiled without the encoder cannot do however
    correctly it is installed.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not on PATH — every video is cut, captioned and muxed "
            "by it. Windows: winget install Gyan.FFmpeg · macOS: brew install "
            "ffmpeg · Debian: sudo apt install ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe.mp4"
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg is present but could not encode H.264: "
                f"{result.stderr.strip()[-300:]}. A build without libx264 "
                f"passes every presence check and renders nothing.")
        size = out.stat().st_size
        if size < 200:
            raise RuntimeError("ffmpeg produced an empty file")
        return f"1s H.264, {size} bytes"


def _build_a_caption_file() -> str:
    """Build a real ASS subtitle file through the code a render uses.

    Not a string comparison: audio_gen.build_ass reads the active format
    profile and the chosen caption style, so this is the one check that would
    catch a style preset whose numbers libass cannot use.
    """
    import audio_gen

    words = [type("W", (), {"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35})()
             for i, w in enumerate(["ninety", "per", "cent", "gone"])]
    segments = [type("S", (), {"words": words})()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ass"
        audio_gen.build_ass(segments, path, 2.0)
        text = path.read_text(encoding="utf-8")
    if "[Events]" not in text or "Style: Default" not in text:
        raise RuntimeError("the subtitle file is missing its style block")
    lines = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    import caption_styles
    style = caption_styles.name()
    if not lines and caption_styles.preset(style).get("enabled", True):
        raise RuntimeError(f"caption style {style!r} produced no lines at all")
    return f"style {style!r}, {len(lines)} caption(s)"


def _round_trip_the_database() -> str:
    """Write a row and read it back, in a throwaway database.

    Deliberately not the real one. A smoke test that writes to the owner's
    database to prove it can write to a database is a test that has to be
    cleaned up after, and the cleanup is where it goes wrong.
    """
    import db_manager

    real = db_manager.DB_FILE
    with tempfile.TemporaryDirectory() as tmp:
        try:
            db_manager.DB_FILE = Path(tmp) / "probe.db"
            db_manager.init_db()
            vid = db_manager.save_video(
                niche="smoke", script_hook="probe", scene_desc="probe",
                video_file="probe.mp4", score=1)
            row = db_manager.video_by_id(vid)
            if not row or row["score"] != 1:
                raise RuntimeError("the row did not come back as written")
            mode = db_manager.journal_mode()
        finally:
            db_manager.DB_FILE = real
    if mode != "wal":
        # Worth failing on. WAL is what lets the dashboard read while a run
        # writes, and a filesystem that silently refuses it (some network
        # shares) turns every concurrent read into a locked database.
        raise RuntimeError(
            f"journal_mode came back {mode!r} rather than WAL — the dashboard "
            f"reads while a run writes, and without WAL that locks. Usually a "
            f"network drive; move the database to a local disk.")
    return "wrote and read a row, journal_mode=wal"


def _serve_a_page() -> str:
    """Render the dashboard's own front page through Flask's test client.

    Imports the whole app, which is a real check on its own — but importing is
    not serving, and a template error only appears when something asks for the
    page.
    """
    import dashboard

    response = dashboard.app.test_client().get("/")
    if response.status_code != 200:
        raise RuntimeError(f"the front page returned {response.status_code}")
    body = response.get_data(as_text=True)
    if "</html>" not in body:
        raise RuntimeError("the page came back truncated")
    return f"{len(body) // 1024}KB of HTML"


def _compose_a_thumbnail() -> str:
    """Draw text onto an image with the bundled font.

    Pillow imports on a broken install and fails at the first draw, and the
    font is the part most likely to be missing from a copy taken by hand.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_file = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"
    if not font_file.exists():
        raise RuntimeError(
            f"{font_file.relative_to(ROOT)} is missing — captions and "
            f"thumbnails fall back to Arial, which is a different channel")
    image = Image.new("RGB", (320, 180), (12, 16, 22))
    draw = ImageDraw.Draw(image)
    draw.text((16, 60), "NINETY PER CENT",
              font=ImageFont.truetype(str(font_file), 28), fill=(255, 210, 63))
    # getcolors rather than getdata: the same answer, without the deprecation
    # warning Pillow 12 prints, and it stops at the first hundred colours
    # instead of walking every pixel.
    colours = image.getcolors(maxcolors=4096) or []
    if len(colours) < 2:
        raise RuntimeError("nothing was drawn — the image is still one colour")
    return f"drew text in Anton, {len(colours)} colour(s)"


def _read_the_configuration() -> str:
    """Load the files a run reads before it does anything.

    A niches.json that lost a brace is a run that dies at step 1 having already
    taken the lock.
    """
    import json

    loaded = []
    for name in ("niches.json", "styles.json", "script_standards.json"):
        path = ROOT / "config" / name
        if not path.exists():
            raise RuntimeError(f"config/{name} is missing")
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"config/{name} is not valid JSON: {e}")
        loaded.append(name)
    import video_format
    profile = video_format.name()
    return f"{len(loaded)} file(s), format {profile!r}"


CHECKS = [
    ("configuration", "the files every run reads first", _read_the_configuration),
    ("database", "a row written and read back", _round_trip_the_database),
    ("encoder", "one second of real H.264", _encode_a_real_clip),
    ("captions", "a subtitle file through the render path", _build_a_caption_file),
    ("thumbnails", "text drawn with the bundled font", _compose_a_thumbnail),
    ("dashboard", "the front page actually served", _serve_a_page),
]


def run() -> list[Check]:
    """Do everything, in order, and return what happened."""
    return [Check(name, what).run(fn) for name, what, fn in CHECKS]


def _cli() -> int:
    try:
        import version
        print(f"\n  {version.stamp()} — smoke test")
    except Exception:
        print("\n  Rufus — smoke test")
    print("  " + "─" * 62)

    results = run()
    for check in results:
        if check.ok is None:
            print(f"  ·  {check.name:14} skipped — {check.skipped_because}")
        elif check.ok:
            print(f"  ✓  {check.name:14} {check.detail}  "
                  f"({check.seconds:.1f}s)")
        else:
            print(f"  ✗  {check.name:14} {check.what}")
            print(f"     {check.detail}")

    failed = [c for c in results if c.ok is False]
    print()
    if failed:
        print(f"  {len(failed)} of {len(results)} failed. This installation "
              f"cannot produce a video yet.\n")
        return 1
    print("  This installation can encode, caption, store and serve.")
    print("  It says nothing about whether the videos are any good — that is")
    print("  measured on the channel.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
