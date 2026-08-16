"""Serving the run keyframes without shipping the masters.

FROM A LIVE SESSION'S LOG: a few minutes of browsing the gallery produced well
over a hundred /debug/<run>/NN.png requests. Every tile, every prompt preview
and every queue strip pointed at the ORIGINAL 1080x1920 png and rendered it
into a 120px box, so the browser downloaded the whole thing each time. The
images were already lazy-loaded, which limits WHEN they are fetched and not
how big they are.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import dashboard  # noqa: E402


@pytest.fixture
def keyframes(tmp_path):
    from PIL import Image
    Image.new("RGB", (1080, 1920), (240, 240, 235)).save(tmp_path / "01.png")
    (tmp_path / "voice.mp3").write_bytes(b"not an image")
    return tmp_path


def test_a_thumbnail_is_a_fraction_of_the_master(keyframes):
    small = dashboard._thumb_of(keyframes, "01.png", 240)
    assert small is not None
    assert small.stat().st_size < (keyframes / "01.png").stat().st_size


def test_the_thumbnail_is_cached_not_reencoded(keyframes):
    first = dashboard._thumb_of(keyframes, "01.png", 240)
    stamp = first.stat().st_mtime_ns
    assert dashboard._thumb_of(keyframes, "01.png", 240) == first
    assert first.stat().st_mtime_ns == stamp


def test_a_regenerated_keyframe_invalidates_its_thumbnail(keyframes):
    """A run can be re-rendered into the same folder; a stale thumbnail would
    then show the previous video's picture."""
    import os
    import time
    small = dashboard._thumb_of(keyframes, "01.png", 240)
    time.sleep(0.01)
    src = keyframes / "01.png"
    os.utime(src, None)
    from PIL import Image
    Image.new("RGB", (1080, 1920), (10, 10, 10)).save(src)
    again = dashboard._thumb_of(keyframes, "01.png", 240)
    assert again.stat().st_mtime >= src.stat().st_mtime


def test_only_a_fixed_set_of_widths_is_served(keyframes):
    """An open ?w= is a cache bomb — one request per width, forever."""
    assert dashboard._thumb_of(keyframes, "01.png", 240) is not None
    assert dashboard._thumb_of(keyframes, "01.png", 241) is None
    assert dashboard._thumb_of(keyframes, "01.png", 99999) is None


def test_no_width_serves_the_master(keyframes):
    """Opening a keyframe in a new tab must still give the full picture."""
    assert dashboard._thumb_of(keyframes, "01.png", None) is None


def test_it_refuses_to_leave_the_run_folder(keyframes):
    assert dashboard._thumb_of(keyframes, "../secret.png", 240) is None


def test_a_non_image_is_served_untouched(keyframes):
    assert dashboard._thumb_of(keyframes, "voice.mp3", 240) is None


def test_any_failure_falls_back_to_the_master(tmp_path):
    """The original always works, so this is an optimisation and never a
    dependency — a missing Pillow must not blank the gallery."""
    assert dashboard._thumb_of(tmp_path, "nothing.png", 240) is None


def test_the_pages_ask_for_the_small_copy():
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert src.count("?w=240") >= 2
    assert "?w=120" in src


# ── the tab icon ────────────────────────────────────────────────────────────

def test_there_is_a_favicon(tmp_path, monkeypatch):
    """Every page load asked for one and got a 404."""
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        r = c.get("/favicon.ico")
    assert r.status_code == 200
    assert "svg" in r.headers["Content-Type"]


def test_the_favicon_is_cacheable():
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        r = c.get("/favicon.ico")
    assert "max-age" in r.headers.get("Cache-Control", "")
