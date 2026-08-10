"""Kokoro's dependency check reports ALL missing packages at once.

Why: Python raises on the first failed import, and _kokoro happens to import
soundfile before kokoro. A box missing both was told "No module named
'soundfile'", the owner installed exactly that, reran a full pipeline, and got
"No module named 'kokoro'" for their trouble — one wasted round trip per
missing package. Listing them costs nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import tts_engine  # noqa: E402


def test_reports_every_missing_dependency_not_just_the_first(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **kw: None if name in ("soundfile", "kokoro")
        else real(name, *a, **kw))
    assert tts_engine._missing_kokoro_deps() == ["soundfile", "kokoro"]


def test_error_names_all_of_them_and_gives_one_command(monkeypatch, tmp_path):
    import importlib.util
    import pytest
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **kw: None if name in ("soundfile", "kokoro")
        else real(name, *a, **kw))
    with pytest.raises(RuntimeError) as exc:
        tts_engine._kokoro("hello", tmp_path / "out.mp3")
    msg = str(exc.value)
    assert "soundfile" in msg and "kokoro" in msg
    assert "pip install soundfile kokoro" in msg


def test_nothing_missing_means_no_early_raise(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **kw: object())
    assert tts_engine._missing_kokoro_deps() == []


def test_the_requirement_list_matches_what_kokoro_actually_imports():
    """A drifted list would report a clean bill of health and then crash on the
    real import — worse than no check."""
    import inspect
    body = inspect.getsource(tts_engine._kokoro)
    for module in tts_engine.KOKORO_REQUIREMENTS:
        assert module in body, f"{module} listed but not imported by _kokoro"


# ── The pin that keeps the good voice ────────────────────────────────────────
# Kokoro fails on numpy 2.x ("Cannot interpret 'torch.float32' as a data type")
# and tts_engine falls back to Edge on ANY Kokoro failure — by design, so a
# render never breaks. The cost of that design is that losing the good voice is
# SILENT: no error, just a flatter video. So the pin is the only thing standing
# between a fresh install and quietly worse audio forever.

_REQS = (Path(__file__).parent.parent / "requirements.txt").read_text()


def test_numpy_is_pinned_below_2():
    assert "numpy<2" in _REQS


def test_the_pin_says_why_it_exists():
    """An unexplained pin is the kind a future cleanup deletes."""
    i = _REQS.index("numpy<2")
    reason = _REQS[max(0, i - 700):i].lower()
    assert "kokoro" in reason
    assert "torch.float32" in reason


def test_opencv_cannot_drag_numpy_2_back_in():
    """opencv-python-headless >=4.12 declares numpy>=2 for Python 3.9+, which
    collides head-on with the pin above. Left at <5 pip installs 4.14 and
    errors on every install — the state the owner's box was actually in."""
    assert "opencv-python-headless>=4.8,<4.12" in _REQS
    assert "opencv-python-headless>=4.8,<5" not in _REQS
