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


# ── Kokoro hands back torch Tensors ─────────────────────────────────────────
# The live failure: "Cannot interpret 'torch.float32' as a data type", every
# run, silently falling back to the flat Edge voice. It reads exactly like a
# numpy-2 incompatibility and was chased as one — but it reproduces on numpy
# 1.26.4. The real cause is np.zeros(gap, dtype=seg_audio.dtype) being handed a
# torch dtype.
#
# It fires ONLY on multi-chunk scripts, because a single chunk never reaches
# the inter-chunk gap. That is why a one-sentence smoke test passed while every
# real script failed — so these tests use TWO chunks on purpose.

class _FakeTensor:
    """Stands in for what Kokoro actually yields: has .detach()/.cpu()/.numpy()
    and a dtype numpy refuses."""

    def __init__(self, arr):
        self._arr = arr
        self.dtype = "torch.float32"       # numpy raises on this, as torch does

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _pipe_yielding(n_chunks):
    import numpy as np

    def pipe(script, voice=None, speed=None):
        for i in range(n_chunks):
            yield (f"Chunk {i}.", "phonemes",
                   _FakeTensor(np.zeros(2400, dtype=np.float32)))
    return pipe


def _run_kokoro(monkeypatch, tmp_path, n_chunks):
    """Drive the real _kokoro with the ONE thing stubbed that needs a GPU-free
    stand-in — the pipeline. soundfile/kokoro are import-only here, so a box
    without them still exercises the dtype path this exists to test."""
    import types
    sf = types.ModuleType("soundfile")
    sf.write = lambda path, data, rate: Path(path).write_bytes(b"\0" * 100)
    kk = types.ModuleType("kokoro")
    kk.KPipeline = object
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    monkeypatch.setitem(sys.modules, "kokoro", kk)
    monkeypatch.setattr(tts_engine, "_missing_kokoro_deps", lambda: [])
    monkeypatch.setattr(tts_engine, "_kokoro_pipe", _pipe_yielding(n_chunks))

    out = tmp_path / "out.mp3"
    monkeypatch.setattr(tts_engine.subprocess, "run",
                        lambda *a, **kw: _ok_ffmpeg(out))
    tts_engine._kokoro("two lines", out)
    return out


def _ok_ffmpeg(out):
    out.write_bytes(b"\0" * 6000)          # ffmpeg's job, stubbed
    class R:
        returncode = 0
        stderr = ""
    return R()


def test_a_multi_chunk_script_survives_torch_tensors(monkeypatch, tmp_path):
    """The exact live failure. Two chunks means one inter-chunk gap, which is
    where the torch dtype used to reach numpy."""
    out = _run_kokoro(monkeypatch, tmp_path, n_chunks=2)
    assert out.exists()


def test_the_single_chunk_case_never_proved_anything(monkeypatch, tmp_path):
    """Kept as documentation: this passed all along, on the broken code too.
    A green one-chunk test is not evidence that Kokoro works."""
    out = _run_kokoro(monkeypatch, tmp_path, n_chunks=1)
    assert out.exists()


def test_the_misleading_numpy_hint_is_gone():
    """The old hint printed 'install numpy<2' on this error and sent two rounds
    of debugging at the environment instead of the code."""
    import inspect
    src = inspect.getsource(tts_engine)
    assert "this is the numpy-2 incompatibility" not in src


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
