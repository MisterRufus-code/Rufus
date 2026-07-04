"""Tests for the production-polish pass: silence trim, two-pass loudnorm,
de-esser gating, caption position, pacing, timeout presence, uploader guards."""

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import audio_gen
from audio_gen import (
    _audio_filter_complex,
    _parse_loudnorm_json,
    _silence_trim_cmd,
    MARGIN_V,
    RENDER_TIMEOUT,
)


# ── B1: silence trim ─────────────────────────────────────────────────────────────

def test_silence_trim_cmd_trims_head_and_tail():
    cmd = _silence_trim_cmd(Path("in.mp3"), Path("out.mp3"))
    af = cmd[cmd.index("-af") + 1]
    # head trim, then the areverse sandwich for the tail
    assert af.count("silenceremove") == 2
    assert af.count("areverse") == 2
    assert "start_periods=1" in af


def test_silence_trim_cmd_outputs_mp3_at_dst():
    cmd = _silence_trim_cmd(Path("a.mp3"), Path("b.mp3"))
    assert cmd[-1] == "b.mp3"
    assert "libmp3lame" in cmd


def test_trim_silence_fail_open_keeps_original(tmp_path):
    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"x" * 10_000)
    with patch("audio_gen.subprocess.run", side_effect=OSError("no ffmpeg")):
        audio_gen._trim_silence(mp3)          # must not raise
    assert mp3.exists() and mp3.stat().st_size == 10_000


# ── B3: loudnorm JSON parser ─────────────────────────────────────────────────────

_LOUDNORM_STDERR = """
[Parsed_loudnorm_0 @ 0x55d] some log line
{
    "input_i" : "-16.62",
    "input_tp" : "-2.32",
    "input_lra" : "6.10",
    "input_thresh" : "-27.04",
    "output_i" : "-14.02",
    "target_offset" : "0.32"
}
"""


def test_parse_loudnorm_json_extracts_measurement():
    m = _parse_loudnorm_json(_LOUDNORM_STDERR)
    assert m is not None
    assert m["input_i"] == "-16.62"
    assert m["target_offset"] == "0.32"


def test_parse_loudnorm_json_takes_last_valid_block():
    noisy = '{"unrelated": 1}\n' + _LOUDNORM_STDERR
    m = _parse_loudnorm_json(noisy)
    assert m and m["input_i"] == "-16.62"


def test_parse_loudnorm_json_none_on_garbage():
    assert _parse_loudnorm_json("no json here { broken") is None
    assert _parse_loudnorm_json("") is None


# ── B4: de-esser gating ──────────────────────────────────────────────────────────

def test_deesser_present_when_enabled():
    fc = _audio_filter_complex(2, 40.0, has_music=False, sfx_events=[],
                               with_deesser=True)
    assert "deesser" in fc


def test_deesser_absent_when_disabled():
    fc = _audio_filter_complex(2, 40.0, has_music=False, sfx_events=[],
                               with_deesser=False)
    assert "deesser" not in fc


# ── B5 + A1: constants ───────────────────────────────────────────────────────────

def test_caption_margin_below_face_zone():
    # ~31% from bottom of a 1920px frame — clear of centered portrait faces
    assert 500 <= MARGIN_V <= 660


def test_render_timeout_configured():
    assert RENDER_TIMEOUT >= 300   # enough for slow renders, but finite


def test_render_calls_carry_timeout():
    src = inspect.getsource(audio_gen.render)
    # both ffmpeg render invocations must pass the timeout
    assert src.count("timeout=RENDER_TIMEOUT") >= 2
    assert "TimeoutExpired" in src


# ── B2: pacing ───────────────────────────────────────────────────────────────────

def test_split_beats_default_allows_ten_scenes():
    import main
    sig = inspect.signature(main._split_beats)
    assert sig.parameters["max_scenes"].default == 10
    # 12 sentences collapse to exactly 10 beats
    script = " ".join(f"This is sentence number {i} spoken aloud." for i in range(12))
    assert len(main._split_beats(script)) == 10


# ── A2/A5: uploader guards ───────────────────────────────────────────────────────

def test_uploader_refresh_is_guarded():
    import youtube_uploader
    src = inspect.getsource(youtube_uploader.get_authenticated_service)
    assert "creds.refresh(Request())" in src
    # the refresh call sits inside a try block and falls back to the flow
    assert src.index("try:") < src.index("creds.refresh(Request())")
    assert "re-auth" in src or "RuntimeError" in src


def test_uploader_no_status_shadowing():
    import youtube_uploader
    src = inspect.getsource(youtube_uploader.upload)
    assert "progress, response = request.next_chunk()" in src


# ── A4: thumbnail None-script guard ──────────────────────────────────────────────

def test_thumbnail_hook_extraction_none_safe():
    import thumbnail_gen
    src = inspect.getsource(thumbnail_gen.make_thumbnail)
    assert '(script or "")' in src


# ── Low-RAM contingency: Whisper model env knob ──────────────────────────────────

def test_whisper_model_env_override(monkeypatch):
    import audio_gen as ag
    monkeypatch.setenv("RUFUS_WHISPER_MODEL", "base")
    monkeypatch.setattr(ag, "_whisper_model", None)
    captured = {}

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None):
            captured["name"] = name

    monkeypatch.setattr(ag, "WhisperModel", FakeModel)
    monkeypatch.setattr(ag, "_GPU", False)
    ag._whisper()
    assert captured["name"] == "base"
    monkeypatch.setattr(ag, "_whisper_model", None)   # don't poison other tests


# ── Whisper CUDA lazy-load failure fallback ──────────────────────────────────────

def test_transcribe_retries_on_cpu_when_cuda_lazy_load_fails(monkeypatch):
    """ctranslate2 lazy-loads its CUDA backend — construction can succeed while
    the first .transcribe() call still fails (e.g. missing cublas64_12.dll).
    _transcribe() must catch that and retry on CPU, not just guard construction."""
    import audio_gen as ag

    calls = {"n": 0}

    class FakeCudaModel:
        def transcribe(self, *a, **kw):
            calls["n"] += 1
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    class FakeCpuModel:
        def transcribe(self, *a, **kw):
            calls["n"] += 1
            return ("segments", "info")

    def fake_whisper(force_cpu=False):
        return FakeCpuModel() if force_cpu else FakeCudaModel()

    monkeypatch.setattr(ag, "_whisper", fake_whisper)
    monkeypatch.setattr(ag, "_whisper_device", "cuda")

    result = ag._transcribe(Path("dummy.mp3"))
    assert result == ("segments", "info")
    assert calls["n"] == 2   # first CUDA attempt, then the CPU retry


def test_transcribe_reraises_when_already_on_cpu(monkeypatch):
    import audio_gen as ag

    class FailingCpuModel:
        def transcribe(self, *a, **kw):
            raise RuntimeError("disk read error")

    monkeypatch.setattr(ag, "_whisper", lambda force_cpu=False: FailingCpuModel())
    monkeypatch.setattr(ag, "_whisper_device", "cpu")

    import pytest
    with pytest.raises(RuntimeError, match="disk read error"):
        ag._transcribe(Path("dummy.mp3"))
