"""Tests for qc_check.py — pure evaluation logic, no ffprobe/ffmpeg needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from qc_check import _extract_info, _evaluate, run_qc, MIN_BYTES


def _good_probe() -> dict:
    return {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920,
             "codec_name": "h264", "avg_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "42.5"},
    }


def _good_info() -> dict:
    return _extract_info(_good_probe())


# ── _extract_info ────────────────────────────────────────────────────────────────

def test_extract_info_reads_streams_and_duration():
    info = _good_info()
    assert info["has_video"] and info["has_audio"]
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["fps"] == 30.0
    assert info["duration"] == 42.5


def test_extract_info_handles_missing_everything():
    info = _extract_info({})
    assert not info["has_video"] and not info["has_audio"]
    assert info["duration"] == 0.0


def test_extract_info_handles_bad_frame_rate():
    probe = _good_probe()
    probe["streams"][0]["avg_frame_rate"] = "0/0"
    assert _extract_info(probe)["fps"] == 0.0


# ── _evaluate: critical failures hold uploads ────────────────────────────────────

def test_perfect_short_passes():
    critical, warnings = _evaluate(_good_info(), size_bytes=5_000_000, mean_vol=-16.0)
    assert critical == [] and warnings == []


def test_missing_audio_is_critical():
    info = _good_info()
    info["has_audio"] = False
    critical, _ = _evaluate(info, size_bytes=5_000_000)
    assert any("audio" in c for c in critical)


def test_wrong_resolution_is_critical():
    info = _good_info()
    info["width"], info["height"] = 1920, 1080   # landscape by mistake
    critical, _ = _evaluate(info, size_bytes=5_000_000)
    assert any("resolution" in c for c in critical)


def test_truncated_render_is_critical():
    info = _good_info()
    info["duration"] = 3.0
    critical, _ = _evaluate(info, size_bytes=5_000_000)
    assert any("duration" in c for c in critical)


def test_tiny_file_is_critical():
    critical, _ = _evaluate(_good_info(), size_bytes=MIN_BYTES - 1)
    assert any("too small" in c for c in critical)


# ── _evaluate: warnings don't block ──────────────────────────────────────────────

def test_long_but_legal_duration_is_warning_only():
    info = _good_info()
    info["duration"] = 90.0   # legal for Shorts (<=180) but past the sweet spot
    critical, warnings = _evaluate(info, size_bytes=5_000_000)
    assert critical == []
    assert any("sweet spot" in w for w in warnings)


def test_quiet_mix_is_warning_only():
    critical, warnings = _evaluate(_good_info(), size_bytes=5_000_000, mean_vol=-45.0)
    assert critical == []
    assert any("near-silent" in w for w in warnings)


# ── run_qc top level ─────────────────────────────────────────────────────────────

def test_run_qc_missing_file_is_critical():
    res = run_qc(Path("/nonexistent/video.mp4"))
    assert res["ok"] is False
    assert any("not found" in c for c in res["critical"])


# ── the last gate before upload ─────────────────────────────────────────────

def test_a_short_is_judged_exactly_as_before():
    """This gate decides whether a finished render ships. Nothing about the
    channel that exists may move."""
    import importlib
    import qc_check
    importlib.reload(qc_check)
    assert (qc_check.REQ_W, qc_check.REQ_H) == (1080, 1920)
    assert (qc_check.MIN_DUR, qc_check.MAX_DUR) == (10.0, 180.0)
    assert (qc_check.IDEAL_MIN, qc_check.IDEAL_MAX) == (25.0, 60.0)


def test_long_form_is_not_failed_for_being_the_shape_it_was_asked_for(monkeypatch):
    """REQ_W/REQ_H were 1080x1920, hard-coded, and this is the LAST thing that
    runs before upload — so every long-form render would have been flagged
    critical, 'wrong resolution', for being exactly the resolution the format
    profile told it to be."""
    import importlib
    import qc_check
    import video_format
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    importlib.reload(video_format)
    importlib.reload(qc_check)
    try:
        assert (qc_check.REQ_W, qc_check.REQ_H) == (1920, 1080)
        info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 540.0,
                "has_video": True, "has_audio": True}
        critical, _warnings = qc_check._evaluate(info, size_bytes=40_000_000)
        assert not critical, critical
    finally:
        monkeypatch.delenv("RUFUS_FORMAT", raising=False)
        importlib.reload(video_format)
        importlib.reload(qc_check)


def test_a_genuinely_wrong_encode_is_still_caught(monkeypatch):
    """The check is still worth having: an encode that came out at the wrong
    size is a real failure and looks identical in every other respect."""
    import qc_check
    info = {"width": 640, "height": 480, "fps": 30.0, "duration": 40.0,
            "has_video": True, "has_audio": True}
    critical, _warnings = qc_check._evaluate(info, size_bytes=5_000_000)
    assert any("resolution" in c for c in critical), critical


def test_long_form_is_not_told_every_time_that_it_is_too_long(monkeypatch):
    """25-60s is the retention sweet spot for a VERTICAL video. A nine-minute
    explainer is outside it by design, and a warning that fires on every
    single run is the noise this repo has twice walked back."""
    import importlib
    import qc_check
    import video_format
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    importlib.reload(video_format)
    importlib.reload(qc_check)
    try:
        assert qc_check.IDEAL_MAX > 600
    finally:
        monkeypatch.delenv("RUFUS_FORMAT", raising=False)
        importlib.reload(video_format)
        importlib.reload(qc_check)
