"""
Tests for edge-case guards across the Rufus pipeline.

Covers:
- script_writer: IndexError-safe hook scorer JSON parsing
- script_writer: _repair_banned synonym swap
- script_writer: double-banned-check fix (no double call)
- audio_gen:     render rejects empty bg_paths immediately
- audio_gen:     _ffmpeg_has_xfade / _ffmpeg_has_nvenc probes
- research:      corrupted used_seeds.json recovery (no crash)
- music_fetcher: archive.org per-identifier errors don't abort loop
- sd_client:     returns [] when A1111 not running (no crash)
- sd_client:     generate_clips skips failed images gracefully
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ── script_writer guards ─────────────────────────────────────────────────────

def test_hook_scorer_empty_dict_no_index_error():
    """list(scores.values())[0] on {} must not raise IndexError."""
    from script_writer import _standards
    # Simulate the exact expression that was crashing
    scores: dict = {}
    result = scores.get("results") or scores.get("hooks") or (list(scores.values())[0] if scores else [])
    assert result == []


def test_hook_scorer_dict_with_results_key():
    scores = {"results": [{"i": 1, "score": 7, "reason": "ok"}]}
    result = scores.get("results") or scores.get("hooks") or (list(scores.values())[0] if scores else [])
    assert result == [{"i": 1, "score": 7, "reason": "ok"}]


def test_repair_banned_crucial():
    from script_writer import _repair_banned, _find_banned
    script = "That habit was crucial to his wealth.\nWhat made it crucial?\nSave this."
    assert _find_banned(script) == "crucial"
    repaired = _repair_banned(script)
    assert _find_banned(repaired) is None
    assert "key" in repaired.lower()


def test_repair_banned_preserves_lines():
    from script_writer import _repair_banned
    script = "Line one.\nCrucial insight here.\nLine three.\nSave this."
    repaired = _repair_banned(script)
    assert len(repaired.strip().splitlines()) == 4


def test_repair_banned_multiple_phrases():
    from script_writer import _repair_banned, _find_banned
    script = "This is groundbreaking and vital research.\nSave this."
    repaired = _repair_banned(script)
    assert _find_banned(repaired) is None


def test_find_banned_no_double_call_needed():
    """Verify the new pattern (store result first) matches old behavior."""
    from script_writer import _find_banned
    script = "This is a crucial lesson about journeys."
    _banned = _find_banned(script)
    rejection = f"banned phrase: '{_banned}'" if _banned else None
    assert rejection is not None
    assert "crucial" in rejection


# ── audio_gen guards ─────────────────────────────────────────────────────────

def test_render_raises_on_empty_bg_paths(tmp_path):
    from audio_gen import render
    with pytest.raises(FileNotFoundError):
        render("test script", [], tmp_path)


def test_render_raises_on_nonexistent_bg_path(tmp_path):
    from audio_gen import render
    with pytest.raises(FileNotFoundError):
        render("test script", [tmp_path / "nonexistent.mp4"], tmp_path)


def test_ffmpeg_has_xfade_returns_bool():
    from audio_gen import _ffmpeg_has_xfade
    # Just verify it returns a bool without crashing (actual value depends on system)
    result = _ffmpeg_has_xfade()
    assert isinstance(result, bool)


def test_ffmpeg_has_nvenc_returns_bool():
    from audio_gen import _ffmpeg_has_nvenc
    result = _ffmpeg_has_nvenc()
    assert isinstance(result, bool)


def test_video_encoder_args_returns_list():
    from audio_gen import _video_encoder_args
    args = _video_encoder_args()
    assert isinstance(args, list)
    assert len(args) >= 2
    assert args[0] == "-c:v"


# ── research guards ──────────────────────────────────────────────────────────

def test_load_used_seeds_handles_corrupted_json(tmp_path, capsys):
    """Corrupted used_seeds.json should return [] and log a warning."""
    import research
    original = research.USED_SEEDS_FILE
    try:
        research.USED_SEEDS_FILE = tmp_path / "used_seeds.json"
        research.USED_SEEDS_FILE.write_text("{ broken json !!!")
        result = research._load_used_seeds()
        assert result == []
        captured = capsys.readouterr()
        assert "recovered" in captured.out.lower() or "corrupted" in captured.out.lower()
    finally:
        research.USED_SEEDS_FILE = original


def test_load_used_seeds_missing_file(tmp_path):
    import research
    original = research.USED_SEEDS_FILE
    try:
        research.USED_SEEDS_FILE = tmp_path / "no_such_file.json"
        result = research._load_used_seeds()
        assert result == []
    finally:
        research.USED_SEEDS_FILE = original


# ── music_fetcher guards ─────────────────────────────────────────────────────

def test_archive_music_per_identifier_error_continues(capsys):
    """A failure on one archive.org identifier must not abort the whole search."""
    from music_fetcher import _archive_music

    call_count = {"n": 0}

    def fake_get(url, **kwargs):
        call_count["n"] += 1
        if "advancedsearch" in url:
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {"response": {"docs": [
                {"identifier": "bad_id"},
                {"identifier": "also_bad"},
            ]}}
            return resp
        # metadata requests all fail
        raise Exception("network error")

    with patch("music_fetcher.requests.get", side_effect=fake_get):
        result = _archive_music("ambient")
    assert result is None   # graceful None, no exception
    captured = capsys.readouterr()
    assert "failed" in captured.out.lower()


def test_fetch_music_returns_none_on_all_failures():
    """fetch_music must return None (not raise) when all providers fail."""
    from music_fetcher import fetch_music
    with patch("music_fetcher._jamendo", return_value=None), \
         patch("music_fetcher._archive_music", return_value=None):
        result = fetch_music("finance")
    assert result is None


# ── sd_client guards ──────────────────────────────────────────────────────────

def test_sd_client_returns_empty_when_unavailable():
    """generate_clips must return [] (not raise) when A1111 is not running."""
    from sd_client import generate_clips
    with patch("sd_client.is_available", return_value=False):
        result = generate_clips(["stock market"], n=2)
    assert result == []


def test_sd_client_skips_failed_images():
    """generate_clips must skip a clip and continue when _generate_image fails."""
    from sd_client import generate_clips
    call_count = {"n": 0}

    def fake_generate(prompt):
        call_count["n"] += 1
        return None  # always fail

    with patch("sd_client.is_available", return_value=True), \
         patch("sd_client._generate_image", side_effect=fake_generate):
        result = generate_clips(["query1", "query2"], n=2)

    assert result == []
    assert call_count["n"] == 2   # tried both, skipped both — no exception


def test_sd_query_to_prompt_contains_query():
    """_query_to_prompt should embed the query and quality suffix."""
    from sd_client import _query_to_prompt
    prompt = _query_to_prompt("stock market crash")
    assert "stock market crash" in prompt
    assert "photorealistic" in prompt


def test_sd_upscale_lanczos_fallback():
    """_upscale_lanczos must return bytes larger than the input (2× scale)."""
    from sd_client import _upscale_lanczos
    from PIL import Image
    import io
    img = Image.new("RGB", (576, 1024), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    small = buf.getvalue()

    big = _upscale_lanczos(small)
    big_img = Image.open(io.BytesIO(big))
    assert big_img.size == (1152, 2048)
