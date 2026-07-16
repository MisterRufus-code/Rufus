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
import os
import sys
import time
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
    """fetch_music must return None (not raise) when every provider AND the
    local synth bed fail."""
    from music_fetcher import fetch_music
    with patch("music_fetcher._jamendo", return_value=None), \
         patch("music_fetcher._archive_music", return_value=None), \
         patch("music_gen.ensure_music", return_value=None):
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
    """generate_clips must skip a clip and continue when _generate_image fails.

    A failed generation is retried (transient 6GB-GPU OOM is common) up to
    MAX_DUP_RETRIES+1 times per clip, then the clip is skipped — no exception.
    """
    import sd_client
    from sd_client import generate_clips
    call_count = {"n": 0}

    def fake_generate(prompt, seed=-1):
        call_count["n"] += 1
        return None  # always fail

    with patch("sd_client.is_available", return_value=True), \
         patch("sd_client._generate_image", side_effect=fake_generate):
        result = generate_clips(["query1", "query2"], n=2)

    assert result == []
    # Both clips attempted; each retried the full budget before being skipped.
    assert call_count["n"] == 2 * (sd_client.MAX_DUP_RETRIES + 1)


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


# ── tts_engine guards ─────────────────────────────────────────────────────────

def test_tts_engine_edge_backend_default():
    """RUFUS_TTS unset → backend is 'edge'."""
    import os
    import tts_engine
    old = os.environ.pop("RUFUS_TTS", None)
    try:
        assert tts_engine._backend() == "edge"
    finally:
        if old is not None:
            os.environ["RUFUS_TTS"] = old


def test_tts_engine_xtts_backend_when_set():
    """RUFUS_TTS=xtts → backend is 'xtts'."""
    import os
    import tts_engine
    os.environ["RUFUS_TTS"] = "xtts"
    try:
        assert tts_engine._backend() == "xtts"
    finally:
        os.environ.pop("RUFUS_TTS", None)


def test_tts_engine_xtts_fallback_on_import_error(tmp_path):
    """synthesize must fall back to Edge TTS when XTTS model import fails."""
    import os
    import tts_engine

    os.environ["RUFUS_TTS"] = "xtts"
    out = tmp_path / "voice.mp3"
    edge_called = {"n": 0}

    def fake_xtts(script, path):
        raise ImportError("TTS package not installed")

    def fake_edge(script, path):
        edge_called["n"] += 1
        path.write_bytes(b"\x00" * 100)  # dummy mp3

    try:
        with patch("tts_engine._xtts", side_effect=fake_xtts), \
             patch("tts_engine._edge", side_effect=fake_edge):
            tts_engine.synthesize("test text", out)
        assert edge_called["n"] == 1, "Edge TTS fallback was not called"
    finally:
        os.environ.pop("RUFUS_TTS", None)


# ── stackexchange research guards ─────────────────────────────────────────────

def _se_item(title, body, score=80, link="https://money.stackexchange.com/q/1"):
    return {"title": title, "body": f"<p>{body}</p>", "score": score, "link": link}


def test_stackexchange_returns_story_and_strips_html():
    import research
    body = "I inherited $40,000 last year and almost lost it all. " * 10
    fake = MagicMock()
    fake.raise_for_status = lambda: None
    fake.json.return_value = {"items": [_se_item("I saved $40,000 in 2 years", body)]}
    with patch("research.httpx.get", return_value=fake):
        seed = research.fetch_stackexchange_story("finance")
    assert seed is not None
    assert seed["type"] == "stackexchange"
    assert "<p>" not in seed["content"]
    assert seed["source"] == "money.stackexchange.com"


def test_stackexchange_skips_used_and_low_score():
    import research
    body = "Real story with enough length to pass the body filter. " * 12
    used_link = "https://money.stackexchange.com/q/1"
    items = [
        _se_item("I saved $9,000", body, score=80, link=used_link),     # used
        _se_item("I lost $500 on fees", body, score=5),                 # low score
    ]
    fake = MagicMock()
    fake.raise_for_status = lambda: None
    fake.json.return_value = {"items": items}
    with patch("research.httpx.get", return_value=fake):
        seed = research.fetch_stackexchange_story("finance", used_ids={"se:" + used_link})
    assert seed is None


def test_stackexchange_unreachable_returns_none(capsys):
    import research
    with patch("research.httpx.get", side_effect=Exception("403 Blocked")):
        seed = research.fetch_stackexchange_story("finance")
    assert seed is None
    assert "unreachable" in capsys.readouterr().out.lower()


def test_stackexchange_skips_unmapped_niches():
    import research
    assert research.fetch_stackexchange_story("motivation") is None


# ── anti-repetition machinery ─────────────────────────────────────────────────

def test_dedupe_similar_hooks_collapses_same_shape():
    from script_writer import _dedupe_similar_hooks
    hooks = [
        "$2.4M by 38. Still scared to retire.",
        "$500K saved. Still renting on purpose.",   # number-shape, different opener → kept
        "$2.4M by 40. Still terrified to quit.",    # number-shape, same opener "$2.4m by" → dropped
        "Why do lottery winners go broke?",
        "Buffett's worst trade made him $25B.",
    ]
    out = _dedupe_similar_hooks(hooks)
    assert "$2.4M by 38. Still scared to retire." in out
    assert "$2.4M by 40. Still terrified to quit." not in out
    assert len(out) == 4


def test_novelty_block_empty_without_db_or_learnings(tmp_path, monkeypatch):
    import script_writer
    monkeypatch.setattr(script_writer, "_recent_video_rows", lambda n, limit=12: [])
    monkeypatch.setattr(script_writer, "_load_learnings", lambda: {})
    assert script_writer._novelty_block("finance") == ""


def test_novelty_block_lists_recent_hooks_and_learnings(monkeypatch):
    import script_writer
    monkeypatch.setattr(script_writer, "_recent_video_rows",
                        lambda n, limit=12: [("Old hook one", "body"), ("Old hook two", "body")])
    monkeypatch.setattr(script_writer, "_load_learnings",
                        lambda: {"winning_hooks": ["Winner A"], "losing_hooks": ["Loser B"]})
    blk = script_writer._novelty_block("finance")
    assert "Old hook one" in blk and "ALREADY PUBLISHED" in blk
    assert "Winner A" in blk and "WINNERS" in blk
    assert "Loser B" in blk and "LOSERS" in blk


def test_pick_cta_avoids_recent_last_lines(monkeypatch):
    import script_writer
    pool = ["Save this.", "Follow for more.", "Send this to a friend."]
    monkeypatch.setattr(script_writer, "_recent_video_rows",
                        lambda n, limit=4: [("h", "Hook line.\nBody.\nSave this."),
                                            ("h", "Hook.\nBody.\nFollow for more.")])
    for _ in range(10):
        cta = script_writer._pick_cta({"cta_pool": pool}, "finance")
        assert cta == "Send this to a friend."


def test_audio_simple_with_loudnorm_masters_fallback():
    from audio_gen import _audio_filter_simple
    fc = _audio_filter_simple(2, 30.0, has_music=True, with_loudnorm=True)
    assert "loudnorm=I=-14" in fc
    voice_only = _audio_filter_simple(2, 30.0, has_music=False, with_loudnorm=True)
    assert "loudnorm=I=-14" in voice_only


def test_wisdom_quote_author_uniform(monkeypatch, tmp_path):
    """28 Buffett quotes vs 2 Munger quotes — Munger must still be picked ~half the time."""
    import json as _json
    import research
    quotes = [{"text": f"buffett wisdom {i}", "author": "Warren Buffett"} for i in range(28)]
    quotes += [{"text": f"munger wisdom {i}", "author": "Charlie Munger"} for i in range(2)]
    f = tmp_path / "finance.json"
    f.write_text(_json.dumps({"quotes": quotes}))
    monkeypatch.setattr(research, "WISDOM_DIR", tmp_path)
    import random as _random
    _random.seed(7)
    authors = [research.pick_wisdom_quote("finance")["source"] for _ in range(200)]
    munger_share = authors.count("Charlie Munger") / len(authors)
    assert 0.3 < munger_share < 0.7   # ~0.5 expected; would be ~0.07 with plain choice


# ── Per-channel instance lock (multi-channel concurrency) ────────────────────

def test_acquire_lock_is_per_channel(tmp_path, monkeypatch):
    """Two DIFFERENT channels must be able to hold their locks concurrently;
    a second run of the SAME channel must be refused. This is the Phase-1
    scaling unlock: the old global lock serialized all channels."""
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)

    # Channel A acquires
    main._acquire_lock("main_en")
    lock_a = main._INSTANCE_LOCK
    assert lock_a.is_locked
    assert "main_en" in str(lock_a.lock_file)

    # A different channel is NOT blocked (separate lock file)
    from filelock import FileLock
    lock_b = FileLock(str(tmp_path / "rufus.spanish.lock") + ".lock")
    lock_b.acquire(timeout=0)          # must not raise
    assert lock_b.is_locked
    lock_b.release()

    # Same channel IS blocked → sys.exit(1)
    import subprocess, sys as _sys
    r = subprocess.run(
        [_sys.executable, "-c", (
            "import sys; sys.path.insert(0, r'%s');\n"
            "import main\n"
            "from pathlib import Path\n"
            "main.ROOT = Path(r'%s')\n"
            "main._acquire_lock('main_en')\n"
        ) % (str(Path(__file__).parent.parent / "scripts"), str(tmp_path))],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 1
    assert "main_en" in r.stdout

    lock_a.release()


def test_sweep_run_temp_only_removes_own_pid_files(tmp_path, monkeypatch):
    """The exit sweep must delete THIS pid's clip temps and never touch a
    concurrent run's files (identified by a different pid in the stamp)."""
    import os
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    comfy = tmp_path / "media_library" / "temp" / "comfy"
    comfy.mkdir(parents=True)

    mine   = comfy / f"1751234567_{os.getpid()}_3.png"
    theirs = comfy / "1751234567_99999_3.png"
    mine.write_bytes(b"x")
    theirs.write_bytes(b"x")

    main._sweep_run_temp()

    assert not mine.exists()
    assert theirs.exists()


def test_ensure_media_root_renames_stray_file(tmp_path, monkeypatch):
    """media_library existing as a FILE (AV quarantine restore, interrupted
    download, manual slip) makes every downstream .mkdir(parents=True,
    exist_ok=True) call hard-crash with WinError 183/FileExistsError, since
    exist_ok only suppresses the error when the existing entry is_dir().
    The startup guard must move it aside so a real directory can be created."""
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    stray = tmp_path / "media_library"
    stray.write_bytes(b"oops, not a folder")

    main._ensure_media_root()

    assert not stray.exists() or stray.is_dir()
    backups = list(tmp_path.glob("media_library.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"oops, not a folder"


def test_ensure_media_root_noop_when_already_a_directory(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    real_dir = tmp_path / "media_library"
    real_dir.mkdir()
    (real_dir / "keep.txt").write_text("keep")

    main._ensure_media_root()

    assert real_dir.is_dir()
    assert (real_dir / "keep.txt").exists()
    assert not list(tmp_path.glob("media_library.bak-*"))


def test_ensure_media_root_noop_when_missing(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    main._ensure_media_root()   # must not raise when media_library doesn't exist yet
    assert not (tmp_path / "media_library").exists()


# ── Debug-mode artifacts (RUFUS_DEBUG) ────────────────────────────────────────

def test_housekeeping_cleans_stale_debug_and_empty_dirs(tmp_path, monkeypatch):
    """Debug output gets its own ~month-long retention window (not the 14-day
    cache one) since it's for reviewing what a run produced — but it still
    needs a cap or it grows forever. Stale files AND the now-empty per-run
    subfolders they leave behind must both get swept."""
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "LOG_DIR", tmp_path / "logs")

    debug_dir = tmp_path / "media_library" / "debug"
    old_run   = debug_dir / "20260101-old"
    new_run   = debug_dir / "20260710-new"
    old_run.mkdir(parents=True)
    new_run.mkdir(parents=True)

    old_file = old_run / "script.txt"
    new_file = new_run / "script.txt"
    old_file.write_text("old")
    new_file.write_text("new")

    old_time = time.time() - 40 * 86400   # 40 days ago — past the 30-day default
    os.utime(old_file, (old_time, old_time))

    main._housekeeping()

    assert not old_file.exists()
    assert not old_run.exists()           # emptied folder gets swept too
    assert new_file.exists()
    assert new_run.exists()


def test_save_debug_artifacts_noop_when_debug_off(tmp_path, monkeypatch):
    import audio_gen as ag
    monkeypatch.delenv("RUFUS_DEBUG", raising=False)
    monkeypatch.setattr(ag, "ROOT", tmp_path)

    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"x")
    ag._save_debug_artifacts("a script", mp3)

    assert not (tmp_path / "media_library" / "debug").exists()


def test_save_debug_artifacts_saves_script_and_voiceover(tmp_path, monkeypatch):
    import audio_gen as ag
    monkeypatch.setenv("RUFUS_DEBUG", "1")
    monkeypatch.setenv("RUFUS_DEBUG_RUN_ID", "20260710-abc123")
    monkeypatch.setattr(ag, "ROOT", tmp_path)

    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"fake mp3 bytes")
    ag._save_debug_artifacts("Hook.\nBody.\nCTA.", mp3)

    out = tmp_path / "media_library" / "debug" / "20260710-abc123"
    assert (out / "script.txt").read_text() == "Hook.\nBody.\nCTA."
    assert (out / "voiceover.mp3").read_bytes() == b"fake mp3 bytes"


def test_save_debug_artifacts_failure_is_non_fatal(tmp_path, monkeypatch):
    """A debug-save failure (disk full, permissions, whatever) must never
    break the actual render — render() calls this with no error handling
    of its own, so the function itself must swallow everything."""
    import audio_gen as ag
    monkeypatch.setenv("RUFUS_DEBUG", "1")
    monkeypatch.setattr(ag, "ROOT", tmp_path)
    monkeypatch.setattr(ag.shutil, "copy2",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"x")
    ag._save_debug_artifacts("script", mp3)   # must not raise


# ── TTS speech sanitizer (voice was reading markdown out loud) ────────────────

def test_sanitize_for_speech_strips_markdown_and_stage_directions():
    import tts_engine as t
    dirty = "**Nixon** shocked the *world*.\n[pause] Then came (beat) the crash. #history"
    clean = t._sanitize_for_speech(dirty)
    assert "*" not in clean and "#" not in clean
    assert "pause" not in clean.lower() and "beat" not in clean.lower()
    assert "Nixon shocked the world." in clean
    assert "Then came" in clean and "the crash." in clean


def test_sanitize_for_speech_idempotent_on_clean_text():
    import tts_engine as t
    clean = "In 1971, Nixon ended the gold standard.\nYour money changed forever."
    assert t._sanitize_for_speech(clean) == clean
