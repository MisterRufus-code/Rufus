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


# ── Cross-run image freshness ─────────────────────────────────────────────────

def test_image_prompt_history_roundtrip_and_channel_scoping(tmp_path, monkeypatch):
    """Prompts remembered for one channel must come back for that channel only —
    two channels must never censor each other's visual ideas."""
    import main
    monkeypatch.setattr(main, "RECENT_PROMPTS_FILE", tmp_path / "recent.json")

    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    main._remember_image_prompts(["a coin on a desk", "a 1948 street market"])
    monkeypatch.setenv("RUFUS_CHANNEL", "other_ch")
    main._remember_image_prompts(["a rocket launch"])

    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    got = main._recent_image_prompts()
    assert "a coin on a desk" in got
    assert "a rocket launch" not in got


def test_image_prompt_history_is_capped(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "RECENT_PROMPTS_FILE", tmp_path / "recent.json")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")

    for i in range(30):   # far past the 24-run cap
        main._remember_image_prompts([f"prompt {i}"])

    data = json.loads((tmp_path / "recent.json").read_text())
    assert len(data["runs"]) == 24
    # oldest runs dropped, newest kept
    assert data["runs"][-1]["prompts"] == ["prompt 29"]


def test_freshness_block_empty_on_first_run(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "RECENT_PROMPTS_FILE", tmp_path / "missing.json")
    assert main._freshness_block() == ""


def test_freshness_block_lists_recent_prompts(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "RECENT_PROMPTS_FILE", tmp_path / "recent.json")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    main._remember_image_prompts(["a wheelbarrow of banknotes on a cobbled street"])

    block = main._freshness_block()
    assert "DO NOT REPEAT" in block
    assert "wheelbarrow of banknotes" in block


def test_comfy_prior_hashes_roundtrip_and_cap(tmp_path, monkeypatch):
    """Accepted-image hashes must persist across runs (capped) so the dup
    check can regen a new image that looks like one from a previous video."""
    import comfy_client
    monkeypatch.setattr(comfy_client, "FRESH_HASH_FILE", tmp_path / "hashes.json")

    comfy_client._save_hashes(list(range(200)))         # over the 120 cap
    got = comfy_client._load_prior_hashes()
    assert len(got) == comfy_client.FRESH_HASH_CAP
    assert got[-1] == 199                                # newest kept, oldest dropped


def test_comfy_prior_hashes_missing_file_is_empty(tmp_path, monkeypatch):
    import comfy_client
    monkeypatch.setattr(comfy_client, "FRESH_HASH_FILE", tmp_path / "nope.json")
    assert comfy_client._load_prior_hashes() == []


def test_fresh_images_env_kill_switch(monkeypatch):
    import comfy_client
    monkeypatch.delenv("RUFUS_FRESH_IMAGES", raising=False)
    assert comfy_client._fresh_images_enabled() is True
    monkeypatch.setenv("RUFUS_FRESH_IMAGES", "0")
    assert comfy_client._fresh_images_enabled() is False


# ── Ken Burns zoom subtlety (stills-only mode) ────────────────────────────────

def test_kenburns_default_zoom_is_subtle(tmp_path, monkeypatch):
    """Default zoom must be a small ('tiny') range, not the old 20% swing —
    a heavy pan/zoom reads as fake on top of already-strong FLUX stills when
    running with Wan/SVD disabled (RUFUS_WAN=0, RUFUS_IMG2VID=0)."""
    import importlib
    import sd_client
    monkeypatch.delenv("RUFUS_KENBURNS_ZOOM", raising=False)
    importlib.reload(sd_client)
    assert sd_client.KENBURNS_ZOOM_RANGE == pytest.approx(0.06)
    assert sd_client.KENBURNS_ZOOM_RANGE < 0.20   # meaningfully subtler than the old default
    importlib.reload(sd_client)  # restore module state for later tests


def test_kenburns_zoom_env_override(monkeypatch):
    import importlib
    import sd_client
    monkeypatch.setenv("RUFUS_KENBURNS_ZOOM", "0.02")
    importlib.reload(sd_client)
    try:
        assert sd_client.KENBURNS_ZOOM_RANGE == pytest.approx(0.02)
    finally:
        monkeypatch.delenv("RUFUS_KENBURNS_ZOOM", raising=False)
        importlib.reload(sd_client)


def test_animate_to_clip_uses_subtle_zoom_in_ffmpeg_filter(tmp_path, monkeypatch):
    """The actual ffmpeg zoompan filter passed to subprocess must cap at the
    configured zoom (not the old hardcoded 1.20)."""
    import importlib
    import sd_client
    monkeypatch.delenv("RUFUS_KENBURNS_ZOOM", raising=False)
    importlib.reload(sd_client)

    img = tmp_path / "still.png"
    from PIL import Image
    Image.new("RGB", (1080, 1920), color=(10, 20, 30)).save(img)
    out = tmp_path / "clip.mp4"

    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out.write_bytes(b"x" * 100_000)
        return types.SimpleNamespace(returncode=0, stderr="")

    with patch("sd_client.subprocess.run", side_effect=fake_run):
        sd_client._animate_to_clip(img, out, duration=2.0, idx=0)

    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert "1.0600" in vf
    assert "1.20" not in vf
    importlib.reload(sd_client)


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

def test_housekeeping_never_deletes_debug(tmp_path, monkeypatch):
    """media_library/debug/ is the permanent quality-review record (every
    run's script/voiceover/keyframes) — it must survive housekeeping
    regardless of age, unlike the rolling cache/temp/log cleanup."""
    import main

    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "LOG_DIR", tmp_path / "logs")

    debug_dir = tmp_path / "media_library" / "debug"
    old_run   = debug_dir / "20260101-old"
    old_run.mkdir(parents=True)

    old_file = old_run / "script.txt"
    old_file.write_text("old")

    old_time = time.time() - 400 * 86400   # over a year old
    os.utime(old_file, (old_time, old_time))

    main._housekeeping()

    assert old_file.exists()
    assert old_run.exists()


def test_save_debug_artifacts_saves_even_when_debug_off(tmp_path, monkeypatch):
    """Every run's script/voiceover gets logged now, not just RUFUS_DEBUG=1
    runs — the quality-review workflow needs every run's artifacts, not an
    opt-in subset a reviewer has to remember to enable."""
    import audio_gen as ag
    monkeypatch.delenv("RUFUS_DEBUG", raising=False)
    monkeypatch.setattr(ag, "ROOT", tmp_path)

    mp3 = tmp_path / "voice.mp3"
    mp3.write_bytes(b"x")
    ag._save_debug_artifacts("a script", mp3)

    assert (tmp_path / "media_library" / "debug").exists()


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


# ── TTS auto-selection (ElevenLabs preferred when key configured) ─────────────

def test_tts_auto_prefers_elevenlabs_when_key_present(monkeypatch):
    """A configured ElevenLabs key means the user opted into the paid, best
    voice — auto-selection must use it, not silently pick Kokoro/Edge."""
    import tts_engine
    monkeypatch.delenv("RUFUS_TTS", raising=False)
    monkeypatch.setattr(tts_engine, "_eleven_key", lambda: "sk-real-key")
    assert tts_engine._backend() == "elevenlabs"


def test_tts_auto_without_key_falls_through(monkeypatch):
    import tts_engine
    monkeypatch.delenv("RUFUS_TTS", raising=False)
    monkeypatch.setattr(tts_engine, "_eleven_key", lambda: "")
    assert tts_engine._backend() in ("kokoro", "edge")


def test_tts_explicit_env_still_wins_over_key(monkeypatch):
    import tts_engine
    monkeypatch.setenv("RUFUS_TTS", "edge")
    monkeypatch.setattr(tts_engine, "_eleven_key", lambda: "sk-real-key")
    assert tts_engine._backend() == "edge"


# ── Beat splitting & prompt echo (Greenback-run bugs) ─────────────────────────

def test_split_beats_does_not_break_on_abbreviations():
    """'...saw the U.S. government issue...' was split at 'U.S.' into two
    broken beats (seen live) — abbreviation periods must not end a beat."""
    import main
    script = ("During the Civil War, 1862 saw the U.S. government issue "
              "United States Notes. People clutched these notes tightly. "
              "Mr. Chase signed every one of them.")
    beats = main._split_beats(script)
    assert len(beats) == 3
    assert "U.S. government issue" in beats[0]
    assert beats[2].startswith("Mr. Chase")


def test_split_beats_normal_sentences_unchanged():
    import main
    script = "First sentence here now. Second sentence follows it. Third one closes."
    assert len(main._split_beats(script)) == 3


def test_strip_beat_echo_removes_full_echo():
    """GPT prefixed prompts with the beat's narration verbatim (seen live) —
    the deterministic guard must cut it and keep the visual description."""
    import main
    beat = "During the Civil War, 1862 saw the U.S. government issue Greenbacks."
    line = ("During the Civil War, 1862 saw the U.S. government issue Greenbacks. "
            "A medium portrait of soldiers being paid in fresh green notes.")
    out = main._strip_beat_echo(line, beat)
    assert out.startswith("A medium portrait")
    assert "Civil War, 1862 saw" not in out


def test_strip_beat_echo_leaves_clean_prompt_alone():
    import main
    beat = "During the Civil War, 1862 saw the U.S. government issue Greenbacks."
    line = ("A medium portrait of Civil War soldiers being paid in fresh green "
            "notes, dusk light through the tent canvas.")
    assert main._strip_beat_echo(line, beat) == line


def test_strip_beat_echo_never_leaves_a_stub():
    """If stripping would leave nothing usable, keep the original line."""
    import main
    beat = "People clutched these notes tightly in the market."
    line = "People clutched these notes tightly in the market. Close-up."
    assert main._strip_beat_echo(line, beat) == line


# ── Audit fixes: lock re-acquire (--rotate), unified OAuth scopes ─────────────

def test_lock_release_allows_reacquire_in_same_process(tmp_path, monkeypatch):
    """Audit C1: filelock's FileLock is NOT reentrant across instances — in
    --rotate, run() #2 re-acquired the same per-channel lock file and died,
    so rotate could only ever produce its first video. run() now releases on
    normal completion; this verifies the release actually unblocks a fresh
    acquire in the same process."""
    import main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "_INSTANCE_LOCK", None, raising=False)

    main._acquire_lock("chan_x")      # run #1 acquires
    main._release_lock()              # run #1 completes normally
    main._acquire_lock("chan_x")      # run #2 must NOT sys.exit
    main._release_lock()


def test_analytics_and_uploader_share_one_scope_list():
    """Audit C3: analytics_fetcher declared its own scope list against the
    SAME token file the uploader writes — the mismatch forced an interactive
    OAuth flow that hung unattended scheduled runs forever (analytics runs
    BEFORE main.py in run_scheduled.bat)."""
    import analytics_fetcher
    import youtube_uploader
    assert analytics_fetcher.SCOPES is youtube_uploader.SCOPES
    assert "https://www.googleapis.com/auth/yt-analytics.readonly" in youtube_uploader.SCOPES


# ── Audit H5: output/ housekeeping (never grew before) ────────────────────────

def test_housekeep_output_deletes_old_approved_but_protects_pending(tmp_path, monkeypatch):
    """output/ was never cleaned — at 5/day it grew forever. Now old approved/
    rejected videos are swept, but a PENDING video (awaiting review) is never
    deleted regardless of age — the reviewer still needs it."""
    import main, db_manager, time as _t
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    (tmp_path / "output").mkdir()

    old = _t.time() - 40 * 86400
    def mk(name, status):
        f = tmp_path / "output" / name
        f.write_bytes(b"video")
        (tmp_path / "output" / name.replace(".mp4", ".thumb.jpg")).write_bytes(b"t")
        os.utime(f, (old, old))
        db_manager.save_video(niche="finance", script_hook="h", scene_desc="s",
                              video_file=str(f), score=9, upload_status=status)
        return f

    approved = mk("short_1.mp4", "approved")
    pending  = mk("short_2.mp4", "pending")

    removed = main._housekeep_output(max_output_days=14)
    assert removed >= 1
    assert not approved.exists()                 # old + approved → swept
    assert not approved.with_suffix(".thumb.jpg").exists()
    assert pending.exists()                      # pending → protected even though old


def test_housekeep_output_keeps_recent_videos(tmp_path, monkeypatch):
    import main, db_manager
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    (tmp_path / "output").mkdir()
    f = tmp_path / "output" / "short_new.mp4"
    f.write_bytes(b"video")                       # fresh mtime
    db_manager.save_video(niche="finance", script_hook="h", scene_desc="s",
                          video_file=str(f), score=9, upload_status="approved")
    main._housekeep_output(max_output_days=14)
    assert f.exists()                             # within window → kept


def test_housekeep_output_skips_when_db_unavailable(tmp_path, monkeypatch, capsys):
    import main, db_manager
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "short_x.mp4").write_bytes(b"v")
    monkeypatch.setattr(db_manager, "_conn",
                        lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    assert main._housekeep_output(max_output_days=1) == 0   # fail-safe: no deletions


# ── Audit H4: scheduled runs wait for the lock instead of dropping the slot ────

def test_acquire_lock_waits_when_wait_seconds_given(tmp_path, monkeypatch):
    """A scheduled run must WAIT for a held lock (up to wait_seconds), not
    sys.exit(1) instantly — otherwise overlapping 5/day triggers silently
    lose slots. Verifies the timeout is actually passed through to filelock."""
    import main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "_INSTANCE_LOCK", None, raising=False)

    captured = {}
    class FakeLock:
        def __init__(self, path): captured["path"] = path
        def acquire(self, timeout=0): captured["timeout"] = timeout
        def release(self): pass
        @property
        def is_locked(self): return True
    monkeypatch.setattr(main, "FileLock", FakeLock)

    main._acquire_lock("chan_x", wait_seconds=1800)
    assert captured["timeout"] == 1800


def test_acquire_lock_default_is_nonblocking(tmp_path, monkeypatch):
    """A manual (non-scheduled) run keeps the old fail-fast behavior."""
    import main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "_INSTANCE_LOCK", None, raising=False)
    captured = {}
    class FakeLock:
        def __init__(self, path): pass
        def acquire(self, timeout=0): captured["timeout"] = timeout
        def release(self): pass
        @property
        def is_locked(self): return True
    monkeypatch.setattr(main, "FileLock", FakeLock)
    main._acquire_lock("chan_x")
    assert captured["timeout"] == 0
