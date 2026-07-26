"""Tests for paths.py — the single source of truth for WHERE Rufus writes.

The debug/media/log roots were hardcoded to the repo folder in six modules, so
a machine whose repo sits on a small SSD had no way to send the bulky output to
a roomier drive. These lock in that every root is relocatable and that the
modules can't drift apart again.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import paths


def test_defaults_live_under_the_repo(monkeypatch):
    for v in ("RUFUS_MEDIA_DIR", "RUFUS_DEBUG_DIR", "RUFUS_OUTPUT_DIR", "RUFUS_LOG_DIR"):
        monkeypatch.delenv(v, raising=False)
    assert paths.media_root() == paths.ROOT / "media_library"
    assert paths.debug_root() == paths.ROOT / "media_library" / "debug"
    assert paths.output_dir() == paths.ROOT / "media_library" / "output"
    assert paths.log_dir() == paths.ROOT / "logs"


def test_media_dir_moves_the_whole_tree(monkeypatch, tmp_path):
    """One var relocates everything media-related — the common case for
    'repo on the SSD, bulky output on the HDD'."""
    for v in ("RUFUS_DEBUG_DIR", "RUFUS_OUTPUT_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("RUFUS_MEDIA_DIR", str(tmp_path / "hdd" / "media"))
    assert paths.media_root() == tmp_path / "hdd" / "media"
    assert paths.debug_root() == tmp_path / "hdd" / "media" / "debug"
    assert paths.output_dir() == tmp_path / "hdd" / "media" / "output"


def test_specific_vars_win_over_media_dir(monkeypatch, tmp_path):
    """Splitting across drives must be possible — the narrower var wins."""
    monkeypatch.setenv("RUFUS_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(tmp_path / "elsewhere" / "debug"))
    assert paths.debug_root() == tmp_path / "elsewhere" / "debug"
    assert paths.output_dir() == tmp_path / "media" / "output"   # still follows media


def test_blank_env_falls_back_to_default(monkeypatch):
    """An empty/whitespace var must not resolve to Path('') (the cwd)."""
    monkeypatch.setenv("RUFUS_LOG_DIR", "   ")
    assert paths.log_dir() == paths.ROOT / "logs"


def test_log_dir_is_independent_of_media_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("RUFUS_LOG_DIR", raising=False)
    assert paths.log_dir() == paths.ROOT / "logs"


# ── run_report.md: one file per run with everything a reviewer needs ──────────

def test_run_report_collects_script_and_prompts(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(tmp_path / "debug"))
    paths.write_run_report("run-1", script="Hook.\nBody.\nCTA.",
                           meta={"score": "9/10", "niche": "money_history"})
    out = paths.write_run_report("run-1", prompts=["a coin, macro", "a vault door"])
    assert out == tmp_path / "debug" / "run-1" / "run_report.md"
    text = out.read_text(encoding="utf-8")
    assert "Hook.\nBody.\nCTA." in text          # the script survives...
    assert "a coin, macro" in text               # ...alongside the prompts
    assert "a vault door" in text
    assert "9/10" in text and "money_history" in text
    assert text.count("# Rufus run") == 1        # header written once, not per append


def test_run_report_never_raises_on_bad_path(monkeypatch, tmp_path):
    """A reporting failure must never break a render."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(blocker))
    assert paths.write_run_report("run-x", script="s") is None


# ── Motion record: the raw material for "why did clip 4 look wrong" ──────────

def test_run_report_logs_motion_prompt_and_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(tmp_path / "debug"))
    out = paths.write_run_report("r", motion=[
        {"beat": 1, "engine": "hunyuan", "ok": True, "seconds": 512.3,
         "motion_prompt": "a vault door, camera drifts slowly",
         "model": "hunyuan_480p_step_distilled.safetensors",
         "steps": 12, "cfg": 1, "shift": 5},
    ])
    t = out.read_text(encoding="utf-8")
    assert "a vault door, camera drifts slowly" in t     # the real prompt used
    assert "512.3s" in t                                  # timing, to spot drift
    assert "hunyuan_480p_step_distilled.safetensors" in t
    assert "**steps**: 12" in t and "**shift**: 5" in t    # reproducible settings


def test_run_report_records_engine_fallthrough(monkeypatch, tmp_path):
    """A clip that fell to Ken Burns must say so, with the failed engine and
    the time it burned — that's the signal an engine is silently broken."""
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(tmp_path / "debug"))
    out = paths.write_run_report("r", motion=[
        {"beat": 2, "engine": "hunyuan", "ok": False, "seconds": 1800.0},
        {"beat": 2, "engine": "kenburns", "ok": True,
         "note": "all motion engines declined/failed"},
    ])
    t = out.read_text(encoding="utf-8")
    assert "hunyuan: failed (1800.0s)" in t
    assert "kenburns: ok" in t


def test_run_report_settings_block_printed_once(monkeypatch, tmp_path):
    """Settings are per-run, not per-beat — repeating them 10x would bury the
    per-beat prompts that actually differ."""
    monkeypatch.setenv("RUFUS_DEBUG_DIR", str(tmp_path / "debug"))
    rec = lambda b: {"beat": b, "engine": "hunyuan", "ok": True,
                     "seconds": 1.0, "steps": 12}
    out = paths.write_run_report("r", motion=[rec(1), rec(2), rec(3)])
    assert out.read_text(encoding="utf-8").count("Motion engine settings") == 1
