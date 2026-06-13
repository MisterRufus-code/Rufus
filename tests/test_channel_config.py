"""Tests for the channel-in-a-box config layer (channel_config.py).

Covers the legacy shim (no channels.json → single-channel behavior unchanged),
niche-override merge precedence, and per-channel path resolution.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import channel_config as cc


# ── Legacy shim ────────────────────────────────────────────────────────────────

def test_legacy_shim_when_no_channels_file(monkeypatch, tmp_path):
    """No channels.json → synthesized 'main_en' using the original paths."""
    monkeypatch.setattr(cc, "CHANNELS_FILE", tmp_path / "channels.json")  # absent
    ch = cc.load_channel()
    assert ch.id == "main_en"
    assert ch.legacy is True
    # legacy paths == the original single-channel locations
    assert ch.token_path("youtube") == cc.CONFIG_DIR / "youtube_token.json"
    assert ch.learnings_path == cc.CONFIG_DIR / "learnings.json"
    assert ch.output_dir.name == "output"          # not nested under an id


def test_legacy_list_channels(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", tmp_path / "channels.json")
    assert cc.list_channels() == ["main_en"]


# ── channels.json present ────────────────────────────────────────────────────────

def _write_channels(tmp_path) -> Path:
    cfg = {
        "default_channel": "alpha",
        "channels": {
            "alpha": {
                "display_name": "Alpha",
                "language": "en",
                "voice": "en-US-AndrewMultilingualNeural",
                "niches": ["finance"],
                "schedule": ["finance"],
                "upload": {"min_score": 9, "privacy": "private"},
                "platforms": {"youtube": {"enabled": True}},
                "niche_overrides": {"finance": {"accent_color": "#000000"}},
            },
            "bravo": {
                "display_name": "Bravo ES",
                "language": "es",
                "platforms": {"youtube": {"enabled": False}},
            },
        },
    }
    p = tmp_path / "channels.json"
    p.write_text(json.dumps(cfg))
    return p


def test_load_named_channel_and_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    ch = cc.load_channel("bravo")
    assert ch.id == "bravo" and ch.legacy is False
    assert ch.language == "es"
    # non-legacy paths nest under the channel id
    assert ch.token_path("youtube") == cc.CONFIG_DIR / "channels" / "bravo" / "youtube_token.json"
    assert ch.output_dir.name == "bravo"
    assert ch.learnings_path == cc.CONFIG_DIR / "channels" / "bravo" / "learnings.json"


def test_default_channel_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    monkeypatch.delenv("RUFUS_CHANNEL", raising=False)
    assert cc.load_channel().id == "alpha"          # default_channel


def test_env_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    monkeypatch.setenv("RUFUS_CHANNEL", "bravo")
    assert cc.load_channel().id == "bravo"


def test_explicit_arg_beats_env(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    monkeypatch.setenv("RUFUS_CHANNEL", "bravo")
    assert cc.load_channel("alpha").id == "alpha"


def test_unknown_channel_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    with pytest.raises(KeyError):
        cc.load_channel("ghost")


def test_upload_defaults_merge(monkeypatch, tmp_path):
    """Channel upload block shallow-merges over DEFAULT_UPLOAD."""
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    ch = cc.load_channel("alpha")
    assert ch.upload["min_score"] == 9              # overridden
    assert ch.upload["privacy"] == "private"        # security default preserved
    assert ch.upload["peak_hours"] == [8, 12, 17, 20]  # from DEFAULT_UPLOAD


def test_platform_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    assert cc.load_channel("alpha").platform_enabled("youtube") is True
    assert cc.load_channel("bravo").platform_enabled("youtube") is False


# ── Niche-override merge precedence ──────────────────────────────────────────────

def test_niche_override_wins_over_base(monkeypatch, tmp_path):
    """channel.niche_overrides[niche] shallow-merges on top of niches.json base."""
    monkeypatch.setattr(cc, "CHANNELS_FILE", _write_channels(tmp_path))
    ch = cc.load_channel("alpha")
    merged = ch.niche_cfg("finance")
    assert merged["accent_color"] == "#000000"      # override won
    # untouched base keys survive (finance has a style_suffix in niches.json)
    assert "style_suffix" in merged
