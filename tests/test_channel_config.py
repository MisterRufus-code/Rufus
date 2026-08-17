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


# ── when a video goes live ───────────────────────────────────────────────────

def test_the_default_is_public(monkeypatch):
    """CHANGED ON REQUEST, and it changes what becomes publicly visible.

    The old default was `private`, which was never really "keep it hidden": a
    private upload also carried a publishAt of the next peak hour, so YouTube
    published it anyway. What it actually did was make publishing depend on
    the timezone database resolving — and on Windows that is a pip package
    nobody had installed, so the schedule degraded to no publishAt at all and
    finished videos sat private forever.

    Nothing uploads on its own: main.py needs RUFUS_AUTO_UPLOAD=1 and a score
    over the bar, and the dashboard needs a human to press approve. Public is
    what happens after somebody has already decided to publish.
    """
    monkeypatch.delenv("RUFUS_PRIVACY", raising=False)
    assert cc.DEFAULT_UPLOAD["privacy"] == "public"
    assert cc.load_channel().upload["privacy"] == "public"


def test_the_dashboard_choice_wins(monkeypatch):
    monkeypatch.setenv("RUFUS_PRIVACY", "private")
    assert cc.load_channel().upload["privacy"] == "private"


@pytest.mark.parametrize("value", ["public", "private", "unlisted"])
def test_every_youtube_privacy_value_is_accepted(monkeypatch, value):
    monkeypatch.setenv("RUFUS_PRIVACY", value)
    assert cc.load_channel().upload["privacy"] == value


def test_a_nonsense_value_is_loud_and_ignored(monkeypatch, capsys):
    """Silently rendering the default from a typo'd setting is
    indistinguishable from the setting not working."""
    monkeypatch.setenv("RUFUS_PRIVACY", "publik")
    assert cc.load_channel().upload["privacy"] == "public"
    assert "is not one of" in capsys.readouterr().out


def test_the_override_reaches_a_channels_json_install(tmp_path, monkeypatch):
    """A box with channels.json must obey the button too, or it looks broken
    on exactly the installs that configured the most."""
    cfg = tmp_path / "channels.json"
    cfg.write_text(json.dumps({
        "default_channel": "c1",
        "channels": {"c1": {"display_name": "C1",
                            "upload": {"privacy": "unlisted"}}},
    }), encoding="utf-8")
    monkeypatch.setattr(cc, "CHANNELS_FILE", cfg)
    monkeypatch.delenv("RUFUS_PRIVACY", raising=False)
    assert cc.load_channel().upload["privacy"] == "unlisted"
    monkeypatch.setenv("RUFUS_PRIVACY", "public")
    assert cc.load_channel().upload["privacy"] == "public"
