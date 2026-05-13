"""Tests for multi-channel manager."""

from __future__ import annotations

import pytest


class TestGetChannel:
    def test_unknown_channel_returns_defaults(self):
        from src.channels.manager import get_channel
        ch = get_channel("unknown_channel_xyz")
        assert ch.channel_id == "unknown_channel_xyz"
        assert ch.niche == "general"
        assert ch.model == "mistral"
        assert ch.tts_voice == "af_heart"

    def test_token_file_auto_resolved(self, tmp_path, monkeypatch):
        from src.channels import manager
        ch_dir = tmp_path / "channels"
        ch_dir.mkdir()
        (ch_dir / "my_finance_channel.yaml").write_text("name: my_finance_channel\nniche: finance\n")
        monkeypatch.setattr(manager, "_CHANNELS_DIR", ch_dir)
        ch = manager.get_channel("my_finance_channel")
        assert "my_finance_channel" in ch.token_file

    def test_secrets_file_auto_resolved(self, tmp_path, monkeypatch):
        from src.channels import manager
        ch_dir = tmp_path / "channels"
        ch_dir.mkdir()
        (ch_dir / "my_tech_channel.yaml").write_text("name: my_tech_channel\nniche: tech\n")
        monkeypatch.setattr(manager, "_CHANNELS_DIR", ch_dir)
        ch = manager.get_channel("my_tech_channel")
        assert "my_tech_channel" in ch.secrets_file

    def test_channel_config_dataclass_fields(self):
        from src.channels.manager import ChannelConfig
        cfg = ChannelConfig(channel_id="test")
        assert cfg.upload is False
        assert cfg.privacy == "private"
        assert cfg.auto_schedule is True


class TestListChannels:
    def test_returns_list(self):
        from src.channels.manager import list_channels
        result = list_channels()
        assert isinstance(result, list)

    def test_no_crash_when_dir_missing(self, tmp_path, monkeypatch):
        import src.channels.manager as mgr
        monkeypatch.setattr(mgr, "_CHANNELS_DIR", tmp_path / "nonexistent")
        result = mgr.list_channels()
        assert result == []
