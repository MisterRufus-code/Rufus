"""Tests for niche config loader."""

from __future__ import annotations

import pytest


class TestGetNicheConfig:
    def test_loads_finance_config(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("finance")
        assert cfg.name == "finance"
        assert cfg.model in ("mistral", "llama3.1")
        assert len(cfg.upload_schedule) > 0
        assert len(cfg.default_tags) > 0

    def test_loads_tech_config(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("tech")
        assert cfg.name == "tech"
        assert cfg.entropy_min_score > 0

    def test_loads_health_config(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("health")
        assert cfg.name == "health"
        assert cfg.shorts_target_duration <= 60

    def test_unknown_niche_falls_back_to_general(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("xyzzy_unknown_niche_9999")
        assert cfg.entropy_min_score >= 0
        assert cfg.model != ""

    def test_general_config_loads(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("general")
        assert cfg.name == "general"

    def test_entropy_thresholds_in_valid_range(self):
        from src.niche_config import get_niche_config
        for niche in ("finance", "tech", "health", "general"):
            cfg = get_niche_config(niche)
            assert 0 < cfg.entropy_min_score < 1
            assert 0 < cfg.entropy_scene_repeat_threshold < 1

    def test_hook_weights_are_floats(self):
        from src.niche_config import get_niche_config
        cfg = get_niche_config("finance")
        for k, v in cfg.hook_weights.items():
            assert isinstance(v, float), f"{k} weight should be float"

    def test_shorts_duration_at_most_60(self):
        from src.niche_config import get_niche_config
        for niche in ("finance", "tech", "health", "general"):
            cfg = get_niche_config(niche)
            assert cfg.shorts_target_duration <= 60


class TestListNiches:
    def test_returns_list_including_known_niches(self):
        from src.niche_config import list_niches
        niches = list_niches()
        assert isinstance(niches, list)
        assert "finance" in niches
        assert "tech" in niches
        assert "health" in niches
        assert "general" in niches
