"""The settings the dashboard saves, read by every launch path.

THE GAP: config/dashboard_settings.json was written by the settings page and
read by exactly one caller — the dashboard's own _launch_run. So the owner
could set the style, the beat count, the Discord webhook and the sound levels
in a form built for the purpose, then start a video with run.bat, the way every
video on this channel has actually been started, and none of it applied. The
dashboard's help text admitted it in a parenthesis, which is a confession
rather than a fix.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import settings_store  # noqa: E402


@pytest.fixture
def saved(tmp_path, monkeypatch):
    f = tmp_path / "dashboard_settings.json"
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", f)

    def _write(**values):
        f.write_text(json.dumps(values), encoding="utf-8")
    return _write


def test_saved_settings_fill_an_empty_environment(saved):
    saved(RUFUS_STYLE="stickman", SD_CLIPS="18")
    env = {}
    assert sorted(settings_store.apply(env, announce=False)) == ["RUFUS_STYLE", "SD_CLIPS"]
    assert env == {"RUFUS_STYLE": "stickman", "SD_CLIPS": "18"}


def test_the_environment_wins_for_this_run(saved):
    """What you type now beats what you saved last week — otherwise a one-off
    experiment cannot be run without editing the form first."""
    saved(RUFUS_STYLE="stickman")
    env = {"RUFUS_STYLE": "ink_woodcut"}
    assert settings_store.apply(env, announce=False) == []
    assert env["RUFUS_STYLE"] == "ink_woodcut"


def test_an_empty_value_is_not_a_setting(saved):
    """"(default)" in the form means don't override, not set to empty."""
    saved(RUFUS_STYLE="", SD_CLIPS="18")
    env = {}
    settings_store.apply(env, announce=False)
    assert "RUFUS_STYLE" not in env


def test_nothing_outside_this_pipeline_can_be_set(saved):
    """The file is written by a web form. A form that could set PATH would be
    a form that could run anything."""
    saved(PATH="/evil", LD_PRELOAD="/evil.so", RUFUS_STYLE="stickman")
    env = {}
    settings_store.apply(env, announce=False)
    assert env == {"RUFUS_STYLE": "stickman"}


def test_a_missing_or_broken_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "nope.json")
    assert settings_store.load() == {}
    assert settings_store.apply({}, announce=False) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", bad)
    assert settings_store.load() == {}


def test_a_list_instead_of_an_object_is_ignored(tmp_path, monkeypatch):
    f = tmp_path / "s.json"
    f.write_text('["RUFUS_STYLE"]', encoding="utf-8")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", f)
    assert settings_store.load() == {}


def test_it_says_what_it_applied(saved, capsys):
    """A settings file that quietly changed a run would be worse than no
    settings file — a surprising run must have its cause in the log."""
    saved(RUFUS_STYLE="stickman")
    settings_store.apply({})
    assert "RUFUS_STYLE=stickman" in capsys.readouterr().out


def test_a_secret_is_counted_not_printed(saved, capsys):
    saved(RUFUS_DISCORD_WEBHOOK="https://discord/api/webhooks/secret")
    settings_store.apply({})
    out = capsys.readouterr().out
    assert "secret" in out
    assert "discord/api/webhooks" not in out


def test_it_says_what_the_environment_overrode(saved, capsys):
    saved(RUFUS_STYLE="stickman")
    settings_store.apply({"RUFUS_STYLE": "ink_woodcut"})
    assert "already sets RUFUS_STYLE" in capsys.readouterr().out


def test_every_run_applies_them_before_anything_reads_them():
    """Applied in main's entry point rather than in one launcher, because
    there are five ways a run starts and a settings page obeyed by some of
    them teaches the owner to trust a form that is sometimes ignored."""
    src = (Path(settings_store.__file__).parent / "main.py").read_text(encoding="utf-8")
    entry = src.split('if __name__ == "__main__":')[1]
    assert "settings_store.apply()" in entry
    assert entry.index("settings_store.apply()") < entry.index("argparse.ArgumentParser")


def test_the_dashboard_no_longer_claims_only_its_own_runs_obey():
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert "needs run_scheduled.bat pointed at the same file" not in src
    assert "every way of starting a run obeys them" in " ".join(src.split())
