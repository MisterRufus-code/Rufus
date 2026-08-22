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


def test_the_ntfy_topic_is_never_echoed_into_a_log(tmp_path, monkeypatch, capsys):
    """ntfy has no accounts — the topic string IS the authentication, and
    anyone holding it can read every alert and push fake ones. It was printed
    in full on every run, into the same logs that get pasted into a chat when
    something goes wrong. WEBHOOK and TOKEN were masked because their names
    say "secret"; this one is a secret because of what it does."""
    f = tmp_path / "dashboard_settings.json"
    f.write_text(json.dumps({"RUFUS_NTFY_TOPIC": "rufus-j10d4dmq5fpm1m",
                             "RUFUS_STYLE": "stickman"}), encoding="utf-8")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", f)
    env: dict = {}
    settings_store.apply(env)
    out = capsys.readouterr().out
    assert "rufus-j10d4dmq5fpm1m" not in out
    assert "1 secret" in out
    assert "RUFUS_STYLE=stickman" in out
    # Masked in the LOG, still applied to the run.
    assert env["RUFUS_NTFY_TOPIC"] == "rufus-j10d4dmq5fpm1m"


# ── every standalone entry point has to read the file ────────────────────────

def test_the_entry_points_that_notify_read_the_saved_settings():
    """main.py applied the settings and nothing else did, so a topic saved
    from the dashboard was live for every real run and absent from three
    processes that send notifications:

      notify.py    the one command whose job is to answer "is this
                   configured" — it printed "No backend configured" at
                   somebody who had just configured it.
      watchdog.py  started by a scheduled task with a bare environment. Its
                   "the dashboard died" alert had nowhere to go, which is the
                   exact failure the watchdog exists to prevent, arriving by
                   another route.
      analytics_fetcher.py  same, for the daily digest.

    A settings page obeyed by some processes and not others teaches the owner
    to distrust the form — this repo's own words, written when the gap was
    between launchers rather than between entry points.
    """
    root = Path(settings_store.__file__).parent
    for name in ("notify.py", "watchdog.py", "analytics_fetcher.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "settings_store" in text, name
        assert "settings_store.apply()" in text, name


def test_the_entry_points_that_RENDER_read_the_saved_settings():
    """The two tools whose entire job is "show me what my style edit did"
    were free to answer about a different style.

      style_probe.py     run from a fresh PowerShell it rendered six probes on
                         the BUILT-IN FALLBACK and wrote style "(default)"
                         into probe.json, under a heading promising "the
                         workflow the channel actually ships". Six pictures
                         were read as evidence about stickman. They were not
                         evidence about stickman.
      workflow_bench.py  same gap, worse consequence: a bench exists to vary
                         ONE thing between columns, and the one thing that
                         must not vary is the style.

    A probe that renders the wrong style is worse than no probe, because you
    act on it. Both are asserted here, next to the notifying entry points, so
    that "which processes read the settings file" is one list in one place —
    the answer is meant to be "all of them", and a future entry point that
    quietly skips it fails this test rather than a gallery.
    """
    root = Path(settings_store.__file__).parent

    probe = (root / "style_probe.py").read_text(encoding="utf-8")
    entry = probe.split("def main(")[1]
    assert "settings_store.apply()" in entry
    # Before argparse, therefore before --style is read and long before
    # anything asks comfy_client what RUFUS_STYLE says.
    assert entry.index("settings_store.apply()") < entry.index("argparse.ArgumentParser")

    bench = (root / "workflow_bench.py").read_text(encoding="utf-8")
    entry = bench.split('if __name__ == "__main__":')[1]
    assert "settings_store.apply()" in entry
    assert entry.index("settings_store.apply()") < entry.index("run()")


def test_a_powershell_written_file_still_loads(tmp_path, monkeypatch):
    """Windows PowerShell 5.1's `Set-Content -Encoding utf8` writes a BOM, and
    editing this file from a PowerShell prompt is a documented way to set it.
    Read as plain utf-8 those three bytes make json.loads throw, the loader
    returned {} without a word, and EVERY saved setting vanished — the style,
    the renderer, the dashboard URL, the ntfy topic — while the run carried on
    using built-in defaults as though the file had never existed."""
    f = tmp_path / "dashboard_settings.json"
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps({"RUFUS_STYLE": "stickman"}).encode())
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", f)
    assert settings_store.load() == {"RUFUS_STYLE": "stickman"}


def test_an_unparseable_file_is_loud(tmp_path, monkeypatch, capsys):
    """A missing file is the ordinary case and stays quiet. A file that EXISTS
    and will not parse is somebody's configuration being ignored."""
    f = tmp_path / "dashboard_settings.json"
    f.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", f)
    assert settings_store.load() == {}
    out = capsys.readouterr().out
    assert "not valid JSON" in out
    assert "EVERY saved setting is being ignored" in out


def test_a_missing_file_says_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "nope.json")
    assert settings_store.load() == {}
    assert capsys.readouterr().out == ""
