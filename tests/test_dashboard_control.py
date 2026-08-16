"""The dashboard as the control surface, not a read-only window.

THE OWNER'S INSTRUCTION: the software is run from the dashboard, not from a
terminal. Running a video the way this channel now runs it meant seven `$env:`
lines in PowerShell before every run, and getting one wrong — a cmd `set` in a
PowerShell prompt, a stale RUFUS_BEAT_MOTION from an earlier experiment — is
invisible until the video comes out wrong.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import dashboard  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


# ── every knob is reachable ─────────────────────────────────────────────────

def test_the_settings_that_shaped_recent_runs_are_all_present():
    """Each of these was set by hand in PowerShell for a real run."""
    keys = {k for k, _l, _kind, _h in dashboard.SETTINGS_SCHEMA}
    for needed in ("RUFUS_STYLE", "RUFUS_STILLS_ONLY", "RUFUS_BEAT_MOTION",
                   "RUFUS_FRAMES_PER_BEAT", "SD_CLIPS", "RUFUS_INSERTS",
                   "RUFUS_INSERT_MODE", "RUFUS_DISCORD_WEBHOOK",
                   "RUFUS_BUBBLE_GAIN", "RUFUS_TTS"):
        assert needed in keys, needed


def test_every_setting_belongs_to_exactly_one_group():
    flat = [k for _t, _b, rows in dashboard.SETTINGS_GROUPS for k, *_ in rows]
    assert len(flat) == len(set(flat)), "a setting is listed twice"
    assert len(flat) == len(dashboard.SETTINGS_SCHEMA)


def test_every_setting_explains_itself():
    """A form of thirty bare variable names is not a control surface."""
    for key, label, kind, help_text in dashboard.SETTINGS_SCHEMA:
        assert label and help_text, key
        assert len(help_text) > 25, key


# ── the fields render as something usable ───────────────────────────────────

def test_a_webhook_gets_a_text_box_not_a_dropdown():
    """The old page made every setting a <select>, so a URL or a beat count
    could not be entered at all."""
    field = dashboard._setting_field("RUFUS_DISCORD_WEBHOOK", "secret", "")
    assert "<input" in field and "password" in field

    number = dashboard._setting_field("SD_CLIPS", "number", "24")
    assert "<input" in number and 'value="24"' in number


def test_a_bool_still_offers_three_states():
    """"Don't override" is not the same as "off" — one leaves the pipeline
    default in place and the other forces it off."""
    field = dashboard._setting_field("RUFUS_SFX", "bool", "")
    assert field.count("<option") == 3
    assert "don&#x27;t override" in field or "don't override" in field


def test_a_stored_secret_is_shown_as_set(client):
    """A password box that renders empty for a stored value is how people
    paste the same webhook in twice and never know which one is live."""
    dashboard._save_settings({"RUFUS_DISCORD_WEBHOOK": "https://x/y"})
    page = client.get("/settings").get_data(as_text=True)
    assert "https://x/y" in page


# ── saving ──────────────────────────────────────────────────────────────────

def test_saving_writes_only_the_filled_fields(client):
    client.post("/settings/save", data={"RUFUS_STYLE": "stickman",
                                        "SD_CLIPS": "", "RUFUS_SFX": "1"})
    saved = json.loads(dashboard.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert saved == {"RUFUS_STYLE": "stickman", "RUFUS_SFX": "1"}


def test_a_number_that_is_not_a_number_is_refused_out_loud(client):
    """Left to reach the run, it becomes an env var whose reader falls back to
    its default silently — the owner sets a value, sees no error, and gets the
    old behaviour with nothing in the log to explain it."""
    r = client.post("/settings/save", data={"SD_CLIPS": "twenty-four"})
    assert "error=" in r.headers["Location"]
    saved = json.loads(dashboard.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert "SD_CLIPS" not in saved


def test_reset_clears_every_override(client):
    dashboard._save_settings({"RUFUS_STYLE": "stickman", "SD_CLIPS": "24"})
    client.get("/settings?reset=1")
    assert json.loads(dashboard.SETTINGS_FILE.read_text(encoding="utf-8")) == {}


def test_saved_settings_reach_the_run_as_environment(client):
    """The whole point: what is set here is what the run sees."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert "env.update(_load_settings())" in src


# ── logs ────────────────────────────────────────────────────────────────────

def test_the_logs_page_lists_both_naming_schemes():
    """rufus_YYYYMMDD.log is a run.bat run; dashboard_run_<epoch>.log is one
    started from this page. Reading one would hide half the runs."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert "dashboard_run_" in src and "*.log" in src


def test_a_log_read_cannot_escape_the_logs_directory():
    """The filename arrives from a query string."""
    assert dashboard._read_log("../config/keys.json") == ""
    assert dashboard._read_log("../../etc/passwd") == ""


def test_a_long_log_is_tailed_and_says_so():
    assert dashboard.LOG_TAIL_BYTES > 100_000
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert "showing the last" in src


def test_logs_are_in_the_nav():
    assert any(href == "/logs" for href, _label, _perm in dashboard.NAV_ITEMS)
