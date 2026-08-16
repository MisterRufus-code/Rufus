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


# ── proving the notification works without a 25-minute run ──────────────────

def test_a_test_notification_can_be_sent_from_the_form(client, monkeypatch):
    """A webhook is exactly the kind of value that is wrong in a way nothing
    reveals until a run finishes: a trailing space, a channel link copied
    instead of the webhook, a webhook deleted months ago."""
    sent = {}

    import notify

    def _fake_send(title, body, *, url=None, priority="normal"):
        sent["title"] = title
        return True

    monkeypatch.setattr(notify, "configured", lambda: ["discord"])
    monkeypatch.setattr(notify, "send", _fake_send)
    monkeypatch.setattr(notify, "_dashboard_url", lambda: "")
    monkeypatch.setattr("importlib.reload", lambda m: m)

    r = client.post("/settings/test-notify",
                    data={"RUFUS_DISCORD_WEBHOOK": "https://discord/api/x"})
    assert "msg=" in r.headers["Location"]
    assert "test" in sent.get("title", "").lower()


def test_testing_with_nothing_configured_says_so(client, monkeypatch):
    import notify
    monkeypatch.setattr(notify, "configured", lambda: [])
    monkeypatch.setattr("importlib.reload", lambda m: m)
    r = client.post("/settings/test-notify", data={})
    assert "error=" in r.headers["Location"]


def test_the_test_uses_what_is_on_screen_not_only_what_is_saved(client):
    """The button sits inside the settings form, so someone who pastes a
    webhook and reaches for "test" before "save" tests the one they pasted."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split('def settings_test_notify')[1].split("@app.route")[0]
    assert "request.form.get(key" in block


def test_the_environment_is_restored_after_a_test(client, monkeypatch):
    """The test layers settings onto os.environ to match what a run sees. This
    Flask process outlives the test, so leaving them set would silently change
    every later launch."""
    import os
    monkeypatch.delenv("RUFUS_DISCORD_WEBHOOK", raising=False)
    import notify
    monkeypatch.setattr(notify, "configured", lambda: [])
    monkeypatch.setattr("importlib.reload", lambda m: m)
    client.post("/settings/test-notify",
                data={"RUFUS_DISCORD_WEBHOOK": "https://leak/me"})
    assert os.environ.get("RUFUS_DISCORD_WEBHOOK") is None


def test_the_dashboard_url_is_settable_so_alerts_can_deep_link():
    keys = {k for k, _l, _kind, _h in dashboard.SETTINGS_SCHEMA}
    assert "RUFUS_DASHBOARD_URL" in keys


# ── insights ────────────────────────────────────────────────────────────────

def test_insights_is_in_the_nav():
    assert any(href == "/insights" for href, _l, _p in dashboard.NAV_ITEMS)


def test_insights_says_what_to_do_when_nothing_is_measured_yet(client, monkeypatch):
    """An empty page that does not say how to fill it is a dead end."""
    import run_review
    monkeypatch.setattr(run_review, "patterns", lambda limit=30: {
        "runs_reviewed": 0, "recurring": [], "rows": []})
    page = client.get("/insights").get_data(as_text=True)
    assert "run_review.py --all" in page


def test_insights_shows_recurring_findings_before_single_runs(client, monkeypatch):
    """The whole reason to keep these across runs: four of six is a code
    change, one of six is a bad seed."""
    import run_review
    monkeypatch.setattr(run_review, "patterns", lambda limit=30: {
        "runs_reviewed": 6,
        "recurring": [{"id": "setting_clause_everywhere", "runs": 4, "share": 0.67}],
        "rows": [{"run_id": "20260816-a", "beats": 24,
                  "clauses": {"thread_share": 0.33, "setting_share": 0.54},
                  "dominant_subject": {"word": "tonic", "share": 0.25},
                  "cuts": {"longest_hold_s": 6.4},
                  "findings": [{"severity": "high", "text": "half the shots"}]}],
    })
    page = client.get("/insights").get_data(as_text=True)
    assert page.index("What keeps happening") < page.index("Run by run")
    assert "setting_clause_everywhere" in page
    assert "4 of 6 runs" in page
    assert "20260816-a" in page


def test_a_review_failure_does_not_break_the_page(client, monkeypatch):
    import run_review
    def _boom(limit=30):
        raise RuntimeError("no debug folder")
    monkeypatch.setattr(run_review, "patterns", _boom)
    page = client.get("/insights")
    assert page.status_code == 200
    assert "no debug folder" in page.get_data(as_text=True)


# ── the look ────────────────────────────────────────────────────────────────

def test_colours_are_defined_once_as_tokens():
    """The old stylesheet hardcoded #171a21 and #2a2d34 across a dozen rules
    and patched light mode with a dozen one-off media queries, so every new
    component had to remember to bring its own override — and the ones that
    forgot were unreadable on a white page."""
    style = dashboard.PAGE_STYLE
    assert "--surface:" in style and "--border:" in style and "--accent:" in style
    # The light palette is a redefinition of the same tokens, not a second set
    # of rules.
    light = style.split("prefers-color-scheme: light")[1][:400]
    assert "--bg:" in light and "--surface:" in light


def test_no_component_hardcodes_the_dark_surface_colour():
    """A hardcoded dark background on a light page is the exact bug this
    replaced: the log viewer rendered a near-black block on white."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    body = src.split("</style></head><body>", 1)[1]
    for literal in ("#171a21", "#0b0d12", "#2a2d34"):
        assert literal not in body, literal


def test_the_review_queue_is_usable_from_a_phone():
    """The owner reviews from a phone; the default 10px tap target is not
    enough for approve/reject."""
    style = dashboard.PAGE_STYLE
    mobile = style.split("max-width: 760px")[-1]
    assert ".btn { padding: 12px" in mobile


def test_the_front_page_leads_with_what_to_change(client, monkeypatch):
    """The page opened on a topic box, which assumes the answer to "what now"
    is always "make another video" — and when most recent runs share a defect,
    another video is precisely the wrong move."""
    monkeypatch.setattr(dashboard, "_advice_now", lambda: (
        [{"title": "Pictures are held too long", "severity": "high"},
         {"title": "second thing", "severity": "medium"}],
        {"state": "needs work", "detail": "Pictures are held too long"}))
    page = client.get("/").get_data(as_text=True)
    assert page.index("Pictures are held too long") < page.index("Make a video about")
    assert "and 1 more" in page


def test_a_broken_advisor_never_breaks_the_front_page(client, monkeypatch):
    def _boom():
        raise RuntimeError("nope")
    monkeypatch.setattr(dashboard, "_advice_now", _boom)
    assert client.get("/").status_code == 200
