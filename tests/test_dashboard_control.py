"""The dashboard as the control surface, not a read-only window.

THE OWNER'S INSTRUCTION: the software is run from the dashboard, not from a
terminal. Running a video the way this channel now runs it meant seven `$env:`
lines in PowerShell before every run, and getting one wrong — a cmd `set` in a
PowerShell prompt, a stale RUFUS_BEAT_MOTION from an earlier experiment — is
invisible until the video comes out wrong.
"""

import json
import os
import re
import sys
import time
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


# ── starting up ─────────────────────────────────────────────────────────────

def test_a_busy_port_is_explained_not_just_reported():
    """run.bat starts a dashboard of its own, so the ordinary way to reach the
    startup path is with one already running — and what Flask prints then is
    WinError 10048 about socket addresses, which does not tell the reader that
    the thing they wanted is already open in another window."""
    # Whitespace-normalised: the message is a wrapped f-string, so a literal
    # search would be asserting on where the lines happen to break.
    src = " ".join(Path(dashboard.__file__).read_text(encoding="utf-8").split())
    src = src.replace('" f"', "").replace('" "', "")
    assert "already in use" in src
    assert "that IS this dashboard" in src
    assert "RUFUS_DASHBOARD_PORT" in src


def test_the_port_check_does_not_raise_on_a_free_port():
    """It runs before Flask binds, so it must never be the thing that stops a
    working start."""
    assert dashboard._port_taken("127.0.0.1", 59999) is False


def test_the_launcher_shows_the_reason_it_exited():
    """Everything the dashboard prints goes to the log, because a scheduled
    task has no console. A person running the bat at a prompt therefore got a
    silent return and no hint at all — which happened: it refused to start
    because the port was held, said so clearly, and said it into a file nobody
    was reading. A launcher that exists to make failure debuggable has to put
    the failure where the person is."""
    bat = (Path(dashboard.__file__).parent.parent / "run_dashboard.bat")
    text = bat.read_text(encoding="utf-8", errors="replace")
    assert "Get-Content" in text and "-Tail" in text
    assert 'if not "%RC%"=="0"' in text
    # And the success path must NOT dump the log — a working start would then
    # end with twenty lines of noise every time.
    tail = text.split('set "RC=%ERRORLEVEL%"')[1]
    assert tail.index('if not "%RC%"=="0"') < tail.index("Get-Content")


# ── the format switch, and picking a look by looking at it ───────────────────

def test_the_format_switch_is_in_the_header_on_every_page(client):
    """Not in Settings. It decides aspect ratio, script length, picture count
    and how long the GPU is busy, and it is the one thing the owner wants to
    change per video rather than per channel. A setting three pages deep that
    changes everything is a setting people forget is set — SD_CLIPS proved
    that on this dashboard already."""
    body = client.get("/").data.decode()
    assert 'action="/format"' in body
    assert "Shorts" in body and "Long-form" in body


def test_switching_format_persists_for_every_launch_path(client, tmp_path,
                                                         monkeypatch):
    """A header button that only changed THIS process would be the
    settings-page-obeyed-by-one-launcher bug wearing a nicer hat."""
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "s.json")
    r = client.post("/format", data={"format": "long"})
    assert r.status_code in (302, 303)
    assert dashboard._load_settings()["RUFUS_FORMAT"] == "long"
    import os as _os
    assert _os.environ["RUFUS_FORMAT"] == "long"
    # Leave the process as we found it: an env var set by one test decides
    # the aspect ratio for every test after it.
    _os.environ.pop("RUFUS_FORMAT", None)


def test_an_unknown_format_changes_nothing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "s.json")
    client.post("/format", data={"format": "vertical-ish"})
    assert "RUFUS_FORMAT" not in dashboard._load_settings()


def test_the_style_page_lists_every_preset(client):
    import comfy_client
    body = client.get("/styles").data.decode()
    for sid in comfy_client.style_presets():
        assert sid in body


def test_a_style_with_no_preview_says_so_rather_than_faking_one(client):
    """Nothing here pretends to show art it has not produced."""
    body = client.get("/styles").data.decode()
    assert "no preview yet" in body


def test_picking_a_style_persists_it(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "s.json")
    client.post("/styles/use", data={"style": "storybook"})
    assert dashboard._load_settings()["RUFUS_STYLE"] == "storybook"


def test_an_unknown_style_is_refused(client, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "s.json")
    client.post("/styles/use", data={"style": "../../etc/passwd"})
    assert "RUFUS_STYLE" not in dashboard._load_settings()


def test_the_preview_scene_is_the_same_for_every_style():
    """A picker where each card shows a different subject compares subjects,
    not styles. The only variable between these frames must be the style
    block."""
    import inspect
    src = inspect.getsource(dashboard.styles_preview)
    assert "STYLE_PREVIEW_SCENE" in src
    assert "add_detail=False" in src, (
        "the automatic suffix is whatever RUFUS_STYLE already is — a picker "
        "that previewed the style you already have is a picker in name only")


def test_a_preview_that_does_not_exist_is_a_404_not_a_traceback(client):
    assert client.get("/styles/preview/not_a_style").status_code == 404


# ── sixteen links is not a navigation, it is an inventory ────────────────────
#
# Flat, they wrapped to two rows on a laptop and filled an entire phone screen
# before any content appeared — on a dashboard whose review queue is worked
# from a phone. NAV_ITEMS stays the flat registry (a page is registered by
# adding one line to it, and four tests unpack it); NAV_GROUPS is a view.
#
# The invariant worth enforcing is coverage. A page that exists and is
# unreachable is worse than one that was never written, and the failure mode
# is silent: you add a route, add its NAV_ITEMS line, and it renders nowhere.

def test_every_registered_page_is_reachable_from_the_nav():
    registered = {href for href, _l, _p in dashboard.NAV_ITEMS}
    grouped = {h for _title, hrefs in dashboard.NAV_GROUPS for h in hrefs}
    assert registered - grouped == set(), "registered but in no group"
    assert grouped - registered == set(), "grouped but not registered"


def test_the_primary_links_are_real_pages():
    registered = {href for href, _l, _p in dashboard.NAV_ITEMS}
    assert set(dashboard.NAV_PRIMARY) <= registered


def test_no_page_is_in_two_groups():
    """Two homes for one page means the reader has to learn which one you
    meant, which is the problem the grouping exists to remove."""
    seen = [h for _t, hrefs in dashboard.NAV_GROUPS for h in hrefs]
    assert len(seen) == len(set(seen))


def test_the_menu_needs_no_javascript_to_open():
    """This dashboard is deliberately self-contained with no build step, and a
    menu that needs a script to open is a menu that does not open when the
    script fails."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert '<details class="navmore">' in src


# ── the front page says what needs you ──────────────────────────────────────
#
# A FINISHED VIDEO ANNOUNCES ITSELF: it lands in the pending list with a
# thumbnail and the count above it goes up. A run that DIED announces nothing —
# Step 6 is what writes the `videos` row, so a run that fell over before it
# leaves no row at all, and the front page looks exactly the way it looked
# yesterday. The owner reads an unchanged screen as "nothing ran last night",
# which is the opposite of what happened.

def _failed(run_id="run-x", ago=120.0, **over):
    p = {"run_id": run_id, "channel": "main_en", "status": "failed",
         "step": 1, "total": 7, "label": "failed",
         "detail": "no seed: every source refused",
         "updated_at": time.time() - ago, "stale": False}
    p.update(over)
    return p


def test_a_crashed_run_is_reported_on_the_front_page(client, monkeypatch):
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [_failed()])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [])
    page = client.get("/").get_data(as_text=True)
    assert "run-x" in page
    assert "no seed: every source refused" in page
    assert "step 1/7" in page


def test_a_run_that_is_still_going_is_not_called_a_crash(client, monkeypatch):
    """A live run has no `videos` row either — its debug folder is an orphan
    right up until Step 6. Without excluding it, every visit during a run would
    report the run in progress as a failure."""
    live = _failed("run-live", status="running", label="writing the script")
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [live])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [
        {"run_id": "run-live", "mtime": time.time(),
         "files": [], "preview": ""}])
    assert dashboard._recent_failures() == []


def test_a_stalled_run_counts_even_though_it_never_said_so(monkeypatch):
    """`stale` means the process died without reaching its finally-block, so
    the file still claims "running" forever. That is a failure that will never
    report itself."""
    stuck = _failed("run-stuck", status="running", stale=True, step=5)
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [stuck])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [])
    found = dashboard._recent_failures()
    assert [f["run_id"] for f in found] == ["run-stuck"]
    assert found[0]["stalled"] is True


def test_a_cancelled_run_is_not_a_failure(monkeypatch):
    """Somebody pressed stop. Reporting a person's own decision back to them as
    a problem is how a notice area becomes something you scroll past."""
    monkeypatch.setattr(dashboard.run_progress, "read_all",
                        lambda: [_failed("run-c", status="cancelled")])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [])
    assert dashboard._recent_failures() == []


def test_an_old_crash_stops_shouting(monkeypatch):
    """A banner that is always on is a banner nobody reads."""
    old = _failed("run-old", ago=dashboard.FAILURE_WINDOW_SECONDS + 3600)
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [old])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [
        {"run_id": "run-old",
         "mtime": time.time() - dashboard.FAILURE_WINDOW_SECONDS - 3600,
         "files": [], "preview": ""}])
    assert dashboard._recent_failures() == []


def test_a_hard_kill_that_wrote_no_progress_is_still_noticed(monkeypatch):
    """Killed hard enough that the finally-block never ran, so the only trace
    is a debug folder with no matching row. That is the case /failures was
    written for; the front page should not need you to go looking."""
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [
        {"run_id": "run-killed", "mtime": time.time() - 300,
         "files": ["script.txt"], "preview": ""}])
    found = dashboard._recent_failures()
    assert [f["run_id"] for f in found] == ["run-killed"]


def test_one_dead_run_is_reported_once(monkeypatch):
    """Most crashes leave BOTH a failed progress file and an orphan folder.
    Counting them separately would say two runs failed when one did."""
    monkeypatch.setattr(dashboard.run_progress, "read_all",
                        lambda: [_failed("run-both")])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [
        {"run_id": "run-both", "mtime": time.time() - 120,
         "files": [], "preview": ""}])
    assert [f["run_id"] for f in dashboard._recent_failures()] == ["run-both"]


def test_a_quiet_week_shows_a_quiet_page(client, monkeypatch):
    monkeypatch.setattr(dashboard.run_progress, "read_all", lambda: [])
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", lambda limit=40: [])
    assert dashboard._failure_notice() == ""


def test_a_broken_failure_lookup_never_breaks_the_front_page(client, monkeypatch):
    def _boom():
        raise RuntimeError("nope")
    monkeypatch.setattr(dashboard.run_progress, "read_all", _boom)
    monkeypatch.setattr(dashboard, "_orphaned_debug_runs", _boom)
    assert client.get("/").status_code == 200


def test_what_needs_you_comes_before_what_you_could_make(client, monkeypatch):
    """The topic box used to open this page, which answers "what now" with
    "make another video" before anybody has said whether the last four are any
    good."""
    waiting = [{"id": 7, "score": 8, "title": "A waiting video", "script_hook": "",
                "niche": "money_history", "upload_status": "pending",
                "run_id": "", "created_at": "2026-08-21 09:30:00",
                "uploaded_at": None}]

    def _videos(limit=60, channel=None, status=None):
        return waiting if status == "pending" else []

    monkeypatch.setattr(dashboard, "_recent_videos", _videos)
    page = client.get("/").get_data(as_text=True)
    assert page.index("Awaiting your review (1)") < page.index("Make a video about")


def test_an_empty_queue_says_so_instead_of_showing_a_bare_zero(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_recent_videos",
                        lambda limit=60, channel=None, status=None: [])
    page = client.get("/").get_data(as_text=True)
    assert "Nothing is waiting on you" in page


# ── the review page is in the order the review happens ──────────────────────

def test_the_decision_buttons_stack_on_a_phone():
    """Three buttons sharing one 390px row give each about a thumb's width,
    and two of the three are irreversible."""
    mobile = dashboard.PAGE_STYLE.split("max-width: 760px")[-1]
    assert ".actions { flex-direction: column" in mobile


# ── the stylesheet is one system, not eleven ────────────────────────────────

def _root_tokens() -> set:
    """Only the tokens on bare :root. A token defined ONLY inside the light
    media query exists in light mode and nowhere else, which is a bug that
    hides from anyone whose phone is in dark mode."""
    block = dashboard.PAGE_STYLE.split(":root {", 1)[1].split("}", 1)[0]
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", block))


def _rule(selector: str) -> str:
    """The declarations of one rule, whitespace removed so a test can ask what
    it sets without caring how it was typed."""
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", dashboard.PAGE_STYLE)
    assert m, f"no rule for {selector}"
    return re.sub(r"\s+", "", m.group(1))


def test_every_token_a_rule_asks_for_actually_exists():
    """CSS DROPS A DECLARATION WHOSE var() DOES NOT RESOLVE, SILENTLY.

    Four names were in use that :root never defined — --line, --card, --muted,
    --fg — all of them near-synonyms of tokens that did exist under another
    name. `border: 1px solid var(--line)` is not a wrong border, it is NO
    border: the grouped nav menu floated over the page with no edge at all,
    and the style cards had none either. Nothing errors, nothing logs, and the
    only symptom is that the page looks slightly wrong in a way you cannot
    grep for. Which is the same shape as every other bug this file has had:
    fail-open without fail-loud is fail-silent."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    used = set(re.findall(r"var\((--[a-z0-9-]+)", src))
    missing = used - _root_tokens()
    assert missing == set(), f"used but never defined: {sorted(missing)}"


def test_corners_come_from_the_radius_tokens():
    """Five different corner radii were in play — 7, 8, 10 and 12px plus the
    pills — which reads as carelessness rather than as a choice. Two tokens:
    panels and controls. The rest are shapes, not corners."""
    found = {r.strip() for r in
             re.findall(r"border-radius:\s*([^;}]+)", dashboard.PAGE_STYLE)}
    allowed = {"var(--radius)", "var(--radius-sm)",
               "999px",   # pills — badges, dots, the progress bar
               "50%",     # the status dots are circles
               "4px"}     # inline code and focus rings
    assert found <= allowed, sorted(found - allowed)


def test_the_stylesheet_uses_one_type_scale():
    sizes = {float(x) for x in
             re.findall(r"font-size:\s*([0-9.]+)px", dashboard.PAGE_STYLE)}
    assert sizes <= {11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 26.0}, sorted(sizes)


def test_every_panel_is_the_same_panel():
    """A card, a table, a log block, a script block and the status bar are all
    the same idea — content raised off the background — and each had arrived
    at its own combination of the four properties that say so."""
    for selector in (".card", ".orphan", ".thumbcard", ".style-card",
                     "#livebar", "table", "pre", ".script"):
        rule = _rule(selector)
        for expected in ("background:var(--surface)", "1pxsolidvar(--border)",
                         "border-radius:var(--radius)", "box-shadow:var(--shadow)"):
            assert expected in rule, f"{selector} is missing {expected}"


def test_the_light_palette_redefines_every_colour_it_needs_to():
    """A token whose dark value survives into light mode is the log viewer bug
    again: a near-black block on a white page."""
    style = dashboard.PAGE_STYLE
    light = style.split("prefers-color-scheme: light", 1)[1].split("}", 1)[0]
    for token in ("--bg", "--surface", "--border", "--text", "--dim"):
        assert f"{token}:" in light, token


def test_the_space_between_blocks_sits_on_one_grid():
    """Gaps were 2, 4, 8, 10, 12, 14, 20 and 22px — seven values for one idea.
    Component PADDING is deliberately not covered by this: those numbers are
    tap targets tuned against a real phone, and rounding a working ergonomic
    to a tidier number is a downgrade dressed up as a system."""
    found = set()
    for m in re.finditer(r"gap:\s*([^;}]+)", dashboard.PAGE_STYLE):
        found.update(m.group(1).strip().split())
    assert found <= {"0", "4px", "8px", "12px", "16px", "20px"}, sorted(found)


# ── the port was held for ten hours and nothing named the holder ────────────

def test_the_two_port_failures_do_not_share_a_message():
    """"The port is taken" was reported as "a dashboard is already running —
    open it", and for ten hours that advice pointed at a python that had
    stopped serving. The health check that disproves it lives one file away and
    was never asked."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split('if _port_taken(host, port):', 1)[1].split("db_manager.init_db()")[0]
    assert "_answers_healthz" in block, "the two states are still one message"
    assert "port_owner.describe" in block, "the holder is still not named"


def test_the_two_port_failures_do_not_share_an_exit_code():
    """The watchdog acts on the exit code, and it cannot act differently on the
    same number: 3 is "a healthy dashboard is already there, leave it alone",
    4 is "something is squatting and serving nothing"."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split('if _port_taken(host, port):', 1)[1].split("db_manager.init_db()")[0]
    assert "sys.exit(3)" in block and "sys.exit(4)" in block


def test_an_unidentified_holder_is_never_offered_up_to_be_killed():
    """The suggested Stop-Process line is only printed for a process positively
    identified as a Rufus dashboard. Telling somebody to kill pid 1888 when
    1888 might be their own work is worse than telling them nothing."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split('if _port_taken(host, port):', 1)[1].split("db_manager.init_db()")[0]
    kill_at = block.index("Stop-Process")
    guard_at = block.index("port_owner.is_rufus_dashboard")
    assert guard_at < kill_at


def test_healthz_is_probed_on_the_loopback_when_bound_to_all_interfaces():
    """0.0.0.0 is an address to bind, not one to connect to — the same
    correction _port_taken already makes."""
    assert dashboard._answers_healthz("0.0.0.0", 1) is False


# ── the interpreter that no launcher chose ─────────────────────────────────

def test_the_venv_interpreter_is_not_complained_about(monkeypatch, tmp_path):
    venv = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv.mkdir(parents=True)
    exe = venv / ("python.exe" if os.name == "nt" else "python")
    exe.write_text("")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard.sys, "executable", str(exe))
    assert dashboard._wrong_interpreter() == ""


def test_a_different_interpreter_is_named_out_loud(monkeypatch, tmp_path):
    """The python that squatted on port 8765 for ten hours was the SYSTEM one.
    Both launchers refuse to use it — so something started this outside a
    launcher, and nothing anywhere said a word."""
    venv = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv.mkdir(parents=True)
    (venv / ("python.exe" if os.name == "nt" else "python")).write_text("")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard.sys, "executable", "/usr/bin/python3")
    msg = dashboard._wrong_interpreter()
    assert "/usr/bin/python3" in msg and ".venv" in msg


def test_no_venv_means_nothing_to_be_wrong_about(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard.sys, "executable", "/usr/bin/python3")
    assert dashboard._wrong_interpreter() == ""


def test_the_notice_can_be_turned_off_by_someone_who_means_it(monkeypatch, tmp_path):
    venv = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv.mkdir(parents=True)
    (venv / ("python.exe" if os.name == "nt" else "python")).write_text("")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard.sys, "executable", "/usr/bin/python3")
    monkeypatch.setenv("RUFUS_ALLOW_ANY_PYTHON", "1")
    assert dashboard._wrong_interpreter() == ""


def test_it_is_a_notice_and_not_a_refusal():
    """The .bat files refuse because the venv is MISSING, which nothing
    downstream recovers from. Here it exists and another interpreter was
    chosen, which may be deliberate — a second dashboard on another port, a
    debugger. Refusing breaks those; saying nothing is how this happened."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("wrong = _wrong_interpreter()", 1)[1].split("app.run(")[0]
    assert "sys.exit" not in block


def test_the_interpreter_shows_up_in_the_status_api(client):
    body = client.get("/api/status").get_json()
    assert "interpreter_warning" in body


# ── the dashboard was waiting on itself ────────────────────────────────────

def test_the_gallery_asks_for_a_downscaled_image():
    """The generated-images gallery sent the FULL png into a 220px card: 36
    cards averaging ~1MB is ~35 megabytes fetched to draw postage stamps, and
    fetched again next visit because nothing said it could be cached. That is
    what "the dashboard is slow" meant on that page."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("def thumbnails_page", 1)[1].split("def thumbnails_generate")[0]
    img = [ln for ln in block.splitlines() if "<img src=" in ln]
    assert img, "no gallery image tag found"
    assert "?w=" in img[0], "the gallery still asks for the full-size file"


def test_the_downscale_width_must_be_one_of_a_fixed_set():
    """An open ?w= lets anyone fill the disk with cache entries."""
    folder = Path(dashboard.__file__).parent
    assert dashboard._thumb_of(folder, "x.png", 137) is None
    assert 480 in dashboard._THUMB_WIDTHS


def test_saving_to_the_phone_still_gets_the_real_file():
    """"Save to phone" that hands over a 480px jpg is worse than useless — the
    whole point of the link is the actual image."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("def thumbnail_file", 1)[1].split("def _setting_field")[0]
    assert "if not download:" in block, "the download path may be downscaled"


def test_both_image_routes_tell_the_browser_it_can_keep_them(client):
    """Rendered stills are written once and never edited. Without a cache
    header every scroll re-fetches megabytes the browser already has."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    for name, nxt in (("def thumbnail_file", "def _setting_field"),
                      ("def debug_file", "def download_video")):
        block = src.split(name, 1)[1].split(nxt)[0]
        assert "_IMAGE_MAX_AGE" in block, f"{name} sends no cache header"


def test_the_env_mutation_is_what_is_locked_not_the_whole_server():
    """threaded=False said "nothing anywhere may overlap" to protect ONE
    route's env mutation, and every keyframe request in the app queued behind
    a single thread for it. The lock says the true, smaller thing."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("def _scoped_env", 1)[1].split("def _redirect_detail")[0]
    assert "_ENV_LOCK.acquire()" in block
    assert "threaded=True" in src


def test_the_environment_is_restored_before_the_lock_is_released():
    """Releasing first lets another approval acquire, apply its overrides, and
    have them overwritten by this block's restore — the exact interleaving the
    lock exists to prevent, moved four lines later."""
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = src.split("def _scoped_env", 1)[1].split("def _redirect_detail")[0]
    tail = block.split("finally:", 1)[1]
    assert tail.index("os.environ") < tail.index("_ENV_LOCK.release()")


def test_two_overlapping_scoped_envs_cannot_interleave():
    """The property, not just the shape of the code."""
    import threading as _t
    seen, errors = [], []

    def worker(value):
        try:
            with dashboard._scoped_env(RUFUS_TEST_CHANNEL=value):
                time.sleep(0.02)
                seen.append((value, os.environ.get("RUFUS_TEST_CHANNEL")))
        except Exception as e:                      # pragma: no cover
            errors.append(e)

    threads = [_t.Thread(target=worker, args=(f"ch{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(asked == got for asked, got in seen), seen
    assert "RUFUS_TEST_CHANNEL" not in os.environ


# ── the thumbnails page makes thumbnails now ───────────────────────────────

@pytest.fixture
def thumbs(tmp_path, monkeypatch):
    """A thumbnails dir with one background in it, wired everywhere it is read."""
    import image_gen
    from PIL import Image
    d = tmp_path / "thumbs"
    d.mkdir()
    monkeypatch.setattr(dashboard.paths, "thumbnails_dir", lambda: d)
    monkeypatch.setattr(image_gen.paths, "thumbnails_dir", lambda: d)
    png = d / "1700000000_hourglass.png"
    Image.new("RGB", (1280, 720), (26, 40, 68)).save(png)
    png.with_suffix(".txt").write_text("PROMPT: a cracked hourglass\nSEED: 7\n",
                                       encoding="utf-8")
    return png


def test_drawing_a_thumbnail_does_not_render_in_the_request(client, monkeypatch):
    """It called image_gen.generate_image() inline and the page waited for it —
    and the code SAID so ("that wait freezes the dashboard for everyone")
    without doing anything about it. A browser tab holding an open connection
    for ninety seconds is still silly on a threaded server, and it cannot draw
    three variants at once."""
    import image_gen
    def _boom(*a, **k):                              # pragma: no cover
        raise AssertionError("rendered inside the request")
    monkeypatch.setattr(image_gen, "generate_image", _boom)
    launched = {}
    monkeypatch.setattr(dashboard, "_launch_thumb",
                        lambda *a, **k: (launched.setdefault("args", (a, k)),
                                         Path("logs/thumb_1.log"))[1:])
    monkeypatch.setattr(dashboard, "_channels", lambda: [])
    r = client.post("/thumbnails/generate",
                    data={"prompt": "a cracked hourglass", "headline": "Gold",
                          "count": "3"})
    assert r.status_code == 302
    assert launched, "the render was never launched"


def test_the_style_override_reaches_the_child_and_not_this_process(monkeypatch):
    """The run style is whatever the videos are being made in. Asking for a
    thumbnail must not quietly change it — and _scoped_env cannot help here
    because the value has to reach a CHILD process."""
    seen = {}

    class _Proc:
        pid = 1

    def _popen(cmd, **kw):
        seen["env"] = kw["env"]
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(dashboard.subprocess, "Popen", _popen)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    dashboard._launch_thumb("a coin", "Gold", 2, style="thumbnail")
    assert seen["env"]["RUFUS_STYLE"] == "thumbnail"
    assert os.environ["RUFUS_STYLE"] == "stickman", "the process style changed"
    assert "--count" in seen["cmd"] and "2" in seen["cmd"]
    assert "--headline" in seen["cmd"]


def test_a_literal_detail_override_cannot_outrank_the_chosen_look(monkeypatch):
    """RUFUS_STILLS_DETAIL beats RUFUS_STYLE in comfy_client._detail_suffix,
    so leaving it set would silently ignore the look picked on the form."""
    class _Proc:
        pid = 1
    seen = {}
    monkeypatch.setattr(dashboard.subprocess, "Popen",
                        lambda cmd, **kw: (seen.update(env=kw["env"]), _Proc())[1])
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "something else entirely")
    dashboard._launch_thumb("a coin", "", 1, style="thumbnail")
    assert "RUFUS_STILLS_DETAIL" not in seen["env"]


def test_retyping_the_headline_never_touches_the_gpu(client, thumbs, monkeypatch):
    """Drawing the picture is seconds of GPU; drawing the words on it is a
    tenth of a second of Pillow. One button for both meant every headline you
    wanted to try cost another render, so nobody tried a second one."""
    import image_gen
    def _boom(*a, **k):                              # pragma: no cover
        raise AssertionError("called ComfyUI to change some words")
    monkeypatch.setattr(image_gen, "generate_image", _boom)
    r = client.post("/thumbnails/compose",
                    data={"name": thumbs.name, "headline": "Rome ran out"})
    assert r.status_code == 302
    assert image_gen.composed_path(thumbs).exists()


def test_the_headline_is_remembered_so_the_box_is_not_empty_next_time(client, thumbs):
    import image_gen
    client.post("/thumbnails/compose",
                data={"name": thumbs.name, "headline": "Rome ran out"})
    assert image_gen.recent_images()[0]["headline"] == "Rome ran out"


def test_the_card_shows_the_composed_thumbnail_not_the_bare_background(client, thumbs):
    """A card showing the raw picture is showing something that will never go
    on YouTube."""
    import image_gen
    client.post("/thumbnails/compose",
                data={"name": thumbs.name, "headline": "Rome ran out"})
    page = client.get("/thumbnails").get_data(as_text=True)
    assert image_gen.composed_path(thumbs).name in page


def test_the_page_shows_it_at_the_size_it_competes_at(client, thumbs):
    """168x94 is the mobile feed. The page showed one size, full width, which
    is the size nobody ever sees it at."""
    client.post("/thumbnails/compose",
                data={"name": thumbs.name, "headline": "Rome ran out"})
    page = client.get("/thumbnails").get_data(as_text=True)
    assert "width:168px;height:94px" in page


def test_an_unknown_image_name_cannot_reach_the_filesystem(client, thumbs):
    r = client.post("/thumbnails/compose",
                    data={"name": "../../etc/passwd", "headline": "x"})
    assert r.status_code == 302
    assert "No%20such%20image" in r.headers["Location"]
