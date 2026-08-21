"""The dashboard as the control surface, not a read-only window.

THE OWNER'S INSTRUCTION: the software is run from the dashboard, not from a
terminal. Running a video the way this channel now runs it meant seven `$env:`
lines in PowerShell before every run, and getting one wrong — a cmd `set` in a
PowerShell prompt, a stale RUFUS_BEAT_MOTION from an earlier experiment — is
invisible until the video comes out wrong.
"""

import json
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
