"""Structural/text checks for schedule_daily.ps1 — no PowerShell runtime here,
so these are the same "read the script, assert on patterns" approach the
repo already uses for run_scheduled.bat (see test_comfy_client.py)."""

from pathlib import Path

SCRIPT = (Path(__file__).parent.parent / "schedule_daily.ps1").read_text()


def test_no_here_strings():
    # Documented PS 5.1/LF parsing pitfall the original script already
    # avoided — a regression here would silently break on some machines.
    assert "@\"" not in SCRIPT
    assert "@'" not in SCRIPT


def test_has_times_param_for_multiple_daily_runs():
    assert "$Times" in SCRIPT
    assert "$Time = " in SCRIPT   # single-time default still supported


def test_unregister_removes_all_prefixed_tasks_not_just_one():
    """5 videos/day means up to 5 registered tasks — -Unregister must find
    and remove all of them by prefix, not a single hardcoded task name."""
    assert "-Unregister" in SCRIPT
    unregister_block = SCRIPT.split("if ($Unregister)")[1].split("if (-not (Test-Path")[0]
    assert "$TaskPrefix" in unregister_block
    assert "foreach" in unregister_block


def test_re_registering_clears_stale_extra_tasks():
    """Going from -Times with 5 entries down to 2 must not leave 3 orphaned
    scheduled tasks behind — re-running the script has to clear old ones
    before creating the new set."""
    create_section = SCRIPT.split("# Clear any previously")[1]
    assert "schtasks /Delete" in create_section.split("$i = 1")[0]


def test_validates_every_time_in_the_list():
    assert "timeList" in SCRIPT
    assert "-notmatch" in SCRIPT   # HH:MM format guard applied per-entry


def test_creates_one_task_per_time_with_distinct_names():
    assert '"$TaskPrefix $i"' in SCRIPT
    assert "schtasks /Create" in SCRIPT


def test_points_at_run_scheduled_bat_not_run_bat():
    # run.bat is the interactive/testing entry point (has --skip-upload
    # available); the autonomous schedule must use run_scheduled.bat.
    assert "run_scheduled.bat" in SCRIPT
    assert "$RunBat" in SCRIPT


def test_mentions_dashboard_for_review_since_nothing_auto_uploads_now():
    assert "dashboard.py" in SCRIPT
