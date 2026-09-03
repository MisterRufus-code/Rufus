"""Credentials that ended up in the log files.

HOW THEY GOT THERE. `auth.py add` prints a sign-in link containing that user's
token, because the token IS the credential and the link is how you hand it
over. serve.ps1 sends the dashboard's stderr to logs/dashboard.log, and
Werkzeug writes one line per request containing the full request target — so
opening `https://…/?token=…` once records an owner credential in plaintext, in
a file that a backup copies, a support bundle would collect and a bug report
gets pasted into.

The dashboard now closes both halves going forward: the sign-in redirects to a
clean URL, and a logging filter strips the value out of the access line. Neither
touches a line written last month.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import logscrub  # noqa: E402


@pytest.fixture
def logs(tmp_path):
    (tmp_path / "dashboard.log").write_text(
        '127.0.0.1 - - [28/Aug/2026 15:00:00] '
        '"GET /?token=abcdefgh12345678 HTTP/1.1" 200 -\n'
        '127.0.0.1 - - [28/Aug/2026 15:00:01] "GET /api/status HTTP/1.1" 200 -\n',
        encoding="utf-8")
    return tmp_path


def test_a_token_in_an_access_line_is_found(logs):
    found = logscrub.scan(logs)
    assert [r["path"].name for r in found] == ["dashboard.log"]
    assert found[0]["hits"] == 1


def test_scanning_changes_nothing(logs):
    """A log is evidence. The decision to alter one is made by a person who can
    see what it costs, which is why the scan is the default."""
    before = (logs / "dashboard.log").read_text(encoding="utf-8")
    logscrub.scan(logs)
    assert (logs / "dashboard.log").read_text(encoding="utf-8") == before


def test_scrubbing_removes_the_value_and_keeps_the_line(logs):
    """A redactor that mangles the whole line makes the log useless, which is
    its own kind of failure — the line is how you find out what happened."""
    logscrub.scrub(logs)
    text = (logs / "dashboard.log").read_text(encoding="utf-8")
    assert "abcdefgh12345678" not in text
    assert "GET /?token=[redacted]" in text
    assert "GET /api/status" in text, "the untouched line survived"


def test_no_copy_of_the_secret_is_left_behind(logs):
    """A .bak beside the file would keep the credential on the same disk,
    which is the thing being undone."""
    logscrub.scrub(logs)
    leftovers = [p.name for p in logs.iterdir() if p.name != "dashboard.log"]
    assert leftovers == [], leftovers
    for p in logs.iterdir():
        assert "abcdefgh12345678" not in p.read_text(encoding="utf-8")


def test_a_scrubbed_file_stops_reporting_itself_as_leaking(logs):
    """Already-redacted values must not count, or a cleaned log would show up
    on every scan forever and the real ones would be lost in it."""
    logscrub.scrub(logs)
    assert logscrub.scan(logs) == []


def test_the_oauth_code_and_state_count_too(tmp_path):
    """A Google callback carries a one-time code that is a credential for as
    long as it is unspent."""
    (tmp_path / "d.log").write_text(
        'GET /auth/google/callback?code=4/0AbCd&state=xyz HTTP/1.1\n',
        encoding="utf-8")
    assert logscrub.scan(tmp_path)[0]["hits"] == 2
    logscrub.scrub(tmp_path)
    text = (tmp_path / "d.log").read_text(encoding="utf-8")
    assert "4/0AbCd" not in text and "xyz" not in text


def test_a_log_with_nothing_in_it_is_not_touched(tmp_path):
    (tmp_path / "run.log").write_text("[research] using wisdom quote\n",
                                      encoding="utf-8")
    assert logscrub.scan(tmp_path) == []
    assert logscrub.scrub(tmp_path) == []


def test_one_pattern_shared_with_the_dashboard():
    """Two regexes for the same job drift, and the one that drifts is the one
    nobody notices has stopped matching."""
    import dashboard
    src = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert "logscrub.redact" in src
    assert "access_token=" not in src, (
        "the dashboard is keeping its own copy of the pattern again")


def test_the_cli_says_it_changed_nothing_unless_asked(logs, monkeypatch,
                                                      capsys):
    monkeypatch.setattr(logscrub, "scan", lambda d=None: [
        {"path": logs / "dashboard.log", "hits": 1}])
    monkeypatch.setattr(sys, "argv", ["logscrub.py"])
    assert logscrub._cli() == 1
    assert "Nothing has been changed" in capsys.readouterr().out


def test_the_cli_says_to_rotate_even_after_a_successful_scrub(
        logs, monkeypatch, capsys):
    """THE HONEST HALF. A secret that sat in a file for a month may already
    have been copied — into a backup, a screenshot, an old disk. Rewriting the
    file revokes nothing, and a clean scan that reads as safety is worse than
    no scan."""
    monkeypatch.setattr(logscrub, "scan", lambda d=None: [
        {"path": logs / "dashboard.log", "hits": 1}])
    monkeypatch.setattr(logscrub, "scrub", lambda d=None: [])
    monkeypatch.setattr(sys, "argv", ["logscrub.py", "--fix"])
    logscrub._cli()
    out = capsys.readouterr().out
    assert "ROTATE THEM ANYWAY" in out
    assert "auth.py rotate" in out
