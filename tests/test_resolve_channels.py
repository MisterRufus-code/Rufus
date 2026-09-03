"""config/competitors.json is the one empty file between a built scout and its
first observation, and it wants UC ids rather than the @handles a person
actually has. These pin the parts that turn one into the other."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import resolve_channels as rc


@pytest.mark.parametrize("raw,expect", [
    ("@EconomicsExplained", ("handle", "EconomicsExplained")),
    ("EconomicsExplained", ("handle", "EconomicsExplained")),
    ("https://www.youtube.com/@Money.Macro", ("handle", "Money.Macro")),
    ("youtube.com/@foo/videos", ("handle", "foo")),
    ("UCY9S57pDCSPNvc8W_j4cQRQ", ("id", "UCY9S57pDCSPNvc8W_j4cQRQ")),
    ("https://www.youtube.com/channel/UCY9S57pDCSPNvc8W_j4cQRQ",
     ("id", "UCY9S57pDCSPNvc8W_j4cQRQ")),
])
def test_it_takes_whatever_a_person_actually_pasted(raw, expect):
    """Asking someone to strip a URL before pasting it is the friction that
    makes a tool go unused, and this one exists to remove friction."""
    assert rc.normalize(raw) == expect


def test_an_id_is_not_mistaken_for_a_handle():
    """UC ids and handles both arrive as bare words. Sending an id to
    forHandle= finds nothing and then burns 100 units on a search for it."""
    kind, _ = rc.normalize("UCabcdefghijklmnopqrstuv")
    assert kind == "id"


def test_a_short_uc_word_is_still_a_handle():
    """"UCLA" starts with UC. It is a handle, and the length check is the only
    thing standing between it and an id lookup that cannot succeed."""
    assert rc.normalize("UCLA") == ("handle", "UCLA")


def _config(tmp_path, monkeypatch, channels):
    p = tmp_path / "competitors.json"
    p.write_text(json.dumps({"channels": channels}), encoding="utf-8")
    monkeypatch.setattr(rc, "COMPETITORS_FILE", p)
    return p


def test_merge_keeps_the_channels_already_chosen(tmp_path, monkeypatch):
    """The file is the owner's list. A resolver adds to it; it does not decide
    what belongs in it."""
    _config(tmp_path, monkeypatch, ["UCk1mVn3pQr7sTuVwXyZaBcD"])
    ids, added = rc.merge([{"id": "UCn9QwErTyUiOpAsDfGhJkLm", "title": "x", "subs": "1"}])
    assert ids[0] == "UCk1mVn3pQr7sTuVwXyZaBcD"
    assert added == ["UCn9QwErTyUiOpAsDfGhJkLm"]


def test_merge_does_not_add_the_same_channel_twice(tmp_path, monkeypatch):
    dup = "UCY9S57pDCSPNvc8W_j4cQRQ"
    _config(tmp_path, monkeypatch, [dup])
    ids, added = rc.merge([{"id": dup, "title": "x", "subs": "1"}])
    assert ids == [dup] and added == []


def test_the_example_placeholders_do_not_survive_a_merge(tmp_path, monkeypatch):
    """Copying the example and not filling it in is the likeliest first-run
    state there is. competitors._is_placeholder already names those ids; a
    resolver that carried them forward would leave the file looking configured
    while every pass reports channels that do not exist."""
    import competitors
    placeholder = "UC" + "x" * 22
    assert competitors._is_placeholder(placeholder)
    _config(tmp_path, monkeypatch, [placeholder])
    ids, _ = rc.merge([{"id": "UCn9QwErTyUiOpAsDfGhJkLm", "title": "x", "subs": "1"}])
    assert placeholder not in ids


def test_an_unreadable_config_is_never_overwritten(tmp_path, monkeypatch):
    """A half-typed JSON file is a list of channels the owner is in the middle
    of choosing. Silently replacing it with two resolved ids loses the rest."""
    p = tmp_path / "competitors.json"
    p.write_text("{ broken", encoding="utf-8")
    monkeypatch.setattr(rc, "COMPETITORS_FILE", p)
    with pytest.raises(SystemExit):
        rc.merge([{"id": "UCn9QwErTyUiOpAsDfGhJkLm", "title": "x", "subs": "1"}])
    assert p.read_text(encoding="utf-8") == "{ broken"


def test_nothing_is_written_without_the_flag(tmp_path, monkeypatch, capsys):
    """Dry by default: the point of naming your own channels is that you look
    at the list before it starts choosing your next video's topic."""
    p = _config(tmp_path, monkeypatch, [])
    before = p.read_text(encoding="utf-8")

    monkeypatch.setattr(rc, "_service", lambda: object())
    monkeypatch.setattr(rc, "resolve",
                        lambda yt, raw: {"id": "UCn9QwErTyUiOpAsDfGhJkLm",
                                         "title": "T", "subs": "9"})
    assert rc.main(["@anything"]) == 0
    assert p.read_text(encoding="utf-8") == before
    assert "dry run" in capsys.readouterr().out
