"""Turning measurements into what to change, and then changing it.

The dashboard was asked to be smart. The tempting reading is a model reading
logs and offering opinions; the better one, for now, is arithmetic — because a
number is checkable, is the same tomorrow, and names the exact lever. "The
setting clause was on half the shots in four of your last six runs" beats a
paragraph of plausible advice and costs nothing to produce.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import advisor  # noqa: E402
import dashboard  # noqa: E402


def _patterns(*findings, runs=6):
    return {"runs_reviewed": runs,
            "recurring": [{"id": i, "runs": int(runs * sh), "share": sh}
                          for i, sh in findings],
            "rows": []}


# ── only what recurs ────────────────────────────────────────────────────────

def test_a_finding_in_one_run_of_six_is_not_advice():
    """One run is a bad seed. Advice that fires every time is advice people
    learn to scroll past — this repo has shipped that mistake twice."""
    items = advisor.advise(_patterns(("repeated_images", 0.17)))
    assert not any(i["id"] == "repeated_images" for i in items)


def test_a_finding_in_most_runs_is_advice():
    items = advisor.advise(_patterns(("setting_clause_everywhere", 0.67)))
    ids = [i["id"] for i in items]
    assert "setting_clause_everywhere" in ids


def test_the_worst_thing_is_first():
    items = advisor.advise(
        _patterns(("pictures_held_too_long", 0.5),
                  ("setting_clause_everywhere", 0.83)))
    assert items[0]["id"] == "setting_clause_everywhere"
    assert items[0]["severity"] == "high"


# ── each one is actionable ──────────────────────────────────────────────────

def test_every_suggestion_names_what_to_do():
    for fid in advisor._REMEDIES:
        items = advisor.advise(_patterns((fid, 1.0)))
        assert items, fid
        assert items[0]["action"], fid
        assert items[0]["why"], fid
        assert items[0]["evidence"], fid


def test_a_suggestion_with_a_setting_carries_the_value_to_set():
    items = advisor.advise(_patterns(("pictures_held_too_long", 0.6)))
    it = next(i for i in items if i["id"] == "pictures_held_too_long")
    assert it["setting"] == "SD_CLIPS" and it["value"]


def test_advice_already_followed_is_demoted_not_just_debuttoned():
    """A live page showed "Too few pictures for the length" as the top HIGH
    finding with "Already set to 24" tacked on the end, and drove the
    readiness line with it. The measurements behind it are of runs made BEFORE
    the change — removing the button was not enough."""
    items = advisor.advise(_patterns(("pictures_held_too_long", 1.0)),
                           settings={"SD_CLIPS": "24"})
    it = next(i for i in items if i["id"] == "pictures_held_too_long")
    assert it["setting"] is None
    assert it["severity"] == "low"
    assert it["done"] is True
    assert "already fixed" in it["title"]
    assert "clears once newer runs are measured" in it["action"]


def test_something_already_fixed_sinks_below_live_problems():
    items = advisor.advise(
        _patterns(("pictures_held_too_long", 1.0), ("one_object_dominates", 0.5)),
        settings={"SD_CLIPS": "24"})
    assert items[-1]["id"] == "pictures_held_too_long"


def test_readiness_ignores_what_was_already_fixed():
    """A readiness line reading "needs work — too few pictures" when the beat
    count was raised an hour ago is reporting the past as the present."""
    pat = _patterns(("pictures_held_too_long", 1.0))
    assert advisor.readiness(pat, {}, {})["state"] == "needs work"
    assert advisor.readiness(pat, {}, {"SD_CLIPS": "24"})["state"] == "good"


def test_every_offered_setting_actually_exists():
    """A button that writes a key nothing reads is worse than no button."""
    keys = {k for k, _l, _kind, _h in dashboard.SETTINGS_SCHEMA}
    for rem in advisor._REMEDIES.values():
        if rem.get("setting"):
            assert rem["setting"] in keys, rem["setting"]


# ── the script is upstream of the pictures ──────────────────────────────────

def test_weak_scripts_are_raised_above_picture_tuning():
    items = advisor.advise(_patterns(("pictures_held_too_long", 0.5)),
                           stats={"avg_score": 5.2, "total": 12})
    ids = [i["id"] for i in items]
    assert "weak_scripts" in ids
    assert ids.index("weak_scripts") < ids.index("pictures_held_too_long")


def test_good_scores_raise_nothing_about_the_writing():
    items = advisor.advise(_patterns(), stats={"avg_score": 8.4, "total": 20})
    assert not any(i["id"] == "weak_scripts" for i in items)


def test_a_review_backlog_is_raised_as_its_own_problem():
    """With nothing published there are no view counts, so the learning loop
    has nothing to learn from."""
    items = advisor.advise(_patterns(), stats={"held": 14, "total": 20,
                                               "avg_score": 8.0})
    assert any(i["id"] == "review_backlog" for i in items)


def test_no_measurements_says_how_to_get_some():
    items = advisor.advise({"runs_reviewed": 0, "recurring": []})
    assert items[-1]["id"] == "no_measurements"
    assert "run_review.py --all" in items[-1]["action"]


# ── readiness ───────────────────────────────────────────────────────────────

def test_readiness_names_the_thing_in_the_way():
    r = advisor.readiness(_patterns(("setting_clause_everywhere", 0.8)))
    assert r["state"] == "needs work"
    assert "location" in r["detail"]


def test_readiness_is_good_when_nothing_recurs():
    r = advisor.readiness(_patterns(), stats={"avg_score": 8.2, "total": 20})
    assert r["state"] == "good"


# ── the page and the button ─────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SETTINGS_FILE", tmp_path / "settings.json")
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def test_the_page_offers_the_button_for_an_applicable_suggestion(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_advice_now", lambda: (
        [{"id": "pictures_held_too_long", "title": "Pictures are held too long",
          "why": "w", "action": "a", "severity": "high", "evidence": "3 of 6",
          "setting": "SD_CLIPS", "value": "24"}],
        {"state": "needs work", "detail": "Pictures are held too long"}))
    page = client.get("/advice").get_data(as_text=True)
    assert "Set SD_CLIPS = 24" in page
    assert "needs work" in page


def test_applying_a_suggestion_writes_the_setting(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_advice_now", lambda: (
        [{"id": "x", "title": "t", "why": "w", "action": "a",
          "severity": "high", "evidence": "e",
          "setting": "SD_CLIPS", "value": "24"}], {"state": "needs work", "detail": "t"}))
    r = client.post("/advice/apply", data={"key": "SD_CLIPS", "value": "24"})
    assert "msg=" in r.headers["Location"]
    saved = json.loads(dashboard.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert saved["SD_CLIPS"] == "24"


def test_applying_something_never_offered_is_refused(client, monkeypatch):
    """The key and value arrive from a form. A POST that could write an
    arbitrary key would be writing arbitrary environment into every run."""
    monkeypatch.setattr(dashboard, "_advice_now", lambda: ([], {"state": "good", "detail": ""}))
    r = client.post("/advice/apply", data={"key": "SD_CLIPS", "value": "999"})
    assert "error=" in r.headers["Location"]
    assert not dashboard.SETTINGS_FILE.exists() or \
        "SD_CLIPS" not in json.loads(dashboard.SETTINGS_FILE.read_text(encoding="utf-8"))


def test_an_unknown_key_is_refused(client):
    r = client.post("/advice/apply", data={"key": "PATH", "value": "/evil"})
    assert "error=" in r.headers["Location"]


def test_advice_is_in_the_nav():
    assert any(href == "/advice" for href, _l, _p in dashboard.NAV_ITEMS)
