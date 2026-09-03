"""Writing a video's script again, without rebuilding the video.

A run produces one script and thirteen minutes of GPU on top of it. When the
SCRIPT is what is wrong, the only lever was to throw the whole run away — which
costs the pictures too, and takes long enough that it does not get used. So a
script nobody was happy with went out anyway.

The two properties worth defending here:

  1. A candidate never replaces anything until someone picks it. Overwriting
     the live script on the way to asking whether the new one is better loses
     the original at the exact moment you need it for comparison.
  2. Pressing it twice gives two different scripts. Otherwise it is not a
     regenerate button, it is a slot machine that keeps landing on the fruit.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import rewrite  # noqa: E402


@pytest.fixture
def video(tmp_path, monkeypatch):
    import paths
    root = tmp_path / "debug"
    (root / "run-1").mkdir(parents=True)
    monkeypatch.setattr(paths, "debug_root", lambda: root)
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    vid = db_manager.save_video(
        niche="money_history", script_hook="the old hook",
        scene_desc="A merchant weighs silver in 1550 Seville.",
        video_file="a.mp4", score=4, run_id="run-1",
        script_full="the old script",
        seed_type="wikipedia", seed_source="Wikipedia",
        seed_content="The price revolution followed American silver.",
        seed_url="https://en.wikipedia.org/wiki/Price_revolution")
    return {"id": vid, "run": root / "run-1"}


def _writer(monkeypatch, script="a brand new script", score=8, calls=None):
    import script_writer

    def _fake(scene, seed=None, **kw):
        if calls is not None:
            calls.append(scene)
        return {"script": script, "score": score,
                "criterion_scores": {"specificity": 3, "hook": 2},
                "reasoning": "SPECIFICITY: 3 — uses the source's own figures.",
                "attempts_used": 1, "cost_usd": 0.012}

    monkeypatch.setattr(script_writer, "write_script_until_good", _fake)


# ── the candidate ────────────────────────────────────────────────────────────

def test_a_candidate_is_written_beside_the_run(video, monkeypatch):
    _writer(monkeypatch)
    got = rewrite.propose(video["id"])
    assert got["ok"] is True
    assert got["score"] == 8
    assert (video["run"] / "rewrite.json").exists()


def test_the_live_script_is_not_touched(video, monkeypatch):
    """THE PROPERTY. You cannot compare two scripts if writing the second one
    destroyed the first."""
    _writer(monkeypatch)
    rewrite.propose(video["id"])
    assert db_manager.video_by_id(video["id"])["script_full"] == "the old script"


def test_the_video_row_is_not_touched_at_all(video, monkeypatch):
    _writer(monkeypatch)
    before = db_manager.video_by_id(video["id"])
    rewrite.propose(video["id"])
    assert db_manager.video_by_id(video["id"]) == before


def test_the_candidate_is_read_back_whole(video, monkeypatch):
    _writer(monkeypatch)
    rewrite.propose(video["id"])
    cand = rewrite.latest("run-1")
    assert cand["script"] == "a brand new script"
    assert cand["criterion_scores"]["specificity"] == 3
    assert "SPECIFICITY" in cand["reasoning"]


def test_discarding_removes_it(video, monkeypatch):
    _writer(monkeypatch)
    rewrite.propose(video["id"])
    assert rewrite.discard("run-1") is True
    assert rewrite.latest("run-1") is None


# ── pressing it twice ────────────────────────────────────────────────────────

def test_the_second_press_is_told_what_the_first_produced(video, monkeypatch):
    """write_script_until_good already feeds a rejected attempt forward. This
    does the same with the candidate, so two presses cannot return the same
    text for the same reason."""
    calls = []
    _writer(monkeypatch, script="first version", calls=calls)
    rewrite.propose(video["id"])
    _writer(monkeypatch, script="second version", calls=calls)
    rewrite.propose(video["id"])
    assert "first version" in calls[1]
    assert "Do not repeat this angle" in calls[1]


def test_the_first_press_is_not_given_a_phantom_previous(video, monkeypatch):
    calls = []
    _writer(monkeypatch, calls=calls)
    rewrite.propose(video["id"])
    assert "Do not repeat" not in calls[0]


def test_the_original_scene_survives_the_do_not_repeat_note(video, monkeypatch):
    calls = []
    _writer(monkeypatch, script="v1", calls=calls)
    rewrite.propose(video["id"])
    _writer(monkeypatch, script="v2", calls=calls)
    rewrite.propose(video["id"])
    assert "1550 Seville" in calls[1], "the rewrite lost its own subject"


# ── the source it writes against ─────────────────────────────────────────────

def test_the_seed_is_rebuilt_from_the_row(video, monkeypatch):
    """Without the seed the writer invents a fresh subject and the fact gate
    has nothing to check against — which is exactly how four videos scored
    specificity 0/3 last week."""
    import script_writer
    seen = {}

    def _fake(scene, seed=None, **kw):
        seen["seed"] = seed
        return {"script": "s", "score": 7, "criterion_scores": {},
                "reasoning": "", "attempts_used": 1, "cost_usd": 0.0}

    monkeypatch.setattr(script_writer, "write_script_until_good", _fake)
    rewrite.propose(video["id"])
    assert seen["seed"]["type"] == "wikipedia"
    assert "price revolution" in seen["seed"]["content"].lower()


def test_a_run_that_saved_no_seed_says_so_and_still_writes(video, monkeypatch, capsys):
    with db_manager._conn() as c:
        c.execute("UPDATE videos SET seed_content='' WHERE id=?", (video["id"],))
    _writer(monkeypatch)
    assert rewrite.propose(video["id"])["ok"] is True
    assert "no source to ground itself" in capsys.readouterr().out


# ── refusals ─────────────────────────────────────────────────────────────────

def test_a_video_with_no_scene_description_cannot_be_rewritten(video, monkeypatch):
    with db_manager._conn() as c:
        c.execute("UPDATE videos SET scene_desc='' WHERE id=?", (video["id"],))
    got = rewrite.propose(video["id"])
    assert got["ok"] is False and "scene description" in got["why"]


def test_a_video_with_no_run_id_has_nowhere_to_put_it(video, monkeypatch):
    vid = db_manager.save_video(niche="x", script_hook="h", scene_desc="s",
                                video_file="a.mp4", score=8)
    got = rewrite.propose(vid)
    assert got["ok"] is False and "run_id" in got["why"]


def test_an_unknown_video_is_a_reason_not_a_traceback(video):
    got = rewrite.propose(999999)
    assert got["ok"] is False and "no video" in got["why"]


def test_a_writer_that_raises_does_not_take_the_page_with_it(video, monkeypatch):
    import script_writer
    monkeypatch.setattr(script_writer, "write_script_until_good",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    got = rewrite.propose(video["id"])
    assert got["ok"] is False and "no key" in got["why"]


def test_an_empty_script_is_not_offered_as_a_candidate(video, monkeypatch):
    _writer(monkeypatch, script="   ")
    got = rewrite.propose(video["id"])
    assert got["ok"] is False
    assert rewrite.latest("run-1") is None


def test_a_corrupt_candidate_file_reads_as_none(video):
    (video["run"] / "rewrite.json").write_text("{ broken", encoding="utf-8")
    assert rewrite.latest("run-1") is None
