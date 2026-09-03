"""Whether the flywheel learns from data it can actually attribute."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _mk(dbm, hook, yt, views, watch, likes):
    vid = dbm.save_video(niche="money_history", script_hook=hook, scene_desc="s",
                         video_file=f"{hook[:6]}.mp4", score=8, channel="main_en")
    with dbm._conn() as c:
        c.execute("UPDATE videos SET youtube_id=? WHERE id=?", (yt, vid))
        c.execute("INSERT INTO metrics (video_id, views, watch_pct, likes) "
                  "VALUES (?,?,?,?)", (vid, views, watch, likes))
    return vid


def test_a_shared_youtube_id_is_kept_out_of_the_learning(tmp_path, monkeypatch):
    """THE DAMAGE THIS UNDOES. Six videos in the owner's real database carried
    the id kGVAHaObJ38 — one link recorded against six different scripts — so
    all six joined to a SEVENTH video's metrics: the same views, the same
    likes, and a watch percentage of zero.

    Zero watch_pct is fatal here specifically, because engagement is
    watch_pct × log(likes) × log(views). Zero times anything is zero, so every
    one of them sorted to the bottom and every one was written into
    losing_hooks. All four entries in that list were these rows. The flywheel
    was not learning that those hooks failed — it was learning that a
    data-entry mistake looks like failure, and feeding that into the hook
    prompt."""
    import db_manager as dbm
    import feedback_analyzer as fa
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "fa.db")
    monkeypatch.setattr(fa, "DB_FILE", tmp_path / "fa.db")
    dbm.init_db()

    # Three honest videos, each with its own link.
    _mk(dbm, "a real winner", "yt_win_0001", 1200, 55.0, 17)
    _mk(dbm, "a middling one", "yt_mid_0001", 400, 40.0, 5)
    _mk(dbm, "a real loser", "yt_los_0001", 120, 12.0, 1)
    # Two sharing a link, both stamped with someone else's numbers.
    _mk(dbm, "wrongly blamed A", "yt_dupe_001", 957, 0.0, 10)
    _mk(dbm, "wrongly blamed B", "yt_dupe_001", 957, 0.0, 10)

    class Ch:
        id = "main_en"
        learnings_path = tmp_path / "learnings.json"
    fa._analyze_channel(Ch())

    import json
    out = json.loads(Ch.learnings_path.read_text(encoding="utf-8"))
    every = out["winning_hooks"] + out["losing_hooks"]
    assert "wrongly blamed A" not in every
    assert "wrongly blamed B" not in every
    assert out["total_videos_analyzed"] == 3, "only the attributable rows vote"
    assert "a real loser" in out["losing_hooks"]


def test_the_exclusion_is_announced_rather_than_silent(tmp_path, monkeypatch, capsys):
    """A silently smaller sample is how this went unnoticed for a week. The
    rows are still in the database and still fixable — the owner clears the
    wrong link and they rejoin the learning."""
    import db_manager as dbm
    import feedback_analyzer as fa
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "loud.db")
    monkeypatch.setattr(fa, "DB_FILE", tmp_path / "loud.db")
    dbm.init_db()
    for i in range(3):
        _mk(dbm, f"clean {i}", f"yt_clean_{i:03}", 300 + i, 40.0, 4)
    _mk(dbm, "dupe A", "yt_same_0001", 957, 0.0, 10)
    _mk(dbm, "dupe B", "yt_same_0001", 957, 0.0, 10)

    class Ch:
        id = "main_en"
        learnings_path = tmp_path / "l.json"
    fa._analyze_channel(Ch())

    printed = capsys.readouterr().out
    assert "yt_same_0001" in printed, "it has to name the id"
    assert "2 video(s) excluded" in printed
    assert "rejoin the learning" in printed, "and say the state is recoverable"


def test_a_video_with_no_link_at_all_still_counts(tmp_path, monkeypatch):
    """Absent is not ambiguous. A row with no youtube_id has no metrics to be
    confused about, and excluding it would shrink the sample for no reason."""
    import db_manager as dbm
    import feedback_analyzer as fa
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "none.db")
    monkeypatch.setattr(fa, "DB_FILE", tmp_path / "none.db")
    dbm.init_db()
    for i in range(3):
        _mk(dbm, f"unlinked {i}", "", 100 + i * 50, 30.0 + i, 2)

    class Ch:
        id = "main_en"
        learnings_path = tmp_path / "l.json"
    fa._analyze_channel(Ch())
    import json
    out = json.loads(Ch.learnings_path.read_text(encoding="utf-8"))
    assert out["total_videos_analyzed"] == 3
