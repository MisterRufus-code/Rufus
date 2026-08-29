"""Taking everything you have made out of Rufus.

Somebody who buys software is entitled to leave with their work — and not as a
copy of rufus.db, which is a proprietary schema readable by this program and
nothing else, a hostage dressed as a backup. It is also the honest half of the
argument for keeping the source closed: a product that will not let you take
your data out is relying on the export being hard.

The test that matters most here is the one about preference pairs. A pair is
not a row anywhere in the database — it is one chosen sibling and the ones it
was chosen over, implied by a grouping. Dumping three tables verbatim would
leave the reader to rediscover that, and this project's own account of itself
is that those pairs are the product.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import export_data  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "rufus.db")
    db_manager.init_db()
    return db_manager


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_an_export_of_an_empty_channel_still_produces_every_file(db, tmp_path):
    """An empty CSV with its header is a true answer — "you have no voice
    takes". A missing file is ambiguous and reads as a failed export."""
    dest = export_data.export(tmp_path / "out")
    for name in ("videos.csv", "decisions.csv", "projects.csv", "spend.csv",
                 "README.txt"):
        assert (dest / name).exists(), name
    assert _read(dest / "videos.csv") == []
    with (dest / "videos.csv").open(encoding="utf-8") as fh:
        assert "score" in fh.readline(), "the header is what makes empty legible"


def test_videos_come_out_with_what_a_person_would_want(db, tmp_path):
    vid = db.save_video(niche="money_history", script_hook="A hook.",
                        scene_desc="s", video_file="a.mp4", score=8,
                        title="The Panic of 1893")
    # A real eleven-character id: mark_published validates the shape, so a
    # made-up short string is silently refused and the row stays unpublished.
    assert db.mark_published(vid, "dQw4w9WgXcQ")
    rows = _read(export_data.export(tmp_path / "o") / "videos.csv")
    assert len(rows) == 1
    assert rows[0]["title"] == "The Panic of 1893"
    assert rows[0]["youtube_id"] == "dQw4w9WgXcQ"
    assert rows[0]["score"] == "8"


# ── the file worth keeping ──────────────────────────────────────────────────

def test_a_script_choice_becomes_a_pair_not_three_rows(db, tmp_path):
    """A PREFERENCE PAIR IS NOT A ROW. It is one chosen sibling and its
    rejected siblings, grouped by whatever made them siblings — and the pairing
    is implied rather than stored. Reassembling it is the work an export exists
    to do rather than leave to whoever opens the file."""
    ids = [db.save_candidate(proposal_id=1, channel="c", niche="n",
                             topic="Tulips", hook_style=style, hook=f"H{i}",
                             script=f"S{i}", score=7 + i)
           for i, style in enumerate(["warning", "shocking_stat",
                                      "counterintuitive"])]
    db.choose_candidate(ids[1], by="daniel")

    rows = _read(export_data.export(tmp_path / "o") / "decisions.csv")
    assert len(rows) == 2, "one chosen against two passed over is two pairs"
    assert {r["kind"] for r in rows} == {"script"}
    assert {r["chosen_id"] for r in rows} == {str(ids[1])}
    assert {r["passed_over_id"] for r in rows} == {str(ids[0]), str(ids[2])}
    assert rows[0]["decided_by"] == "daniel"
    assert rows[0]["chosen_label"] == "shocking_stat"


def test_an_undecided_set_is_not_a_preference_yet(db, tmp_path):
    """Three scripts nobody ruled between say nothing about what anybody
    prefers, and a row claiming otherwise is worse than no row."""
    for style in ("warning", "shocking_stat"):
        db.save_candidate(proposal_id=2, channel="c", niche="n", topic="T",
                          hook_style=style, hook="h", script="s", score=8)
    assert _read(export_data.export(tmp_path / "o") / "decisions.csv") == []


def test_a_per_shot_picture_swap_is_one_pair_per_shot(db, tmp_path):
    """The whole reason the gallery is judged shot by shot: A wins shot 3 and
    B wins shot 9, so picking a bundle throws away the good half of the other.
    That is one labelled pair per shot, not one per video."""
    sid = db.save_gallery_set(candidate_id=1, channel="c", niche="n",
                              topic="Rome", script_file="s.txt", n_variants=2)
    for beat in range(3):
        for variant in range(2):
            db.save_gallery_image(set_id=sid, variant=variant, beat_index=beat,
                                  path=f"/n/{variant}_{beat}.png",
                                  prompt=f"shot {beat}", seed=1)
    db.choose_gallery_base(sid, 0, by="daniel")
    db.swap_gallery_beat(sid, 1, 1, by="daniel")

    rows = _read(export_data.export(tmp_path / "o") / "decisions.csv")
    picture_rows = [r for r in rows if r["kind"] == "picture"]
    assert len(picture_rows) == 3, "three shots, one pair each"
    swapped = next(r for r in picture_rows if "shot:2" in r["group"])
    assert swapped["chosen_label"] == "variant 1"
    assert swapped["passed_over_label"] == "variant 0"


def test_the_three_kinds_of_choice_are_one_file_and_labelled(db, tmp_path):
    """Script, picture and read are different decisions and the same SHAPE.
    One file with a kind column beats three files somebody has to join."""
    ids = [db.save_candidate(proposal_id=3, channel="c", niche="n", topic="T",
                             hook_style=s, hook="h", script="s", score=8)
           for s in ("warning", "shocking_stat")]
    db.choose_candidate(ids[0])
    sid = db.save_gallery_set(candidate_id=ids[0], channel="c", niche="n",
                              topic="T", script_file="s.txt", n_variants=2)
    for variant in range(2):
        db.save_gallery_image(set_id=sid, variant=variant, beat_index=0,
                              path=f"/n/{variant}.png", prompt="p", seed=1)
    db.choose_gallery_base(sid, 0)
    takes = [db.save_voice_take(set_id=sid, channel="c", topic="T", tone=t,
                                text="t", path=f"/n/{t}.mp3")
             for t in ("curiosity", "tension")]
    db.choose_voice_take(takes[0])

    rows = _read(export_data.export(tmp_path / "o") / "decisions.csv")
    assert {r["kind"] for r in rows} == {"script", "picture", "voice"}


def test_the_scripts_come_out_as_files_a_person_can_read(db, tmp_path):
    """A script is something somebody reads. Buried in a CSV cell with its
    newlines escaped, it is not."""
    cid = db.save_candidate(proposal_id=4, channel="c", niche="n",
                            topic="The Panic of 1893", hook_style="warning",
                            hook="A hook.", script="A hook.\nAnd a body.",
                            score=9)
    db.choose_candidate(cid)
    dest = export_data.export(tmp_path / "o")
    files = list((dest / "scripts").glob("*.txt"))
    assert len(files) == 1
    assert "the-panic-of-1893" in files[0].name
    assert files[0].read_text(encoding="utf-8") == "A hook.\nAnd a body."


# ── what must not come out ──────────────────────────────────────────────────

def test_no_credential_leaves_with_the_export(db, tmp_path, monkeypatch):
    """An export gets emailed, uploaded to a spreadsheet service and left in a
    downloads folder. Sign-in tokens are not the owner's data, they are the
    keys to the building."""
    db.save_video(niche="n", script_hook="h", scene_desc="s",
                  video_file="a.mp4", score=8)
    dest = export_data.export(tmp_path / "o")
    for path in dest.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            assert "rufus_auth" not in text
            assert "sk-" not in text
    assert not (dest / "users.csv").exists()
    assert not (dest / "keys.json").exists()


def test_the_readme_says_what_was_left_out(db, tmp_path):
    """Somebody about to hand this to a stranger is owed that list without
    auditing the folder themselves."""
    text = (export_data.export(tmp_path / "o") / "README.txt").read_text(
        encoding="utf-8")
    assert "WHAT IS NOT IN HERE" in text
    assert "token" in text.lower()


def test_a_missing_table_costs_that_file_and_not_the_export(db, tmp_path,
                                                            monkeypatch):
    """A database from an older version is missing tables a newer one writes,
    and somebody exporting is usually about to stop using the program — the
    worst possible moment to hand them an exception instead of their data."""
    import sqlite3
    with sqlite3.connect(str(db.DB_FILE)) as c:
        c.execute("DROP TABLE IF EXISTS voice_takes")
    dest = export_data.export(tmp_path / "o")
    assert (dest / "videos.csv").exists()
    assert (dest / "decisions.csv").exists()


def test_nothing_here_needs_rufus_to_read_it(db, tmp_path):
    """The point of the exercise. Every file opens in a spreadsheet or a text
    editor; none of it is a database."""
    dest = export_data.export(tmp_path / "o")
    for path in dest.rglob("*"):
        if path.is_file():
            assert path.suffix in (".csv", ".txt", ".json"), path.name
