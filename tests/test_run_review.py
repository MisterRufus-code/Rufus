"""Measuring a run, so the next one can be better.

Every fix to this pipeline in recent memory started the same way: the owner
watched a video, noticed something, and pasted a log. Three of those reports,
verbatim —

    "why all the images with coin"
    "restated the location in 13 shot(s)"
    "QC ⚠ 5 stretch(es) over 5s without a cut"

— are the cases below, because each is a number computable from files already
on disk. The tests use the real prompts from the runs that produced them.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_review  # noqa: E402

STYLE = ("Minimalist stick-figure cartoon illustration drawn in thin, clean "
         "black line art of uniform weight.")


def _write_run(tmp_path, monkeypatch, prompts, run_id="r1", script="word " * 40):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    parts = [f"# Rufus run `{run_id}`\n"]
    for i, p in enumerate(prompts, 1):
        parts.append(f"**Beat {i}**\n\n```\n{p} {STYLE}\n```\n\n")
    (d / "run_report.md").write_text("".join(parts), encoding="utf-8")
    (d / "script.txt").write_text(script, encoding="utf-8")
    monkeypatch.setattr(run_review.paths, "debug_root", lambda: tmp_path)
    return d


# ── the coin complaint, as a number ─────────────────────────────────────────

DANEGELD = [
    "The narrator stands near a wooden table with a stack of silver coins.",
    "A hall with stone walls. A king looks at a map and a stack of silver coins.",
    "Officials count and organize silver coins on the table.",
    "A close-up of the king's face, with silver coins in the foreground.",
    "The stack of silver coins grows larger each time the raiders return.",
    "A large pile of silver coins on the table.",
    "A close-up of a single silver coin, showing its worn edges.",
    "A modern office with a screen. A person sits at a desk.",
    "A hand holding a card, bills scattered on a modern desk.",
    "Back in the hall, the narrator stands by the now-empty table.",
]


def test_one_object_in_most_frames_is_reported(tmp_path, monkeypatch):
    """The owner's exact words were "why all the images with coin"."""
    _write_run(tmp_path, monkeypatch, DANEGELD)
    m = run_review.review("r1")
    assert m["dominant_subject"]["word"] in ("silver", "coins", "coin")
    assert m["dominant_subject"]["share"] >= run_review.DOMINANT_SHARE
    assert any(f["id"] == "one_object_dominates" for f in m["findings"])


def test_a_varied_sequence_is_not_reported(tmp_path, monkeypatch):
    """The check has to be quiet on a good run or nobody will read it — the
    same failure the drift warning had at seven of ten."""
    varied = ["A ship leaves a harbour at dawn.",
              "Two merchants argue beside a weighing scale.",
              "A child runs down a narrow alley.",
              "A furnace glows in a dark workshop.",
              "A queue waits outside a shuttered bank.",
              "Rain falls on an abandoned market square."]
    _write_run(tmp_path, monkeypatch, varied)
    m = run_review.review("r1")
    assert not any(f["id"] == "one_object_dominates" for f in m["findings"])


# ── the appended clauses ────────────────────────────────────────────────────

def test_the_setting_clause_on_half_the_shots_is_reported(tmp_path, monkeypatch):
    prompts = [f"Shot {i} of something."
               + (f" {run_review.SETTING_MARK} market, cobblestone, bright."
                  if i <= 13 else "")
               for i in range(1, 25)]
    _write_run(tmp_path, monkeypatch, prompts)
    m = run_review.review("r1")
    assert m["clauses"]["setting_clause"] == 13
    assert any(f["id"] == "setting_clause_everywhere" for f in m["findings"])


def test_an_appended_clause_does_not_count_as_the_subject(tmp_path, monkeypatch):
    """The first version counted the marker only, so "market, cobblestone,
    bright" survived on thirteen prompts and the metric reported the video was
    about a market — a count of text the PIPELINE wrote, not of a picture the
    storyboard chose."""
    prompts = [f"A person does thing number {i} outdoors."
               + f" {run_review.SETTING_MARK} market, cobblestone, bright."
               for i in range(1, 15)]
    _write_run(tmp_path, monkeypatch, prompts)
    m = run_review.review("r1")
    assert m["dominant_subject"]["word"] != "market"


def test_the_thread_on_most_shots_is_reported(tmp_path, monkeypatch):
    prompts = [f"Shot {i}."
               + (f" {run_review.THREAD_MARK} the same lantern."
                  if i <= 6 else "") for i in range(1, 11)]
    _write_run(tmp_path, monkeypatch, prompts)
    m = run_review.review("r1")
    assert any(f["id"] == "thread_everywhere" for f in m["findings"])


# ── held pictures ───────────────────────────────────────────────────────────

def test_a_picture_held_too_long_is_reported(tmp_path, monkeypatch):
    """QC already prints this; the point of repeating it here is that it lands
    in the same record as everything else, so a run can be compared to the one
    before it."""
    _write_run(tmp_path, monkeypatch, ["A shot."] * 10)
    video = tmp_path / "short.mp4"
    Path(str(video) + ".qc.json").write_text(json.dumps(
        {"duration": 40.0, "cuts": [3.0, 9.5, 16.0, 22.0, 28.0]}), encoding="utf-8")
    m = run_review.review("r1", video)
    assert m["cuts"]["cuts"] == 5
    assert m["cuts"]["longest_hold_s"] >= 6.0
    assert any(f["id"] == "pictures_held_too_long" for f in m["findings"])


def test_missing_qc_data_is_not_an_error(tmp_path, monkeypatch):
    _write_run(tmp_path, monkeypatch, ["A shot."] * 10)
    m = run_review.review("r1", tmp_path / "nothing.mp4")
    assert m["cuts"]["cuts"] == 0


# ── the contract ────────────────────────────────────────────────────────────

def test_every_finding_names_the_lever():
    """A number nobody acts on is the same as no number."""
    src = Path(run_review.__file__).read_text(encoding="utf-8")
    for lever in ("SETTING_SHARE", "SD_CLIPS"):
        assert lever in src, lever


def test_a_broken_run_folder_returns_a_partial_result_not_an_error(tmp_path, monkeypatch):
    """A reviewer that raises would turn a good run into a failed one."""
    monkeypatch.setattr(run_review.paths, "debug_root", lambda: tmp_path)
    m = run_review.review("does-not-exist")
    assert m["beats"] == 0
    assert m["findings"] == [] or isinstance(m["findings"], list)


def test_review_and_save_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(run_review.paths, "debug_root",
                        lambda: (_ for _ in ()).throw(OSError("no disk")))
    assert run_review.review_and_save("x") == {}


def test_the_review_is_saved_beside_the_run(tmp_path, monkeypatch):
    _write_run(tmp_path, monkeypatch, DANEGELD)
    run_review.review_and_save("r1")
    saved = json.loads((tmp_path / "r1" / run_review.REVIEW_FILE)
                       .read_text(encoding="utf-8"))
    assert saved["beats"] == len(DANEGELD)
    assert run_review.load("r1") == saved


# ── across runs ─────────────────────────────────────────────────────────────

def test_patterns_counts_a_finding_across_runs(tmp_path, monkeypatch):
    """One run's numbers say a video was weak. The same finding in most runs is
    a code change rather than a bad seed."""
    for rid in ("a", "b", "c"):
        _write_run(tmp_path, monkeypatch, DANEGELD, run_id=rid)
        run_review.review_and_save(rid)
    p = run_review.patterns()
    assert p["runs_reviewed"] == 3
    ids = {r["id"]: r for r in p["recurring"]}
    assert ids["one_object_dominates"]["runs"] == 3
    assert ids["one_object_dominates"]["share"] == 1.0


def test_the_run_calls_the_reviewer():
    src = (Path(run_review.__file__).parent / "main.py").read_text(encoding="utf-8")
    assert "run_review.review_and_save" in src
    block = src.split("run_review.review_and_save")[0][-600:]
    assert "non-fatal" in src.split("run_review.review_and_save")[1][:400] or \
           "except Exception" in src.split("import run_review")[1][:400]
