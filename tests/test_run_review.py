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


# ── what the first pass over fifty real runs exposed ────────────────────────
# Running this against the owner's whole debug folder was the first time the
# instrument was pointed at more than one run, and it reported four things
# about itself.

OLD_NICHE_STYLE = (
    " flat 2D vector illustration, warm sepia and antique-gold color palette, "
    "bold clean outlines of consistent weight, simplified flat color fills "
    "with no gradients, storybook illustration style, no photographic "
    "texture, no film grain, no lens blur or depth of field, consistent "
    "hairstyles across every modern figure")

VARIED = ["A ship leaves a harbour at dawn.",
          "Two merchants argue beside a weighing scale.",
          "A child runs down a narrow alley.",
          "A furnace glows in a dark workshop.",
          "A queue waits outside a shuttered bank.",
          "Rain falls on an abandoned market square.",
          "A clerk stamps a ledger with a wooden press.",
          "Snow settles on a silent quayside.",
          "A lamp is lit in an upstairs window."]


def _write_styled(tmp_path, monkeypatch, beats, style, run_id="r1", words=36):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    parts = [f"# run\n"]
    for i, b in enumerate(beats, 1):
        parts.append(f"**Beat {i}**\n\n```\n{b}{style}\n```\n\n")
    (d / "run_report.md").write_text("".join(parts), encoding="utf-8")
    (d / "script.txt").write_text("word " * words, encoding="utf-8")
    monkeypatch.setattr(run_review.paths, "debug_root", lambda: tmp_path)
    return d


def test_a_style_suffix_is_never_the_subject(tmp_path, monkeypatch):
    """Across fifty real runs this reported nine of nine prompts about "lens"
    (from "no lens blur"), ten of ten about "hairstyles", nine of nine about
    "modern" — three confident findings, all of them a description of the
    style block. Splitting on the three style names this repo ships missed
    every older run using a niche's own style_suffix."""
    _write_styled(tmp_path, monkeypatch, VARIED, OLD_NICHE_STYLE)
    m = run_review.review("r1")
    assert m["dominant_subject"]["word"] not in ("lens", "hairstyles", "modern")
    assert not any(f["id"] == "one_object_dominates" for f in m["findings"])


def test_the_style_block_is_found_as_the_shared_tail():
    """Found this way rather than by matching its opening words, so it
    survives a style nobody has written yet.

    Beats that happen to END alike would let the shared tail absorb that too,
    which is harmless — it can only ever strip MORE boilerplate, never a word
    unique to one shot."""
    prompts = [b + OLD_NICHE_STYLE for b in VARIED[:3]]
    tail = run_review._common_suffix(prompts)
    # The shared full stop that ends every beat comes along with it. Harmless:
    # the tail is only ever used to be REMOVED before counting words.
    assert tail.lstrip(". ").startswith("flat 2D vector")
    assert "lens" in tail and "hairstyles" in tail


def test_a_shared_tail_too_short_to_be_a_style_is_ignored():
    """Two prompts ending in the same few characters share punctuation, not a
    style block."""
    assert run_review._common_suffix(["A coin on a table.",
                                      "A ship on a table."]) == ""


def test_a_single_picture_run_cannot_report_dominance(tmp_path, monkeypatch):
    """One prompt is trivially 100% of one prompt. A live pass reported
    "numbers appears in 1 of 1 prompts" as a dominance failure."""
    _write_styled(tmp_path, monkeypatch, ["Rising numbers on a chart."], "")
    m = run_review.review("r1")
    assert not any(f["id"] == "one_object_dominates" for f in m["findings"])


def test_few_pictures_is_measured_against_the_script(tmp_path, monkeypatch):
    """The absolute version fired on 48 of 50 real runs — one fact about the
    past, repeated forty-eight times. By this module's own standard that is
    noise, and noise is what people learn to scroll past."""
    # A short script legitimately gets few pictures.
    _write_styled(tmp_path, monkeypatch, VARIED, OLD_NICHE_STYLE, words=36)
    assert not any(f["id"] == "few_pictures"
                   for f in run_review.review("r1")["findings"])
    # A long one with the same nine does not.
    _write_styled(tmp_path, monkeypatch, VARIED, OLD_NICHE_STYLE,
                  run_id="r2", words=110)
    long_run = run_review.review("r2")
    assert any(f["id"] == "few_pictures" for f in long_run["findings"])
    assert "110-word" in next(f["text"] for f in long_run["findings"]
                              if f["id"] == "few_pictures")


def test_sub_frames_of_one_beat_are_not_duplicates(tmp_path, monkeypatch):
    """In cut mode a beat is saved as 07.png, 07a.png, 07b.png — the same
    scene a moment apart, on one seed, near-identical BY DESIGN. Counting them
    reported 73 duplicate pairs on a run whose whole point was that the shot
    advances inside the beat."""
    d = _write_styled(tmp_path, monkeypatch, VARIED, OLD_NICHE_STYLE)
    for i in range(1, 10):
        for suffix in ("", "a", "b"):
            (d / f"{i:02d}{suffix}.png").write_bytes(b"x")
    frames = run_review._keyframes(d)
    assert len(frames) == 9, [f.name for f in frames]
    assert all(f.stem.isdigit() for f in frames)


def test_duplicates_are_reported_as_frames_not_pairs():
    """Pairs grow with the square of the sequence, so "73 pairs" says nothing
    a reader can picture: five identical frames among forty is ten pairs, and
    five among fifteen is also ten."""
    src = Path(run_review.__file__).read_text(encoding="utf-8")
    assert "duplicate_share" in src
    assert "have a near-identical twin" in src


def test_the_all_summary_covers_everything_it_measured():
    """The first pass reviewed fifty-five runs and reported patterns across
    twenty, which reads as most of the work having been discarded."""
    src = Path(run_review.__file__).read_text(encoding="utf-8")
    assert "patterns(limit=len(ids))" in src


def test_a_deliberate_tone_hold_is_not_a_defect(tmp_path, monkeypatch):
    """The cut planner now stretches a revelation to about 4.4s on a
    38-second video. Reporting that as "held too long" would be the
    measurement contradicting the feature — the same mistake as counting a
    beat's own sub-frames as duplicates."""
    _write_run(tmp_path, monkeypatch, ["A shot."] * 9)
    video = tmp_path / "s.mp4"
    # 2.2s neutrals with a 4.4s revelation in the middle: the intended shape.
    cuts, at = [], 0.0
    for gap in (2.2, 2.2, 2.75, 2.2, 4.4, 3.85, 2.2, 2.2):
        at += gap
        cuts.append(round(at, 2))
    Path(str(video) + ".qc.json").write_text(json.dumps(
        {"duration": round(at + 3.35, 2), "cuts": cuts}), encoding="utf-8")
    m = run_review.review("r1", video)
    assert not any(f["id"] == "pictures_held_too_long" for f in m["findings"]), \
        m["cuts"]


def test_a_genuine_stall_is_still_reported(tmp_path, monkeypatch):
    """Past the line QC already draws, a hold stops reading as emphasis."""
    _write_run(tmp_path, monkeypatch, ["A shot."] * 5)
    video = tmp_path / "s.mp4"
    Path(str(video) + ".qc.json").write_text(json.dumps(
        {"duration": 40.0, "cuts": [3.0, 9.5, 16.0, 22.0]}), encoding="utf-8")
    m = run_review.review("r1", video)
    assert any(f["id"] == "pictures_held_too_long" for f in m["findings"])


def test_all_really_means_all(tmp_path, monkeypatch):
    """--all measured sixty of the owner's eighty-six runs and said nothing
    about the other twenty-six. A silent cap is the one thing this repo treats
    as always a bug."""
    for i in range(65):
        (tmp_path / f"run{i:03d}").mkdir()
    monkeypatch.setattr(run_review.paths, "debug_root", lambda: tmp_path)
    assert len(run_review.all_run_ids()) == 60, "the dashboard's cap still holds"
    assert len(run_review.all_run_ids(limit=None)) == 65

    src = Path(run_review.__file__).read_text(encoding="utf-8")
    # The last mention of --all is the branch that acts on it; the first is
    # the line that strips it out of argv.
    assert "all_run_ids(limit=None)" in src.rsplit('"--all"', 1)[1]
