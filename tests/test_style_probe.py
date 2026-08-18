"""Seeing a style edit without paying for a video to find out.

THE COST THIS REMOVES. Every style change this week was evaluated by running
the whole pipeline — script, voiceover, storyboard, stills, render — thirteen
minutes to answer a question about six sentences of text. And each run wrote a
different script on different seeds, so two galleries differed for reasons
unrelated to the edit. One conclusion had to be walked back because of exactly
that.

So the probe fixes everything except the style block: same scenes, same seeds,
same workflow. When two runs differ, the text is the only thing that could
have done it — and the tests below are mostly about that property holding,
because it is the entire value of the tool.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import style_probe  # noqa: E402
import workflow_bench as bench  # noqa: E402


# ── the control ──────────────────────────────────────────────────────────────

def test_the_probes_are_the_bench_s_probes_not_a_second_list():
    """A hand-copied scene list drifts from the bench's within a month, and
    then two tools disagree about what a face test is. This repo has paid for
    duplicated constants before."""
    assert style_probe.selected(None) == list(bench.PROBES)


def test_the_seeds_come_from_the_bench_too():
    """The seeds are the control. If this file grew its own, a probe run and a
    bench run would be comparing different noise."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    assert "bench._PROBE_SEEDS[name]" in src
    assert "_PROBE_SEEDS = " not in src, "the probe defined its own seeds"


def test_a_subset_keeps_the_bench_s_own_order():
    got = style_probe.selected("crowd,face")
    assert [n for n, _ in got] == ["face", "crowd"]


def test_a_typo_in_probes_is_named_rather_than_ignored(capsys):
    """Silently running all six after a typo wastes the minutes this exists to
    save, and does it while looking like it worked."""
    assert style_probe.selected("fce,crowd") == []
    out = capsys.readouterr().out
    assert "no such probe" in out
    assert "face" in out, "it has to say what the real names are"


# ── the style under test ─────────────────────────────────────────────────────

def test_a_named_style_is_the_one_that_gets_rendered(monkeypatch):
    import comfy_client
    monkeypatch.setattr(comfy_client, "style_presets",
                        lambda: {"stickman": "STICK", "storybook": "SOFT"})
    assert style_probe.style_text("storybook", False) == ("storybook", "SOFT")


def test_an_unknown_style_lists_the_real_ones(monkeypatch, capsys):
    import comfy_client
    monkeypatch.setattr(comfy_client, "style_presets",
                        lambda: {"stickman": "STICK"})
    label, tail = style_probe.style_text("stikman", False)
    assert label == ""
    assert "stickman" in capsys.readouterr().out


def test_plain_renders_the_scene_with_no_style_at_all():
    """The comparison that says what the block is contributing at all."""
    assert style_probe.style_text(None, True) == ("plain", "")


# ── the diff, which is the point ─────────────────────────────────────────────

def _run(style_text, shas):
    return {"style_text": style_text,
            "renders": {k: {"sha": v} for k, v in shas.items()}}


def test_it_reports_which_pictures_changed():
    before = _run("A. B.", {"face": "aaa", "crowd": "bbb"})
    now = _run("A. C.", {"face": "aaa", "crowd": "zzz"})
    said = " ".join(style_probe.compare(now, before))
    assert "pictures changed: crowd" in said
    assert "pictures identical: face" in said


def test_it_shows_which_sentences_of_the_style_changed():
    before = _run("The sky is blue. The ground is green.", {"face": "a"})
    now = _run("The sky is blue. The ground is brown.", {"face": "b"})
    said = "\n".join(style_probe.compare(now, before))
    assert "sentence(s) differ" in said
    assert "brown" in said


def test_an_unchanged_style_that_moved_the_pictures_is_called_out():
    """THE TRAP THIS TOOL EXISTS TO CLOSE. Identical prompts and identical
    seeds should give identical pixels. If they did not, the difference is the
    model's own variance and reading it as the effect of an edit is how a
    conclusion gets walked back."""
    before = _run("same text", {"face": "aaa"})
    now = _run("same text", {"face": "bbb"})
    said = " ".join(style_probe.compare(now, before))
    assert "identical" in said
    assert "variance" in said


def test_the_first_run_has_nothing_to_compare_against():
    """Diffing a whole style block against an absent run is true and useless.
    run() says "first run" in words instead."""
    assert style_probe.compare(_run("x", {"face": "a"}), {}) == []


# ── the record on disk ───────────────────────────────────────────────────────

def test_a_run_is_listed_with_its_style(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(style_probe, "probe_root", lambda: tmp_path)
    d = tmp_path / "20260818_120000"
    d.mkdir()
    (d / style_probe.MANIFEST).write_text(json.dumps(
        {"style": "stickman", "renders": {"face": {"ok": True}}}),
        encoding="utf-8")
    assert style_probe.show_runs() == 0
    out = capsys.readouterr().out
    assert "stickman" in out and "1 image" in out


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(style_probe, "probe_root", lambda: tmp_path)
    (tmp_path / "not_a_run").mkdir()
    assert style_probe.runs() == []


def test_a_corrupt_manifest_does_not_raise(tmp_path, monkeypatch):
    d = tmp_path / "20260818_120000"
    d.mkdir()
    (d / style_probe.MANIFEST).write_text("{ broken", encoding="utf-8")
    assert style_probe.load(d) == {}


def test_nothing_yet_says_how_to_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(style_probe, "probe_root", lambda: tmp_path)
    assert style_probe.show_runs() == 1
    assert "style_probe.py" in capsys.readouterr().out


# ── the contract ─────────────────────────────────────────────────────────────

def test_no_comfyui_is_a_reason_not_a_traceback(monkeypatch, capsys):
    import comfy_client, comfy_template
    monkeypatch.setattr(comfy_template, "load_template", lambda p: {"1": {}})
    monkeypatch.setattr(comfy_client, "is_available", lambda: False)
    assert style_probe.run() == 1
    assert "not reachable" in capsys.readouterr().out


def test_no_workflow_says_how_to_export_one(monkeypatch, capsys):
    import comfy_template
    monkeypatch.setattr(comfy_template, "load_template", lambda p: None)
    assert style_probe.run() == 1
    out = capsys.readouterr().out
    assert "Export (API)" in out and "RUFUS_PROMPT" in out
