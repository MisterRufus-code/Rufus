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


# ── where the style came from, which the pictures cannot tell you ────────────
#
# A run of this probe reported style "(default)" and rendered flat-vector
# people in realistic proportions. They were read as a stickman regression.
# Stickman was never rendered: nothing in a fresh terminal sets RUFUS_STYLE,
# the probe fell back to the built-in block, and said so only in line 3 of a
# JSON file nobody opens when the pictures are right there.

def test_the_built_in_fallback_is_named_as_a_fallback(monkeypatch):
    monkeypatch.delenv("RUFUS_STYLE", raising=False)
    assert style_probe.style_source(None, False) == "built-in default"


def test_a_style_typed_for_this_run_is_not_credited_to_the_settings(monkeypatch):
    """--style beats everything, and has to READ as beating everything —
    otherwise the manifest of a one-off experiment claims the channel ships
    it."""
    monkeypatch.setattr(style_probe, "_STYLE_CAME_FROM_SETTINGS", True)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    assert style_probe.style_source("ink_woodcut", False) == "--style"


def test_the_settings_file_and_the_environment_are_told_apart(monkeypatch):
    """Same os.environ either way by the time run() looks. The difference
    matters: "saved settings" means the dashboard and the channel agree;
    "environment" means somebody typed it here and a scheduled run will do
    something else."""
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    monkeypatch.setattr(style_probe, "_STYLE_CAME_FROM_SETTINGS", True)
    assert style_probe.style_source(None, False) == "saved settings"
    monkeypatch.setattr(style_probe, "_STYLE_CAME_FROM_SETTINGS", False)
    assert style_probe.style_source(None, False) == "environment"


def test_an_empty_style_variable_is_still_the_fallback(monkeypatch):
    """`$env:RUFUS_STYLE = ""` sets the name and not the style."""
    monkeypatch.setenv("RUFUS_STYLE", "   ")
    assert style_probe.style_source(None, False) == "built-in default"


def test_plain_is_reported_as_plain(monkeypatch):
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    assert style_probe.style_source(None, True) == "--plain"


def test_the_manifest_records_where_the_style_came_from():
    """probe.json is what a probe is FOR — it is the thing still readable a
    week later, and compare() diffs against it. A label without a provenance
    was how "(default)" got mistaken for the channel's style."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    assert '"style_source": source' in src


def test_the_fallback_is_loud_on_the_way_past(monkeypatch, capsys):
    """Fail-open without fail-loud is fail-silent. Falling back is right —
    the probe should still render — but not quietly, because the output is
    then evidence about a style nobody chose."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    body = src.split("def run(")[1]
    assert 'if source == "built-in default":' in body
    assert "NOT the style this channel ships" in body


# ── the probe composes the prompt the way the pipeline does ──────────────────
#
# THE DRIFT, CAUGHT BY ITS OWN OUTPUT. This loop hand-assembled
# `f"{scene}. {tail}"` under a comment promising "the same composition the
# pipeline builds". Two things had already gone wrong with that copy:
#
#   1. It sent the literal separator line "--- FIGURE ONLY ---" into the
#      prompt. _detail_for_shot strips it; the hand-copy never called
#      _detail_for_shot, so every probe rendered so far had that string in it
#      as prompt text — in a style block whose own rule is that it has no meta
#      level and every word in it is a word in the prompt.
#   2. When _with_detail grew the RUFUS_SHOT_LAST branch, the copy did not.
#      Four probe runs came back as two pairs of byte-identical SHAs while
#      their manifests recorded shot_last true and false. A probe that reports
#      a condition it did not apply is worse than one that cannot apply it.

def test_the_probe_does_not_assemble_its_own_prompt():
    """One composer. A second copy of "scene, then style" cannot be kept in
    step with the first — this file has now proved that twice in one night."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    assert "comfy_client._with_detail(" in src
    assert 'f"{text.rstrip().rstrip(\'.\')}. {tail}"' not in src


def test_the_separator_never_reaches_the_prompt(monkeypatch):
    import comfy_client
    monkeypatch.setenv("RUFUS_STILLS_DETAIL",
                       f"SHARED PART.\n{comfy_client.STYLE_FIGURE_MARKER}\n"
                       f"FIGURE PART.")
    out = comfy_client._with_detail("a figure on a hill")
    assert comfy_client.STYLE_FIGURE_MARKER not in out
    assert "SHARED PART" in out and "FIGURE PART" in out


def test_the_probe_restores_the_style_override_it_borrowed(monkeypatch):
    """RUFUS_STILLS_DETAIL is how the tail is handed to _with_detail. It is
    also a real setting somebody may have set for this shell, and a probe that
    leaves it rewritten would change the next command they run."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    body = src.split("def run(")[1]
    assert "prev_detail" in body
    assert "finally:" in body


# ── a place shot is not a figure shot ────────────────────────────────────────

def test_a_shot_with_no_person_in_it_is_tagged_as_one():
    """THE FIGURE THAT NOBODY ASKED FOR. weather_place is rain, a cave, a
    flooded bank and hills — no person anywhere in the sentence. Untagged,
    shot_kind() reads "figure" and the whole FIGURE ONLY half goes into the
    prompt: oval heads, eyebrow strokes, five separate limbs. The gallery came
    back with a stick figure standing in the middle of the landscape, wearing
    a small blue mouth-shape that poured water down its front — "the mouth of
    a cave" plus a paragraph about faces, arriving together.

    The pipeline tags its beats. A probe that does not is measuring a prompt
    the channel never sends."""
    import comfy_client
    probes = dict(bench.PROBES)
    assert comfy_client.shot_kind(probes["weather_place"]) == "object"
    for has_a_person in ("face", "animal", "action", "writing_surface", "crowd"):
        assert comfy_client.shot_kind(probes[has_a_person]) == "figure", has_a_person


def test_the_place_shot_is_never_sent_the_figure_rules(monkeypatch):
    import comfy_client
    monkeypatch.setenv("RUFUS_STYLE", "stickman_lean")
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    out = comfy_client._with_detail(dict(bench.PROBES)["weather_place"])
    assert "STICK FIGURE" not in out
    assert "eyebrow strokes" not in out
    assert "[SHOT=" not in out, "the tag is an instruction to us, not to the model"


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


# ── probing a candidate without clobbering the live template ─────────────────

def test_a_named_workflow_is_used_instead_of_the_shipping_one(monkeypatch, tmp_path, capsys):
    """Trying a LoRA by saving over config/stills_api.json and forgetting to
    put it back is a broken run discovered at 3am."""
    import comfy_client, comfy_template
    seen = {}
    monkeypatch.setattr(comfy_template, "load_template",
                        lambda p: seen.setdefault("path", p) and None or None)
    monkeypatch.setattr(comfy_client, "is_available", lambda: False)
    cand = tmp_path / "zimage-doodle.json"
    cand.write_text("{}", encoding="utf-8")
    style_probe.run(workflow=str(cand))
    assert seen["path"] == cand


def test_a_missing_workflow_is_named_not_silently_ignored(tmp_path, capsys):
    assert style_probe.run(workflow=str(tmp_path / "nope.json")) == 1
    assert "no such workflow" in capsys.readouterr().out


def test_the_manifest_records_which_workflow_drew_the_pictures():
    """Two probe runs that differ because they used different workflows, with
    nothing on disk saying so, is the confusion this tool exists to remove."""
    src = Path(style_probe.__file__).read_text(encoding="utf-8")
    assert '"workflow": str(template)' in src
