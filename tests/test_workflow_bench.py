"""Comparing two ComfyUI exports, which nothing here could do before.

The /styles page renders one fixed scene through MANY STYLES and one workflow.
This is the transpose — one style, MANY WORKFLOWS — because the gallery that
started this had a style block forbidding washed-out backgrounds twice and
came back pale beige anyway. When a model overrides an explicit instruction,
the next sentence does not fix it and the checkpoint might.

The most valuable thing in here is not the renderer. It is `advisories`: the
owner's real export runs a KSampler at cfg 1, where the negative prompt has no
effect at all, which is the actual reason readable lettering reached a finished
gallery while the negative led with "text, letters, words".
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import workflow_bench as wb  # noqa: E402


def _graph(cfg=7.5, width=832, height=1472, negative=True, placeholder=True):
    """A miniature stills export, shaped like the owner's real one."""
    g = {
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT" if placeholder else "a coin"}},
        "13": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": 1, "steps": 9, "cfg": cfg,
                         "positive": ["6", 0], "latent_image": ["13", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "x", "images": ["8", 0]}},
    }
    if negative:
        g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}
        g["3"]["inputs"]["negative"] = ["7", 0]
    return g


def _write(tmp_path, name, graph):
    p = tmp_path / name
    p.write_text(json.dumps(graph), encoding="utf-8")
    return p


# ── the advisory that explains the lettering ─────────────────────────────────

def test_cfg_one_is_reported_because_the_negative_cannot_work():
    """THE FINDING THIS MODULE EXISTS TO HAVE MADE. At CFG 1.0 the guided
    result is exactly the conditional one — the unconditional branch cancels —
    so the negative prompt is computed, wired, sent and discarded. A real
    gallery came back with readable sentences across two frames while the
    negative led with "text, letters, words", and this is why."""
    notes = wb.advisories(_graph(cfg=1))
    assert any("cfg is 1" in n for n in notes)
    assert any("no classifier-free guidance" in n for n in notes)
    assert any("turbo" in n for n in notes), "and why it is not a bug"


def test_a_guided_workflow_gets_no_such_note():
    """A check that fires on every workflow tells you nothing about any of
    them."""
    assert not any("cfg" in n for n in wb.advisories(_graph(cfg=7.5)))


def test_a_portrait_export_on_a_landscape_format_is_reported(monkeypatch):
    """The same crop the stills path warns about at render time, said here
    before a single picture is drawn."""
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    import importlib
    import video_format
    importlib.reload(video_format)
    notes = wb.advisories(_graph(width=832, height=1472))
    assert any("crop most of every picture" in n for n in notes)


def test_a_matching_export_is_not_reported(monkeypatch):
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    import importlib
    import video_format
    importlib.reload(video_format)
    assert not any("crop" in n for n in wb.advisories(_graph(832, 1472)))


# ── validation before anything is rendered ───────────────────────────────────

def test_a_good_export_validates(tmp_path):
    ok, problems = wb.validate(_write(tmp_path, "good.json", _graph()))
    assert ok and problems == []


def test_an_export_without_the_placeholder_is_refused(tmp_path):
    """Six probes across four candidates is twenty-four renders. A broken
    export should cost zero of them."""
    ok, problems = wb.validate(
        _write(tmp_path, "noph.json", _graph(placeholder=False)))
    assert not ok
    assert any("RUFUS_PROMPT" in p for p in problems)


def test_an_export_with_no_negative_node_is_refused(tmp_path):
    ok, problems = wb.validate(
        _write(tmp_path, "noneg.json", _graph(negative=False)))
    assert not ok
    assert any("negative" in p for p in problems)


def test_a_graph_with_no_sizable_latent_is_refused(tmp_path):
    g = _graph()
    del g["13"]["inputs"]["width"]
    ok, problems = wb.validate(_write(tmp_path, "nolatent.json", g))
    assert not ok
    assert any("width and height" in p for p in problems)


def test_a_file_that_is_not_an_export_is_refused(tmp_path):
    p = tmp_path / "junk.json"
    p.write_text("not json at all", encoding="utf-8")
    ok, problems = wb.validate(p)
    assert not ok
    assert "Export (API)" in problems[0]


# ── the candidates ───────────────────────────────────────────────────────────

def test_the_live_workflow_is_always_the_first_column(tmp_path, monkeypatch):
    """A candidate measured against nothing is a picture. Measured against
    what the channel ships today, it is a decision."""
    monkeypatch.setattr(wb, "WORKFLOWS_DIR", tmp_path)
    _write(tmp_path, "b_second.json", _graph())
    _write(tmp_path, "a_first.json", _graph())
    labels = [label for label, _ in wb.candidates()]
    assert labels[0] == wb.BASELINE_NAME
    assert labels[1:] == ["a_first", "b_second"]


def test_no_workflows_folder_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "WORKFLOWS_DIR", tmp_path / "does-not-exist")
    assert [l for l, _ in wb.candidates()] == [wb.BASELINE_NAME]


# ── the probes ───────────────────────────────────────────────────────────────

def test_every_probe_is_a_defect_this_project_has_shipped():
    """The probes are not a pretty test scene. Each one is a way this channel
    has actually been let down, so a workflow is measured against those."""
    names = [n for n, _ in wb.PROBES]
    assert names == ["face", "animal", "action", "writing_surface", "crowd",
                     "weather_place"]
    for _name, text in wb.PROBES:
        assert len(text.split()) >= 8, text


def test_the_action_probe_asks_for_something_happening():
    """Thirteen frames of sixteen had nobody doing anything — the probe has to
    be the thing that failed."""
    text = dict(wb.PROBES)["action"]
    assert "goes over" in text and "scatter" in text


def test_every_candidate_sees_the_same_seed_for_a_probe():
    """A difference between two columns has to be the workflow and not the
    noise it started from."""
    assert len(set(wb._PROBE_SEEDS.values())) == len(wb.PROBES)
    assert set(wb._PROBE_SEEDS) == {n for n, _ in wb.PROBES}


# ── the contract ─────────────────────────────────────────────────────────────

def test_the_bench_does_nothing_when_comfy_is_down(monkeypatch, capsys):
    import comfy_client
    monkeypatch.setattr(comfy_client, "is_available", lambda: False)
    out = wb.run()
    assert out["workflows"] == []
    assert "not reachable" in capsys.readouterr().out


def test_a_missing_result_reads_as_no_result(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "bench_root", lambda: tmp_path / "nothing")
    assert wb.latest() == {}


def test_the_newest_result_is_the_one_read(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "bench_root", lambda: tmp_path)
    for stamp, marker in (("20260101_000000", "old"), ("20260817_000000", "new")):
        d = tmp_path / stamp
        d.mkdir(parents=True)
        (d / "bench.json").write_text(json.dumps({"stamp": marker}),
                                      encoding="utf-8")
    assert wb.latest()["stamp"] == "new"


def test_a_corrupt_result_is_skipped_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "bench_root", lambda: tmp_path)
    (tmp_path / "20260817_000000").mkdir(parents=True)
    (tmp_path / "20260817_000000" / "bench.json").write_text("{{", encoding="utf-8")
    (tmp_path / "20260101_000000").mkdir(parents=True)
    (tmp_path / "20260101_000000" / "bench.json").write_text(
        json.dumps({"stamp": "old"}), encoding="utf-8")
    assert wb.latest()["stamp"] == "old"
