"""The pipeline could never see its own pictures.

run_review has found real defects — one object in most frames, the thread
restated everywhere, too few pictures for the length — and found every one of
them by reading TEXT. It has never seen a pixel. So the defects that live in
the image were all found the same way: the owner opened the gallery, looked,
and said "why is everything coins", "the faces are all the same", "it put
images on top of images".

A pipeline that renders sixty stills a night and can only be checked by a
human looking at them has its quality capped by how often that human looks.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import vision_review as vr  # noqa: E402


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("RUFUS_VISION", "1")
    monkeypatch.setenv("RUFUS_VISION_MODEL", "qwen2.5vl:7b")


# A HEALTHY run: pictures that match, no lettering, and faces that DIFFER.
# The first version of this gave every frame the same expression, so a test
# asserting "one bad frame is not a finding" tripped
# one_expression_everywhere instead — correctly. A fixture that is itself
# defective cannot be the baseline the defect tests measure against.
_FACES = ("brows flat, mouth a short straight line",
          "brows raised high, mouth a small open oval",
          "brows angled down and inward, mouth flat")


def _seen(n, **over):
    out = []
    for i in range(n):
        f = {"shows_it": True, "missing": "", "lettering": False,
             "lettering_note": "", "faces": 1, "expression": _FACES[i % 3],
             "frame": f"{i:02d}.png"}
        f.update(over)
        out.append(f)
    return out


# ── off by default, because it costs seconds a frame ─────────────────────────

def test_it_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("RUFUS_VISION", raising=False)
    assert vr.enabled() is False
    assert vr.review_frames([Path("a.png")], ["p"])["looked_at"] == 0


# ── reading the model's answer ───────────────────────────────────────────────

def test_a_fenced_reply_is_still_an_answer():
    """Local models fence their JSON in ```json blocks more often than the
    cloud ones do, and a fenced reply is a correct answer this would
    otherwise throw away."""
    got = vr._parse('```json\n{"shows_it": false, "missing": "the queue", '
                    '"lettering": true, "lettering_note": "a sign", '
                    '"faces": 2, "expression": "brows raised"}\n```')
    assert got["shows_it"] is False
    assert got["missing"] == "the queue"
    assert got["faces"] == 2


def test_junk_is_not_an_answer():
    for junk in ("", "no idea", "{not json", None):
        assert vr._parse(junk) is None


def test_a_missing_field_does_not_crash_the_run():
    got = vr._parse('{"shows_it": true}')
    assert got["faces"] == 0 and got["expression"] == ""


# ── the sample ───────────────────────────────────────────────────────────────

def test_a_short_run_is_looked_at_whole():
    frames = [Path(f"{i}.png") for i in range(10)]
    assert len(vr._sample(frames, ["p"] * 10)) == 10


def test_a_long_run_is_sampled_across_the_whole_sequence():
    """Truncating at the front would leave the last third of every long-form
    video unlooked-at — exactly the part nobody scrolls to, and therefore the
    part a defect survives in."""
    frames = [Path(f"{i:03d}.png") for i in range(150)]
    picked = vr._sample(frames, ["p"] * 150)
    assert len(picked) == vr.MAX_FRAMES
    names = [p.name for p, _ in picked]
    assert names[0] == "000.png"
    assert int(names[-1].split(".")[0]) > 140, "the end must be looked at"


def test_each_frame_is_asked_about_its_own_prompt():
    frames = [Path("0.png"), Path("1.png")]
    pairs = vr._sample(frames, ["first prompt", "second prompt"])
    assert pairs[1][1] == "second prompt"


# ── the findings ─────────────────────────────────────────────────────────────

def test_one_bad_frame_is_a_seed_and_not_a_finding():
    """A check that fires on one frame in twelve is the noise this repo has
    twice had to walk back."""
    seen = _seen(12)
    seen[0]["shows_it"] = False
    assert vr.findings(seen) == []


def test_a_third_of_them_missing_the_prompt_is_a_finding():
    seen = _seen(12)
    for f in seen[:5]:
        f["shows_it"] = False
        f["missing"] = "the queue outside"
    ids = [f["id"] for f in vr.findings(seen)]
    assert "pictures_miss_their_prompt" in ids


def test_lettering_getting_through_is_reported_with_what_was_seen():
    seen = _seen(10)
    for f in seen[:6]:
        f["lettering"] = True
        f["lettering_note"] = "garbled word on a banner"
    got = next(f for f in vr.findings(seen)
               if f["id"] == "lettering_got_through")
    assert "banner" in got["text"]
    assert "defusal" in got["text"]


def test_the_same_face_everywhere_is_finally_countable():
    """The complaint that started this. "the faces are all the same" was
    something the owner saw across sixty stills; nothing in the pipeline
    could count it, because the prompts all said something different while
    the renders all looked the same."""
    seen = _seen(10, expression="brows slanted up, mouth curved down")
    got = next(f for f in vr.findings(seen)
               if f["id"] == "one_expression_everywhere")
    assert "10 of 10" in got["text"]
    assert "every prompt differs" in got["text"]


def test_varied_faces_draw_no_finding():
    seen = _seen(9)
    for i, f in enumerate(seen):
        f["expression"] = ["brows flat", "brows raised", "brows down"][i % 3]
    assert not any(f["id"] == "one_expression_everywhere"
                   for f in vr.findings(seen))


def test_empty_rooms_are_reported():
    seen = _seen(10, faces=0, expression="")
    ids = [f["id"] for f in vr.findings(seen)]
    assert "nobody_in_the_pictures" in ids


def test_too_few_frames_to_conclude_anything():
    assert vr.findings(_seen(2, shows_it=False)) == []


# ── fail-open ────────────────────────────────────────────────────────────────

def test_no_endpoint_means_no_findings_and_a_reason(monkeypatch, capsys):
    import llm
    monkeypatch.setattr(llm, "usable", lambda: False)
    out = vr.review_frames([Path("a.png")], ["p"])
    assert out["looked_at"] == 0
    assert "no endpoint" in capsys.readouterr().out


def test_a_model_that_errors_costs_the_frame_and_not_the_run(monkeypatch,
                                                             tmp_path, capsys):
    img = tmp_path / "01.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 50)

    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("model not loaded")
    assert vr.look(img, "a prompt", client=_Boom()) is None
    assert "model not loaded" in capsys.readouterr().out


def test_an_unreadable_file_is_not_a_failed_picture():
    assert vr.look(Path("/nope/missing.png"), "p", client=object()) is None


def test_the_question_forbids_naming_an_emotion():
    """The renderer can only draw five faces, and it draws them from geometry.
    A reviewer answering "he looks sad" could not be compared with a prompt
    that asked for brows slanted up over a downward mouth curve."""
    assert "Never name an emotion" in vr._PROMPT
    assert "BROWS" in vr._PROMPT and "MOUTH" in vr._PROMPT


def test_a_stylised_drawing_is_not_counted_as_missing_its_prompt():
    """This is a flat cartoon by design; a reviewer marking every frame
    "not photographic" would fire on all of them, which is no signal."""
    assert "not because it is stylised" in vr._PROMPT
