"""The re-roll a person does by hand, as a check the renderer can make.

Every fix in this repo starts with something the owner saw in a folder. This
one starts with a gallery of sixteen where two frames were six-panel contact
sheets of the same stick figure and thirteen were figures standing in an empty
street — all of them correct, non-duplicate drawings, which is why the only
gate that existed (a perceptual-duplicate check) passed every one.

The thresholds matter more than the checks. A gate that rejects a frame costs
a re-render, so one that fires on good frames costs an hour of GPU and teaches
nobody anything — the tests below pin BOTH directions, and the legitimate
styles that look superficially like the defects (a monochrome engraving is
mostly white paper; a landscape has a plain sky band) are the ones that must
pass.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import frame_gate  # noqa: E402

PIL = pytest.importorskip("PIL")


def _img(path: Path, draw, size=(512, 512), bg=(255, 255, 255)):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", size, bg)
    draw(ImageDraw.Draw(im), size)
    im.save(str(path))
    return path


def _contact_sheet(path: Path):
    """Three columns by two rows of panels with white gutters between them —
    the shape two frames of the real gallery came back as."""
    def draw(d, size):
        w, h = size
        for r in range(2):
            for c in range(3):
                x0 = int(c * w / 3) + 14
                y0 = int(r * h / 2) + 14
                x1 = int((c + 1) * w / 3) - 14
                y1 = int((r + 1) * h / 2) - 14
                d.rectangle([x0, y0, x1, y1], fill=(70, 90, 120))
    return _img(path, draw)


def _landscape(path: Path):
    """A plain sky over a dark ground: one horizontal band of near-uniform
    white and nothing vertical. Legitimate, and the closest a real drawing
    gets to looking like a grid."""
    def draw(d, size):
        w, h = size
        d.rectangle([0, int(h * 0.55), w, h], fill=(60, 120, 70))
        d.ellipse([int(w * 0.1), int(h * 0.3), int(w * 0.6), int(h * 0.7)],
                  fill=(40, 90, 55))
        d.rectangle([int(w * 0.7), int(h * 0.2), int(w * 0.8), int(h * 0.6)],
                    fill=(90, 70, 50))
    return _img(path, draw)


def _almost_empty(path: Path):
    """The white-void bug at its worst: a small figure and nothing else."""
    def draw(d, size):
        w, h = size
        d.line([int(w * 0.5), int(h * 0.45), int(w * 0.5), int(h * 0.6)],
               fill=(0, 0, 0), width=3)
        d.ellipse([int(w * 0.48), int(h * 0.40), int(w * 0.52), int(h * 0.45)],
                  outline=(0, 0, 0))
    return _img(path, draw)


def _engraving(path: Path):
    """Monochrome hatching on white paper — mostly white BY DESIGN. ink_woodcut
    renders like this and must never be mistaken for an empty frame."""
    def draw(d, size):
        w, h = size
        for x in range(0, w, 6):
            d.line([x, int(h * 0.25), x, int(h * 0.95)], fill=(20, 20, 20))
        for y in range(int(h * 0.3), int(h * 0.9), 7):
            d.line([int(w * 0.1), y, int(w * 0.9), y], fill=(40, 40, 40))
    return _img(path, draw)


# ── the contact sheet ────────────────────────────────────────────────────────

def test_a_contact_sheet_is_rejected(tmp_path):
    ok, reason, detail = frame_gate.check(_contact_sheet(tmp_path / "grid.png"))
    assert not ok
    assert reason == "contact_sheet"
    assert "gutters" in detail


def test_a_landscape_with_a_plain_sky_is_not_a_grid(tmp_path):
    """One band alone is a horizon, a tabletop or a wall. Requiring a
    horizontal AND a vertical gutter is what keeps this check off ordinary
    drawings."""
    h, v = frame_gate.grid_bands(_landscape(tmp_path / "land.png"))
    assert h >= 1, "the sky really is a near-uniform band"
    assert v == 0
    assert frame_gate.check(tmp_path / "land.png")[0]


# ── the empty frame ──────────────────────────────────────────────────────────

def test_an_almost_empty_frame_is_rejected(tmp_path):
    ok, reason, _ = frame_gate.check(_almost_empty(tmp_path / "void.png"))
    assert not ok and reason == "blank_frame"


def test_a_monochrome_engraving_is_not_an_empty_frame(tmp_path):
    """THE FALSE POSITIVE THIS CHECK IS MOST LIKELY TO HAVE. ink_woodcut is
    ink on white paper and a legitimate frame in it is largely white — which
    is why the threshold is 90% and not 75%, and why it catches only the
    version of the bug nobody would defend."""
    p = _engraving(tmp_path / "wood.png")
    assert frame_gate.blank_share(p) < frame_gate.BLANK_SHARE
    assert frame_gate.check(p)[0]


# ── the contract ─────────────────────────────────────────────────────────────

def test_a_frame_that_cannot_be_read_is_kept(tmp_path, capsys):
    """A frame that could not be looked at is not a frame that failed."""
    bad = tmp_path / "not-an-image.png"
    bad.write_text("nonsense", encoding="utf-8")
    assert frame_gate.check(bad)[0]


def test_a_missing_file_is_kept(tmp_path):
    assert frame_gate.check(tmp_path / "nothing.png")[0]


def test_the_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_FRAME_GATE", raising=False)
    monkeypatch.delenv("RUFUS_VISION_GATE", raising=False)
    assert frame_gate.enabled() is False
    assert frame_gate.vision_enabled() is False


def test_the_vision_question_is_its_own_switch(monkeypatch):
    """RUFUS_VISION reviews a finished run when the GPU is free. This one runs
    while ComfyUI is holding the card, so it cannot ride on the same flag."""
    monkeypatch.setenv("RUFUS_VISION", "1")
    monkeypatch.delenv("RUFUS_VISION_GATE", raising=False)
    assert frame_gate.vision_enabled() is False


def test_the_vision_model_is_not_asked_when_it_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("RUFUS_VISION_GATE", raising=False)
    called = []
    import vision_review
    monkeypatch.setattr(vision_review, "look",
                        lambda *a, **k: called.append(1) or None)
    frame_gate.check(_landscape(tmp_path / "l.png"), prompt="a hill")
    assert not called


def test_the_vision_model_rejects_a_frame_that_misses_its_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_VISION_GATE", "1")
    import vision_review
    monkeypatch.setattr(vision_review, "look", lambda *a, **k: {
        "shows_it": False, "missing": "the overturned table",
        "lettering": False, "lettering_note": "", "faces": 1, "expression": ""})
    ok, reason, detail = frame_gate.check(_landscape(tmp_path / "l.png"),
                                          prompt="the table goes over")
    assert not ok and reason == "misses_the_prompt"
    assert detail == "the overturned table"


def test_lettering_is_a_rejection_when_the_model_is_looking(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_VISION_GATE", "1")
    import vision_review
    monkeypatch.setattr(vision_review, "look", lambda *a, **k: {
        "shows_it": True, "missing": "", "lettering": True,
        "lettering_note": "the word Proof on a document",
        "faces": 1, "expression": ""})
    ok, reason, _ = frame_gate.check(_landscape(tmp_path / "l.png"), prompt="x")
    assert not ok and reason == "lettering"


def test_a_vision_model_that_will_not_answer_keeps_the_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_VISION_GATE", "1")
    import vision_review
    monkeypatch.setattr(vision_review, "look", lambda *a, **k: None)
    assert frame_gate.check(_landscape(tmp_path / "l.png"), prompt="x")[0]


# ── the re-roll says what was wrong ──────────────────────────────────────────

def test_every_rejection_has_something_to_tell_the_next_attempt():
    """A re-roll with the same prompt and a new seed IS the same prompt. A
    person who re-rolls says what was wrong with the last one."""
    for reason in ("contact_sheet", "blank_frame", "lettering"):
        assert frame_gate.retry_hint(reason), reason
    assert "overturned table" in frame_gate.retry_hint(
        "misses_the_prompt", "the overturned table")


def test_the_grid_hint_names_the_shape_and_not_the_subject():
    hint = frame_gate.retry_hint("contact_sheet")
    assert "never a grid" in hint and "ONE camera" in hint


# ── the dry run ──────────────────────────────────────────────────────────────

def test_the_dry_run_reports_without_rendering(tmp_path, capsys):
    """How the thresholds get set: against frames already on disk, where the
    answer can be checked by looking."""
    _contact_sheet(tmp_path / "01.png")
    _landscape(tmp_path / "02.png")
    _engraving(tmp_path / "03.png")
    out = frame_gate.dry_run(tmp_path)
    assert out["looked_at"] == 3
    assert [r["frame"] for r in out["rejected"]] == ["01.png"]


def test_the_dry_run_says_so_when_it_would_reject_too_many(tmp_path, capsys):
    """A quarter of a gallery is not a gate, it is a re-render loop with a
    thesaurus."""
    for i in range(3):
        _contact_sheet(tmp_path / f"{i}.png")
    frame_gate.dry_run(tmp_path)
    assert "too many" in capsys.readouterr().out
