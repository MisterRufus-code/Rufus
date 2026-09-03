"""The background plate: one place, drawn once, figures composited onto it."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import plates  # noqa: E402


def _make_plate(root: Path, slug: str, meta: dict, front: bool = False) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    Image.new("RGB", (200, 360), (60, 120, 90)).save(d / "back.png")
    if front:
        fg = Image.new("RGBA", (200, 360), (0, 0, 0, 0))
        ImageDraw.Draw(fg).rectangle([(0, 300), (200, 360)], fill=(90, 60, 30, 255))
        fg.save(d / "front.png")
    (d / "plate.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _figure(w=80, h=240, bg=(245, 245, 240), touch_bottom=False):
    im = Image.new("RGB", (w + 120, h + 120), bg)
    d = ImageDraw.Draw(im)
    cx = im.width // 2
    d.ellipse((cx - 30, 40, cx + 30, 110), fill=(255, 255, 255), outline=(20, 20, 20), width=5)
    d.line((cx, 110, cx, h), fill=(20, 20, 20), width=6)
    end = im.height - 1 if touch_bottom else h + 60
    d.line((cx, h, cx - 30, end), fill=(20, 20, 20), width=6)
    d.line((cx, h, cx + 30, end), fill=(20, 20, 20), width=6)
    return im


# ── the library ─────────────────────────────────────────────────────────────
def test_a_directory_without_a_back_image_is_not_a_plate(tmp_path):
    d = tmp_path / "half_made"
    d.mkdir()
    (d / "plate.json").write_text('{"name": "half made"}', encoding="utf-8")
    assert plates.library(tmp_path) == []


def test_a_broken_manifest_does_not_take_the_run_down(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    Image.new("RGB", (10, 10)).save(d / "back.png")
    (d / "plate.json").write_text("{not json", encoding="utf-8")
    assert plates.library(tmp_path) == []


def test_no_library_means_no_plate_rather_than_an_error(tmp_path):
    assert plates.pick("a hand pushes a coin across the counter",
                       root=tmp_path / "nothing here") is None


# ── choosing one ────────────────────────────────────────────────────────────
def test_the_plate_whose_words_the_shot_uses_is_the_one_chosen(tmp_path):
    _make_plate(tmp_path, "counting_house", {
        "name": "counting house", "keywords": ["ledger", "counting", "desk", "strongbox"]})
    _make_plate(tmp_path, "quay", {
        "name": "quay", "keywords": ["crate", "rope", "hull", "harbour"]})
    got = plates.pick("a hand slams the ledger shut on the counting house desk",
                      root=tmp_path)
    assert got is not None and got.slug == "counting_house"


def test_a_shot_that_merely_brushes_one_word_gets_no_plate(tmp_path):
    _make_plate(tmp_path, "market_street", {
        "name": "market", "keywords": [
            "market", "stall", "awning", "crate", "barrel", "cobblestone",
            "produce", "basket", "trader", "shutter", "cart", "wares"]})
    # One hit out of twelve is a mention, not a location.
    assert plates.pick("a cart of grain leaves the village", root=tmp_path) is None


def test_a_place_from_the_wrong_century_is_ruled_out(tmp_path):
    _make_plate(tmp_path, "mint_forge", {
        "name": "mint", "when": [1200, 1700],
        "keywords": ["mint", "furnace", "anvil", "coin"]})
    shot = "the furnace mouth glows as a coin is struck on the anvil at the mint"
    assert plates.pick(shot, year=1550, root=tmp_path) is not None
    assert plates.pick(shot, year=1980, root=tmp_path) is None


def test_an_unknown_year_never_rules_a_place_out(tmp_path):
    """The era check exists to reject a wrong century, never to be the reason
    nothing matches at all."""
    _make_plate(tmp_path, "mint_forge", {
        "name": "mint", "when": [1200, 1700],
        "keywords": ["mint", "furnace", "anvil", "coin"]})
    shot = "the furnace mouth glows as a coin is struck on the anvil at the mint"
    assert plates.pick(shot, year=None, root=tmp_path) is not None


# ── cutting the figure out ──────────────────────────────────────────────────
def test_the_white_body_fill_survives_the_cutout():
    """The figures are white fill inside a black outline, so keying on colour
    would erase the body along with the background. Only pale that reaches the
    edge of the frame is background."""
    cut = plates.cutout(_figure())
    assert cut is not None
    assert cut.getpixel((cut.width // 2, 30))[:3] == (255, 255, 255)


def test_a_figure_touching_the_frame_edge_is_not_eaten():
    """Feet reach the bottom of the frame in most shots. Seeding the fill from
    the bottom edge would start it inside the figure."""
    tall = plates.cutout(_figure(touch_bottom=True))
    assert tall is not None
    assert tall.height > 200


def test_a_render_that_kept_its_scenery_is_refused():
    """Too little filled means the background was not plain, and cutting into
    scenery is worse than not compositing at all."""
    busy = _figure()
    d = ImageDraw.Draw(busy)
    for i in range(0, busy.width, 12):
        d.line((i, 0, i, busy.height), fill=(120, 90, 60), width=7)
    assert plates.cutout(busy) is None


def test_an_empty_frame_produces_no_cutout():
    assert plates.cutout(Image.new("RGB", (200, 300), (245, 245, 240))) is None


# ── standing the figure in the place ────────────────────────────────────────
def test_the_figure_is_scaled_and_stood_on_the_plates_own_ground_line(tmp_path):
    """Two shots of one room agree about how big a person is and where the
    floor runs only because both come from these numbers."""
    _make_plate(tmp_path, "hall", {
        "name": "hall", "keywords": ["hall"],
        "ground_y": 0.75, "figure_height": 0.5, "safe_x": [0.5, 0.5]})
    plate = plates.library(tmp_path)[0]
    out = plates.compose(plate, plates.cutout(_figure()), seed=1)
    assert out.size == (200, 360)

    # The figure is half the frame high and its feet land on the ground line.
    import numpy as np
    arr = np.asarray(out)
    ink = (arr.sum(2) < 200)
    rows = np.where(ink.any(1))[0]
    assert abs(rows.max() - 0.75 * 360) <= 6, "feet are not on the ground line"
    assert abs((rows.max() - rows.min()) - 0.5 * 360) <= 8, "wrong height"


def test_the_same_beat_puts_the_figure_in_the_same_place_twice(tmp_path):
    _make_plate(tmp_path, "hall", {"name": "hall", "keywords": ["hall"]})
    plate = plates.library(tmp_path)[0]
    fig = plates.cutout(_figure())
    a = plates.compose(plate, fig, seed=7)
    b = plates.compose(plate, fig, seed=7)
    assert list(a.getdata()) == list(b.getdata())


def test_different_beats_do_not_stack_the_figure_in_one_spot(tmp_path):
    """Dead centre every time is the composition that gives the trick away."""
    _make_plate(tmp_path, "hall", {
        "name": "hall", "keywords": ["hall"], "safe_x": [0.15, 0.85]})
    plate = plates.library(tmp_path)[0]
    fig = plates.cutout(_figure())
    seen = {plates.compose(plate, fig, seed=s).tobytes() for s in range(6)}
    assert len(seen) > 1


def test_the_foreground_layer_passes_in_front_of_the_figure(tmp_path):
    """Without it every figure floats in front of every counter and crate."""
    _make_plate(tmp_path, "hall", {
        "name": "hall", "keywords": ["hall"],
        "ground_y": 0.95, "figure_height": 0.6, "safe_x": [0.5, 0.5]},
        front=True)
    plate = plates.library(tmp_path)[0]
    out = plates.compose(plate, plates.cutout(_figure()), seed=0)
    assert out.getpixel((100, 330)) == (90, 60, 30), "the figure is over the counter"


# ── what the figure-only render is asked for ────────────────────────────────
def test_the_figure_only_prompt_names_one_flat_colour_and_forbids_a_gradient():
    """The cutout takes the background from what the corners agree on and
    floods inward, so a wash or a vignette across the frame is exactly what
    makes an otherwise clean figure uncuttable."""
    import comfy_client
    got = comfy_client._figure_only_prompt("A hand slams the ledger shut.").lower()
    assert "flat mid-blue" in got
    assert "no gradient" in got and "no vignette" in got
    assert "no shadow under the figure" in got


def test_the_figure_only_prompt_drops_the_place_building_region():
    """A room drawn behind the figure would be composited on top of the real
    one, so the style block's own place-building block has to come out."""
    import comfy_client
    prompt = ("A hand slams the ledger shut. "
              f"{comfy_client.STYLE_FARSHOT_OPEN} BUILD THE WHOLE PLACE: a "
              f"surface for the subject to stand on. {comfy_client.STYLE_FARSHOT_CLOSE}")
    got = comfy_client._figure_only_prompt(prompt)
    assert "BUILD THE WHOLE PLACE" not in got
    assert "slams the ledger shut" in got
