"""The word-synced insert layer, end to end — planner, images, renderer, sound.

The format: a picture per NOUN. The narrator says "palace" and a palace pops in
on the word, twenty-odd times across forty seconds. It is the shape that does
well on TikTok, and this pipeline can do it because it already transcribes the
finished voiceover into word-level timestamps for the captions.

Every test here is about a SEAM, because the planner's own logic is covered in
test_insert_director.py and the seams are where an optional layer usually goes
wrong: it must be inert when off, invisible when it fails, and never able to
turn a working render into a broken one.

No GPU, no ComfyUI, no network.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

ROOT = Path(__file__).parent.parent

import comfy_client  # noqa: E402
import insert_director  # noqa: E402
import remotion_renderer  # noqa: E402
import sfx_gen  # noqa: E402


# ── the pop ──────────────────────────────────────────────────────────────────

def test_the_insert_sound_is_not_the_whoosh():
    """A whoosh is a TRANSITION — "we are moving from here to there" — which is
    right for a scene cut and wrong for an object landing on top of a scene
    that has not changed. Twenty whooshes in forty seconds is traffic noise."""
    cmd = sfx_gen._sfx_cmd("pop", Path("/tmp/x.wav"))
    assert cmd is not None
    assert sfx_gen._sfx_cmd("whoosh", Path("/tmp/x.wav")) != cmd


def test_the_pop_is_short_enough_not_to_smear():
    """Inserts can be 0.45s apart. Anything with a tail overlaps the next one
    and the two blur into mush."""
    cmd = " ".join(sfx_gen._sfx_cmd("pop", Path("/tmp/x.wav")))
    import re
    durations = [float(d) for d in re.findall(r"duration=([\d.]+)", cmd)]
    assert durations and max(durations) <= 0.15


def test_a_box_that_cannot_synthesize_the_pop_keeps_its_other_sounds(
        monkeypatch, tmp_path):
    """The pop serves one optional layer. Losing the whole sound design to a
    feature the owner is not using would be a bad trade — and the original
    all-or-nothing rule would have done exactly that."""
    monkeypatch.setattr(sfx_gen, "SFX_DIR", tmp_path)
    for name in ("hit", "whoosh", "riser"):          # already cached
        (tmp_path / f"{name}.wav").write_bytes(b"\0" * (sfx_gen.MIN_BYTES + 1))
    monkeypatch.setattr(sfx_gen, "_sfx_cmd",
                        lambda name, out: None if name == "pop" else [])

    got = sfx_gen.ensure_sfx()
    assert {"hit", "whoosh", "riser"} <= set(got)
    assert "pop" not in got


# ── the images ───────────────────────────────────────────────────────────────

def test_a_failed_insert_is_dropped_not_fatal(monkeypatch, tmp_path):
    """One picture missing is a quieter video. A raised exception here would
    cost the whole render, after the voice and every beat were already paid
    for."""
    monkeypatch.setattr(comfy_client, "_render_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = comfy_client.render_inserts(
        [{"word": "palace", "at": 1.0, "hold": 0.7, "prompt": "a palace"}],
        tmp_path)
    assert out == []


def test_a_rendered_insert_carries_the_filename_the_renderer_needs(monkeypatch, tmp_path):
    monkeypatch.setattr(comfy_client, "_render_image", lambda *a, **k: b"PNGDATA")
    out = comfy_client.render_inserts(
        [{"word": "palace", "at": 1.0, "hold": 0.7, "prompt": "a palace"}],
        tmp_path)
    assert len(out) == 1
    assert out[0]["file"].endswith(".png")
    assert (tmp_path / out[0]["file"]).read_bytes() == b"PNGDATA"
    # The timing survives the render step untouched — it came from Whisper and
    # nothing downstream may reinterpret it.
    assert out[0]["at"] == 1.0


def test_each_insert_gets_its_own_seed(monkeypatch, tmp_path):
    """One seed for all of them renders near-identical pictures, which is the
    duplicate problem this repo already hit once on beats."""
    seeds = []
    monkeypatch.setattr(comfy_client, "_render_image",
                        lambda p, seed, cid, niche=None: seeds.append(seed) or b"X")
    comfy_client.render_inserts(
        [{"word": w, "at": i, "hold": 0.7, "prompt": w}
         for i, w in enumerate(["palace", "sword", "crown"])], tmp_path)
    assert len(set(seeds)) == 3


def test_no_inserts_means_no_work(tmp_path):
    assert comfy_client.render_inserts([], tmp_path) == []


# ── the renderer seam ────────────────────────────────────────────────────────

def test_the_renderer_plans_inserts_after_transcription():
    """An insert is pinned to the second its word is SPOKEN, and only the
    Whisper pass over the finished voiceover knows that. Planning earlier would
    mean guessing from the script."""
    src = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    assert src.index("_transcribe") < src.index("insert_director")


def test_the_layer_is_wrapped_so_it_can_never_break_a_render():
    src = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    block = src.split("import insert_director")[1].split("props = {")[0]
    assert "except Exception" in block
    assert "rendering without them" in block


def test_inserts_reach_the_composition_as_a_prop():
    src = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    assert '"inserts":' in src


def test_the_insert_style_is_the_channels_own():
    """An insert in a different look from the beat behind it reads as a bug —
    exactly what text-to-video did to this channel's flat-vector style."""
    src = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    assert "_detail_suffix" in src


# ── the composition ──────────────────────────────────────────────────────────

def _short_tsx() -> str:
    return (ROOT / "remotion" / "src" / "Short.tsx").read_text(encoding="utf-8")


def test_the_composition_accepts_inserts_optionally():
    """Optional throughout: a run without a plan renders exactly as it always
    did, which is the same contract `edit` follows."""
    tsx = _short_tsx()
    assert "inserts?: Insert[] | null" in tsx
    assert "inserts && inserts.length ?" in tsx


def test_an_insert_pops_rather_than_fades():
    """It is not a transition. The scene underneath does not change — an object
    lands on top of it, so it springs in and hard-cuts out."""
    tsx = _short_tsx()
    layer = tsx.split("const InsertLayer")[1].split("export type")[0] \
        if "export type" in tsx.split("const InsertLayer")[1] else \
        tsx.split("const InsertLayer")[1]
    assert "spring(" in layer


def test_inserts_render_under_the_captions():
    """Captions are the thing the viewer is reading. An insert that covers them
    costs comprehension for decoration."""
    tsx = _short_tsx()
    assert tsx.index("<InsertLayer") < tsx.index("<Captions")


def test_the_default_props_include_inserts():
    root = (ROOT / "remotion" / "src" / "Root.tsx").read_text(encoding="utf-8")
    assert "inserts: []" in root


def test_the_image_component_is_imported():
    """Remotion's Img waits for the asset before advancing the frame; a bare
    <img> renders a blank box on the frame the pop happens."""
    assert "\n  Img," in _short_tsx()
