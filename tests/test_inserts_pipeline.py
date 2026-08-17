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

def test_the_insert_sound_is_not_the_cut_sound():
    """The cut sound is a TRANSITION — "we are moving from here to there" — which is
    right for a scene cut and wrong for an object landing on top of a scene
    that has not changed. Twenty of them in forty seconds is traffic noise."""
    cmd = sfx_gen._sfx_cmd("pop", Path("/tmp/x.wav"))
    assert cmd is not None
    assert sfx_gen._sfx_cmd("bubble", Path("/tmp/x.wav")) != cmd


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
    for name in ("hit", "bubble", "riser"):          # already cached
        (tmp_path / f"{name}.wav").write_bytes(b"\0" * (sfx_gen.MIN_BYTES + 1))
    monkeypatch.setattr(sfx_gen, "_sfx_cmd",
                        lambda name, out: None if name == "pop" else [])

    got = sfx_gen.ensure_sfx()
    assert {"hit", "bubble", "riser"} <= set(got)
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
                        lambda p, seed, cid, niche=None, px=None:
                        seeds.append(seed) or b"X")
    comfy_client.render_inserts(
        [{"word": w, "at": i, "hold": 0.7, "prompt": w}
         for i, w in enumerate(["palace", "sword", "crown"])], tmp_path)
    assert len(set(seeds)) == 3


def test_no_inserts_means_no_work(tmp_path):
    assert comfy_client.render_inserts([], tmp_path) == []


# ── the renderer seam ────────────────────────────────────────────────────────

def _renderer_sources() -> dict:
    """Both renderers. THE FFMPEG ONE IS THE ONE THAT ACTUALLY RUNS —
    RUFUS_RENDERER defaults to "ffmpeg" — and the insert layer shipped into
    the Remotion path alone, so the owner set RUFUS_INSERT_MODE and
    RUFUS_INSERT_MAX=40 and got the same ten beat pictures twice, with nothing
    in the log to explain it. Every seam test below therefore asks the
    question of BOTH paths."""
    import audio_gen
    return {
        "remotion": Path(remotion_renderer.__file__).read_text(encoding="utf-8"),
        "ffmpeg": Path(audio_gen.__file__).read_text(encoding="utf-8"),
    }


def test_both_renderers_plan_inserts_after_transcription():
    """An insert is pinned to the second its word is SPOKEN, and only the
    Whisper pass over the finished voiceover knows that. Planning earlier would
    mean guessing from the script."""
    for name, src in _renderer_sources().items():
        assert src.index("insert_director") > 0, name
        transcribe = min(i for i in (src.find("_transcribe"), src.find("words"))
                         if i >= 0)
        assert transcribe < src.rindex("insert_director.plan_for"), name


def test_both_renderers_wrap_the_layer_so_it_can_never_break_a_render():
    for name, src in _renderer_sources().items():
        block = src.rsplit("import insert_director", 1)[1][:1500]
        assert "except Exception" in block, name
        assert "without them" in block, name


def test_inserts_reach_the_composition_as_a_prop():
    src = Path(remotion_renderer.__file__).read_text(encoding="utf-8")
    assert '"inserts":' in src


def test_the_insert_style_is_the_channels_own():
    """An insert in a different look from the beat behind it reads as a bug —
    exactly what text-to-video did to this channel's flat-vector style. Both
    renderers ask ONE function so the two paths cannot drift into two looks."""
    import insert_director
    src = Path(insert_director.__file__).read_text(encoding="utf-8")
    assert "_detail_suffix" in src
    for name, rsrc in _renderer_sources().items():
        assert "style_suffix" in rsrc, name


def test_an_insert_overrides_the_style_block_background():
    """The channel style now asks for scenery behind the figures, which is
    right for a beat and wrong for a 460px cutout composited over one: two
    backgrounds in one frame is two competing scenes. The override has to come
    AFTER the style block, because that is the half a model reads last."""
    import insert_director
    out = insert_director.insert_prompt("coins", "STYLE ASKING FOR A ROOM.")
    assert out.index("STYLE ASKING FOR A ROOM") < out.index("no background")
    phrase = insert_director.phrase_prompt("he paid the men",
                                           "STYLE ASKING FOR A ROOM.")
    assert "no background scenery" in phrase


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


# ── named looks ──────────────────────────────────────────────────────────────

def test_the_bundled_styles_load():
    got = comfy_client.style_presets()
    assert {"stickman", "flat_vector"} <= set(got)
    assert all(isinstance(v, str) and v.strip() for v in got.values())


def test_readme_keys_are_not_offered_as_styles():
    """config/styles.json documents itself in a "_readme" key. Rendering a
    video in the readme would be a memorable bug."""
    assert not any(k.startswith("_") for k in comfy_client.style_presets())


def test_a_named_style_becomes_the_look(monkeypatch):
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    assert "stick-figure" in comfy_client._detail_suffix()


def test_a_literal_style_outranks_a_preset(monkeypatch):
    """A one-off experiment must never require editing a config file first."""
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "my own look")
    assert comfy_client._detail_suffix() == "my own look"


def test_an_unknown_style_is_loud(monkeypatch, capsys):
    """A typo'd style name that quietly rendered the default look would be
    indistinguishable from the preset not working at all."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stikman")
    comfy_client._detail_suffix()
    out = capsys.readouterr().out
    assert "not a known style" in out
    assert "stickman" in out          # names the ones that exist


def test_no_style_set_keeps_the_current_look(monkeypatch):
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.delenv("RUFUS_STYLE", raising=False)
    assert comfy_client._detail_suffix() == comfy_client.DEFAULT_DETAIL_SUFFIX.strip()


def test_a_broken_styles_file_does_not_stop_a_run(monkeypatch, tmp_path):
    bad = tmp_path / "styles.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(comfy_client, "STYLES_FILE", bad)
    assert comfy_client.style_presets() == {}


def test_inserts_inherit_whatever_style_is_active(monkeypatch):
    """An insert drawn in a different look from the beat behind it reads as a
    bug — the same failure text-to-video produced against flat-vector."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    p = insert_director.insert_prompt("sword", comfy_client._detail_suffix())
    assert "stick-figure" in p


# ── inserts are cheap only if they are actually rendered small ───────────────

def test_an_insert_renders_smaller_than_a_beat_still():
    """The docstring claimed "small and simple" before the code did it, and at
    full still size twenty-eight inserts is six minutes of GPU — which would
    have made this format MORE expensive than the video it replaces."""
    assert comfy_client._insert_px() <= 768


def test_the_shrink_only_touches_real_numbers():
    """[node, slot] is a wire. Overwriting one with an integer severs the graph
    and the failure appears at submit time, after the beats are paid for."""
    g = {"1": {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1080, "height": 1920, "batch_size": 1}},
         "2": {"class_type": "K", "inputs": {"width": ["1", 0], "height": ["1", 1]}}}
    out = comfy_client._shrink(g, 512)
    assert out["1"]["inputs"]["width"] == out["1"]["inputs"]["height"] == 512
    assert out["2"]["inputs"]["width"] == ["1", 0]


def test_the_shrink_does_not_mutate_the_cached_template():
    """_stills_template() is loaded once and reused for every beat. Shrinking
    it in place would render the REST of the video at insert size."""
    g = {"1": {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1080, "height": 1920}}}
    comfy_client._shrink(g, 512)
    assert g["1"]["inputs"]["width"] == 1080


def test_the_insert_size_is_a_legal_latent_dimension(monkeypatch):
    """Dimensions that are not a multiple of 64 break most samplers."""
    for raw in ("700", "513", "100"):
        monkeypatch.setenv("RUFUS_INSERT_PX", raw)
        assert comfy_client._insert_px() % 64 == 0
        assert comfy_client._insert_px() >= 256


def test_a_junk_insert_size_falls_back_loudly(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_INSERT_PX", "big")
    assert comfy_client._insert_px() == 512
    assert "not a number" in capsys.readouterr().out


def test_render_inserts_asks_for_the_small_size(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(comfy_client, "_render_image",
                        lambda p, s, c, niche=None, px=None: got.update(px=px) or b"X")
    comfy_client.render_inserts(
        [{"word": "sword", "at": 1.0, "hold": 0.7, "prompt": "a sword"}], tmp_path)
    assert got["px"] == comfy_client._insert_px()


# ── the ffmpeg overlay layer ─────────────────────────────────────────────────
# Verified once against a real ffmpeg 6.1 render before these were written: an
# insert is absent at t=0.2, red on the left at t=1.0, and gone again with the
# second one blue on the right at t=2.5. These keep the graph that did it.

def test_the_overlay_chain_switches_each_insert_on_at_its_own_second():
    import audio_gen
    parts, last = audio_gen._insert_overlay_parts(
        [{"at": 0.5, "hold": 1.0}, {"at": 2.0, "hold": 0.7}], base=4)
    graph = ";".join(parts)
    assert "enable='between(t,0.500,1.500)'" in graph
    assert "enable='between(t,2.000,2.700)'" in graph
    assert "[4:v]" in graph and "[5:v]" in graph, "inputs are indexed from base"
    assert last == "ov1"


def test_an_insert_never_truncates_the_video():
    """eof_action=pass: the worst an insert input can do is stop early. Without
    it a short overlay input ends the whole output stream."""
    import audio_gen
    parts, _ = audio_gen._insert_overlay_parts([{"at": 1.0, "hold": 0.5}], base=2)
    assert "eof_action=pass" in ";".join(parts)


def test_inserts_sit_under_the_captions():
    """A word is illustrated by its picture, never covered by it — the same
    stacking order Short.tsx uses on the Remotion path."""
    import audio_gen
    fc = audio_gen._finish_video(
        ["[0:v]null[vcat]"], 10.0, "eq=contrast=1.0", "sub.ass", "fonts",
        "#FFD23F", inserts=[{"at": 1.0, "hold": 0.5}], insert_base=3)
    assert fc.index("overlay=x=") < fc.index("ass='sub.ass'")


def test_no_plan_leaves_the_graph_byte_for_byte_unchanged():
    """The layer is additive or it is nothing: a run without inserts must
    render the video this pipeline rendered before it existed."""
    import audio_gen
    # A fresh parts list per call — _finish_video appends to the one it is
    # given, and sharing it would compare a graph against itself twice over.
    def graph(**kw):
        return audio_gen._finish_video(["[0:v]null[vcat]"], 10.0,
                                       "eq=contrast=1.0", "sub.ass", "fonts",
                                       "#FFD23F", **kw)
    assert graph() == graph(inserts=[], insert_base=3)


def test_the_ffmpeg_ladder_can_drop_inserts_without_losing_the_render():
    """Forty extra inputs and forty overlays go into the most fragile string in
    this repo. A failure there has to cost the pictures, not the video — and it
    has to SAY so, because a video that quietly lost its inserts looks exactly
    like one that never planned any."""
    import audio_gen
    src = Path(audio_gen.__file__).read_text(encoding="utf-8")
    assert "full mix without inserts" in src
    assert "insert(s) dropped" in src


# ── captions are for reading, the planner is not reading them ────────────────

def test_caption_cards_are_useless_to_the_insert_planner(monkeypatch):
    """The failure this pins. The planner looks up ONE noun and needs the
    second it was spoken; a caption card holding four words matches nothing it
    asks for, so the whole layer plans zero inserts — silently, and on one
    renderer only."""
    import insert_director as ins
    monkeypatch.setenv("RUFUS_INSERTS", "1")
    cards = [{"text": "the palace held a", "start": 0.0, "end": 1.6},
             {"text": "coin of solid gold", "start": 1.6, "end": 3.2}]
    spoken = [{"text": w, "start": i * 0.4, "end": i * 0.4 + 0.3}
              for i, w in enumerate("the palace held a coin of solid gold".split())]
    script = "The palace held a coin of solid gold."
    assert ins.plan(script, cards) == []
    assert ins.plan(script, spoken), "the same script, one word per entry"


def test_the_remotion_path_plans_from_the_raw_stream():
    """Both renderers build their own `spoken` list for this reason. Remotion
    reused its caption list, which was correct only while a caption was one
    word — the moment long-form grouped them into phrases it would have
    stopped planning inserts and said nothing."""
    src = (Path(__file__).parent.parent / "scripts" / "remotion_renderer.py").read_text(encoding="utf-8")
    assert "plan_for(script, spoken," in src
    assert "plan_for(script, words," not in src
