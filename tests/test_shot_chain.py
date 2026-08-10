"""Carrying the picture forward from one beat to the next.

storyboard.py plans the shots as a sequence and writes down what each one
continues — "the same coin from shot 1, now thinner". That thread reached the
image model as TEXT only, and comfy_client rendered every beat from fresh noise,
so the model read "the same coin" having never seen the coin and invented a new
one. Ten different coins is exactly the incoherence the owner reported.

These tests pin the two halves of the fix: the marker is detected and turned
into an edit instruction, and a template that is really img2img in disguise is
refused — that mistake already cost this project a full run once.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import shot_chain  # noqa: E402
import storyboard  # noqa: E402

_SCENE = "A worn silver coin lies alone on a bare wooden counter"
_CARRY = "the same coin from shot 1, now thinner"
_CHAINED = f"{_SCENE}. {shot_chain.MARKER} {_CARRY}."


# ── The signal: storyboard's own wording, no plumbing between the modules ────

def test_the_marker_is_exactly_what_storyboard_writes():
    """If these two drift apart the chain silently never fires."""
    plan = {"shots": [{"n": 1, "visual": f"{_SCENE} in the light.",
                       "carries_over": None},
                      {"n": 2, "visual": f"{_SCENE} in the dark.",
                       "carries_over": _CARRY}]}
    out = storyboard._clean(plan, 2)
    assert shot_chain.carried(out[1]) == _CARRY
    assert shot_chain.carried(out[0]) == "", "shot 1 continues nothing"


def test_a_beat_that_carries_nothing_is_not_chained():
    assert shot_chain.carried(f"{_SCENE}.") == ""
    assert shot_chain.carried("") == ""


def test_the_scene_is_recoverable_without_the_clause():
    assert shot_chain.scene(_CHAINED) == _SCENE
    assert shot_chain.MARKER not in shot_chain.scene(_CHAINED)


# ── The instruction handed to the edit model ────────────────────────────────

def test_the_edit_prompt_names_what_survives_before_what_changes():
    """An edit model reads this as a change against the image it holds. Naming
    the carried object first keeps it from being treated as one more thing to
    reimagine."""
    p = shot_chain.edit_prompt(_CHAINED)
    assert p.index(_CARRY) < p.index(_SCENE)
    assert "Keep" in p and "Change" in p


def test_the_edit_prompt_demands_a_different_picture():
    """Without this the safest thing an edit model can do is hand back its
    input, which is the exact failure the copy check downstream catches."""
    p = shot_chain.edit_prompt(_CHAINED)
    assert "not the same picture again" in p


def test_the_edit_prompt_carries_the_house_look_forward():
    p = shot_chain.edit_prompt(_CHAINED)
    for word in ("drawing style", "colour palette", "lighting"):
        assert word in p


# ── Preflight: refuse img2img wearing an edit model's wiring ────────────────

def _graph(denoise=1.0, placeholder=True, load_image=True):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT" if placeholder else "a coin"}},
        "5": {"class_type": "KSampler",
              "inputs": {"denoise": denoise, "seed": 1, "latent_image": ["4", 0],
                         "positive": ["2", 0]}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }
    if load_image:
        g["3"] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
    else:
        g["3"] = {"class_type": "EmptyLatentImage", "inputs": {}}
    return g


def _install(tmp_path, monkeypatch, graph):
    p = tmp_path / "shot_chain_api.json"
    p.write_text(json.dumps(graph))
    monkeypatch.setenv("RUFUS_SHOT_CHAIN_TEMPLATE", str(p))
    monkeypatch.delenv("RUFUS_SHOT_CHAIN", raising=False)
    return p


def test_a_real_edit_template_is_accepted(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _graph(denoise=1.0))
    ok, why = shot_chain.ready()
    assert ok, why


def test_an_img2img_template_is_refused(tmp_path, monkeypatch):
    """config/character_stills_api.json was LoadImage -> VAEEncode ->
    KSampler(denoise=0.55) and all ten beats came back as the reference
    portrait. Wiring alone cannot tell the two apart; the denoise can."""
    _install(tmp_path, monkeypatch, _graph(denoise=0.55))
    ok, why = shot_chain.ready()
    assert not ok
    assert "0.55" in why and "img2img" in why


def test_a_template_with_nowhere_to_put_the_previous_shot_is_refused(
        tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _graph(load_image=False))
    ok, why = shot_chain.ready()
    assert not ok and "LoadImage" in why


def test_a_template_without_the_placeholder_is_refused(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _graph(placeholder=False))
    ok, why = shot_chain.ready()
    assert not ok and "RUFUS_PROMPT" in why


def test_no_template_is_the_normal_state_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_SHOT_CHAIN_TEMPLATE",
                       str(tmp_path / "absent.json"))
    monkeypatch.delenv("RUFUS_SHOT_CHAIN", raising=False)
    ok, why = shot_chain.ready()
    assert not ok
    assert "shot_chain_api.json" in why      # says how to enable it
    assert shot_chain.template() is None


def test_disabled_by_env(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _graph())
    monkeypatch.setenv("RUFUS_SHOT_CHAIN", "0")
    assert shot_chain.ready()[0] is False
    assert shot_chain.template() is None


def test_a_beat_carrying_nothing_never_reaches_the_server(tmp_path, monkeypatch):
    """Chaining an unrelated scene would be worse than not chaining at all."""
    _install(tmp_path, monkeypatch, _graph())
    assert shot_chain.continue_shot(tmp_path / "prev.png", _SCENE, 1, "cid") is None


def test_a_broken_render_falls_through(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _graph())
    import svd_client
    monkeypatch.setattr(svd_client, "_upload_image",
                        lambda p: (_ for _ in ()).throw(RuntimeError("down")))
    assert shot_chain.continue_shot(tmp_path / "prev.png", _CHAINED, 1, "cid") is None


# ── comfy_template: the denoise is the discriminator ────────────────────────

def test_loaded_image_denoise_reads_the_sampler():
    import comfy_template
    assert comfy_template.loaded_image_denoise(_graph(denoise=0.55)) == 0.55
    assert comfy_template.loaded_image_denoise(_graph(denoise=1.0)) == 1.0


def test_loaded_image_denoise_is_none_when_nothing_starts_from_an_image():
    import comfy_template
    assert comfy_template.loaded_image_denoise(_graph(load_image=False)) is None


def test_starts_from_loaded_image_still_works():
    import comfy_template
    assert comfy_template.starts_from_loaded_image(_graph()) is True
    assert comfy_template.starts_from_loaded_image(_graph(load_image=False)) is False


# ── The wiring in comfy_client ──────────────────────────────────────────────

_SRC = (Path(__file__).parent.parent / "scripts" / "comfy_client.py").read_text()


def test_comfy_client_only_chains_on_the_first_attempt():
    """A retry exists because the chained result was unusable — chaining again
    would just reproduce it."""
    assert "chain_ready and retry == 0 and anchor_png is not None" in _SRC


def test_comfy_client_exempts_chained_shots_from_the_dup_gate():
    """Resembling the previous shot is the whole point of a chained one."""
    assert "is_dup = (h is not None and not chained" in _SRC


def test_comfy_client_rejects_a_chained_shot_that_is_a_copy():
    assert "CHAIN_COPY_THRESHOLD" in _SRC
    assert shot_chain.MIN_EDIT_DENOISE == 0.9


def test_comfy_client_anchors_the_next_beat_on_raw_output_not_the_fitted_frame():
    """Re-feeding the upscaled/cropped 1080x1920 frame would re-resample on
    every link and compound down the whole video."""
    assert "anchor_png.write_bytes(accepted_raw)" in _SRC
