"""Tests for make_comfy_templates.py — deriving img2img ComfyUI graphs from
the channel owner's already-proven txt2img export.

The point of deriving rather than hand-writing: comfy_template.py bans building
a graph from documentation, because guessed wiring and sampler settings were
subtly wrong when that was tried. Everything the proven export verified must
survive untouched; only what img2img actually requires may change."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import make_comfy_templates as mct


def _proven_graph():
    """A stand-in for the real config/stills_api.json — the shipped Z-Image
    export's actual shape (empty-latent txt2img, separate VAELoader)."""
    return {
        "16": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "z_image_turbo_bf16.safetensors",
                          "weight_dtype": "fp8_e4m3fn"}},
        "18": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "11": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"shift": 3, "model": ["16", 0]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT", "clip": ["18", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["18", 0]}},
        "13": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": 832, "height": 1472, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": 47447417949230, "steps": 9, "cfg": 1,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1, "model": ["11", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["13", 0]}},
        "17": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["17", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "rufus_stills", "images": ["8", 0]}},
    }


# ── What must be PRESERVED — the whole reason for deriving ───────────────────

def test_every_verified_setting_survives_untouched():
    """Loaders, weight dtype, sampler, scheduler, cfg and shift are what the
    proven export actually verified. Changing any of them silently would be
    exactly the failure comfy_template.py warns about."""
    src = _proven_graph()
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="rufus_i2i")

    assert out["16"] == src["16"]           # UNETLoader + fp8 dtype
    assert out["18"] == src["18"]           # CLIPLoader + lumina2 type
    assert out["11"] == src["11"]           # ModelSamplingAuraFlow shift
    assert out["17"] == src["17"]           # VAELoader
    for key in ("cfg", "sampler_name", "scheduler", "model", "positive", "negative"):
        assert out["3"]["inputs"][key] == src["3"]["inputs"][key], key


def test_prompt_placeholder_survives():
    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=14, save_prefix="p")
    assert out["6"]["inputs"]["text"] == "RUFUS_PROMPT"


# ── The four changes img2img requires ────────────────────────────────────────

def test_adds_loadimage_and_vaeencode():
    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=14, save_prefix="p")
    assert out[mct.LOAD_ID]["class_type"] == "LoadImage"
    assert out[mct.ENCODE_ID]["class_type"] == "VAEEncode"
    assert out[mct.ENCODE_ID]["inputs"]["pixels"] == [mct.LOAD_ID, 0]


def test_vae_encode_reuses_the_exact_wire_vaedecode_uses():
    """Not a hunted-for VAELoader id — reusing the reference keeps this right
    when the VAE comes bundled out of a checkpoint loader instead."""
    src = _proven_graph()
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")
    assert out[mct.ENCODE_ID]["inputs"]["vae"] == src["8"]["inputs"]["vae"]


def test_vae_encode_follows_a_checkpoint_bundled_vae():
    """Same graph shape but the VAE arrives on output slot 2 of a checkpoint
    loader — the derivation must follow that, not assume a VAELoader."""
    src = _proven_graph()
    del src["17"]
    src["20"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}}
    src["8"]["inputs"]["vae"] = ["20", 2]
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")
    assert out[mct.ENCODE_ID]["inputs"]["vae"] == ["20", 2]


def test_sampler_takes_the_encoded_latent_and_the_new_denoise():
    out = mct.derive_i2i(_proven_graph(), denoise=0.45, steps=14, save_prefix="p")
    assert out["3"]["inputs"]["latent_image"] == [mct.ENCODE_ID, 0]
    assert out["3"]["inputs"]["denoise"] == 0.45


def test_steps_are_raised_because_denoise_multiplies_them():
    """A distilled model's 9 steps at denoise 0.4 leaves ~4 effective steps —
    too few to move the picture."""
    src = _proven_graph()
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")
    assert out["3"]["inputs"]["steps"] == 14 > src["3"]["inputs"]["steps"]


def test_steps_left_alone_when_not_requested():
    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=None, save_prefix="p")
    assert out["3"]["inputs"]["steps"] == 9


def test_orphaned_empty_latent_is_dropped():
    """Its size inputs no longer do anything — the size comes from the init
    image now — and leaving it invites someone to 'fix' the resolution there."""
    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=14, save_prefix="p")
    assert "13" not in out


def test_shared_latent_source_is_kept():
    """Only drop it if genuinely unreferenced."""
    src = _proven_graph()
    src["21"] = {"class_type": "SomeOtherNode", "inputs": {"latent": ["13", 0]}}
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")
    assert "13" in out


def test_save_prefix_is_rebranded():
    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=14, save_prefix="rufus_i2i")
    assert out["9"]["inputs"]["filename_prefix"] == "rufus_i2i"


# ── Refuses to guess ─────────────────────────────────────────────────────────

def test_raises_without_a_sampler():
    src = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}}}
    with pytest.raises(ValueError, match="sampler"):
        mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")


def test_raises_without_a_vaedecode():
    src = _proven_graph()
    del src["8"]
    with pytest.raises(ValueError, match="VAE"):
        mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")


def test_sampler_found_by_inputs_not_class_name():
    """A renamed/custom sampler still resolves."""
    src = _proven_graph()
    src["3"]["class_type"] = "SomeCustomSampler"
    out = mct.derive_i2i(src, denoise=0.4, steps=14, save_prefix="p")
    assert out["3"]["inputs"]["latent_image"] == [mct.ENCODE_ID, 0]


# ── The produced files must satisfy the pipeline's own loaders ───────────────

def test_derived_graph_passes_comfy_template_contract():
    import comfy_template

    out = mct.derive_i2i(_proven_graph(), denoise=0.4, steps=14, save_prefix="rufus_i2i")
    assert comfy_template.has_placeholder(out)

    prepared = comfy_template.prepare(out, prompt="a florin, a moment later",
                                      image_name="prev.png", seed=999,
                                      save_prefix="rufus_i2i")
    assert prepared["6"]["inputs"]["text"] == "a florin, a moment later"
    assert prepared[mct.LOAD_ID]["inputs"]["image"] == "prev.png"
    assert prepared["3"]["inputs"]["seed"] == 999
    # prepare() must not disturb what makes this img2img at all
    assert prepared["3"]["inputs"]["denoise"] == 0.4
    assert prepared["3"]["inputs"]["latent_image"] == [mct.ENCODE_ID, 0]


def test_shipped_templates_load_through_comfy_client():
    """The files actually committed must be loadable by the code that uses
    them — not merely valid JSON."""
    import comfy_client
    import comfy_template

    # character_stills_api.json is deliberately NOT in this list any more — see
    # test_derived_character_template_is_rejected below for why deriving one
    # was wrong.
    for path in (Path(__file__).parent.parent / "config" / "stills_i2i_api.json",):
        assert path.exists(), f"{path.name} missing"
        tpl = comfy_template.load_template(path)
        assert tpl is not None, f"{path.name} is not a loadable API export"
        assert comfy_template.has_placeholder(tpl), f"{path.name} lost RUFUS_PROMPT"
        assert any(n.get("class_type") == "LoadImage" for n in tpl.values()), \
            f"{path.name} has no LoadImage for the init frame"
        sampler = mct._sampler_id(tpl)
        assert sampler and tpl[sampler]["inputs"]["denoise"] < 1.0, \
            f"{path.name} is not actually img2img (denoise must be < 1.0)"


# ── A derived character template is the wrong thing entirely ─────────────────
# config/character_stills_api.json used to be derived here, mechanically, as
# plain img2img at denoise 0.55. That is not an identity lock — it makes the
# reference portrait the START LATENT, so the sampler can only redraw it.
#
# Run #59 is the proof. All ten beats came back as the same hooded figure
# standing centred on a plain background; the prompts had asked for miners
# swinging pickaxes, a newspaper office, a mining camp, a portrait wall and a
# classroom door, and not one of those scenes appeared. The near-duplicate
# detector fired on every single clip and was RIGHT — they genuinely were the
# same picture, three times over per beat counting the wasted retries.
#
# An identity lock needs the reference as CONDITIONING (IPAdapter / PuLID /
# InstantID) with the latent still starting from noise. That node cannot be
# invented from an existing txt2img export, which is exactly the case
# comfy_template.py's header reserves for a user-exported proven workflow.

def _img2img_graph():
    """The shape that was shipped: LoadImage → VAEEncode → sampler latent."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": "rufus_init.png"}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0]}},
        "3": {"class_type": "KSampler",
              "inputs": {"denoise": 0.55, "positive": ["2", 0],
                         "latent_image": ["enc", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }


def test_generator_no_longer_writes_a_character_template(tmp_path, monkeypatch):
    src = Path(__file__).parent.parent / "config" / "stills_api.json"
    monkeypatch.setattr(mct, "CONFIG", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_comfy_templates.py", "--source", str(src)])
    mct.main()
    assert (tmp_path / "stills_i2i_api.json").exists(), "the i2i template is still derived"
    assert not (tmp_path / "character_stills_api.json").exists()


def test_img2img_is_detected_as_not_image_conditioned():
    import comfy_template
    g = _img2img_graph()
    assert comfy_template.starts_from_loaded_image(g)
    assert not comfy_template.is_image_conditioned(g)


def test_an_ipadapter_graph_is_accepted():
    """The real thing: reference as conditioning, latent from noise."""
    import comfy_template
    g = _img2img_graph()
    g["ipa"] = {"class_type": "IPAdapterApply", "inputs": {"image": ["img", 0]}}
    g["3"]["inputs"]["latent_image"] = ["empty", 0]
    g["empty"] = {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 832, "height": 1472}}
    assert comfy_template.is_image_conditioned(g)
    assert not comfy_template.starts_from_loaded_image(g)


def test_character_path_refuses_an_img2img_template(tmp_path, monkeypatch, capsys):
    """The durable guard: even if this file reappears, the run must not spend
    ten renders reproducing one portrait."""
    import comfy_client
    bad = tmp_path / "character_stills_api.json"
    bad.write_text(json.dumps(_img2img_graph()))
    monkeypatch.setattr(comfy_client, "CHARACTER_TEMPLATE", bad)
    monkeypatch.delenv("RUFUS_CHARACTER_TEMPLATE", raising=False)
    assert comfy_client._character_template() is None
    assert "plain img2img" in capsys.readouterr().out


def test_character_path_accepts_a_real_conditioning_template(tmp_path, monkeypatch):
    import comfy_client
    g = _img2img_graph()
    g["ipa"] = {"class_type": "PuLIDApply", "inputs": {"image": ["img", 0]}}
    g["3"]["inputs"]["latent_image"] = ["empty", 0]
    g["empty"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 832}}
    good = tmp_path / "character_stills_api.json"
    good.write_text(json.dumps(g))
    monkeypatch.setattr(comfy_client, "CHARACTER_TEMPLATE", good)
    monkeypatch.delenv("RUFUS_CHARACTER_TEMPLATE", raising=False)
    assert comfy_client._character_template() is not None


def test_the_i2i_template_is_still_allowed_to_be_img2img():
    """Chained frames genuinely SHOULD continue the previous frame — the guard
    must not spill onto the mode it was never about."""
    import comfy_template
    tpl = comfy_template.load_template(
        Path(__file__).parent.parent / "config" / "stills_i2i_api.json")
    assert comfy_template.starts_from_loaded_image(tpl)
