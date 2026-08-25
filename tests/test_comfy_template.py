"""Tests for comfy_template.py — user-exported API workflows as Rufus engines."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_template as ct


def _i2v_graph():
    """A miniature HunyuanVideo-style i2v API export."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "hy15.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "bad quality", "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "5": {"class_type": "HunyuanVideo15ImageToVideo",
              "inputs": {"width": 1280, "height": 720, "length": 121,
                         "positive": ["2", 0], "negative": ["3", 0],
                         "start_image": ["4", 0]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0]}},
        "8": {"class_type": "CreateVideo", "inputs": {"images": ["7", 0], "fps": 24}},
        "9": {"class_type": "SaveVideo",
              "inputs": {"video": ["8", 0], "filename_prefix": "video"}},
    }


# ── load_template ─────────────────────────────────────────────────────────────

def test_load_template_bare_graph(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_i2v_graph()))
    assert ct.load_template(p) is not None


def test_load_template_prompt_wrapper(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"prompt": _i2v_graph()}))
    g = ct.load_template(p)
    assert g is not None and "1" in g


def test_load_template_missing_file(tmp_path):
    assert ct.load_template(tmp_path / "nope.json") is None


def test_load_template_invalid_json(tmp_path):
    p = tmp_path / "g.json"
    p.write_text("{ not json")
    assert ct.load_template(p) is None


def test_load_template_not_a_graph(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"a": {"no_class": 1}}))
    assert ct.load_template(p) is None


# ── placeholder detection ─────────────────────────────────────────────────────

def test_has_placeholder_true():
    assert ct.has_placeholder(_i2v_graph())


def test_has_placeholder_false():
    g = _i2v_graph()
    g["2"]["inputs"]["text"] = "a real prompt"
    assert not ct.has_placeholder(g)


# ── prepare: substitutions ────────────────────────────────────────────────────

def test_prepare_substitutes_prompt_image_seed_dims():
    g = ct.prepare(_i2v_graph(), prompt="a man reading", image_name="up.png",
                   seed=777, dims=(480, 832, 121))
    assert g["2"]["inputs"]["text"] == "a man reading"
    assert g["3"]["inputs"]["text"] == "bad quality"      # negative untouched
    assert g["4"]["inputs"]["image"] == "up.png"
    assert g["6"]["inputs"]["noise_seed"] == 777
    assert g["5"]["inputs"]["width"] == 480
    assert g["5"]["inputs"]["height"] == 832
    assert g["5"]["inputs"]["length"] == 121


def test_prepare_does_not_mutate_original():
    g0 = _i2v_graph()
    ct.prepare(g0, prompt="x", seed=1)
    assert g0["2"]["inputs"]["text"] == "RUFUS_PROMPT"
    assert g0["6"]["inputs"]["noise_seed"] == 42


def test_prepare_swaps_video_writers_for_saveimage():
    """SaveVideo/CreateVideo produce an mp4 Rufus can't post-process — they
    must be replaced by SaveImage frames wired to the VAEDecode output."""
    g = ct.prepare(_i2v_graph(), prompt="x", save_prefix="rufus_hy")
    classes = {n["class_type"] for n in g.values()}
    assert "SaveVideo" not in classes and "CreateVideo" not in classes
    assert g["rufus_save"]["class_type"] == "SaveImage"
    assert g["rufus_save"]["inputs"]["images"] == ["7", 0]   # the VAEDecode
    assert g["rufus_save"]["inputs"]["filename_prefix"] == "rufus_hy"


def test_prepare_image_workflow_rebrands_saveimage_prefix():
    g0 = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "2": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "flux2", "images": ["1", 0]}},
    }
    g = ct.prepare(g0, prompt="p", save_prefix="rufus_flux2")
    assert g["2"]["inputs"]["filename_prefix"] == "rufus_flux2"


def test_prepare_dims_only_on_nodes_with_all_three():
    g0 = {
        "1": {"class_type": "SomeNode", "inputs": {"width": 1, "height": 2}},
        "2": {"class_type": "I2V", "inputs": {"width": 1, "height": 2, "length": 3}},
    }
    g = ct.prepare(g0, dims=(480, 832, 121))
    assert g["1"]["inputs"]["width"] == 1                  # untouched (no length)
    assert g["2"]["inputs"]["length"] == 121


def test_prepare_placeholder_inside_longer_text():
    g0 = {"1": {"class_type": "CLIPTextEncode",
                "inputs": {"text": "cinematic. RUFUS_PROMPT. no text"}}}
    g = ct.prepare(g0, prompt="a coin spins")
    assert g["1"]["inputs"]["text"] == "cinematic. a coin spins. no text"


# ── missing_nodes ─────────────────────────────────────────────────────────────

def test_missing_nodes_reports_unknown_classes():
    def fake_get(url, timeout=10):
        class R:
            status_code = 200
            def json(self):
                ct_name = url.rsplit("/", 1)[-1]
                return {ct_name: {"input": {}}} if ct_name != "SaveVideo" else {}
        return R()

    with patch.object(ct.requests, "get", side_effect=fake_get):
        missing = ct.missing_nodes(_i2v_graph(), "http://x")
    assert missing == ["SaveVideo"]


def test_missing_nodes_server_down_reports_all():
    with patch.object(ct.requests, "get", side_effect=OSError("down")):
        missing = ct.missing_nodes(_i2v_graph(), "http://x")
    assert len(missing) == len({n["class_type"] for n in _i2v_graph().values()})


# ── missing_models: catch a stale model filename at PREFLIGHT ─────────────────
# missing_nodes() passes on a graph whose node classes all exist but whose
# weights file is gone — ComfyUI then rejects the submit with
# "value_not_in_list". Live, that surfaced only after the whole stills phase
# had run, wasting the run.

def _loader_graph(unet="model_a.safetensors", class_type="UNETLoader"):
    return {"12": {"class_type": class_type,
                   "inputs": {"unet_name": unet, "weight_dtype": "default"}}}


def _object_info(class_type, names):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {class_type: {"input": {"required": {
        "unet_name": [names, {"tooltip": "x"}]}}}}
    return r


def test_missing_models_flags_filename_not_on_disk():
    """The exact live failure: config named the superseded 480p fp16 build
    while only the step-distilled fp8 file was actually present."""
    graph = _loader_graph("hunyuanvideo1.5_480p_i2v_fp16.safetensors")
    resp = _object_info("UNETLoader",
                        ["hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors"])
    with patch.object(ct.requests, "get", return_value=resp):
        missing = ct.missing_models(graph, "http://x")
    assert len(missing) == 1
    assert "hunyuanvideo1.5_480p_i2v_fp16.safetensors" in missing[0]
    assert "UNETLoader.unet_name" in missing[0]


def test_missing_models_empty_when_file_present():
    graph = _loader_graph("model_a.safetensors")
    with patch.object(ct.requests, "get",
                      return_value=_object_info("UNETLoader", ["model_a.safetensors"])):
        assert ct.missing_models(graph, "http://x") == []


def test_missing_models_fails_open_when_object_info_unreadable():
    """Must never block a run it can't actually verify."""
    graph = _loader_graph("whatever.safetensors")
    with patch.object(ct.requests, "get", side_effect=OSError("down")):
        assert ct.missing_models(graph, "http://x") == []


def test_missing_models_flags_gguf_loader_filename_not_on_disk():
    """UnetLoaderGGUF (city96's ComfyUI-GGUF pack) — the loader a GGUF-
    quantized LTX checkpoint swap depends on. Same live-failure shape as
    UNETLoader: a stale/misnamed .gguf file must be caught at preflight."""
    graph = _loader_graph("LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
                          class_type="UnetLoaderGGUF")
    resp = _object_info("UnetLoaderGGUF", ["some-other-quant.gguf"])
    with patch.object(ct.requests, "get", return_value=resp):
        missing = ct.missing_models(graph, "http://x")
    assert len(missing) == 1
    assert "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf" in missing[0]
    assert "UnetLoaderGGUF.unet_name" in missing[0]


def test_missing_models_empty_when_gguf_file_present():
    graph = _loader_graph("model.gguf", class_type="UnetLoaderGGUF")
    with patch.object(ct.requests, "get",
                      return_value=_object_info("UnetLoaderGGUF", ["model.gguf"])):
        assert ct.missing_models(graph, "http://x") == []


def test_missing_models_ignores_non_loader_nodes():
    graph = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}
    with patch.object(ct.requests, "get") as get:
        assert ct.missing_models(graph, "http://x") == []
    get.assert_not_called()   # no probe for a node that names no model file


# ── Shipped configs stay valid Rufus templates ────────────────────────────────
# The repo ships hand-built API exports (config/hunyuan_i2v_api.json for motion,
# config/stills_api.json for the commercial-free Z-Image stills model). A typo
# in either — a broken JSON, or losing the RUFUS_PROMPT placeholder — silently
# disables the engine (falls back to FLUX/Ken Burns) with no error, so guard it.

CONFIG_DIR = Path(__file__).parent.parent / "config"


@pytest.mark.parametrize("name", ["hunyuan_i2v_api.json", "stills_api.json"])
def test_shipped_config_is_a_valid_rufus_template(name):
    tpl = ct.load_template(CONFIG_DIR / name)
    assert tpl is not None, f"{name} is not a loadable API graph"
    assert ct.has_placeholder(tpl), f"{name} lost its RUFUS_PROMPT placeholder"


def test_shipped_stills_config_substitutes_and_is_portrait():
    tpl = ct.load_template(CONFIG_DIR / "stills_api.json")
    g = ct.prepare(tpl, prompt="a real photo", seed=123, save_prefix="rufus_stills")
    # prompt substituted, no placeholder left anywhere
    assert not ct.has_placeholder(g)
    # portrait latent (taller than wide) so Rufus' 1080x1920 crop keeps the subject
    latent = next(n for n in g.values() if n["class_type"] == "EmptySD3LatentImage")
    assert latent["inputs"]["height"] > latent["inputs"]["width"]


# ── LTX-style nodes size in SECONDS, not frames ───────────────────────────────

def test_prepare_handles_duration_style_dims():
    """LTX's all-in-one i2v node takes width/height/duration(+fps) instead of
    width/height/length. Without this the dims substitution silently no-ops and
    the export's own 1280x720 LANDSCAPE wins over the portrait pipeline."""
    g = {"1": {"class_type": "LTXImageToVideo",
               "inputs": {"width": 1280, "height": 720, "duration": 5, "fps": 25,
                          "prompt": "RUFUS_PROMPT"}}}
    out = ct.prepare(g, prompt="x", dims=(832, 1472, 121))
    ins = out["1"]["inputs"]
    assert (ins["width"], ins["height"]) == (832, 1472)
    assert ins["duration"] == 5          # 121 frames / 25fps, rounded


def test_prepare_duration_survives_a_bad_fps():
    g = {"1": {"class_type": "LTXImageToVideo",
               "inputs": {"width": 1, "height": 2, "duration": 5, "fps": 0}}}
    out = ct.prepare(g, dims=(832, 1472, 121))     # must not raise
    assert out["1"]["inputs"]["width"] == 832


def test_prepare_still_prefers_length_when_present():
    g = {"1": {"class_type": "HunyuanVideo15ImageToVideo",
               "inputs": {"width": 1, "height": 2, "length": 3}}}
    out = ct.prepare(g, dims=(480, 832, 121))
    assert out["1"]["inputs"]["length"] == 121


# ── a negative that lands nowhere ────────────────────────────────────────────

def _sampler_graph(with_negative: bool):
    """A minimal graph, with and without a text node wired to `negative`."""
    g = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}},
        "3": {"class_type": "KSampler",
              "inputs": {"positive": ["1", 0], "seed": 1}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    if with_negative:
        g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}}
        g["3"]["inputs"]["negative"] = ["2", 0]
    return g


def test_a_negative_that_lands_nowhere_is_loud(capsys):
    """FAIL-OPEN WITHOUT FAIL-LOUD IS FAIL-SILENT, in the one place that
    suppresses the most obvious AI tell. A workflow with no negative text node
    — a turbo or flow-match graph that runs without CFG — takes the
    substitution and renders with no suppression at all, and everything
    downstream looks fine: the terms are in the log, the prompt is right, the
    render succeeds. A real gallery came back with readable sentences written
    across two of its sixteen frames while the negative led with "text,
    letters, words"."""
    ct._NEG_WARNED = False
    ct.prepare(_sampler_graph(False), prompt="x", negative="text, letters")
    out = capsys.readouterr().out
    assert "landed NOWHERE" in out
    assert "RUFUS_NEGATIVE" in out, "the warning has to name the fix"


def test_a_workflow_that_takes_the_negative_says_nothing(capsys):
    """The warning fires on a real defect or it is noise — and this is the
    normal case."""
    ct._NEG_WARNED = False
    g = ct.prepare(_sampler_graph(True), prompt="x",
                               negative="text, letters")
    assert "text, letters" in g["2"]["inputs"]["text"]
    assert capsys.readouterr().out == ""


def test_the_placeholder_counts_as_landing(capsys):
    ct._NEG_WARNED = False
    g = dict(_sampler_graph(False))
    g["2"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": f"blurry, {ct.NEG_PLACEHOLDER}"}}
    out_graph = ct.prepare(g, prompt="x", negative="text, letters")
    assert "text, letters" in out_graph["2"]["inputs"]["text"]
    assert capsys.readouterr().out == ""


def test_the_warning_is_said_once_not_per_beat(capsys):
    """A hundred and fifty identical lines is not a louder warning."""
    ct._NEG_WARNED = False
    for _ in range(5):
        ct.prepare(_sampler_graph(False), prompt="x", negative="text")
    assert capsys.readouterr().out.count("landed NOWHERE") == 1


# ── the file is exported on Windows, and Windows writes three encodings ──────
#
# "no stills model configured" for a workflow the owner had saved correctly by
# every measure visible to them. load_template read the file as strict utf-8 and
# caught only OSError and JSONDecodeError, so a BOM came back as a bare None and
# UTF-16 raised straight through. Notepad writes the first; PowerShell's `>`
# writes the second.

def test_a_workflow_saved_by_notepad_still_loads(tmp_path):
    """UTF-8 with a BOM is what Notepad writes, and json.loads rejects it."""
    p = tmp_path / "stills_api.json"
    body = '{"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "RUFUS_PROMPT"}}}'
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    g = ct.load_template(p)
    assert g is not None and ct.has_placeholder(g)


def test_a_utf16_export_is_refused_by_name_not_by_traceback(tmp_path, capsys):
    """PowerShell's `>` writes UTF-16. This used to raise UnicodeDecodeError
    out of a function whose contract is "returns the graph or None"."""
    p = tmp_path / "stills_api.json"
    p.write_bytes('{"1": {"class_type": "X", "inputs": {}}}'.encode("utf-16"))
    assert ct.load_template(p) is None
    assert "not UTF-8" in capsys.readouterr().out


def test_malformed_json_says_what_is_wrong_with_it(tmp_path, capsys):
    p = tmp_path / "stills_api.json"
    p.write_text("{not json", encoding="utf-8")
    assert ct.load_template(p) is None
    assert "not valid JSON" in capsys.readouterr().out
