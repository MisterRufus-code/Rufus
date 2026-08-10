"""Negative-conditioning substitution (comfy_template + comfy_client).

Why this file exists: a live money_history batch of 40 stills came back with
invented lettering on a coin ("national"), a newspaper ("NEVLES / NAOTRO"), a
ledger, a bank facade and two documents. Every one of those prompts already
carried main._DETEXT_CLAUSE — but that clause was a NEGATION living in the
POSITIVE prompt, where CLIP reads its tokens (text, numbers, readable,
lettering) as things to paint. Suppression only works from the negative
conditioning, so comfy_template now finds the text node the export already
wired to the sampler's `negative` input and appends to it.

The safety property under test is that this is a SUBSTITUTION into a proven
wire, never a re-wire: no node is added, removed, or reconnected, and the
export's own negative text survives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_template  # noqa: E402


def _graph(**overrides):
    """A Z-Image-Turbo-shaped stills graph: two CLIPTextEncodes into one
    KSampler, positive and negative distinguished only by which sampler input
    they are wired to (they are the same class, with the same input names —
    which is exactly why the negative can't be found by class name)."""
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "z-image-turbo.safetensors"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "RUFUS_PROMPT", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "ugly, deformed", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 832, "height": 1472, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": 42, "steps": 8, "cfg": 1.0,
                         "model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "out", "images": ["6", 0]}},
    }
    g.update(overrides)
    return g


# ── Finding the negative node ────────────────────────────────────────────────

def test_finds_the_node_wired_to_the_samplers_negative_input():
    assert comfy_template.negative_text_nodes(_graph()) == ["3"]


def test_walks_back_through_conditioning_ops():
    """Real exports put FluxGuidance / ConditioningZeroOut / timestep-range
    nodes between the encode and the sampler. The encode is still the thing
    that carries text."""
    g = _graph()
    g["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}}
    g["9"] = {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["8", 0], "guidance": 3.5}}
    g["5"]["inputs"]["negative"] = ["9", 0]
    assert comfy_template.negative_text_nodes(g) == ["3"]


def test_shared_encode_is_never_touched():
    """Some minimal workflows wire ONE encode into both positive and negative.
    Appending suppression terms there would inject the very words ("text",
    "letters", "watermark") into the positive prompt — strictly worse than
    doing nothing."""
    g = _graph()
    g["5"]["inputs"]["negative"] = ["2", 0]      # same node as positive
    assert comfy_template.negative_text_nodes(g) == []


def test_graph_with_no_negative_input_yields_nothing():
    g = _graph()
    del g["5"]["inputs"]["negative"]
    assert comfy_template.negative_text_nodes(g) == []


# ── Applying it ──────────────────────────────────────────────────────────────

def test_prepare_appends_to_the_exports_own_negative():
    """The export's negative text was part of the run the owner verified — it
    is not ours to discard, only to extend."""
    out = comfy_template.prepare(_graph(), prompt="a coin", negative="text, letters")
    assert out["3"]["inputs"]["text"] == "ugly, deformed, text, letters"


def test_prepare_leaves_the_positive_prompt_alone():
    out = comfy_template.prepare(_graph(), prompt="a coin", negative="text, letters")
    assert out["2"]["inputs"]["text"] == "a coin"
    assert "text, letters" not in out["2"]["inputs"]["text"]


def test_prepare_fills_an_empty_negative_without_a_leading_comma():
    g = _graph()
    g["3"]["inputs"]["text"] = ""
    out = comfy_template.prepare(g, prompt="a coin", negative="text, letters")
    assert out["3"]["inputs"]["text"] == "text, letters"


def test_explicit_placeholder_is_replaced_not_appended():
    """RUFUS_NEGATIVE is the author saying "this text is mine to control"."""
    g = _graph()
    g["3"]["inputs"]["text"] = "RUFUS_NEGATIVE"
    out = comfy_template.prepare(g, prompt="a coin", negative="text, letters")
    assert out["3"]["inputs"]["text"] == "text, letters"


def test_applying_twice_does_not_duplicate():
    once = comfy_template.prepare(_graph(), prompt="a coin", negative="text, letters")
    twice = comfy_template.prepare(once, prompt="a coin", negative="text, letters")
    assert twice["3"]["inputs"]["text"] == "ugly, deformed, text, letters"


def test_no_negative_leaves_the_graph_byte_identical():
    """Every existing caller and every existing template must be unaffected."""
    base = _graph()
    assert comfy_template.prepare(base, prompt="a coin") == \
           comfy_template.prepare(base, prompt="a coin", negative=None)


def test_substitution_never_rewires_the_graph():
    """The proven-template contract: same nodes, same classes, same links —
    only string values change."""
    before = _graph()
    after = comfy_template.prepare(before, prompt="a coin", negative="text")
    after.pop("rufus_save", None)
    assert set(after) == set(before)
    for nid, node in after.items():
        assert node["class_type"] == before[nid]["class_type"]
        for key, val in node["inputs"].items():
            if isinstance(val, list):        # a link
                assert val == before[nid]["inputs"][key], f"{nid}.{key} rewired"


def test_template_without_a_negative_still_prepares_fine():
    """Fail-open: a graph we can't find a negative in must still render."""
    g = _graph()
    del g["5"]["inputs"]["negative"]
    out = comfy_template.prepare(g, prompt="a coin", negative="text, letters")
    assert out["2"]["inputs"]["text"] == "a coin"


# ── The terms themselves ─────────────────────────────────────────────────────

def test_default_negative_leads_with_the_text_terms():
    """Garbled words are the single most obvious AI tell in a finished Short,
    and early terms carry more weight in the encoding."""
    import comfy_client
    neg = comfy_client.DEFAULT_STILLS_NEGATIVE
    assert neg.startswith("text, letters, words")
    for term in ("watermark", "signature", "logo", "gibberish text"):
        assert term in neg


def test_stills_negative_respects_the_off_switch(monkeypatch):
    import comfy_client
    monkeypatch.setenv("RUFUS_STILLS_NEGATIVE", "0")
    assert comfy_client._stills_negative() == ""
    monkeypatch.setenv("RUFUS_STILLS_NEGATIVE", "blurry")
    assert comfy_client._stills_negative() == "blurry"
    monkeypatch.delenv("RUFUS_STILLS_NEGATIVE")
    assert comfy_client._stills_negative() == comfy_client.DEFAULT_STILLS_NEGATIVE


def test_empty_negative_is_not_applied():
    """RUFUS_STILLS_NEGATIVE=0 must leave the export's negative untouched, not
    append an empty string."""
    out = comfy_template.prepare(_graph(), prompt="a coin", negative="")
    assert out["3"]["inputs"]["text"] == "ugly, deformed"


def test_negative_blocks_style_contamination_not_only_text(monkeypatch):
    """A flat-2D look drifts toward whatever medium the subject usually appears
    in — a 1923 street toward sepia photography, a coin toward a 3D product
    render — and one drifted beat among nine flat ones reads worse than either
    look on its own. Naming the mediums to stay out of holds the style far
    better than asking for it once in the positive prompt."""
    import comfy_client
    neg = comfy_client.DEFAULT_STILLS_NEGATIVE
    for term in ("watercolor", "oil painting", "3d render", "pencil sketch",
                 "bokeh", "rough texture", "gradient shading"):
        assert term in neg, term


def test_text_terms_still_come_first():
    """Ordering is load-bearing: early terms weigh more, and garbled words are
    the most visible defect."""
    import comfy_client
    neg = comfy_client.DEFAULT_STILLS_NEGATIVE
    assert neg.index("text") < neg.index("watercolor") < neg.index("extra fingers")
