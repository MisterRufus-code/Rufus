#!/usr/bin/env python3
"""
comfy_template.py — run a user-exported ComfyUI API workflow as a Rufus engine.

Why templates instead of hand-built graphs: the Wan 2.2 integration proved
that blind-wiring a ComfyUI graph from documentation gets the node wiring or
the settings subtly wrong, and only an API export of a proven-good run on the
actual rig sets it right. This module makes that the FIRST-CLASS path for new
engines (HunyuanVideo 1.5 motion, FLUX.2 stills): the channel owner runs the
built-in ComfyUI template once, verifies the output, sets the positive prompt
to the literal placeholder RUFUS_PROMPT, exports with
"Export (API)" and drops the JSON into config/. Rufus then reuses EXACTLY that
proven graph, substituting only:

  - any input string equal to "RUFUS_PROMPT"      → the per-clip prompt
  - any input string equal to "RUFUS_NEGATIVE"    → the per-clip negative, or
    else the text node already wired to the         appended to whatever the
    sampler's `negative` input                      export already had there
  - the first LoadImage node's image              → the uploaded init frame
  - any seed / noise_seed input                   → a fresh random seed
  - width/height/length on nodes that have all 3  → Rufus portrait dims+frames
  - video-writer nodes (SaveVideo etc.)           → replaced with SaveImage on
                                                    the VAEDecode output, so the
                                                    existing frame-polling +
                                                    ffmpeg assembly pipeline
                                                    works unchanged

Everything else — samplers, steps, cfg, shift, model files, resolution tricks —
stays exactly as the export captured it, because that's what was verified.
"""

import copy
import json
from pathlib import Path

import requests

PLACEHOLDER = "RUFUS_PROMPT"
NEG_PLACEHOLDER = "RUFUS_NEGATIVE"

# Nodes that write a video container — replaced with SaveImage frames because
# Rufus assembles its own mp4 (interpolation, upscale, freeze-extend).
_VIDEO_WRITER_CLASSES = {
    "SaveVideo", "CreateVideo", "SaveWEBM", "SaveAnimatedWEBP",
    "SaveAnimatedPNG", "VHS_VideoCombine",
}


def load_template(path: Path) -> dict | None:
    """Parse an API-export JSON. Returns the graph dict or None (missing/bad).

    Accepts both a bare graph ({"1": {...}, ...}) and the occasional wrapper
    ({"prompt": {...}}) some exporters produce."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        data = data["prompt"]
    if not isinstance(data, dict):
        return None
    # A graph is a dict of node dicts each having class_type.
    nodes = [v for v in data.values() if isinstance(v, dict)]
    if not nodes or not all("class_type" in n for n in nodes):
        return None
    return data


def has_placeholder(graph: dict) -> bool:
    """True if some node input carries the RUFUS_PROMPT placeholder."""
    for node in graph.values():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, str) and PLACEHOLDER in v:
                return True
    return False


def missing_nodes(graph: dict, host: str) -> list[str]:
    """Which of the template's node classes the running ComfyUI doesn't have.
    Probes /object_info per unique class. Raises nothing — an unreachable
    server reports every class as missing (callers treat that as not-ready)."""
    missing = []
    for ct in sorted({n["class_type"] for n in graph.values()}):
        try:
            r = requests.get(f"{host}/object_info/{ct}", timeout=10)
            if r.status_code != 200 or not r.json().get(ct):
                missing.append(ct)
        except Exception:
            missing.append(ct)
    return missing


# Loader inputs whose value is a MODEL FILENAME that ComfyUI validates against
# the files actually on disk. class_type -> input names to check.
_MODEL_FILE_INPUTS = {
    "UNETLoader":        ("unet_name",),
    "CheckpointLoaderSimple": ("ckpt_name",),
    "VAELoader":         ("vae_name",),
    "CLIPLoader":        ("clip_name",),
    "DualCLIPLoader":    ("clip_name1", "clip_name2"),
    "CLIPVisionLoader":  ("clip_name",),
    "LoraLoaderModelOnly": ("lora_name",),
    "LatentUpscaleModelLoader": ("model_name",),
    # city96's ComfyUI-GGUF pack — needed for GGUF-quantized checkpoints
    # (e.g. the LTX-2.3 Q4_K_M build recommended for 24GB-VRAM boxes).
    "UnetLoaderGGUF":    ("unet_name",),
}


def missing_models(graph: dict, host: str) -> list[str]:
    """Model FILENAMES the template references that ComfyUI can't load.

    missing_nodes() only checks node CLASSES exist — a graph can pass that and
    still be rejected at submit time with
    "value_not_in_list: unet_name '<file>' not in [...]" when the referenced
    weights file isn't on disk. That happened live: a config still naming the
    superseded 480p fp16 checkpoint failed only AFTER the whole stills phase
    had run, wasting the run. Checking it in the same preflight as the node
    classes turns that into an upfront, actionable message.

    Each loader's valid filenames come from /object_info/<class>, whose
    required-input spec carries the enum list. Unreadable/unknown shapes are
    skipped (fail-open) — this must never block a run it can't actually
    verify."""
    missing: list[str] = []
    cache: dict[str, set[str] | None] = {}
    for node in graph.values():
        ct = node.get("class_type")
        fields = _MODEL_FILE_INPUTS.get(ct)
        if not fields:
            continue
        if ct not in cache:
            cache[ct] = _loader_choices(ct, host)
        for field in fields:
            value = (node.get("inputs") or {}).get(field)
            choices = cache[ct]
            if isinstance(value, str) and choices and value not in choices:
                missing.append(f"{value} (for {ct}.{field})")
    return missing


def _loader_choices(class_type: str, host: str) -> set[str] | None:
    """Filenames a loader node will accept, or None when it can't be read."""
    try:
        r = requests.get(f"{host}/object_info/{class_type}", timeout=10)
        if r.status_code != 200:
            return None
        required = (r.json().get(class_type, {})
                    .get("input", {}).get("required", {}))
    except Exception:
        return None
    names: set[str] = set()
    for spec in required.values():
        # A file-enum input is [[...names...], {...opts}] or [[...names...]]
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            names.update(s for s in spec[0] if isinstance(s, str))
    return names or None


# ── Image conditioning vs. img2img ───────────────────────────────────────────
# These are two different things and confusing them produced a whole ruined
# run. A recurring-character workflow must take the reference portrait as
# CONDITIONING (IPAdapter / PuLID / InstantID / a reference style model) while
# the latent still starts from noise, so the scene is free to be whatever the
# prompt says and only the identity is carried over.
#
# Feed the same reference in as the STARTING LATENT instead — LoadImage →
# VAEEncode → KSampler.latent_image, i.e. ordinary img2img — and the sampler
# can only redraw the reference. At denoise 0.55 the composition is locked to
# it. Live: config/character_stills_api.json had exactly that shape, and all
# ten beats of run #59 came back as the same hooded figure standing centred on
# a plain background. The prompts asked for miners swinging pickaxes, a
# newspaper office, a mining camp and a classroom; none of them appeared. The
# near-duplicate detector fired on every clip and was RIGHT — they genuinely
# were the same picture.
_IMAGE_CONDITIONING_HINTS = (
    "ipadapter", "pulid", "instantid", "faceid", "reference",
    "stylemodel", "redux", "controlnet", "clipvision",
)


def is_image_conditioned(graph: dict) -> bool:
    """True if the reference image reaches the sampler as CONDITIONING.

    False for a plain img2img graph, where the reference is the start latent
    and the sampler can only reproduce it."""
    for node in graph.values():
        ct = str(node.get("class_type", "")).lower()
        if any(h in ct for h in _IMAGE_CONDITIONING_HINTS):
            return True
    return False


def _samplers_starting_from_loaded_image(graph: dict) -> list[dict]:
    """Nodes whose `latent_image` traces back to a LoadImage through a VAEEncode."""
    load_ids = {nid for nid, n in graph.items()
                if str(n.get("class_type", "")) == "LoadImage"}
    if not load_ids:
        return []
    encode_ids = set()
    for nid, node in graph.items():
        if "VAEEncode" not in str(node.get("class_type", "")):
            continue
        for v in (node.get("inputs") or {}).values():
            if _link_target(v) in load_ids:
                encode_ids.add(nid)
    if not encode_ids:
        return []
    return [node for node in graph.values()
            if _link_target((node.get("inputs") or {}).get("latent_image")) in encode_ids]


def starts_from_loaded_image(graph: dict) -> bool:
    """True if a LoadImage becomes the sampler's starting latent (img2img).

    Legitimate for chained frames — stills_i2i_api.json genuinely wants to
    continue the previous frame — and wrong for a character reference, where
    it means every beat renders the reference portrait again."""
    return bool(_samplers_starting_from_loaded_image(graph))


def loaded_image_denoise(graph: dict) -> float | None:
    """Denoise of the sampler that starts from a loaded image, else None.

    This one number separates two graphs that are wired identically. An EDIT
    model (Qwen-Image-Edit, Kontext) legitimately encodes the source image into
    the start latent and samples at denoise 1.0 — the picture is rebuilt from
    noise and the source acts as instruction, so the scene can change completely
    while the world holds. A plain img2img graph has the same wiring at denoise
    0.4-0.6, where the sampler can only redraw what it was handed.

    That distinction is not academic here: config/character_stills_api.json was
    the second kind at denoise 0.55, and every one of the ten beats came back as
    the reference portrait. Wiring alone could not tell them apart; this can.

    None when no sampler starts from a loaded image, or when the value isn't a
    plain number (a link, say) — callers treat None as "can't tell", not "safe".
    """
    for node in _samplers_starting_from_loaded_image(graph):
        val = (node.get("inputs") or {}).get("denoise")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


# ── Negative conditioning ────────────────────────────────────────────────────
# Why this exists: every text-suppression rule Rufus had lived in the POSITIVE
# prompt, phrased as a negation ("absolutely no readable text, numbers, or
# interface elements"). CLIP has no "not" operator — it sees the tokens text,
# numbers, readable, lettering and happily paints them. A live money_history
# batch came back with garbled words on a coin, a newspaper, a ledger, a bank
# facade and two documents; the de-text clause was present in every one of
# those prompts. Suppression only works from the negative conditioning, so
# find the text node the export already wired to the sampler's `negative`
# input and append to it — a substitution into a proven wire, exactly like the
# seed/dims ones, never a re-wire.

# Inputs that carry prompt text on an encode node, most specific first.
_TEXT_INPUT_KEYS = ("text", "text_g", "text_l", "prompt", "string")

# How far back to walk from a sampler's `negative` link before giving up.
# Real graphs put 0-3 conditioning ops (FluxGuidance, ConditioningZeroOut,
# ConditioningSetTimestepRange) between the encode and the sampler.
_NEG_WALK_DEPTH = 5


def _link_target(value) -> str | None:
    """The node id a ComfyUI API-format input link points at, else None."""
    if isinstance(value, list) and len(value) == 2:
        return str(value[0])
    return None


def _text_nodes_behind(graph: dict, root: str) -> set[str]:
    """Ids of text-carrying nodes reachable backwards from node `root`."""
    found: set[str] = set()
    seen: set[str] = set()
    frontier = [(root, 0)]
    while frontier:
        nid, depth = frontier.pop()
        if nid in seen or depth > _NEG_WALK_DEPTH or nid not in graph:
            continue
        seen.add(nid)
        inputs = graph[nid].get("inputs") or {}
        if any(isinstance(inputs.get(k), str) for k in _TEXT_INPUT_KEYS):
            found.add(nid)
            continue          # the encode itself — no reason to walk past it
        for v in inputs.values():
            tgt = _link_target(v)
            if tgt is not None:
                frontier.append((tgt, depth + 1))
    return found


def negative_text_nodes(graph: dict) -> list[str]:
    """Text-encode node ids feeding a sampler's `negative` input.

    Excludes anything also reachable from a `positive` input: some minimal
    workflows wire one encode into both, and appending Rufus' suppression
    terms there would poison the positive prompt with the very words it is
    trying to keep out of the image."""
    negative: set[str] = set()
    positive: set[str] = set()
    for node in graph.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, bucket in (("negative", negative), ("positive", positive)):
            tgt = _link_target(inputs.get(key))
            if tgt is not None:
                bucket |= _text_nodes_behind(graph, tgt)
    return sorted(negative - positive)


def _apply_negative(g: dict, negative: str) -> bool:
    """Substitute `negative` into the graph. True if it landed somewhere.

    An explicit RUFUS_NEGATIVE placeholder wins and is REPLACED — the author
    put it there to say "this text is mine to control". Otherwise the terms
    are APPENDED to the export's own negative, because that text was part of
    the run the owner verified and is not ours to discard."""
    placed = False
    for node in g.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for k, v in list(inputs.items()):
            if isinstance(v, str) and NEG_PLACEHOLDER in v:
                inputs[k] = v.replace(NEG_PLACEHOLDER, negative)
                placed = True
    if placed:
        return True
    for nid in negative_text_nodes(g):
        inputs = g[nid]["inputs"]
        for k in _TEXT_INPUT_KEYS:
            if isinstance(inputs.get(k), str):
                existing = inputs[k].strip().rstrip(",")
                if negative.lower() in existing.lower():
                    placed = True
                    continue
                inputs[k] = f"{existing}, {negative}" if existing else negative
                placed = True
    return placed


def prepare(graph: dict, *, prompt: str | None = None,
            image_name: str | None = None, seed: int | None = None,
            dims: tuple[int, int, int] | None = None,
            save_prefix: str = "rufus_tpl",
            negative: str | None = None,
            keep_video_writers: bool = False) -> dict:
    """Deep-copy the template and substitute Rufus' per-run values (see module
    docstring for the exact substitution contract)."""
    g = copy.deepcopy(graph)

    if negative:
        _apply_negative(g, negative)

    image_set = False
    for node in g.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        ct = node.get("class_type", "")

        if prompt is not None:
            for k, v in list(inputs.items()):
                if isinstance(v, str) and PLACEHOLDER in v:
                    inputs[k] = v.replace(PLACEHOLDER, prompt)

        if image_name is not None and ct == "LoadImage" and not image_set:
            inputs["image"] = image_name
            image_set = True

        if seed is not None:
            for key in ("seed", "noise_seed"):
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = seed

        if dims is not None and all(k in inputs for k in ("width", "height", "length")):
            inputs["width"], inputs["height"], inputs["length"] = dims
        elif dims is not None and all(k in inputs for k in ("width", "height", "duration")):
            # LTX-style all-in-one nodes size clips in SECONDS + fps rather than
            # a frame count. Without this branch the whole dims substitution is
            # skipped silently and the export's own resolution wins — which for
            # the stock LTX template means 1280x720 LANDSCAPE clips fed into a
            # 1080x1920 portrait pipeline, i.e. a pillarboxed mess that still
            # "succeeds". Convert frames -> seconds with the node's own fps.
            w, h, frames = dims
            fps = inputs.get("fps") or 25
            inputs["width"], inputs["height"] = w, h
            try:
                inputs["duration"] = max(1, round(frames / float(fps)))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # Swap video-writer nodes for SaveImage frames (wired to the decode the
    # writer chain was consuming — found by walking back through the writers).
    # keep_video_writers: some all-in-one i2v nodes (LTX) emit a VIDEO with no
    # VAEDecode behind it. Stripping the writer there deletes the graph's ONLY
    # output and leaves something that renders nothing, so those engines keep
    # the writer and download the finished container instead.
    writer_ids = [] if keep_video_writers else [
        nid for nid, n in g.items()
        if n.get("class_type") in _VIDEO_WRITER_CLASSES]
    if writer_ids:
        decode_id = _find_decode_source(g, writer_ids)
        for nid in writer_ids:
            del g[nid]
        if decode_id is not None:
            g["rufus_save"] = {"class_type": "SaveImage",
                               "inputs": {"filename_prefix": save_prefix,
                                          "images": [decode_id, 0]}}
    else:
        # Image workflow — just rebrand the SaveImage prefix for traceability.
        for n in g.values():
            if n.get("class_type") == "SaveImage":
                n["inputs"]["filename_prefix"] = save_prefix
    return g


def _find_decode_source(g: dict, writer_ids: list[str]) -> str | None:
    """The node id whose IMAGE output ultimately feeds the video writers —
    normally a VAEDecode; walk link-chains through intermediate writer nodes."""
    writer_set = set(writer_ids)
    for nid in writer_ids:
        for v in (g[nid].get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in g:
                src = str(v[0])
                if src in writer_set:
                    continue
                if g[src].get("class_type") == "VAEDecode":
                    return src
    # Fallback: any VAEDecode in the graph (an i2v template has exactly one).
    for nid, n in g.items():
        if n.get("class_type") == "VAEDecode":
            return nid
    return None
