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


def prepare(graph: dict, *, prompt: str | None = None,
            image_name: str | None = None, seed: int | None = None,
            dims: tuple[int, int, int] | None = None,
            save_prefix: str = "rufus_tpl") -> dict:
    """Deep-copy the template and substitute Rufus' per-run values (see module
    docstring for the exact substitution contract)."""
    g = copy.deepcopy(graph)

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

    # Swap video-writer nodes for SaveImage frames (wired to the decode the
    # writer chain was consuming — found by walking back through the writers).
    writer_ids = [nid for nid, n in g.items()
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
