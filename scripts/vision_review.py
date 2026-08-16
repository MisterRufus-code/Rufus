#!/usr/bin/env python3
"""
vision_review.py — look at the pictures.

THE GAP THIS FILLS, and it is the only one of its kind in this pipeline.
run_review measures a finished run and has found real defects: one object in
most of the frames, the thread restated everywhere, too few pictures for the
length. Every one of those it found by reading TEXT — the prompts, the script,
the QC sidecar. It has never seen a single pixel, and cannot.

So the defects that live in the image have all been found the same way: the
owner opened the gallery, looked, and said "why is everything coins", "the
faces are all the same", "it added images on top of images". A pipeline that
renders sixty stills a night and can only be checked by a human looking at
them is a pipeline whose quality is capped by how often that human looks.

This asks a vision model the three questions the text can never answer:

  1. Does the picture show what its prompt asked for?
  2. Is there readable lettering in it? (garbled text is the clearest tell
     that a machine made the video, and the defusal clause only tries)
  3. Is the face doing the geometry the storyboard specified?

WHY NOT A ComfyUI WORKFLOW. That was the first plan — config/vision_api.json,
the same template contract as stills. But every vision node in ComfyUI is a
custom node with its own install, and the pipeline already speaks
OpenAI-compatible chat: Ollama serves llava, qwen2.5vl and moondream on the
exact endpoint llm.py already points at. One base_url and this works against a
local 7B on the owner's own 3090, or against gpt-4o, with no export step and
no new dependency. The workflow route is still open if a model only ships as
a node; nothing here forecloses it.

    RUFUS_VISION=1                       turn it on (off by default: it costs
                                         seconds per frame)
    RUFUS_VISION_MODEL=qwen2.5vl:7b      the model that looks
    RUFUS_LLM_BASE=http://localhost:11434/v1

COST, honestly. A 7B VLM on a 3090 answers in two to four seconds a frame, so
a fourteen-picture run is under a minute and a 150-picture long-form run is
ten. That is why it is off by default and why it samples rather than reading
every sub-frame.

SHARING ONE CARD. This and ComfyUI both want the 3090, and 24GB does not hold
a stills model and a 7B VLM comfortably at the same time. The run's own order
mostly solves it: the writers talk to the model, THEN ComfyUI renders, THEN
this looks at what came out — three phases that do not overlap. What does
overlap is RESIDENCY, because Ollama keeps a model loaded for five minutes
after the last request by default and ComfyUI holds its weights between jobs.
Set OLLAMA_KEEP_ALIVE=30s (or 0) and Ollama gives the memory back before the
image phase starts. run_review's command line additionally stands down while
a render holds the channel lock — see _gpu_is_busy.

CONTRACT: fail-open and never fatal. No model, no endpoint, a refusal, a
malformed reply — all yield no findings and a printed reason. A picture that
could not be looked at is not a picture that failed.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

# How many frames to look at in one pass. A Short's fourteen fit easily; a
# long-form run's hundred-and-fifty would take ten minutes, so the sample is
# spread across the sequence rather than truncated at the front — a defect in
# the last third is exactly the one nobody scrolls to.
MAX_FRAMES = 24

# Below this share the finding is one bad frame, which is a seed, not a defect.
# Above it, the same thing went wrong repeatedly and that is a code or prompt
# problem. The number matches run_review's own systematic threshold, because a
# measurement that fires on one frame in twelve is the noise this repo has
# twice had to walk back.
SYSTEMATIC_SHARE = 0.34

_TIMEOUT = 90


def enabled() -> bool:
    return os.environ.get("RUFUS_VISION", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def model() -> str:
    import llm
    return llm.model_for("vision",
                         os.environ.get("RUFUS_VISION_MODEL") or "gpt-4o-mini")


def _data_url(path: Path) -> str | None:
    """A PNG as a data URL. None if it cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# The question. Deliberately asks for JSON with three narrow fields rather than
# a description: a paragraph about a picture reads well and cannot be counted,
# and this module's whole value is turning "the faces are all the same" from
# something the owner noticed into something with a number beside it.
_PROMPT = (
    "You are checking one frame from an animated explainer video against the "
    "prompt it was generated from.\n\n"
    "THE PROMPT THAT ASKED FOR THIS PICTURE:\n{prompt}\n\n"
    "Answer ONLY with this JSON:\n"
    '{{"shows_it": true|false,\n'
    ' "missing": "<if shows_it is false, the ONE thing the prompt asked for '
    'that is not in the picture; else \\"\\">",\n'
    ' "lettering": true|false,\n'
    ' "lettering_note": "<if lettering is true, what words or letter-shapes '
    'are visible; else \\"\\">",\n'
    ' "faces": <how many faces are visible, as a number>,\n'
    ' "expression": "<if any face is visible, describe the BROWS and the '
    'MOUTH in plain physical terms — for example \\"brows flat, mouth a short '
    'straight line\\". Never name an emotion. If no face, \\"\\">"}}\n\n'
    "shows_it is false ONLY when the picture is missing something the prompt "
    "specifically asked for — not because it is stylised, simple, or drawn "
    "rather than photographic. This is a flat cartoon by design.\n"
    "lettering is true if ANY readable or half-readable writing appears: "
    "signs, labels, numbers on a coin face, text on a page. Squiggles that "
    "imitate writing count."
)


def look(image: Path, prompt: str, client=None) -> dict | None:
    """Ask about one frame. None on any failure — never raises."""
    import llm
    url = _data_url(image)
    if not url:
        return None
    client = client or llm.client()
    try:
        resp = client.chat.completions.create(
            model=model(),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT.format(prompt=prompt[:1200])},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            temperature=0,
            max_tokens=300,
            timeout=_TIMEOUT,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[vision] {image.name}: {e}")
        return None
    return _parse(raw)


def _parse(raw: str) -> dict | None:
    """The model's JSON, however it wrapped it.

    Local models fence their JSON in ```json blocks more often than the cloud
    ones do, and a fenced reply is a correct answer this module would
    otherwise throw away.
    """
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    return {
        "shows_it": bool(d.get("shows_it", True)),
        "missing": str(d.get("missing") or "")[:200],
        "lettering": bool(d.get("lettering", False)),
        "lettering_note": str(d.get("lettering_note") or "")[:200],
        "faces": int(d.get("faces") or 0) if str(d.get("faces", "")).strip().lstrip("-").isdigit() else 0,
        "expression": str(d.get("expression") or "")[:200],
    }


def _sample(frames: list[Path], prompts: list[str]) -> list[tuple[Path, str]]:
    """Frames paired with the prompt that made them, evenly spread.

    Truncating at the front would leave the last third of every long-form
    video unlooked-at, which is exactly the part nobody scrolls to and
    therefore the part a defect survives in.
    """
    pairs = [(f, prompts[i] if i < len(prompts) else "")
             for i, f in enumerate(frames)]
    if len(pairs) <= MAX_FRAMES:
        return pairs
    step = len(pairs) / float(MAX_FRAMES)
    return [pairs[min(len(pairs) - 1, int(i * step))] for i in range(MAX_FRAMES)]


def review_frames(frames: list[Path], prompts: list[str]) -> dict:
    """Look at a run's pictures. Always returns a dict; never raises."""
    out = {"looked_at": 0, "model": model(), "frames": [], "findings": []}
    if not enabled():
        return out
    if not frames:
        return out

    import llm
    if not llm.usable():
        print("[vision] no endpoint or key — skipping the picture review")
        return out

    try:
        client = llm.client()
    except Exception as e:
        print(f"[vision] could not reach the model ({e}) — skipping")
        return out

    pairs = _sample(frames, prompts)
    print(f"[vision] looking at {len(pairs)} of {len(frames)} frame(s) "
          f"with {model()}")
    for path, prompt in pairs:
        got = look(path, prompt, client=client)
        if got is None:
            continue
        got["frame"] = path.name
        out["frames"].append(got)
        out["looked_at"] += 1

    out["findings"] = findings(out["frames"])
    return out


def findings(seen: list[dict]) -> list[dict]:
    """What the pictures say, as countable findings.

    RARE, SPECIFIC, ACTIONABLE — the standard this repo arrived at after
    walking back two warnings that fired on most runs. One frame with
    lettering is a seed; a third of them is the defusal clause not working.
    """
    out: list[dict] = []
    n = len(seen)
    if n < 3:
        return out

    missed = [f for f in seen if not f.get("shows_it")]
    if len(missed) / n > SYSTEMATIC_SHARE:
        examples = "; ".join(f["missing"] for f in missed[:3] if f.get("missing"))
        out.append({
            "id": "pictures_miss_their_prompt",
            "severity": "high",
            "text": (f"{len(missed)} of {n} pictures do not show what their "
                     f"prompt asked for. This is the one thing no text check "
                     f"can see — the prompt reads fine in every case."
                     + (f" Missing: {examples}." if examples else "")),
        })

    lettering = [f for f in seen if f.get("lettering")]
    if len(lettering) / n > SYSTEMATIC_SHARE:
        note = next((f["lettering_note"] for f in lettering
                     if f.get("lettering_note")), "")
        out.append({
            "id": "lettering_got_through",
            "severity": "high",
            "text": (f"{len(lettering)} of {n} pictures contain readable or "
                     f"half-readable lettering. The blank-surfaces defusal "
                     f"asks for none, so it is being overridden or ignored, "
                     f"and garbled words are the clearest sign a machine made "
                     f"the video." + (f" Seen: {note}." if note else "")),
        })

    # THE COMPLAINT THAT STARTED THIS, as a number. "the same face expressions"
    # was something the owner saw in a gallery of sixty; nothing in the
    # pipeline could count it, because the prompts all said something
    # different while the renders all looked the same.
    faces = [f["expression"].strip().lower() for f in seen
             if f.get("expression") and f.get("faces")]
    if len(faces) >= 4:
        from collections import Counter
        common, hits = Counter(faces).most_common(1)[0]
        if hits / len(faces) > 0.6:
            out.append({
                "id": "one_expression_everywhere",
                "severity": "medium",
                "text": (f"{hits} of {len(faces)} faces are drawn the same "
                         f"way ({common}). The prompts asked for different "
                         f"ones, so this is the renderer falling back to its "
                         f"default — the defect a text check cannot see, "
                         f"because on paper every prompt differs."),
            })

    empty = [f for f in seen if not f.get("faces")]
    if n >= 6 and len(empty) / n > 0.7:
        out.append({
            "id": "nobody_in_the_pictures",
            "severity": "medium",
            "text": (f"{len(empty)} of {n} pictures have no face in them. The "
                     f"storyboard is asked for people in at least half the "
                     f"shots, because a number lands on a face and not on an "
                     f"empty room."),
        })
    return out
