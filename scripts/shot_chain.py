#!/usr/bin/env python3
"""
shot_chain.py — carry the picture forward from one beat to the next.

THE GAP THIS CLOSES. storyboard.py now plans the ten shots as one sequence and
says, in words, what each shot continues from the one before it: "the same coin
from shot 1, now thinner". That thread reaches the image model as TEXT only.
comfy_client then renders every beat from fresh noise on its own seed, so the
model reads "the same coin" having never seen the coin. It invents a new one.
Ten different coins, ten different tables, ten different hands — which is
exactly what the owner reported: the video doesn't hold together.

Text cannot fix this. "The same coin" is not a description of a coin. The only
way shot 2 can contain the thing shot 1 contained is if shot 2 is generated
FROM shot 1.

WHAT THIS IS NOT. It is not img2img. That distinction cost this project a full
run once: config/character_stills_api.json was LoadImage → VAEEncode →
KSampler(denoise=0.55), and all ten beats came back as the same reference
portrait, because at that denoise the sampler can only redraw what it is given.
An EDIT model is wired the same way and behaves oppositely: it samples at
denoise 1.0, rebuilding the picture from noise while the source image steers it
through conditioning. The scene is free to change completely; the world, the
palette and the recurring objects are not. `ready()` measures that denoise and
refuses a template that is really img2img in disguise — see
comfy_template.loaded_image_denoise for why the number, and not the wiring, is
the test.

WHICH ENGINE. Qwen-Image-Edit-2509 is the one to export this from: Apache-2.0
(so it is safe for a monetised channel, unlike FLUX.1/FLUX.2), built for
multi-image editing with facial and object identity preserved across shots, and
its Q4_K_M GGUF build is ~13GB — comfortable on a 24GB card, which matters here
because system RAM, not VRAM, is this box's real ceiling. Nothing in this file
is Qwen-specific though; any edit model that satisfies the contract works.

TEMPLATE-DRIVEN, like every other engine here. Never hand-wired:

  1. Build the edit workflow in ComfyUI and RUN IT ONCE on two real images.
  2. Set the edit instruction text to exactly:  RUFUS_PROMPT
  3. Workflow -> Export (API) -> save as:  <Rufus>/config/shot_chain_api.json

CONTRACT: fail-open, and inert until that file exists. No template, no
LoadImage, a copy-grade denoise, a failed render — every one of them returns
None and the caller renders the beat the way it does today. A chained shot is a
better way to get the picture, never a prerequisite for getting one.
"""

import os
import re
from pathlib import Path

import comfy_template

ROOT = Path(__file__).parent.parent

# storyboard.py writes this exact clause when a shot continues the last one.
# Detecting it here means the two modules need no plumbing between them: the
# thread the storyboard already puts in the prompt IS the signal to chain.
MARKER = "Continuing from the previous shot:"

_MARKER_RE = re.compile(re.escape(MARKER) + r"\s*(.*?)\s*$", re.S)

# Below this denoise a graph that starts from the loaded image is img2img: it
# can only redraw its input. An edit model samples at 1.0.
MIN_EDIT_DENOISE = 0.9


def enabled() -> bool:
    return os.environ.get("RUFUS_SHOT_CHAIN", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def template_path() -> Path:
    return Path(os.environ.get("RUFUS_SHOT_CHAIN_TEMPLATE",
                               str(ROOT / "config" / "shot_chain_api.json")))


def carried(prompt: str) -> str:
    """What this shot continues from the previous one, or "" if nothing does.

    Empty means render this beat normally — not every shot can carry something
    forward, and forcing one to would chain unrelated scenes together."""
    m = _MARKER_RE.search(prompt or "")
    return m.group(1).strip().rstrip(".").strip() if m else ""


def scene(prompt: str) -> str:
    """The prompt with the continuity clause removed — the new scene alone."""
    return _MARKER_RE.sub("", prompt or "").strip().rstrip(".").strip()


def edit_prompt(prompt: str) -> str:
    """The instruction handed to the edit model.

    Phrased as a CHANGE against the image it is given, because that is what an
    edit model reads. Naming what must survive first and the new scene second
    keeps the carried object from being treated as one more thing to reimagine.
    """
    keep = carried(prompt)
    new = scene(prompt)
    return (f"Keep {keep} exactly as it appears in this image, along with the "
            f"same drawing style, colour palette and lighting. Change "
            f"everything else to a new scene: {new}. This is a different "
            f"moment, not the same picture again — the composition, the "
            f"framing and the surroundings should all be new.")


def template() -> dict | None:
    """The validated edit workflow, or None to render the beat normally."""
    ok, _ = ready()
    return comfy_template.load_template(template_path()) if ok else None


def ready() -> tuple[bool, str]:
    """Fail-closed preflight, same contract as the other engines.

    Deliberately does NOT reach the ComfyUI server: this runs once per beat
    inside the stills loop, and an engine that is simply not configured — the
    normal state — must cost nothing."""
    if not enabled():
        return False, "shot chaining disabled (RUFUS_SHOT_CHAIN=0)"
    tpl = comfy_template.load_template(template_path())
    if tpl is None:
        return False, ("no API export at config/shot_chain_api.json — build an "
                       "image-EDIT workflow (Qwen-Image-Edit-2509) in ComfyUI, "
                       "set the instruction to RUFUS_PROMPT, Export (API) — "
                       "see shot_chain.py header")
    if not comfy_template.has_placeholder(tpl):
        return False, ("export found but no RUFUS_PROMPT placeholder — set the "
                       "edit instruction to RUFUS_PROMPT and re-export")
    if not any(str(n.get("class_type", "")) == "LoadImage" for n in tpl.values()):
        return False, ("export has no LoadImage node — the previous shot has "
                       "nowhere to go, so this cannot chain anything")
    denoise = comfy_template.loaded_image_denoise(tpl)
    if denoise is not None and denoise < MIN_EDIT_DENOISE:
        return False, (f"export samples at denoise {denoise:g} from the loaded "
                       f"image — that is img2img, which can only redraw its "
                       f"input, not an edit model. Every beat would come back "
                       f"as the previous picture. Re-export at denoise 1.0")
    return True, "shot chaining ready (edit template loaded)"


def continue_shot(prev_png: Path, prompt: str, seed: int, client_id: str,
                  negative: str = "") -> bytes | None:
    """Render this beat as an edit of `prev_png`. Raw PNG bytes, or None.

    None always means "render this beat the ordinary way" — the caller keeps
    its own retry loop and beat-alignment reuse, so nothing here can cost a
    clip.

    `prev_png` should be the previous beat's RAW model output, not the finished
    1080x1920 frame: _fit_to_frame upscales and crops, and feeding that back
    in would re-resample on every link, so the degradation would compound down
    the whole video rather than stopping at one beat.
    """
    if not carried(prompt):
        return None
    tpl = template()
    if tpl is None:
        return None
    try:
        from svd_client import _upload_image
        from comfy_client import _submit, _await_image

        image_name = _upload_image(prev_png)
        if not image_name:
            return None
        g = comfy_template.prepare(tpl, prompt=edit_prompt(prompt),
                                   image_name=image_name, seed=seed,
                                   save_prefix="rufus_chain", negative=negative)
        pid = _submit(g, client_id)
        if not pid:
            return None
        return _await_image(pid)
    except Exception as e:
        print(f"[chain] skipped (non-fatal): {e}")
        return None
