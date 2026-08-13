#!/usr/bin/env python3
"""comfy_doctor.py — what ComfyUI can actually see, and what each engine needs.

WHY THIS EXISTS. Every ComfyUI-backed engine here is inert until its API export
lands in config/ (see comfy_template.py — the graph is never hand-wired, it is
exported from a workflow the owner has personally watched succeed). That rule
is right, and it leaves one bad gap: when an engine says "off", the owner has
no way to tell WHICH of four things is wrong — ComfyUI down, nodes not
installed, model files in the wrong folder, or the export simply not done yet.
The answer used to require starting a full run and reading the log.

So ask ComfyUI directly. /object_info carries every loader's list of filenames
it will accept, which is the same list the dropdown shows in the UI — if a
downloaded file is not in it, the file is in the wrong folder, and that is the
single most common reason a "downloaded" model does not work.

    python scripts/comfy_doctor.py                      # every engine
    python scripts/comfy_doctor.py wan_t2v              # one engine
    python scripts/comfy_doctor.py wan_t2v --dry-run    # + what would be submitted

--dry-run answers a different question from the checks above. They ask "is this
export loadable". It asks "will MY settings land on MY export" — which is not
the same, because prepare() writes into inputs that exist and skips the ones
that do not, so a perfectly valid export can ignore every environment variable
the owner typed. It runs prepare() and prints the result. No submit, no
sampling, no GPU.

Read-only. Makes no changes and starts nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests  # noqa: E402

import comfy_template  # noqa: E402
from comfy_client import _host  # noqa: E402

# The loaders worth listing, and the words that mark a file as belonging to a
# given engine. Substring matching on purpose: the community re-names these
# constantly (wan2.2, Wan2_2, wan22) and an exact list would rot in a week.
#
# LoraLoader/LoraLoaderModelOnly ARE ON THIS LIST BECAUSE LEAVING THEM OFF
# PRODUCED A FALSE NEGATIVE. The first real run reported
#
#     ✗ 4-step LoRA (t2v)          none visible
#
# on a box that had wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise and its
# low_noise partner sitting in models/loras, already wired into the workflow —
# because nothing here ever asked ComfyUI for the LoRA enum. The advice built
# on top then sent the owner to re-download files they already had. A doctor
# that reports "not present" for something present is worse than no doctor:
# every other line it prints becomes suspect.
_LOADERS = ("UNETLoader", "CheckpointLoaderSimple", "VAELoader", "CLIPLoader",
            "DualCLIPLoader", "UnetLoaderGGUF", "CLIPVisionLoader",
            "LoraLoaderModelOnly", "LoraLoader")

ENGINES = {
    "stills":    ("comfy_client",      "config/stills_api.json",      ("flux", "sd", "qwen")),
    "stills_i2i": (None,               "config/stills_i2i_api.json",  ("flux", "qwen")),
    # shot_chain carries its OWN check that nothing else here can do: it
    # measures the denoise on the path from the loaded image, because an edit
    # workflow and an img2img workflow are wired identically and behave
    # oppositely. At denoise 0.55 every beat comes back as the previous
    # picture — that mistake already cost this project a full run
    # (config/character_stills_api.json, ten identical hooded figures). Running
    # its ready() here means the export is judged before a run, not after.
    "shot_chain": ("shot_chain",       "config/shot_chain_api.json",  ("qwen", "edit")),
    "hunyuan":   ("hunyuan_client",    "config/hunyuan_i2v_api.json", ("hunyuan", "hy")),
    "wan_i2v":   ("wan_client",        None,                          ("wan",)),
    "wan_t2v":   ("wan_t2v_client",    "config/wan_t2v_api.json",     ("wan", "umt5", "t2v")),
    "ltx":       ("ltx_client",        "config/ltx_api.json",         ("ltx",)),
}

ROOT = Path(__file__).parent.parent

# --dry-run: also show what prepare() would produce against this export. Off by
# default because it is a second, longer report and the common question is just
# "is this thing ready".
DRY_RUN = False


# ── Wan: which variant is on disk, and what it costs to run ──────────────────
#
# Wan 2.2 ships as two different things with confusingly similar names, and the
# difference decides whether this box is usable:
#
#   T2V/I2V-A14B  mixture-of-experts, two ~14B experts, ~28GB of weights. A
#                 24GB card holds ONE expert, so ComfyUI swaps them mid-clip.
#                 The evicted expert should land in system RAM — on a 16GB box
#                 there is no room, so it is re-read from disk every clip.
#   TI2V-5B       one dense 5B model, ~10GB. Fits in VRAM whole. No swap, so
#                 system RAM stops mattering at all.
#
# And the step count matters more than either: measured on this owner's 3090
# (see wan_client.py) 20 steps is ~19 min per clip and 12 steps is ~11-12 min,
# i.e. ~57s per step. The lightx2v/Lightning distillation LoRAs cut 20 steps to
# 4. That is a 5x saving against maybe one minute from fixing the swap, which
# is why this report leads with the LoRA and not with the RAM.
_WAN_KINDS = {
    # "ti2v" CONTAINS "i2v", so every i2v rule has to exclude it or the 5B
    # model — the one variant that solves the RAM problem — gets filed as a
    # 14B image-to-video model and the advice inverts.
    "t2v_model":  (("wan", "t2v"), ("lora", "lightx", "lightning")),
    "i2v_model":  (("wan", "i2v"), ("ti2v", "lora", "lightx", "lightning")),
    "ti2v_5b":    (("ti2v",), ()),
    "lora_t2v":   (("t2v", "lightx"), ()),
    "lora_i2v":   (("i2v", "lightx"), ("ti2v",)),
    "text_enc":   (("umt5",), ()),
    "vae_21":     (("wan", "2.1", "vae"), ()),
    "vae_22":     (("wan", "2.2", "vae"), ()),
}


def _classify_wan(seen: dict[str, set[str]]) -> dict[str, list[str]]:
    """Bucket every Wan-ish filename ComfyUI can load.

    Substring matching, because these files get renamed constantly (wan2.2,
    Wan2_2, wan22) and an exact list would rot inside a week. A file may land
    in more than one bucket; that is fine, the buckets are advisory.
    """
    out: dict[str, list[str]] = {k: [] for k in _WAN_KINDS}
    for names in seen.values():
        for f in names:
            low = f.lower()
            # "5b" is only meaningful next to a wan/ti2v name — plenty of
            # unrelated checkpoints have a 5b in them.
            if "wan" in low and "5b" in low and f not in out["ti2v_5b"]:
                out["ti2v_5b"].append(f)
            for kind, (need, block) in _WAN_KINDS.items():
                if all(n in low for n in need) and not any(b in low for b in block):
                    if f not in out[kind]:
                        out[kind].append(f)
    return {k: sorted(v) for k, v in out.items()}


def _wan_advice(found: dict[str, list[str]], have_export: bool) -> list[str]:
    """The next thing to do, in the order that saves the most time first."""
    tips: list[str] = []

    if not found["t2v_model"] and not found["ti2v_5b"]:
        if found["i2v_model"]:
            tips.append(
                "You have the IMAGE-to-video models but no TEXT-to-video ones. "
                "They are different downloads — having i2v does not give you "
                "t2v. Fetch the T2V models, or TI2V-5B (below).")
        else:
            tips.append("No Wan video models are visible to ComfyUI at all.")

    if found["t2v_model"] and not found["lora_t2v"]:
        # Say only what the inventory actually shows. An earlier version of
        # this line asserted "you already have the i2v version" because
        # wan_client.py's header lists it among the files the I2V template
        # installs — but that is what the template WOULD install, not what is
        # on this disk, and the first real run of this report printed the claim
        # directly above "4-step LoRA (i2v): none visible".
        familiar = (" You already have the i2v version of this LoRA, so you "
                    "know the family; the t2v files are separate."
                    if found["lora_i2v"] else "")
        tips.append(
            "BIGGEST WIN, and it is missing: the 4-step lightx2v/Lightning "
            "LoRA for T2V. Without it the workflow samples ~20 steps at "
            "roughly 57s/step on this card — about 19 minutes a clip. With it, "
            "4 steps, about 4 minutes." + familiar +
            " Until you have it, cutting the KSampler steps by hand (20 → 8) "
            "in ComfyUI before exporting is most of the win at some quality "
            "cost, and needs no download.")
    elif found["lora_t2v"]:
        tips.append(
            "The 4-step T2V LoRA is present. Turn it ON in ComfyUI BEFORE you "
            "Export (API) — comfy_template.prepare() substitutes only prompt, "
            "image, seed and dimensions, so the step count and the LoRA toggle "
            "are frozen into whatever you export. There is no env var for "
            "them.")

    if found["ti2v_5b"]:
        tips.append(
            "TI2V-5B is on disk — this is the variant that removes the 16GB "
            "RAM penalty entirely: one dense 5B model, ~10GB, fits in 24GB "
            "VRAM whole, so there is no expert swap and nothing streams from "
            "disk. Worth exporting as a second workflow to compare.")
        if not found["vae_22"]:
            tips.append(
                "…but the Wan 2.2 VAE is NOT visible, and TI2V-5B needs it — "
                "it does not use wan_2.1_vae. A workflow exported with the "
                "wrong VAE is rejected at submit time with value_not_in_list, "
                "after the stills phase has already run.")

    if not found["text_enc"]:
        tips.append(
            "The umt5 text encoder is not visible. Every Wan text prompt goes "
            "through it, so nothing will run without it.")

    if not have_export and (found["t2v_model"] or found["ti2v_5b"]):
        tips.append(
            "Models are here, the export is not. That is the only manual step "
            "and it is three clicks: run the workflow once in ComfyUI, set the "
            "positive prompt to exactly RUFUS_PROMPT, Workflow → Export (API).")
    return tips


# ── the edit model shot_chain needs, and the half of it that is easy to miss ──
#
# The first real check on the owner's box listed five Qwen files ComfyUI could
# load and every one of them was a TEXT ENCODER:
#
#     · qwen_2.5_vl_7b_fp8_scaled.safetensors  [CLIPLoader]
#     · qwen_3_4b.safetensors                  [CLIPLoader]
#
# No diffusion model, no VAE. Read quickly that looks like "Qwen is installed",
# and the natural next move is to go build a workflow out of parts that are not
# there. An edit workflow needs three separate downloads and they are easy to
# conflate, because the text encoder is the one that ships with several other
# Qwen workflows and so tends to arrive first.
def _classify_edit(seen: dict[str, set[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"unet": [], "vae": [], "text_enc": []}
    for cls, names in seen.items():
        for f in sorted(names):
            low = f.lower()
            if not (low.endswith(".safetensors") or low.endswith(".gguf")):
                continue          # loader `type` enums are not files
            if "vae" in low and ("qwen" in low or "edit" in low):
                out["vae"].append(f)
            elif cls in ("CLIPLoader", "DualCLIPLoader") and "qwen" in low:
                out["text_enc"].append(f)
            elif cls in ("UNETLoader", "UnetLoaderGGUF",
                         "CheckpointLoaderSimple") and (
                    "edit" in low or ("qwen" in low and "image" in low)):
                out["unet"].append(f)
    return out


def _report_edit(seen: dict[str, set[str]]) -> None:
    found = _classify_edit(seen)
    print("\n─── image-edit inventory (shot_chain) " + "─" * 24)
    for key, label in (("unet", "edit model (Qwen-Image-Edit)"),
                       ("text_enc", "Qwen text encoder"),
                       ("vae", "Qwen image VAE")):
        files = found[key]
        mark = "✓" if files else "✗"
        shown = ", ".join(f[:44] for f in files[:2]) if files else "none visible"
        print(f"  {mark} {label:<30} {shown}")

    if not found["unet"]:
        print("\n  The EDIT MODEL itself is not installed. A text encoder is "
              "not an\n  edit model — it ships with several other Qwen "
              "workflows, so it tends\n  to arrive first and make the rest "
              "look present. Fetch\n  Qwen-Image-Edit-2509 (the Q4_K_M GGUF is "
              "~13GB, comfortable on 24GB)\n  into models/diffusion_models, and "
              "its VAE into models/vae.")
    elif not found["vae"]:
        print("\n  Edit model present, VAE missing — the graph will not decode.")
    else:
        print("\n  All three parts present. Build the workflow, RUN IT ONCE on "
              "two real\n  images, set the edit instruction to exactly "
              "RUFUS_PROMPT, then\n  Export (API) → config/shot_chain_api.json. "
              "Sample at denoise 1.0:\n  at 0.55 it is img2img and every beat "
              "returns the previous picture.")


def _report_wan(seen: dict[str, set[str]]) -> None:
    found = _classify_wan(seen)
    labels = {
        "t2v_model": "T2V models", "i2v_model": "I2V models",
        "ti2v_5b": "TI2V-5B (no expert swap)", "lora_t2v": "4-step LoRA (t2v)",
        "lora_i2v": "4-step LoRA (i2v)", "text_enc": "umt5 text encoder",
        "vae_21": "Wan 2.1 VAE", "vae_22": "Wan 2.2 VAE",
    }
    print("\n─── Wan inventory " + "─" * 44)
    for kind, label in labels.items():
        files = found[kind]
        mark = "✓" if files else "✗"
        shown = ", ".join(f[:44] for f in files[:2]) if files else "none visible"
        extra = f" (+{len(files) - 2})" if len(files) > 2 else ""
        print(f"  {mark} {label:<26} {shown}{extra}")

    tips = _wan_advice(found, (ROOT / "config" / "wan_t2v_api.json").exists())
    if tips:
        print("\n  next, in the order that saves the most time:")
        for n, tip in enumerate(tips, 1):
            body = tip.split(". ")
            print(f"    {n}. {body[0]}.")
            for rest in body[1:]:
                if rest.strip():
                    print(f"       {rest.rstrip('.')}.")


def _reachable(host: str) -> bool:
    try:
        return requests.get(f"{host}/object_info/UNETLoader", timeout=8).status_code == 200
    except Exception:
        return False


def _visible_files(host: str) -> dict[str, set[str]]:
    """Every filename each loader will accept, keyed by loader class."""
    out: dict[str, set[str]] = {}
    for cls in _LOADERS:
        names = comfy_template._loader_choices(cls, host)
        if names:
            out[cls] = names
    return out


def _is_off(value) -> bool:
    """Whether a toggle reads as OFF, however the export spelled it.

    A boolean survives a round trip through JSON, a ComfyUI widget and a
    hand-edited graph in at least five shapes. Comparing with `is False` sees
    exactly one of them and silently passes the rest, which turns a check into
    a false reassurance.
    """
    if value is None:
        return False                       # absent is not "off", it is unknown
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in ("false", "0", "off", "no", "disable",
                                         "disabled")
    return False


def _as_int(value) -> int | None:
    """An integer setting, whether the export stored it as int or as text.

    A wire is [node_id, slot] and must never be read as a value — "7" from
    ["7", 0] would report as a 7-step sampler.
    """
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _export_facts(tpl: dict) -> list[str]:
    """What the export actually SAYS about speed and size.

    WHY POSITIVE REPORTING AND NOT JUST WARNINGS. _speed_notes only speaks when
    something is wrong, and silence turned out to be ambiguous: it reads
    identically whether the settings are good or whether nothing recognisable
    was found. After exporting, the owner's real question is "did the 4-step
    LoRA toggle actually make it into the file?" — and a quiet report cannot
    answer that, so the answer was being inferred from an absence.

    Print the numbers instead. "steps 4, cfg 1.0, 480x832" is checkable against
    what was set in the UI; an empty list here is itself informative, because it
    means the export exposes none of these and every guess about its speed is
    just a guess.
    """
    facts: list[str] = []
    for node in tpl.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        ct = node.get("class_type", "node")
        bits = []
        for key in ("steps", "cfg", "sampler_name", "scheduler",
                    "enable_turbo_mode", "denoise"):
            if key in ins and not isinstance(ins[key], list):
                bits.append(f"{key}={ins[key]}")
        if "width" in ins and "height" in ins:
            size = f"{ins['width']}x{ins['height']}"
            for k in ("length", "duration"):
                if k in ins and not isinstance(ins[k], list):
                    size += f" {k}={ins[k]}"
            bits.append(size)
        if bits:
            facts.append(f"{ct}: " + ", ".join(bits))
    return facts


def _speed_notes(tpl: dict) -> list[str]:
    """Settings frozen into an export that will make every run slow.

    These are invisible once exported. prepare() substitutes prompt, image,
    seed and dims and nothing else, so a graph exported at 20 steps is 20 steps
    on every run forever, and the only symptom is that clips take a long time —
    which reads as "the model is slow" rather than "the export is wrong".

    Measured on this rig: ~57 seconds per sampling step. The difference between
    an export made with the 4-step LoRA on and one made with it off is roughly
    nineteen minutes per clip.
    """
    notes: list[str] = []
    for node in tpl.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        # ComfyUI's packaged Wan node folds the whole 4-step LoRA path behind
        # one boolean. Exported false, the LoRA files can be sitting right
        # there in the graph and contribute nothing.
        # `is False` MATCHED ONLY A PYTHON BOOL, and that is not what an export
        # necessarily contains. The owner's ComfyUI showed enable_turbo_mode
        # false in the open workflow while this check stayed silent on the
        # export made from it — the two cannot both be right, and a check that
        # under-reports is the worse of the two, because silence here reads as
        # "your settings are fine". JSON, ComfyUI widget state and hand-edited
        # graphs all render booleans differently: false, "false", 0, "0", "off".
        if _is_off(ins.get("enable_turbo_mode")):
            notes.append(
                "enable_turbo_mode is FALSE in this export — the 4-step LoRA "
                "path is off. Turn it on in ComfyUI and Export (API) again; "
                "this is worth about 5x.")
        steps = _as_int(ins.get("steps"))
        if steps is not None and steps > 10:
            notes.append(
                f"a sampler is exported at {steps} steps (~{steps * 57 // 60} "
                f"min/clip at this rig's ~57s/step). 4-8 is the fast range.")
    return notes


def _substitution_gaps(tpl: dict) -> list[str]:
    """Rufus values this export will silently ignore.

    prepare() writes prompt, image, seed and dims into inputs THAT EXIST. When
    an input is absent the write is skipped and nothing anywhere reports it —
    the substitution "succeeds", the run proceeds, and the setting the owner
    typed did nothing. That is this repo's oldest failure shape.

    It matters most for packaged all-in-one nodes like ComfyUI's "Text to
    Video (Wan2.2)", which collapse a whole graph into one node and expose only
    some of its knobs. Without a seed input, wan_t2v_client's entire seed
    lineage — the mechanism that keeps neighbouring beats sharing noise
    structure, and that makes a run reproducible from one number — is inert.
    """
    gaps: list[str] = []
    has_seed = has_dims = False
    for node in tpl.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        if any(k in ins for k in ("seed", "noise_seed")):
            has_seed = True
        if "width" in ins and "height" in ins and (
                "length" in ins or "duration" in ins):
            has_dims = True

    if not has_seed:
        gaps.append(
            "no seed input — RUFUS_T2V_SEED and the per-beat seed lineage will "
            "do nothing, so the run is not reproducible and neighbouring beats "
            "no longer share noise structure. Not fatal, but silent: expose a "
            "seed on the node if the workflow has one.")
    if not has_dims:
        gaps.append(
            "no width/height + length-or-duration on any node — "
            "RUFUS_T2V_W/H/FRAMES will do nothing and the export's own "
            "resolution wins every run. A landscape export feeding a 1080x1920 "
            "pipeline still 'succeeds', pillarboxed.")
    return gaps


def _dry_run(name: str, tpl: dict) -> list[str]:
    """What the graph would look like AFTER Rufus substitutes into it.

    The structural checks above answer "is this export loadable". They cannot
    answer the question that actually costs GPU time: will MY env vars land on
    MY export. Those are different questions, because prepare() writes into
    inputs that exist and skips the ones that do not — so an export can be
    perfectly valid and still ignore every setting the owner typed.

    Running prepare() against the real settings and printing the result answers
    it for nothing: no ComfyUI submit, no sampling, no clip.
    """
    mod_name = ENGINES[name][0]
    if not mod_name:
        return []
    try:
        mod = __import__(mod_name)
        cfg = mod.settings()
        w, h = int(cfg["width"]), int(cfg["height"])
        frames = int(cfg["frames"])
        fps = getattr(mod, "WAN_FPS", None)
    except Exception as e:
        return [f"cannot resolve this engine's settings ({e})"]

    before = comfy_template.load_template  # noqa: F841  (kept for symmetry)
    out = comfy_template.prepare(tpl, prompt="RUFUS_DRY_RUN", seed=12345,
                                 dims=(w, h, frames), keep_video_writers=True,
                                 **({"fps": fps} if fps else {}))

    lines = [f"asking for {w}x{h}, {frames} frames"
             + (f" (= {frames / fps:.2f}s at {fps}fps)" if fps else "")]
    landed = False
    for node in out.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        ct = node.get("class_type", "node")
        if ins.get("width") == w and ins.get("height") == h:
            landed = True
            length = ins.get("length", ins.get("duration"))
            lines.append(f"{ct} received {w}x{h}, length/duration={length}")
        if ins.get("seed") == 12345 or ins.get("noise_seed") == 12345:
            lines.append(f"{ct} received the seed")
    if not landed:
        lines.append("NOTHING received the dimensions — this export will run "
                     "at whatever size it was saved with, every time")
    if not any("received the seed" in l for l in lines):
        lines.append("nothing received a seed — seed lineage is inert here")
    if not any("RUFUS_DRY_RUN" in str(n.get("inputs")) for n in out.values()):
        lines.append("the prompt placeholder did not substitute — this export "
                     "would render its saved prompt on every beat")
    return lines


def _report_engine(name: str, host: str, seen: dict[str, set[str]]) -> bool:
    """Print the report; return whether this engine could run right now.

    The return value is what lets a launcher GATE on this. A preflight that
    always exits 0 cannot stop anything, and the failure it would have stopped
    is expensive: a bad template is only rejected at submit time, which is
    after the entire stills phase has run and the GPU time is already spent.
    """
    usable = True
    mod_name, tpl_rel, marks = ENGINES[name]
    print(f"\n─── {name} " + "─" * max(0, 60 - len(name)))

    # 1. Do the weights exist where ComfyUI looks? This is the question the
    #    owner is actually asking when they say "I downloaded the models".
    hits: list[str] = []
    for cls, names in seen.items():
        for f in sorted(names):
            if any(m in f.lower() for m in marks):
                hits.append(f"{f}  [{cls}]")
    if hits:
        print(f"  models ComfyUI can load ({len(hits)}):")
        for h in hits[:12]:
            print(f"    · {h}")
        if len(hits) > 12:
            print(f"    … and {len(hits) - 12} more")
    else:
        print(f"  ⚠ ComfyUI lists NO file matching {marks}.")
        print(f"    If you have downloaded them, they are in the wrong folder —")
        print(f"    ComfyUI only sees what is under its own models/ tree, and")
        print(f"    the loader dropdown in the UI shows exactly this same list.")

    # 2. Is the API export done? This is the step that is never automatic.
    if tpl_rel:
        tpl_path = ROOT / tpl_rel
        if not tpl_path.exists():
            usable = False
            print(f"  ✗ no export at {tpl_rel}")
            print(f"    ComfyUI → Workflow → Browse Templates → run it once →")
            print(f"    set the positive prompt to exactly  RUFUS_PROMPT  →")
            print(f"    Workflow → Export (API) → save as {tpl_rel}")
        else:
            tpl = comfy_template.load_template(tpl_path)
            if tpl is None:
                usable = False
                print(f"  ✗ {tpl_rel} exists but is not a valid API export")
                print(f"    (\"Export (API)\", not \"Export\" — the plain export is")
                print(f"    a UI workflow and has a different shape)")
            elif not comfy_template.has_placeholder(tpl):
                usable = False
                print(f"  ✗ {tpl_rel} has no RUFUS_PROMPT placeholder")
                print(f"    Set the positive prompt text to exactly RUFUS_PROMPT")
                print(f"    in ComfyUI and export again.")
            else:
                bad_nodes = comfy_template.missing_nodes(tpl, host)
                bad_files = comfy_template.missing_models(tpl, host)
                if bad_nodes:
                    usable = False
                    print(f"  ✗ ComfyUI is missing node(s): {', '.join(bad_nodes[:5])}")
                elif bad_files:
                    usable = False
                    print(f"  ✗ ComfyUI cannot load: {'; '.join(bad_files[:4])}")
                    print(f"    The export names a file this ComfyUI does not have —")
                    print(f"    re-export after picking a file that IS in the dropdown.")
                else:
                    print(f"  ✓ {tpl_rel} valid — nodes and model files all resolve")
                    facts = _export_facts(tpl)
                    for fact in facts:
                        print(f"  · {fact}")
                    if not facts:
                        print(f"  · this export exposes no steps/cfg/size "
                              f"inputs — nothing here can confirm how it was "
                              f"configured, so judge it by the clip time")
                    for note in _speed_notes(tpl):
                        print(f"  ⏱ {note}")
                    for gap in _substitution_gaps(tpl):
                        print(f"  ⚠ {gap}")
                    if DRY_RUN:
                        print("  dry run — what Rufus would actually submit:")
                        for line in _dry_run(name, tpl):
                            print(f"    · {line}")

    # 3. What the engine itself says, which is the line a run will print.
    if mod_name:
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "ready"):
                ok, why = mod.ready()
                print(f"  engine: {'READY' if ok else 'not ready'} — {why}")
            if hasattr(mod, "enabled") and not mod.enabled():
                print(f"  engine: switched OFF for this shell (env var) — being "
                      f"ready is not the same as being on")
        except Exception as e:
            print(f"  engine: could not load {mod_name} ({e})")
    return usable


def main(argv: list[str]) -> int:
    global DRY_RUN

    # A MISTYPED FLAG MUST NOT LOOK LIKE SUCCESS. Two commands once arrived
    # pasted together as `--dry-rungit pull origin <branch>`; every token fell
    # into the "unknown engine" list, the ordinary report printed in full, and
    # the dry run the owner asked for simply did not happen. That is this
    # repo's standing failure shape — a degraded path nobody could see — in a
    # tool written to catch exactly that, so an unrecognised flag stops here
    # instead of being folded in with engine names.
    flags = [a for a in argv if a.startswith("-")]
    bad_flags = [f for f in flags if f != "--dry-run"]
    if bad_flags:
        print(f"unknown option(s): {', '.join(bad_flags)}")
        print(f"usage: comfy_doctor.py [engine ...] [--dry-run]")
        print(f"       engines: {', '.join(ENGINES)}")
        if any("dry" in f for f in bad_flags):
            print("did you mean --dry-run? (check for two commands pasted "
                  "onto one line)")
        return 2

    DRY_RUN = "--dry-run" in argv
    argv = [a for a in argv if not a.startswith("-")]
    host = _host()
    print(f"ComfyUI at {host}")
    if not _reachable(host):
        print("✗ not reachable. Start ComfyUI first — everything below would "
              "be a guess without it.")
        return 1
    print("✓ reachable")

    seen = _visible_files(host)
    total = sum(len(v) for v in seen.values())
    print(f"✓ {total} model file(s) visible across {len(seen)} loader(s)")

    wanted = [a for a in argv if a in ENGINES]
    unusable = [name for name in (wanted or list(ENGINES))
                if not _report_engine(name, host, seen)]

    # The Wan inventory answers a question the per-engine report cannot: not
    # "is this engine ready" but "which variant do I have, and what is it
    # going to cost me per clip".
    if not wanted or any(w.startswith("wan") for w in wanted):
        _report_wan(seen)
    if not wanted or "shot_chain" in wanted or "stills_i2i" in wanted:
        _report_edit(seen)

    unknown = [a for a in argv if a not in ENGINES]
    if unknown:
        print(f"\nunknown engine(s): {', '.join(unknown)} — "
              f"known: {', '.join(ENGINES)}")

    # Exit 2 only when the caller ASKED about specific engines. A bare run is a
    # survey of everything, and most engines being un-exported is the normal
    # resting state of this repo, not an error.
    if wanted and unusable:
        print(f"\nnot runnable yet: {', '.join(unusable)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
