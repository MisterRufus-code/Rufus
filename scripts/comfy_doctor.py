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

    python scripts/comfy_doctor.py            # every engine
    python scripts/comfy_doctor.py wan_t2v    # one engine

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
    "shot_chain": (None,               "config/shot_chain_api.json",  ("qwen", "edit")),
    "hunyuan":   ("hunyuan_client",    "config/hunyuan_i2v_api.json", ("hunyuan", "hy")),
    "wan_i2v":   ("wan_client",        None,                          ("wan",)),
    "wan_t2v":   ("wan_t2v_client",    "config/wan_t2v_api.json",     ("wan", "umt5", "t2v")),
    "ltx":       ("ltx_client",        "config/ltx_api.json",         ("ltx",)),
}

ROOT = Path(__file__).parent.parent


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
        if ins.get("enable_turbo_mode") is False:
            notes.append(
                "enable_turbo_mode is FALSE in this export — the 4-step LoRA "
                "path is off. Turn it on in ComfyUI and Export (API) again; "
                "this is worth about 5x.")
        steps = ins.get("steps")
        if isinstance(steps, int) and steps > 10:
            notes.append(
                f"a sampler is exported at {steps} steps (~{steps * 57 // 60} "
                f"min/clip at this rig's ~57s/step). 4-8 is the fast range.")
    return notes


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
                    for note in _speed_notes(tpl):
                        print(f"  ⏱ {note}")

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
