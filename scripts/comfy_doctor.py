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
_LOADERS = ("UNETLoader", "CheckpointLoaderSimple", "VAELoader", "CLIPLoader",
            "DualCLIPLoader", "UnetLoaderGGUF", "CLIPVisionLoader")

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


def _report_engine(name: str, host: str, seen: dict[str, set[str]]) -> None:
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
            print(f"  ✗ no export at {tpl_rel}")
            print(f"    ComfyUI → Workflow → Browse Templates → run it once →")
            print(f"    set the positive prompt to exactly  RUFUS_PROMPT  →")
            print(f"    Workflow → Export (API) → save as {tpl_rel}")
        else:
            tpl = comfy_template.load_template(tpl_path)
            if tpl is None:
                print(f"  ✗ {tpl_rel} exists but is not a valid API export")
                print(f"    (\"Export (API)\", not \"Export\" — the plain export is")
                print(f"    a UI workflow and has a different shape)")
            elif not comfy_template.has_placeholder(tpl):
                print(f"  ✗ {tpl_rel} has no RUFUS_PROMPT placeholder")
                print(f"    Set the positive prompt text to exactly RUFUS_PROMPT")
                print(f"    in ComfyUI and export again.")
            else:
                bad_nodes = comfy_template.missing_nodes(tpl, host)
                bad_files = comfy_template.missing_models(tpl, host)
                if bad_nodes:
                    print(f"  ✗ ComfyUI is missing node(s): {', '.join(bad_nodes[:5])}")
                elif bad_files:
                    print(f"  ✗ ComfyUI cannot load: {'; '.join(bad_files[:4])}")
                    print(f"    The export names a file this ComfyUI does not have —")
                    print(f"    re-export after picking a file that IS in the dropdown.")
                else:
                    print(f"  ✓ {tpl_rel} valid — nodes and model files all resolve")

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
    for name in (wanted or list(ENGINES)):
        _report_engine(name, host, seen)

    unknown = [a for a in argv if a not in ENGINES]
    if unknown:
        print(f"\nunknown engine(s): {', '.join(unknown)} — "
              f"known: {', '.join(ENGINES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
