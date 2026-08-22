#!/usr/bin/env python3
"""
workflow_bench.py — which exported ComfyUI workflow actually draws this look?

WHY THIS EXISTS. The visual ceiling of this pipeline is the workflow, not the
prompt. The evidence is a real gallery of sixteen stills: `stickman` forbids a
washed-out background twice, in plain language, in the block appended to every
single prompt — and every background came back pale beige. When a model
overrides an explicit, repeated instruction, the next sentence will not fix
it. The checkpoint, the LoRA and the sampler settings will.

But "which workflow is better" has been unanswerable, because nothing here can
compare two of them. The /styles page renders one fixed scene through MANY
STYLES and one workflow. This is the transpose: one style, MANY WORKFLOWS,
same prompt, same seed, side by side.

    config/workflows/*.json        drop an API export in; it becomes a column
    config/stills_api.json         the baseline — what the channel ships today

HOW TO USE IT WITHOUT FOOLING YOURSELF. Change ONE thing between candidates —
the checkpoint, or the LoRA, or its weight, or the step count — and name the
file after that thing. Two changes at once produce a winner you cannot explain
and therefore cannot repeat.

    python scripts/workflow_bench.py --check     validate the exports, no GPU
    python scripts/workflow_bench.py             render the grid

THE PROBES ARE NOT DECORATIVE. Each of the six is a defect this project has
actually shipped, so a workflow is measured against the ways this channel has
been let down rather than against a pretty test scene:

    face             ten shots of a country losing its money, ten mild smiles
    animal           the stick-figure/real-animal contrast that IS this style
    action           thirteen of sixteen frames had nobody doing anything
    writing_surface  readable lettering got through twice in one gallery
    crowd            two frames came back as six-panel contact sheets
    weather_place    every background pale beige, in defiance of the style

NO OVERALL SCORE, on purpose. A single number would pretend a precision that
does not exist, and the picture is still the arbiter — you look at the grid and
decide which column is the channel's look. The numbers are here to stop the
obvious failures being argued about: how many probes each workflow passed, and
what it cost in seconds.

CONTRACT: read-only with respect to the pipeline. It renders into
media_library/bench/ and touches nothing a run depends on.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402

ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / "config" / "workflows"
BASELINE_NAME = "current (stills_api.json)"

# One render per probe per workflow, so this is the number that decides how
# long a bench takes. Six probes is about three minutes per candidate on a
# 3090 — cheap enough to re-run after every change to a workflow, which is the
# whole point of having it.
PROBES: list[tuple[str, str]] = [
    ("face",
     "Close on one figure's face at the moment the news lands: brows raised "
     "high, mouth a small open oval."),
    ("animal",
     "A figure stands beside a leopard sheltering under a rock ledge, the "
     "animal drawn in full with its spots and its real proportions."),
    ("action",
     "The table goes over and coins scatter across the floorboards, one figure "
     "still holding the edge of it."),
    ("writing_surface",
     "A clerk pushes an open ledger across a counter, its pages turned toward "
     "us."),
    ("crowd",
     "Five people crowd against the shutters of a closed bank, one hammering "
     "on the wood."),
    # TAGGED, BECAUSE THE PIPELINE TAGS ITS BEATS. Untagged, shot_kind() reads
    # "figure" and _detail_for_shot sends the whole FIGURE ONLY half — a
    # paragraph about oval heads and five separate limbs — attached to a shot
    # with no person in it. A real gallery run showed exactly what that buys:
    # a stick figure standing in the middle of a landscape that never asked
    # for one, with a small blue mouth-shape stuck to its face pouring water,
    # because "the mouth of a cave" and a paragraph about faces arrived in the
    # same prompt. A probe that sends what the pipeline would not send is
    # measuring a prompt this channel never renders.
    ("weather_place",
     "[SHOT=object] Rain pours past the mouth of a cave onto a flooded river "
     "bank below, hills behind it under a low grey sky."),
]

# Fixed per probe and shared by every candidate: the difference between two
# columns has to be the workflow, not the noise it started from.
_PROBE_SEEDS = {name: 1_000_003 + i * 7919 for i, (name, _) in enumerate(PROBES)}

RENDER_TIMEOUT = 300


def bench_root() -> Path:
    return paths.media_root() / "bench"


# ── the candidates ───────────────────────────────────────────────────────────

def candidates() -> list[tuple[str, Path]]:
    """(label, path) for every workflow to compare, baseline first.

    The baseline is what the channel ships today. A candidate measured against
    nothing is a picture; measured against the current export it is a decision.
    """
    import comfy_client
    out: list[tuple[str, Path]] = []
    if comfy_client.STILLS_TEMPLATE.exists():
        out.append((BASELINE_NAME, comfy_client.STILLS_TEMPLATE))
    if WORKFLOWS_DIR.is_dir():
        for p in sorted(WORKFLOWS_DIR.glob("*.json")):
            out.append((p.stem, p))
    return out


def advisories(graph: dict) -> list[str]:
    """Things that render fine and change what you get. Never fatal.

    THE ONE THAT MATTERS MOST, and it was found by reading the owner's own
    export rather than by guessing: a KSampler at **cfg 1**. At CFG 1.0 the
    guided result is exactly the conditional one — the unconditional branch
    cancels out — so the negative prompt is computed, wired, sent, and then
    has no effect whatsoever. Every term in it (no lettering, no photo drift,
    no extra fingers) does nothing.

    That is normal and correct for a distilled turbo checkpoint, which is what
    z_image_turbo is: it runs at CFG 1 by design. It is also the answer to
    "why is there readable lettering in my gallery when the negative leads with
    text, letters, words". Suppression has to come from the positive prompt or
    from a checkpoint that runs with guidance — not from a longer negative.
    """
    import video_format
    notes: list[str] = []
    for node in graph.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        cfg = ins.get("cfg")
        if isinstance(cfg, (int, float)) and cfg <= 1.0:
            notes.append(
                f"cfg is {cfg:g} — at CFG 1 there is no classifier-free "
                f"guidance, so the negative prompt has NO effect at all. "
                f"Normal for a turbo/distilled checkpoint; it means lettering "
                f"and photo drift can only be suppressed from the positive "
                f"prompt or by another checkpoint.")
            break
    want_w, want_h = video_format.still_dimensions()
    for node in graph.values():
        ins = node.get("inputs")
        if not isinstance(ins, dict):
            continue
        w, h = ins.get("width"), ins.get("height")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            if (w > h) != (want_w > want_h):
                notes.append(
                    f"the latent is {int(w)}×{int(h)} and this format wants "
                    f"{want_w}×{want_h} — the fit will crop most of every "
                    f"picture away")
            break
    return notes


def validate(path: Path) -> tuple[bool, list[str]]:
    """(usable, problems) for one export, without rendering anything.

    Six probes across four candidates is twenty-four renders. A broken export
    should cost zero of them, and should say which of the four things is wrong
    rather than failing at submit time with a ComfyUI stack trace.
    """
    import comfy_template
    problems: list[str] = []
    graph = comfy_template.load_template(path)
    if graph is None:
        return (False, ["not a readable API export (Export (API), not Export)"])
    if not comfy_template.has_placeholder(graph):
        problems.append(f"no {comfy_template.PLACEHOLDER} in any text node — "
                        f"set the positive prompt to that word and re-export")
    if not comfy_template.negative_text_nodes(graph):
        problems.append("no text node wired to a sampler's negative input, so "
                        "the negative prompt lands nowhere: lettering, photo "
                        "drift and extra fingers go unsuppressed")
    if not any(isinstance(n.get("inputs"), dict)
               and isinstance(n["inputs"].get("width"), (int, float))
               and isinstance(n["inputs"].get("height"), (int, float))
               for n in graph.values()):
        problems.append("no node with a plain width and height, so the frame "
                        "size cannot be set and the export's own wins")
    return (not problems, problems)


# ── one render ───────────────────────────────────────────────────────────────

def render_probe(graph: dict, prompt: str, seed: int, out_path: Path,
                 width: int, height: int) -> tuple[bool, float]:
    """Render one probe through one workflow. (ok, seconds). Never raises."""
    import comfy_client
    import comfy_template
    import image_gen

    t0 = time.time()
    try:
        g = comfy_template.prepare(graph, prompt=prompt, seed=seed,
                                   save_prefix="rufus_bench",
                                   negative=comfy_client._stills_negative())
        image_gen._apply_image_dims(g, width, height)
        pid = comfy_client._submit(g, uuid.uuid4().hex)
        if not pid:
            print("      ComfyUI rejected the workflow")
            return (False, time.time() - t0)
        png = comfy_client._await_image(pid, timeout=RENDER_TIMEOUT)
        if not png:
            print("      no image came back")
            return (False, time.time() - t0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        return (True, time.time() - t0)
    except Exception as e:
        print(f"      render failed: {e}")
        return (False, time.time() - t0)


def judge(png: Path, prompt: str) -> dict:
    """The objective half. Reuses the checks the pipeline already ships.

    Not a verdict on the workflow — a note under the picture. `frame_gate`
    catches a blank frame and a contact sheet from the pixels alone; the vision
    model, when it is switched on, answers whether the picture shows what was
    asked and whether lettering got through. Both are the same checks a real
    run applies, which is the point: a workflow is being measured against the
    standard its output will actually face.
    """
    out = {"gate": "ok", "detail": ""}
    try:
        import frame_gate
        ok, reason, detail = frame_gate.check(png, prompt=prompt)
        if not ok:
            out["gate"], out["detail"] = reason, detail
    except Exception as e:
        out["gate"], out["detail"] = "unchecked", str(e)[:120]
    return out


# ── the bench ────────────────────────────────────────────────────────────────

def run(styled: bool = True) -> dict:
    """Render every probe through every candidate. Returns the result dict."""
    import comfy_client
    import comfy_template
    import video_format

    cands = candidates()
    if not cands:
        print("[bench] no workflows to compare. Export one from ComfyUI to "
              f"{WORKFLOWS_DIR}/ (Export (API), prompt set to "
              f"{comfy_template.PLACEHOLDER}).")
        return {"workflows": [], "probes": []}

    if not comfy_client.is_available():
        print(f"[bench] ComfyUI is not reachable at {comfy_client._host()} — "
              f"start it first.")
        return {"workflows": [], "probes": []}

    width, height = video_format.still_dimensions()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = bench_root() / stamp
    result = {"stamp": stamp, "dir": str(out_dir), "width": width,
              "height": height, "probes": [p for p, _ in PROBES],
              "workflows": []}

    print(f"[bench] {len(cands)} workflow(s) × {len(PROBES)} probe(s) "
          f"at {width}×{height}")

    for label, path in cands:
        ok, problems = validate(path)
        graph = comfy_template.load_template(path)
        notes = advisories(graph) if graph else []
        entry = {"label": label, "path": str(path), "usable": ok,
                 "problems": problems, "notes": notes, "renders": {}}
        result["workflows"].append(entry)
        if not ok:
            print(f"[bench] {label}: SKIPPED")
            for p in problems:
                print(f"      · {p}")
            continue

        print(f"[bench] {label}")
        for n in notes:
            print(f"      ⚠ {n}")
        for probe, text in PROBES:
            prompt = comfy_client._with_detail(text) if styled else text
            png = out_dir / _slug(label) / f"{probe}.png"
            rendered, secs = render_probe(graph, prompt, _PROBE_SEEDS[probe],
                                          png, width, height)
            row = {"ok": rendered, "seconds": round(secs, 1),
                   "file": str(png) if rendered else ""}
            if rendered:
                row.update(judge(png, prompt))
            entry["renders"][probe] = row
            mark = row.get("gate", "-") if rendered else "FAILED"
            print(f"      {probe:<16} {secs:5.1f}s  {mark}")

        done = [r for r in entry["renders"].values() if r["ok"]]
        entry["mean_seconds"] = round(
            sum(r["seconds"] for r in done) / len(done), 1) if done else 0.0
        entry["passed"] = sum(1 for r in done if r.get("gate") == "ok")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bench.json").write_text(json.dumps(result, indent=2),
                                            encoding="utf-8")
        print(f"\n[bench] {out_dir}/bench.json")
    except OSError as e:
        print(f"[bench] could not write bench.json ({e})")
    return result


def _slug(label: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "_" for c in label.lower()]
    return "".join(keep).strip("_") or "workflow"


def latest() -> dict:
    """The most recent bench result, or {} — what the dashboard reads."""
    root = bench_root()
    if not root.is_dir():
        return {}
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        f = d / "bench.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    return {}


def check() -> int:
    """Validate every candidate and render nothing. Returns an exit code."""
    cands = candidates()
    if not cands:
        print(f"[bench] nothing to check. Drop an API export in "
              f"{WORKFLOWS_DIR}/")
        return 1
    import comfy_template
    bad = 0
    for label, path in cands:
        ok, problems = validate(path)
        graph = comfy_template.load_template(path)
        print(f"{'ok  ' if ok else 'BAD '} {label}")
        for p in problems:
            print(f"       · {p}")
        for n in (advisories(graph) if graph else []):
            print(f"       ⚠ {n}")
        bad += 0 if ok else 1
    print(f"\n{len(cands) - bad} of {len(cands)} usable")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    # THE STYLE THE CHANNEL SHIPS LIVES IN THE DASHBOARD, NOT IN THIS TERMINAL.
    # `run(styled=True)` appends comfy_client._with_detail(), which reads
    # RUFUS_STYLE — and RUFUS_STYLE is set on the Settings page, into
    # config/dashboard_settings.json. main.py and watchdog.py read that file
    # before they do anything; this file did not, so a bench started from a
    # fresh PowerShell compared workflows through the BUILT-IN FALLBACK style
    # while the channel renders on another one. The whole point of a bench is
    # that only one thing varies between the columns, and the one thing that
    # must not vary is the style. (The same gap put style_probe.py's manifest
    # on record as style "(default)".)
    #
    # apply() fills gaps only, so a `$env:RUFUS_STYLE` typed for this one
    # comparison still wins. Best-effort: an unreadable settings file must not
    # stop a bench that would otherwise run.
    try:
        import settings_store
        settings_store.apply()
    except Exception as e:
        print(f"[bench] saved settings not read ({e})")

    if "--check" in sys.argv[1:]:
        raise SystemExit(check())
    run()
