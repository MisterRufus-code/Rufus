#!/usr/bin/env python3
"""
style_probe.py — see a style change in two minutes instead of thirteen.

WHY THIS EXISTS. Every style edit this week cost a full pipeline run to
evaluate: script, voiceover, storyboard, stills, render — thirteen minutes to
answer a question about six sentences of text. Worse, each run drew a
different script on different seeds, so two galleries differed for reasons
that had nothing to do with the edit. Half the comparisons were against noise,
and at least one conclusion had to be walked back because of it.

This renders a fixed set of scenes, on fixed seeds, through the workflow the
channel actually ships, with only the style block varying. Same prompts, same
seeds, every time — so when two runs differ, the text is the only thing that
could have done it.

    python scripts/style_probe.py                      the current style
    python scripts/style_probe.py --style ink_explainer
    python scripts/style_probe.py --probes face,crowd  just those
    python scripts/style_probe.py --plain              no style at all
    python scripts/style_probe.py --list               what ran before

IT SHARES THE BENCH'S PROBES ON PURPOSE. workflow_bench.py already keeps six
scenes, each one a defect this project has actually shipped, and their seeds.
A second hand-written list would drift from that one within a month, and then
two tools would disagree about what a face test is. This is the transpose of
the bench, not a rival to it: the bench holds the style still and varies the
workflow, this holds the workflow still and varies the style.

WHAT IT PRINTS. Where the pictures are, what the automatic checks made of
them, and — the useful part — which probes changed since the previous run and
how the style text differs. "These words changed and these pictures changed"
is the sentence a style edit needs; "the pictures look different to me" is
what you get without it.

CONTRACT: read-only with respect to the pipeline. It renders into
media_library/bench/style/ and touches nothing a run depends on.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import console  # noqa: E402
import paths  # noqa: E402
import workflow_bench as bench  # noqa: E402

MANIFEST = "probe.json"


def probe_root() -> Path:
    return paths.media_root() / "bench" / "style"


def _digest(png: Path) -> str:
    """Content hash, so "did this picture change" is not a judgement call."""
    try:
        return hashlib.sha256(png.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def runs() -> list[Path]:
    """Previous probe directories, newest first."""
    root = probe_root()
    if not root.is_dir():
        return []
    out = [d for d in root.iterdir() if (d / MANIFEST).is_file()]
    return sorted(out, key=lambda d: d.name, reverse=True)


def load(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def selected(names: str | None) -> list[tuple[str, str]]:
    """The probes to run. Unknown names are named rather than ignored — a
    typo'd --probes that silently ran all six would waste the time this exists
    to save."""
    if not names:
        return list(bench.PROBES)
    want = [n.strip() for n in names.split(",") if n.strip()]
    known = {n for n, _ in bench.PROBES}
    unknown = [n for n in want if n not in known]
    if unknown:
        print(f"[probe] no such probe(s): {', '.join(unknown)}")
        print(f"[probe] available: {', '.join(sorted(known))}")
        return []
    return [(n, t) for n, t in bench.PROBES if n in want]


def style_text(name: str | None, plain: bool) -> tuple[str, str]:
    """(label, the block that will be appended). ("plain", "") for --plain."""
    import comfy_client
    if plain:
        return ("plain", "")
    if name:
        presets = comfy_client.style_presets()
        if name not in presets:
            print(f"[probe] no style named {name!r}. Available: "
                  f"{', '.join(sorted(presets))}")
            return ("", "")
        return (name, presets[name])
    import os
    label = (os.environ.get("RUFUS_STYLE") or "").strip() or "(default)"
    return (label, comfy_client._detail_suffix())


def compare(now: dict, before: dict) -> list[str]:
    """What changed between two probe runs, in the order a person asks it."""
    lines: list[str] = []
    if not before:
        # Nothing to diff against. Reporting the whole style block as "changed"
        # against an absent run would be true and useless, and run() already
        # says "first run" in words.
        return lines
    a, b = before.get("style_text", ""), now.get("style_text", "")
    if a != b:
        diff = [l for l in difflib.unified_diff(
            a.split(". "), b.split(". "), lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        lines.append(f"style text: {len(diff)} sentence(s) differ")
        for l in diff[:8]:
            lines.append(f"    {l[0]} {l[1:].strip()[:100]}")
        if len(diff) > 8:
            lines.append(f"    … {len(diff) - 8} more")
    elif before:
        lines.append("style text: identical — any difference below is the "
                     "model's own variance, not your edit")

    old_r, new_r = before.get("renders", {}), now.get("renders", {})
    changed = [p for p in new_r
               if p in old_r and old_r[p].get("sha") != new_r[p].get("sha")]
    same = [p for p in new_r
            if p in old_r and old_r[p].get("sha") == new_r[p].get("sha")]
    if changed:
        lines.append(f"pictures changed: {', '.join(changed)}")
    if same:
        lines.append(f"pictures identical: {', '.join(same)}")
    return lines


def run(style_name: str | None = None, probes: str | None = None,
        plain: bool = False) -> int:
    import comfy_client
    import comfy_template
    import video_format

    chosen = selected(probes)
    if not chosen:
        return 1
    label, tail = style_text(style_name, plain)
    if not label:
        return 1

    template = comfy_client.STILLS_TEMPLATE
    graph = comfy_template.load_template(template)
    if not graph:
        print(f"[probe] no usable workflow at {template} — export one from "
              f"ComfyUI (Export (API), prompt set to "
              f"{comfy_template.PLACEHOLDER}).")
        return 1
    if not comfy_client.is_available():
        print(f"[probe] ComfyUI is not reachable at {comfy_client._host()} — "
              f"start it first.")
        return 1

    for note in bench.advisories(graph):
        print(f"[probe] ⚠ {note}")

    width, height = video_format.still_dimensions()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = probe_root() / stamp
    previous = runs()

    print(f"[probe] style {label!r} · {len(chosen)} probe(s) at {width}×{height}")
    manifest = {"stamp": stamp, "style": label, "style_text": tail,
                "width": width, "height": height,
                "workflow": str(template), "renders": {}}

    for name, text in chosen:
        # The same composition the pipeline builds: the scene, then the style
        # block appended byte for byte. Rendering the bare scene would test
        # something this channel never ships.
        prompt = f"{text.rstrip().rstrip('.')}. {tail}" if tail else text
        png = out_dir / f"{name}.png"
        ok, secs = bench.render_probe(graph, prompt, bench._PROBE_SEEDS[name],
                                      png, width, height)
        row = {"ok": ok, "seconds": round(secs, 1),
               "file": str(png) if ok else "", "sha": _digest(png) if ok else ""}
        if ok:
            row.update(bench.judge(png, prompt))
        manifest["renders"][name] = row
        flag = "" if not ok else ("" if row.get("gate") == "ok"
                                  else f"  ⚠ {row['gate']}")
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<16} {secs:5.1f}s{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                    encoding="utf-8")
    print(f"\n[probe] {out_dir}")

    if previous:
        prev = load(previous[0])
        print(f"[probe] vs {previous[0].name}:")
        for line in compare(manifest, prev) or ["  (nothing to compare)"]:
            print(f"  {line}")
    else:
        print("[probe] first run — the next one will diff against this.")
    return 0


def show_runs() -> int:
    have = runs()
    if not have:
        print(f"[probe] nothing yet. Run it once: python scripts/style_probe.py")
        return 1
    for d in have[:15]:
        m = load(d)
        rendered = sum(1 for r in m.get("renders", {}).values() if r.get("ok"))
        print(f"  {d.name}  style={m.get('style', '?'):<16} {rendered} image(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--style", help="preset name (default: RUFUS_STYLE)")
    ap.add_argument("--probes", help="comma-separated subset, e.g. face,crowd")
    ap.add_argument("--plain", action="store_true",
                    help="render with no style block, to see what it adds")
    ap.add_argument("--list", action="store_true", help="previous probe runs")
    args = ap.parse_args(argv)
    if args.list:
        return show_runs()
    return run(args.style, args.probes, args.plain)


if __name__ == "__main__":
    raise SystemExit(main())
