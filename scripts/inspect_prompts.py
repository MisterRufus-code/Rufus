#!/usr/bin/env python3
"""
inspect_prompts.py — Diagnose SD prompt quality before wasting a full render.

Shows the FULL prompt for each beat (main.py truncates to 90 chars), grades each
one against the 7 required sections, flags issues, and optionally sends one prompt
to A1111 / diffusers so you can visually verify the output.

Usage:
    # Use real seed from active niche (writes script, shows all prompts)
    python scripts/inspect_prompts.py

    # Specific niche
    python scripts/inspect_prompts.py --niche finance

    # Test with your own script text
    python scripts/inspect_prompts.py --script "I paid off 34k. Here is how."

    # Also generate the images (needs A1111 running or diffusers installed)
    python scripts/inspect_prompts.py --generate

    # Only generate the first N beats (faster test)
    python scripts/inspect_prompts.py --generate --beats 2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

DIVIDER  = "─" * 72
SECTION_RE = {
    "ACTIVATOR":   re.compile(r"\bRAW\s+photo\b", re.IGNORECASE),
    "SUBJECT":     re.compile(r"\b(\d{2}yo|\d{2}-year|year.old|man|woman|person|figure|worker|trader|entrepreneur)\b", re.IGNORECASE),
    "ACTION/POSE": re.compile(r"\b(holding|gripping|sitting|standing|typing|reading|staring|reviewing|walking|looking|checking|signing|opening|closing|pressing)\b", re.IGNORECASE),
    "SETTING":     re.compile(r"\b(office|desk|room|kitchen|street|city|floor|building|home|studio|cafe|bank|market|table|counter)\b", re.IGNORECASE),
    "LIGHTING":    re.compile(r"\b(light|lamp|glow|sun|shadow|ray|beam|warm|cold|neon|tungsten|led|overcast|window|backlit|rim\s*light)\b", re.IGNORECASE),
    "OPTICS":      re.compile(r"\b(mm|f/\d|aperture|nikon|canon|sony|sigma|fuji|leica|lens|bokeh|depth.of.field|dof)\b", re.IGNORECASE),
    "COLOR_GRADE": re.compile(r"\b(grade|teal|orange|cinematic|muted|tone|warm|cool|desaturated|film\s*grain|kodak|portra|fuji)\b", re.IGNORECASE),
}

ABSTRACT_WARNS = re.compile(
    r"\b(signifying|representing|symbol|metaphor|concept|idea|notion|"
    r"abstract|surreal|allegorical|dream|vision|spirit)\b",
    re.IGNORECASE,
)
STOCK_WARNS = re.compile(
    r"\b(stock\s*photo|posed|smiling\s*at\s*camera|looking\s*at\s*camera|"
    r"corporate\s*headshot|handshake|thumbs.up)\b",
    re.IGNORECASE,
)
QUALITY_BANS = re.compile(
    r"\b(8k\s*uhd|masterpiece|best\s*quality|ultra\s*detailed|highly\s*detailed)\b",
    re.IGNORECASE,
)


def _grade_prompt(prompt: str, beat: str) -> dict:
    """Return a dict of pass/fail checks + word count + issues list."""
    words    = len(prompt.split())
    issues   = []
    sections = {}

    for name, pattern in SECTION_RE.items():
        found = bool(pattern.search(prompt))
        sections[name] = found
        if not found:
            issues.append(f"missing {name} section")

    if words < 60:
        issues.append(f"too short ({words} words — aim for 90-110)")
    elif words < 90:
        issues.append(f"short ({words} words — aim for 90-110)")
    elif words > 130:
        issues.append(f"very long ({words} words — may cut off in A1111)")

    if ABSTRACT_WARNS.search(prompt):
        m = ABSTRACT_WARNS.search(prompt)
        issues.append(f"abstract language: '{m.group()}' — show the LITERAL thing")

    if STOCK_WARNS.search(prompt):
        m = STOCK_WARNS.search(prompt)
        issues.append(f"stock-photo tell: '{m.group()}'")

    if QUALITY_BANS.search(prompt):
        m = QUALITY_BANS.search(prompt)
        issues.append(f"banned quality token: '{m.group()}' — these are appended by sd_client")

    # Check subject specificity — does it have age/ethnicity/expression?
    has_age        = bool(re.search(r"\b\d{2}\s*(?:yo|year)", prompt, re.IGNORECASE))
    has_expression = bool(re.search(r"\b(frown|grimace|exhausted|tense|anxious|relieved|determined|frustrated|proud|worried|neutral)\b", prompt, re.IGNORECASE))
    if not has_age:
        issues.append("subject lacks age detail — add e.g. '34yo'")
    if not has_expression:
        issues.append("subject lacks facial expression — add e.g. 'jaw tight, quiet dread'")

    # Beat alignment — check that key words from the beat appear in the prompt
    beat_nouns = [w.lower().strip(".,;:!?") for w in beat.split() if len(w) > 4 and w[0].isupper() or w[0].isdigit()]
    beat_nouns = [w for w in beat_nouns if w not in {"this","that","those","these","with","from","when","then"}]
    unmatched = [w for w in beat_nouns[:3] if w.lower() not in prompt.lower()]
    if unmatched:
        issues.append(f"beat words not reflected in prompt: {unmatched}")

    score = max(0, 10 - len(issues))
    return {
        "words":    words,
        "sections": sections,
        "issues":   issues,
        "score":    score,
    }


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def _ok(s):  return _color(s, "32")
def _warn(s): return _color(s, "33")
def _bad(s):  return _color(s, "31")
def _bold(s): return _color(s, "1")
def _dim(s):  return _color(s, "2")


def inspect(prompts: list[str], beats: list[str]) -> None:
    """Pretty-print full prompts with grading."""
    total_score = 0

    for i, (prompt, beat) in enumerate(zip(prompts, beats)):
        grade = _grade_prompt(prompt, beat)
        score = grade["score"]
        total_score += score

        score_label = (
            _ok(f"{score}/10")   if score >= 8  else
            _warn(f"{score}/10") if score >= 5  else
            _bad(f"{score}/10")
        )

        print(f"\n{DIVIDER}")
        print(f"  {_bold(f'BEAT {i+1}')}  score: {score_label}  words: {grade['words']}")
        print(f"{DIVIDER}")
        print(f"  {_dim('Spoken line:')} {beat}")
        print()

        # Full prompt with word wrap at 72 chars
        words  = prompt.split()
        line   = "  "
        for w in words:
            if len(line) + len(w) + 1 > 74:
                print(line)
                line = "  " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

        print()
        print(f"  {_bold('Sections:')}", end="  ")
        for sec, ok in grade["sections"].items():
            label = _ok(f"✓{sec}") if ok else _bad(f"✗{sec}")
            print(label, end="  ")
        print()

        if grade["issues"]:
            print(f"\n  {_bold('Issues:')}")
            for issue in grade["issues"]:
                print(f"    {_warn('⚠')} {issue}")

    avg = total_score / len(prompts) if prompts else 0
    avg_label = _ok(f"{avg:.1f}/10") if avg >= 7 else _warn(f"{avg:.1f}/10") if avg >= 5 else _bad(f"{avg:.1f}/10")
    print(f"\n{DIVIDER}")
    print(f"  {_bold('BATCH AVERAGE:')} {avg_label}  ({len(prompts)} prompts)")
    print(f"{DIVIDER}\n")

    if avg < 6:
        print(_bad("  Prompts are weak — likely GPT key missing or fallback templates used."))
        print("  Check: OpenAI key in config/keys.json → GPT-4o-mini will rewrite these.\n")
    elif avg < 8:
        print(_warn("  Prompts are decent but some beats lack specificity."))
        print("  Common fix: add age, expression, and setting texture.\n")
    else:
        print(_ok("  Prompts look strong. Ready to generate.\n"))


def _generate_one(prompt: str, out_dir: Path, index: int) -> Path | None:
    """Send one prompt to A1111 or diffusers and return the output image path."""
    # Try A1111 first
    try:
        from sd_client import generate_clips
        clips = generate_clips([prompt], n=1, prebuilt=True)
        if clips:
            print(f"  [A1111] clip: {clips[0]}")
            return clips[0]
    except Exception as e:
        print(f"  [A1111] failed ({e})")

    # Fallback to diffusers
    try:
        from diffusers_client import generate_clips as diff_gen
        clips = diff_gen([prompt])
        if clips:
            print(f"  [diffusers] clip: {clips[0]}")
            return clips[0]
    except Exception as e:
        print(f"  [diffusers] failed ({e})")

    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect and grade Rufus SD prompts")
    parser.add_argument("--niche",    type=str, help="Niche to use (overrides active)")
    parser.add_argument("--script",   type=str, help="Provide script text directly (skip GPT)")
    parser.add_argument("--generate", action="store_true", help="Also generate test images")
    parser.add_argument("--beats",    type=int, default=6, help="Max beats / scenes (default 6)")
    args = parser.parse_args()

    if args.niche:
        os.environ["RUFUS_NICHE_OVERRIDE"] = args.niche

    from main import _build_sd_prompts, _split_beats, load_niche_cfg

    niche_cfg, active = load_niche_cfg(args.niche)
    print(f"\n{_bold('Rufus Prompt Inspector')}  niche={active}")

    # ── Get a script to work from ──────────────────────────────────────────────
    if args.script:
        script = args.script
        print(f"  Using provided script ({len(script.split())} words)\n")
    else:
        print("  Generating real seed + script (this uses GPT — ~$0.002)...")
        try:
            from research import get_seed
            from script_writer import write_script, preanalyze
            seed     = get_seed(active)
            analysis, run_id, _ = preanalyze(seed)
            result   = write_script("", seed=seed, precomputed_analysis=analysis, run_id=run_id)
            script   = result["script"]
            print(f"  Script score: {result['score']}/10")
            print(f"  Script ({len(script.split())} words):")
            print(f"  {_dim(script[:200])}{'…' if len(script) > 200 else ''}\n")
        except Exception as e:
            print(f"  GPT script gen failed ({e}) — using built-in demo script")
            script = (
                "I paid off $34,000 in student loans in 14 months on a $48,000 salary. "
                "Most people think that's impossible. Here's what I actually did. "
                "I automated $1,200 every payday before I could touch it. "
                "I worked weekends delivering groceries for an extra $600 a month. "
                "And I cut my rent by moving in with two roommates. "
                "Fourteen months later, I made the final payment."
            )

    # ── Build prompts ──────────────────────────────────────────────────────────
    beats   = _split_beats(script, max_scenes=args.beats)
    prompts = _build_sd_prompts(script, active, max_scenes=args.beats)
    print(f"  Generated {len(prompts)} prompts for {len(beats)} beats")

    # ── Inspect ────────────────────────────────────────────────────────────────
    inspect(prompts, beats)

    # ── Optionally generate ────────────────────────────────────────────────────
    if args.generate:
        n = args.beats if args.beats < len(prompts) else len(prompts)
        print(f"  Generating {n} test image(s)...\n")
        out_dir = ROOT / "media_library" / "prompt_tests"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            print(f"  Beat {i+1}: {prompts[i][:80]}…")
            result = _generate_one(prompts[i], out_dir, i)
            if not result:
                print(f"  ✗ Beat {i+1}: no output — check A1111 is running or diffusers is installed")
        print(f"\n  Test images saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
