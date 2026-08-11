#!/usr/bin/env python3
"""
edit_director.py — decide how each beat is CUT, not just what it shows.

Why this exists: every Rufus Short is edited identically. Short.tsx picks its
camera move with `KB_PATTERNS[index % KB_PATTERNS.length]` — a fixed cycle of
six, in the same order, every video. A Short about hyperinflation and one about
the Bank of France get the same push-in on beat 1, the same pull-back on beat
2, forever. The images changed; the edit never did.

That is the difference between a slideshow and an edit. A director watching
this script would not zoom mechanically: they would hold dead still on the
number so it lands, push in on the turn, and let the closing line sit.

So a small model reads the finished script and returns, per beat, how that beat
should MOVE — plus which words carry the line. Timing is deliberately NOT the
director's to set: cut points come from the voiceover's own word timestamps,
and letting a model stretch a beat would desync picture from speech. What it
controls is what happens INSIDE each slot it is given.

CONTRACT: fail-open, like every optional step here. No key, no reply, bad JSON,
wrong beat count — all return None, and Short.tsx falls back to the cycle it
uses today. An edit plan is an improvement on a working render, never a
prerequisite for one.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONFIG_DIR = Path(__file__).parent.parent / "config"

# The moves Short.tsx can actually perform. The director picks from THIS list
# and nothing else — a name the renderer doesn't know would silently fall back
# to the mechanical cycle, which looks like the feature doing nothing.
MOTIONS = ("push_in", "pull_back", "hold_still", "drift_left",
           "drift_right", "rise")

# How hard the move is. "hold_still" ignores it; everything else scales by it.
INTENSITIES = ("subtle", "normal", "strong")

# The emotional map rides along in this same reply rather than costing a second
# model call — the director is already the one stage that reads the finished
# script end to end, so it is the cheapest place to ask what each beat FEELS
# like on top of how it moves. Consumed by emotional_map for per-beat grading
# and SFX weight; see that module for why this is not a gate.
import emotional_map  # noqa: E402
from emotional_map import TONES  # noqa: E402  (vocabulary, not behaviour)

MODEL_DEFAULT = "gpt-4o-mini"
MAX_EMPHASIS_WORDS = 4


def enabled() -> bool:
    return os.environ.get("RUFUS_EDIT_DIRECTOR", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _model() -> str:
    return os.environ.get("RUFUS_DIRECTOR_MODEL", MODEL_DEFAULT).strip() or MODEL_DEFAULT


def _prompt(beats: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(beats))
    return (
        "You are the editor of a 40-second vertical documentary Short. The "
        "narration is already recorded and the cut points are already fixed by "
        "the voice — you are NOT setting timing. For each beat you decide how "
        "the picture MOVES while that line is spoken, and which words carry "
        "it.\n\n"
        f"BEATS:\n{numbered}\n\n"
        f"MOTIONS you may use (exact strings): {', '.join(MOTIONS)}\n"
        f"INTENSITY: {', '.join(INTENSITIES)}\n\n"
        "HOW AN EDITOR ACTUALLY CHOOSES:\n"
        "- hold_still is the strongest move you have, and the most underused. "
        "A number, a reveal, or the line the whole video was built toward "
        "lands harder on a frame that does not move. If every beat moves, "
        "nothing does.\n"
        "- push_in for a turn or a tightening — the moment the story narrows "
        "onto one fact or one person.\n"
        "- pull_back to open out: scale, aftermath, a crowd, a consequence "
        "larger than the person it started with.\n"
        "- drift_left / drift_right / rise for continuity — a beat that "
        "carries the same scene onward rather than starting a new idea.\n"
        "- Vary. Two identical moves in a row read as an accident. But do not "
        "cycle for the sake of it either — repetition with PURPOSE (two "
        "hold_stills around the turn) is real editing.\n"
        "- The opening beat has one job: stop the scroll. The closing beat has "
        "one job: let the question sit.\n\n"
        f"TONE — how the beat FEELS, one of exactly: {', '.join(TONES)}\n"
        "- tension: something is wrong, or about to be.\n"
        "- curiosity: the question is open and the answer is withheld.\n"
        "- revelation: the turn, the number, the thing the video was built for.\n"
        "- weight: the consequence, the cost, the aftermath.\n"
        "- resolution: the line that lets it sit.\n"
        "- neutral: narration carrying information. Most beats are this one, "
        "and a video where every beat is a revelation has none.\n\n"
        "EMPHASIS: 0-3 words per beat that the caption should hit hardest — "
        "the figure, the name, the reversal. Copy them EXACTLY as they appear "
        "in the beat, including case. Most beats need none; a beat where every "
        "word is emphasised has no emphasis.\n\n"
        "Reply with ONLY this JSON, no prose:\n"
        '{"peak_beat": <the beat number where the story turns>,\n'
        ' "beats": [{"n": 1, "motion": "...", "intensity": "...", '
        '"tone": "...", "emphasis": ["..."]}, ...]}\n'
        f"Exactly {len(beats)} entries, n from 1 to {len(beats)}."
    )


def _clean(plan: dict, n_beats: int) -> dict | None:
    """Validate and normalise a model reply. None if it can't be trusted.

    Strict on purpose: a plan with an unknown motion name renders as the
    mechanical cycle for that beat only, which looks like the director working
    on some beats and not others — harder to diagnose than it plainly not
    running."""
    if not isinstance(plan, dict):
        return None
    raw = plan.get("beats")
    if not isinstance(raw, list) or len(raw) != n_beats:
        return None

    beats = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None
        motion = str(entry.get("motion", "")).strip().lower()
        if motion not in MOTIONS:
            return None
        intensity = str(entry.get("intensity", "normal")).strip().lower()
        if intensity not in INTENSITIES:
            intensity = "normal"
        emphasis = entry.get("emphasis") or []
        if not isinstance(emphasis, list):
            emphasis = []
        # Filter BEFORE capping: slicing first lets a blank entry eat one of
        # the four slots, silently dropping a word the director meant to hit.
        emphasis = [s for s in (str(w).strip() for w in emphasis) if s
                    ][:MAX_EMPHASIS_WORDS]
        # Tone degrades where motion rejects. An unknown motion name is a plan
        # the renderer cannot execute, so the whole plan is refused; an unknown
        # tone just grades that beat neutral. Refusing a good edit plan over a
        # cosmetic field would trade a working feature for a new one.
        tone = emotional_map.normalise(entry.get("tone"))
        beats.append({"n": i + 1, "motion": motion, "intensity": intensity,
                      "tone": tone, "emphasis": emphasis})

    peak = plan.get("peak_beat")
    if not isinstance(peak, int) or not 1 <= peak <= n_beats:
        peak = (n_beats + 1) // 2
    return {"peak_beat": peak, "beats": beats}


# Memo for one process, keyed on the exact beats. Both renderers now ask for a
# plan, and the Remotion→FFmpeg fallback asks twice for the same script inside
# a single run — without this that is two model calls and, worse, two DIFFERENT
# plans for one video (temperature 0.7), so the grade would not match the edit
# the director actually chose.
_plan_cache: dict[tuple[str, ...], dict | None] = {}


def direct(beats: list[str]) -> dict | None:
    """An edit plan for these beats, or None to use the renderer's default."""
    if not enabled() or not beats:
        return None
    cache_key = tuple(beats)
    if cache_key in _plan_cache:
        return _plan_cache[cache_key]
    plan = _direct_uncached(beats)
    _plan_cache[cache_key] = plan
    return plan


def _direct_uncached(beats: list[str]) -> dict | None:
    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        key = ""
        if keys_file.exists():
            key = json.loads(keys_file.read_text(encoding="utf-8")).get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            return None
        resp = OpenAI(api_key=key).chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": _prompt(beats)}],
            temperature=0.7, max_tokens=700, timeout=60,
            response_format={"type": "json_object"},
        )
        plan = _clean(json.loads(resp.choices[0].message.content or "{}"), len(beats))
    except Exception as e:
        print(f"[director] skipped (non-fatal): {e}")
        return None

    if plan is None:
        print("[director] reply didn't validate — using the default cycle")
        return None
    moves = ", ".join(b["motion"] for b in plan["beats"])
    print(f"[director] peak at beat {plan['peak_beat']} — {moves}")
    print(f"[director] tones: "
          f"{emotional_map.describe([b['tone'] for b in plan['beats']])}")
    return plan
