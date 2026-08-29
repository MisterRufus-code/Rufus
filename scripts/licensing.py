#!/usr/bin/env python3
"""What Rufus is made of, and what each part lets you do with it.

WHY A PIPELINE NEEDS THIS AND A SCRIPT DOES NOT. Rufus assembles a video out of
other people's work: a text model writes the script, an image model draws every
shot, a voice model reads it, a music source scores it, a font sets the
captions. Each of those arrives under a licence, and the licences do not all
say the same thing. Some are Apache 2.0 and ask nothing. Some — Stability's
research licences, Coqui's model licence, MusicGen's CC-BY-NC weights — permit
use and forbid COMMERCIAL use, which is precisely what a monetised channel is.

Nothing in this repository knew that. The engine picker in tts_engine's
docstring describes XTTS as "Free, local GPU, near-ElevenLabs quality" with no
mention of what its licence permits, and the same is true of every other
choice. An owner switching engines to improve a video has no way to see that
one of the switches changes what they may legally do with the result.

TWO DIFFERENT QUESTIONS, AND CONFLATING THEM IS THE MISTAKE.

  SELL COPIES OF RUFUS — constrained by the licences of the code shipped or
  imported. A GPL dependency is a problem here; a non-commercial MODEL is not,
  because the model is not shipped.

  MONETISE THE VIDEOS — constrained by the licences of the model WEIGHTS and
  the asset sources, plus the terms of service of any cloud API. A permissively
  licensed program can still produce output you may not sell.

Both are reported, separately, because a component can be fine for one and
fatal for the other.

WHAT THIS FILE IS NOT. It is not legal advice and it does not pretend to
certainty it has not got. Every entry carries the primary source URL, and a
claim is only made where it was actually READ. Python package licences below
are read at runtime from the installed distribution's own metadata, which is as
primary as it gets. Model weights and cloud services cannot be read from here,
so they are carried as OPEN QUESTIONS with the URL that answers them, and the
preflight names them until somebody records an answer in
config/licences.json.

A hard-coded licence table would have been quicker and would have been wrong
within a year, silently — the same shape of bug as every other stale constant
in this tree. An unchecked entry that says so is worth more than a checked-in
guess that reads like a fact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANSWERS_FILE = ROOT / "config" / "licences.json"

# The two questions, kept apart everywhere below.
SELL = "sell_rufus"          # may I sell copies of this software?
MONETISE = "monetise_output"  # may I make money from the videos it produces?

# What an answer may be. UNKNOWN is the default and is not a failure state — it
# is the honest state of a question nobody has looked up yet.
YES, NO, CONDITIONAL, UNKNOWN = "yes", "no", "conditional", "unknown"


class Component:
    """One part of the pipeline whose licence constrains something.

    `applies` is a callable taking the environment and returning whether this
    component is actually in play for the CURRENT configuration — the whole
    point of the report is that it answers for the setup in front of you rather
    than listing every model the repository has ever mentioned.
    """

    def __init__(self, key: str, name: str, what: str, *, question: str,
                 source: str, applies=None, note: str = "",
                 measured_from: str = ""):
        self.key = key
        self.name = name
        self.what = what
        self.question = question
        self.source = source
        self.note = note
        self.measured_from = measured_from
        self._applies = applies or (lambda env: True)

    def applies(self, env: dict) -> bool:
        try:
            return bool(self._applies(env))
        except Exception:
            # A predicate that cannot decide must not hide the component. The
            # cost of an extra line in the report is nothing; the cost of a
            # silently omitted non-commercial model is the whole point of this
            # file.
            return True


def _tts(env: dict) -> str:
    return (env.get("RUFUS_TTS") or "").strip().lower() or "edge"


def _source(env: dict) -> str:
    return (env.get("RUFUS_VIDEO_SOURCE") or "").strip().lower()


# ── the parts, and what each one gates ───────────────────────────────────────
#
# Ordered by how likely they are to bite: the engines a run actually uses come
# first, the ones behind an explicit opt-in after.

COMPONENTS: list[Component] = [
    # ── voice ───────────────────────────────────────────────────────────────
    Component(
        "edge-tts", "Edge TTS", "the default narration voice",
        question=SELL,
        source="https://github.com/rany2/edge-tts",
        applies=lambda env: _tts(env) in ("edge", ""),
        measured_from="edge_tts",
        note="Two separate things to settle, and only the first is a licence. "
             "The PACKAGE's licence is read from its own metadata below. The "
             "SERVICE it calls is Microsoft's Read Aloud endpoint, used "
             "without a published API agreement — whether a commercial "
             "product may drive it is a terms-of-service question, not a "
             "licensing one, and no file in this repository has ever asked "
             "it. Kokoro is the escape hatch: Apache 2.0, local, no service.",
    ),
    Component(
        "xtts-v2", "Coqui XTTS v2 weights", "narration when RUFUS_TTS=xtts",
        question=MONETISE,
        source="https://coqui.ai/cpml",
        applies=lambda env: _tts(env) == "xtts",
        note="Released under the Coqui Public Model Licence rather than a "
             "software licence, and the distinction matters: it governs what "
             "the model may be USED FOR. Settle this before a video voiced "
             "by it earns anything.",
    ),
    Component(
        "kokoro-82m", "Kokoro-82M weights", "narration when RUFUS_TTS=kokoro",
        question=MONETISE,
        source="https://huggingface.co/hexgrad/Kokoro-82M",
        applies=lambda env: _tts(env).startswith("kokoro"),
    ),
    Component(
        "elevenlabs", "ElevenLabs", "narration when RUFUS_TTS=elevenlabs",
        question=MONETISE,
        source="https://elevenlabs.io/terms",
        applies=lambda env: _tts(env) == "elevenlabs",
        note="A paid service, so this is a plan question rather than a licence "
             "one: their free and lowest tiers have historically not carried "
             "commercial rights. Check which tier the key belongs to.",
    ),

    # ── pictures ────────────────────────────────────────────────────────────
    Component(
        "z-image", "Z-Image / Z-Image-Turbo weights",
        "every gallery still on the ComfyUI path",
        question=MONETISE,
        source="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        applies=lambda env: _source(env) in ("comfy", "", "sd"),
        note="THE ONE THAT MATTERS MOST HERE. It draws every picture in every "
             "video this channel ships, so whatever it permits is the ceiling "
             "on what the channel may do — there is no per-video escape.",
    ),
    Component(
        "flux", "FLUX.1 weights", "stills when the FLUX workflow is selected",
        question=MONETISE,
        source="https://blackforestlabs.ai/",
        applies=lambda env: bool((env.get("RUFUS_FLUX") or "").strip()),
        note="The dev and schnell releases do not carry the same terms as each "
             "other. Check which checkpoint is actually loaded, not which "
             "family it belongs to.",
    ),
    Component(
        "svd-xt", "Stable Video Diffusion (svd_xt)",
        "image-to-video motion on the SVD engine",
        question=MONETISE,
        source="https://stability.ai/license",
        applies=lambda env: "svd" in _source(env)
        or bool((env.get("RUFUS_MOTION") or "").lower().count("svd")),
        note="Stability have licensed their video models under research and "
             "community terms that have changed more than once, and have "
             "carried revenue thresholds. Read the current one against this "
             "channel's actual revenue.",
    ),
    Component(
        "wan22", "Wan 2.2 weights", "text-to-video clips on the Wan engine",
        question=MONETISE,
        source="https://github.com/Wan-Video/Wan2.2",
        applies=lambda env: "wan" in _source(env),
    ),
    Component(
        "hunyuanvideo", "HunyuanVideo 1.5 weights",
        "image-to-video clips on the Hunyuan engine",
        question=MONETISE,
        source="https://github.com/Tencent-Hunyuan/HunyuanVideo",
        applies=lambda env: "hunyuan" in _source(env),
        note="Tencent's community licences have carried TERRITORY limits as "
             "well as scale limits — a clause that has nothing to do with "
             "revenue and everything to do with where the operator sits.",
    ),

    # ── words ───────────────────────────────────────────────────────────────
    Component(
        "openai", "OpenAI API", "the script, the storyboard and the metadata",
        question=MONETISE,
        source="https://openai.com/policies/business-terms",
        note="Ownership of model output and the rules on publishing it are set "
             "by the terms in force for the account holding the key.",
    ),

    # ── sound and stock ─────────────────────────────────────────────────────
    Component(
        "musicgen", "MusicGen weights", "generated music beds (audiocraft)",
        question=MONETISE,
        source="https://github.com/facebookresearch/audiocraft",
        applies=lambda env: "musicgen" in (env.get("RUFUS_MUSIC") or "").lower(),
        note="Audiocraft splits its code licence from its WEIGHTS licence, and "
             "the weights are the half that governs the bed under a monetised "
             "video. An optional extra, so the safe default is simply not "
             "installing it.",
    ),
    Component(
        "jamendo", "Jamendo", "music, first in the chain when a key is set",
        question=MONETISE,
        source="https://developer.jamendo.com/v3.0/docs",
        applies=lambda env: bool((env.get("JAMENDO_CLIENT_ID") or "").strip()),
        note="Jamendo distinguishes personal from commercial use at the "
             "licence level and sells the second separately. A key obtained "
             "for the free tier is not the same permission.",
    ),
    Component(
        "archive-org", "archive.org audio", "music, second in the chain",
        question=MONETISE,
        source="https://archive.org/about/terms.php",
        note="NOT ONE LICENCE BUT THOUSANDS. archive.org hosts items under "
             "everything from public domain to all-rights-reserved, per item, "
             "so 'it came from archive.org' answers nothing on its own. "
             "music_fetcher records which item it took — that record is what "
             "has to be checkable per video.",
    ),
    Component(
        "pexels", "Pexels", "stock footage when RUFUS_VIDEO_SOURCE=pexels",
        question=MONETISE,
        source="https://www.pexels.com/license/",
        applies=lambda env: _source(env) == "pexels",
    ),

    # ── shipped with the product ────────────────────────────────────────────
    Component(
        "anton", "Anton font", "every burned-in caption and thumbnail",
        question=SELL,
        source="https://fonts.google.com/specimen/Anton/about",
        note="Bundled in assets/fonts, so it is redistributed with the "
             "product rather than merely used by it — which is a stricter "
             "test, and usually a satisfiable one that comes with conditions "
             "about carrying the licence file alongside.",
    ),
    Component(
        "ffmpeg", "FFmpeg", "the render itself — every frame and every mux",
        question=SELL,
        source="https://ffmpeg.org/legal.html",
        note="FFMPEG'S LICENCE DEPENDS ON THE BUILD, NOT ON FFMPEG. The same "
             "version is LGPL or GPL according to which encoders it was "
             "compiled with — a build carrying libx264 is GPL. Rufus calls it "
             "as a separate process and does not bundle it, which is the "
             "arrangement that keeps this simple; bundling one in an "
             "installer is what would change the answer. Run "
             "`ffmpeg -version` and read the configuration line.",
    ),
]

# Python distributions whose licence is read from their own installed metadata.
# Measured rather than asserted: this is the one part of the picture the
# machine can answer for itself, and the answer changes with the lockfile.
PACKAGES = [
    "openai", "httpx", "requests", "edge-tts", "faster-whisper", "Pillow",
    "numpy", "opencv-python-headless", "filelock", "flask",
    "google-api-python-client", "google-auth", "google-auth-oauthlib",
    "google-auth-httplib2", "tzdata",
]

# Licence families that are fine to build a proprietary product on without
# further thought, and the ones that are not. Anything outside both lists is
# reported rather than judged.
PERMISSIVE = ("mit", "bsd", "apache", "isc", "python software foundation",
              "unlicense", "public domain", "historical permission",
              "mit-cmu", "zlib")
COPYLEFT = ("gpl", "agpl", "lgpl", "mpl", "epl", "cddl")


def package_licences() -> list[dict]:
    """Every core dependency and the licence its own metadata declares.

    Read at runtime because a requirements floor is not a version: the file
    says `openai>=1.40` and the machine has whatever pip resolved. A table
    written by hand answers for the day it was written.
    """
    import importlib.metadata as md

    out = []
    for name in PACKAGES:
        row = {"package": name, "version": "", "licence": "", "family": UNKNOWN}
        try:
            meta = md.metadata(name)
            row["version"] = md.version(name)
            classifiers = [c for c in (meta.get_all("Classifier") or [])
                           if c.startswith("License ::")]
            if classifiers:
                row["licence"] = classifiers[0].split("::")[-1].strip()
            else:
                # Older metadata puts a free-text licence here, sometimes with
                # the entire licence body in it — take the first line only.
                raw = (meta.get("License") or "").strip()
                row["licence"] = raw.splitlines()[0][:80] if raw else ""
        except Exception:
            row["licence"] = ""
            row["family"] = "not installed"
            out.append(row)
            continue

        low = row["licence"].lower()
        if not low:
            row["family"] = UNKNOWN
        elif any(c in low for c in COPYLEFT):
            row["family"] = "copyleft"
        elif any(p in low for p in PERMISSIVE):
            row["family"] = "permissive"
        else:
            row["family"] = UNKNOWN
        out.append(row)
    return out


def answers() -> dict:
    """Recorded answers from config/licences.json, or {} when nobody has looked.

    Deliberately a separate file from the manifest above. The manifest is code
    — it says what the questions ARE and travels with the release. The answers
    are a fact about a moment: who read which page, when, and what it said.
    Keeping them apart is what lets an answer go stale visibly instead of
    hiding inside a constant somebody edited two versions ago.
    """
    try:
        raw = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        # Same rule as the user store: unreadable is not the same as absent,
        # and must not read as "everything is cleared".
        print(f"[licensing] {ANSWERS_FILE.name} could not be read ({e}) — "
              f"treating every question as unanswered")
        return {}
    return raw.get("components", {}) if isinstance(raw, dict) else {}


def record(key: str, verdict: str, *, source_read: str = "", by: str = "",
           note: str = "") -> None:
    """Write down what a page actually said, and who read it."""
    import datetime
    if verdict not in (YES, NO, CONDITIONAL, UNKNOWN):
        raise ValueError(f"verdict must be one of yes/no/conditional/unknown, "
                         f"not {verdict!r}")
    comp = next((c for c in COMPONENTS if c.key == key), None)
    if comp is None:
        raise ValueError(f"{key!r} is not a component — one of "
                         f"{', '.join(sorted(c.key for c in COMPONENTS))}")
    # PROVENANCE IS NOT THE CALLER'S JOB. Every verdict has to say which page
    # produced it or it cannot be re-checked when the page changes, and a
    # caller that forgets the argument would write an answer indistinguishable
    # from an opinion. The manifest already knows the URL.
    source_read = source_read or comp.source
    try:
        doc = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        doc = {}
    doc.setdefault("components", {})[key] = {
        "verdict": verdict,
        "source_read": source_read,
        "by": by,
        "on": datetime.date.today().isoformat(),
        "note": note,
    }
    ANSWERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS_FILE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def report(env: dict | None = None) -> dict:
    """What is switched on right now, and what is known about each part.

    {"active": [...], "inactive": [...], "packages": [...],
     "blocking": [...], "open": [...]}

    `blocking` is anything recorded as a NO on the question it gates — a
    component that has been looked up and came back forbidding what this
    channel does. `open` is everything nobody has looked up. They are separate
    lists because they need separate actions: one is a decision to change the
    configuration, the other is twenty minutes of reading.
    """
    env = dict(os.environ if env is None else env)
    known = answers()
    active, inactive, blocking, open_qs = [], [], [], []

    for c in COMPONENTS:
        recorded = known.get(c.key) or {}
        verdict = recorded.get("verdict", UNKNOWN)
        row = {
            "key": c.key, "name": c.name, "what": c.what,
            "question": c.question, "source": c.source, "note": c.note,
            "verdict": verdict, "recorded": recorded,
        }
        if not c.applies(env):
            inactive.append(row)
            continue
        active.append(row)
        if verdict == NO:
            blocking.append(row)
        elif verdict == UNKNOWN:
            open_qs.append(row)

    return {"active": active, "inactive": inactive, "blocking": blocking,
            "open": open_qs, "packages": package_licences()}


def _cli() -> int:
    import sys

    args = sys.argv[1:]
    if args and args[0] == "record":
        if len(args) < 3:
            print("usage: licensing.py record <component> "
                  "<yes|no|conditional|unknown> [--by NAME] [--note TEXT]")
            return 2
        by = note = ""
        rest = args[3:]
        for flag, target in (("--by", "by"), ("--note", "note")):
            if flag in rest:
                i = rest.index(flag)
                val = rest[i + 1] if i + 1 < len(rest) else ""
                if target == "by":
                    by = val
                else:
                    note = val
        record(args[1], args[2], by=by, note=note)
        print(f"recorded: {args[1]} → {args[2]}")
        return 0

    r = report()
    print("\nWhat this configuration is made of")
    print("=" * 64)

    copyleft = [p for p in r["packages"] if p["family"] == "copyleft"]
    unclear = [p for p in r["packages"] if p["family"] == UNKNOWN]
    print(f"\n  Dependencies: {len(r['packages'])} read from installed "
          f"metadata")
    for p in copyleft:
        print(f"    ! {p['package']} {p['version']} — {p['licence']}")
        print(f"      copyleft. Fine to USE; obligations attach if you "
              f"redistribute.")
    for p in unclear:
        print(f"    ? {p['package']} {p['version']} — licence not declared in "
              f"metadata")
    if not copyleft and not unclear:
        print("    all permissive")

    print(f"\n  In play for this configuration: {len(r['active'])} component(s)")
    for row in r["active"]:
        mark = {NO: "!", UNKNOWN: "?", YES: "+", CONDITIONAL: "~"}.get(
            row["verdict"], "?")
        which = ("selling copies of Rufus" if row["question"] == SELL
                 else "money from the videos")
        print(f"    {mark} {row['name']} — {row['what']}")
        print(f"      gates: {which}")
        if row["verdict"] == UNKNOWN:
            print(f"      UNANSWERED — read {row['source']}")
        else:
            rec = row["recorded"]
            print(f"      {row['verdict']} (read by {rec.get('by') or '—'} "
                  f"on {rec.get('on') or '—'})")

    if r["blocking"]:
        print(f"\n  BLOCKING: {len(r['blocking'])} component(s) recorded as "
              f"forbidding what this channel does.")
        for row in r["blocking"]:
            print(f"    {row['name']} — {row['recorded'].get('note') or ''}")
    if r["open"]:
        print(f"\n  {len(r['open'])} question(s) nobody has answered yet. "
              f"Record each with:")
        print(f"    python scripts/licensing.py record <component> "
              f"<yes|no|conditional> --by <name> --note \"what the page said\"")
    if not r["blocking"] and not r["open"]:
        print("\n  Every part in play has been checked and cleared.")
    print()
    return 1 if r["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
