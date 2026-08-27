#!/usr/bin/env python3
"""
voice_takes.py — three reads of the hook, so a person picks how it opens.

ONLY THE HOOK, and the reason is what it costs to listen rather than what it
costs to make. Audio is the one thing on this pipeline's choose-from-several
list that cannot be skimmed: it plays at one times speed and there is no
glancing at it. Three full forty-five-second takes is two and a half minutes of
attention; three eight-second hooks is twenty-four seconds. And if the opening
read lands, the rest of the script follows it — the hook is where a Short is
won or lost anyway.

THE VOICE DOES NOT VARY. A channel whose narrator changes every video has no
narrator; that is channel identity, chosen once, and re-rolling it per video
spends the one thing a viewer uses to recognise you. What varies is the TONE
the director assigns beat 0 — the same lever that already sizes that beat's
pauses and grades its picture, so choosing here is choosing something the rest
of the pipeline already understands.

PRONUNCIATION IS NOT ON THIS LIST, and it is worth saying why rather than
leaving it looking forgotten. A word the voice says wrongly — a currency name,
a historical figure — is a fault, not a preference. Re-rolling three whole
takes hoping one lands "Rentenmark" is a lottery ticket where a dictionary
entry belongs. There is no lexicon anywhere in this codebase yet; that is a
separate thing to build, not a variant to choose between.

    RUFUS_VOICE_TAKES  3   how many reads to record
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths

DEFAULT_TAKES = 3

# THE DIRECTOR'S OWN CHOICE FIRST. A set that never offers what the pipeline
# would have done on its own turns a choice into a forced change — and the
# director reads the actual beat, which a fixed list cannot. The rest are the
# two openings that most differ from each other, so the set spans the range
# instead of sampling near one point.
_CONTRAST = ("curiosity", "tension", "revelation", "weight")


def how_many() -> int:
    try:
        return max(1, int(os.environ.get("RUFUS_VOICE_TAKES", DEFAULT_TAKES)))
    except ValueError:
        return DEFAULT_TAKES


def takes_dir(set_id: int) -> Path:
    return paths.media_root() / "voice_takes" / str(set_id)


def hook_of(script: str) -> str:
    """The opening line — the eight seconds this stage is about."""
    for line in (script or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def tones_for(script: str, n: int) -> list[str]:
    """The tones to read the hook in: the director's own, then contrasts.

    Fail-open to the plain list if the director cannot be reached — a set of
    three reads is still a choice, it just no longer leads with the one the
    pipeline would have picked by itself.
    """
    chosen: list[str] = []
    try:
        import edit_director
        import emotional_map
        import main as rufus_main
        beats = rufus_main._split_beats(script, max_scenes=n or 1, grow=False)
        plan = edit_director.direct(beats) if beats else None
        tones = emotional_map.tones_from_plan(plan, len(beats or [1]))
        if tones:
            chosen.append(emotional_map.normalise(tones[0]))
    except Exception as e:
        print(f"[takes] the director did not answer ({e}) — using the "
              f"contrast list alone")
    for t in _CONTRAST:
        if len(chosen) >= n:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen[:n]


def build(script_file: str, *, set_id: int, channel: str = "main_en",
          topic: str = "", n: int | None = None) -> list[dict]:
    """Record the hook once per tone. Returns the rows saved.

    Fail-open per take, like every other loop here: a tone the backend chokes
    on leaves two reads to choose between rather than none.
    """
    import db_manager
    import tts_engine

    script = Path(script_file).read_text(encoding="utf-8")
    hook = hook_of(script)
    if not hook:
        print("[takes] the script has no opening line to read")
        return []

    n = n or how_many()
    try:
        import emotional_map
        print(f"[takes] this backend varies {emotional_map.speaks_tone()}")
    except Exception:
        pass
    out_dir = takes_dir(set_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []

    for tone in tones_for(script, n):
        mp3 = out_dir / f"{tone}.mp3"
        try:
            # One beat, one tone: synthesize's per-beat path takes a tone list
            # and the text those tones describe, which for a single line is
            # exactly this pair. Going through it rather than around it means
            # the take a person hears is produced the same way the render will
            # produce it.
            tts_engine.synthesize(hook, mp3, [tone], [hook])
        except Exception as e:
            print(f"[takes] {tone}: no audio ({e})")
            continue
        if not mp3.exists() or mp3.stat().st_size < 1_000:
            print(f"[takes] {tone}: the file came back empty")
            continue
        row_id = db_manager.save_voice_take(
            set_id=set_id, channel=channel, topic=topic, tone=tone,
            text=hook, path=str(mp3))
        saved.append({"id": row_id, "tone": tone, "path": str(mp3)})
        print(f"[takes] #{row_id} {tone} — {mp3.name}")

    print(f"[takes] {len(saved)} read(s) of the hook — choose at /voice")
    return saved


if __name__ == "__main__":
    # THE SCHEMA, BEFORE ANYTHING TRIES TO WRITE TO IT. The dashboard calls
    # init_db at startup and every test fixture calls it too, so every path
    # that had ever been exercised already had the tables — and the one path
    # nobody had run, the command line, died on "no such table" after paying
    # for a script. Built, tested, and never actually run, which is this
    # repo's oldest bug wearing a new hat.
    import argparse
    import db_manager
    db_manager.init_db()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("script_file")
    ap.add_argument("--set", type=int, required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    build(a.script_file, set_id=a.set, topic=a.topic, n=a.n)
