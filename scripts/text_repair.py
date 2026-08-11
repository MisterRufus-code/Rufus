#!/usr/bin/env python3
"""
text_repair.py
Last line of defence against text that was decoded with the wrong code page.

The root cause is fixed elsewhere: every read_text/write_text in scripts/ now
states utf-8, and the launchers set PYTHONUTF8=1, so a fresh run no longer
mis-decodes a config file. This module exists for the two things that fix
cannot reach:

  1. Text already at rest. Rows written to rufus.db before the encoding fix
     hold the corrupted form permanently — re-reading the config correctly
     does not repair a string that was saved wrong months ago.

  2. Any future mis-decode. The failure mode that let this run for months is
     that corrupted text looks like a console rendering problem, so nothing
     ever objected. A Hebrew letter reached the TTS backend and Kokoro read it
     out loud in an English video; every gate said pass.

So: repair what is recognisably repairable, and make anything left over say so
in the log rather than quietly reaching a viewer's ears.

How the repair works. UTF-8 bytes decoded as a single-byte code page produce a
fixed, reversible signature — an em-dash is three bytes, so it becomes three
characters:

    "—".encode("utf-8")            == b"\\xe2\\x80\\x94"
    b"\\xe2\\x80\\x94".decode("cp1255") == "ג€”"      # the owner's Hebrew locale
    b"\\xe2\\x80\\x94".decode("cp1252") == "â€”"      # western locale, same bug

Re-encoding to that code page and decoding as UTF-8 undoes it exactly. The
round trip validates itself: genuine Hebrew ("שלום") encodes to cp1255 bytes
that are not valid UTF-8, so the decode raises and the original is kept. That
is what makes this safe to run over text that is *supposed* to be Hebrew.
"""

from __future__ import annotations

import re
import unicodedata

# Code pages worth trying, in order. cp1255 is the owner's box; cp1252 is the
# same bug on a western Windows install and costs nothing to cover.
CODEPAGES = ("cp1255", "cp1252")

# Scripts an English-voice channel should never contain. Deliberately not a
# blanket "non-ASCII" test — a script may legitimately hold é, £, — or a smart
# quote, and stripping those would damage good text to catch a rare bad case.
_FOREIGN_BLOCKS = (
    (0x0400, 0x04FF),   # Cyrillic
    (0x0530, 0x058F),   # Armenian
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E7F),   # Thai
    (0x3040, 0x30FF),   # Kana
    (0x4E00, 0x9FFF),   # CJK
    (0xAC00, 0xD7AF),   # Hangul
)

# Characters that only turn up as mis-decode debris in Latin text. Used to
# decide whether a repair is even worth attempting, so clean text is never
# round-tripped on a hunch.
_MOJIBAKE_MARKERS = frozenset("ÂÃâ€šž¬")

_MAX_PASSES = 3     # doubly-encoded text exists; three unwinds is beyond ample


def _is_foreign(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _FOREIGN_BLOCKS)


def _damage(text: str) -> int:
    """How much of `text` looks like mis-decode debris.

    Counts foreign-script letters plus the marker characters. The repair is
    accepted only when this number strictly drops, which is what stops a
    'repair' from mangling text that was fine.
    """
    return sum(1 for ch in text if _is_foreign(ch) or ch in _MOJIBAKE_MARKERS)


def looks_corrupted(text: str) -> bool:
    """True if `text` carries the signature of a wrong-code-page decode."""
    return bool(text) and _damage(text) > 0


def repair_mojibake(text: str) -> str:
    """Undo a UTF-8-decoded-as-code-page round trip, if that is what happened.

    Returns `text` unchanged when it is clean, when it is genuinely non-Latin,
    or when no candidate decode strictly reduces the damage. Idempotent.
    """
    if not text or not looks_corrupted(text):
        return text

    current = text
    for _ in range(_MAX_PASSES):
        best = current
        for cp in CODEPAGES:
            try:
                candidate = current.encode(cp).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _damage(candidate) < _damage(best):
                best = candidate
        if best == current:
            break
        current = best
    return current


def foreign_characters(text: str) -> list[str]:
    """The distinct foreign-script characters in `text`, in order of appearance."""
    seen: list[str] = []
    for ch in text:
        if _is_foreign(ch) and ch not in seen:
            seen.append(ch)
    return seen


def describe(chars: list[str]) -> str:
    """Name characters for a log line — 'ג' alone in a console is unreadable."""
    return ", ".join(f"{ch!r} ({unicodedata.name(ch, 'U+%04X' % ord(ch))})"
                     for ch in chars)


# Above this share of foreign letters the text is not debris, it is the
# content — a non-English channel narrating in its own script. Stripping there
# would silence the video instead of saving it.
_DEBRIS_SHARE_MAX = 0.10


def clean_for_speech(text: str, label: str = "text") -> str:
    """Repair `text`, then strip foreign debris that survived, loudly.

    A TTS voice pronounces whatever it is handed. Kokoro reading a Hebrew
    letter in an English short is not a cosmetic defect — it ships in the
    audio, where no later gate looks. Repair first (the common case is a
    recoverable em-dash), strip second, and name the dropped characters so the
    cause shows up in the log rather than only in the finished video.

    Stripping applies only when the foreign characters are a small minority of
    the letters. `channels.json` supports non-English channels, and a script
    that is *meant* to be Hebrew or Japanese is majority-foreign by definition
    — this must repair that text, never gut it.
    """
    repaired = repair_mojibake(text)
    if repaired != text:
        print(f"[text] repaired mis-decoded {label} before speech "
              f"(wrong code page upstream — check PYTHONUTF8=1)")

    leftover = foreign_characters(repaired)
    if not leftover:
        return repaired

    letters = sum(1 for ch in repaired if ch.isalpha())
    foreign = sum(1 for ch in repaired if _is_foreign(ch))
    if letters and foreign / letters > _DEBRIS_SHARE_MAX:
        return repaired      # a non-English script, not debris — leave it alone

    print(f"[text] ⚠ dropped {len(leftover)} foreign character(s) from {label} "
          f"so the voice cannot read them aloud: {describe(leftover)}")
    stripped = "".join(" " if _is_foreign(ch) else ch for ch in repaired)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


# ---------------------------------------------------------------- db repair

# Columns holding text a human or a viewer eventually sees.
_TEXT_COLUMNS = ("script_hook", "script_full", "scene_desc", "seed_content")


def repair_database(apply: bool = False) -> list[tuple[int, str, str, str]]:
    """Find (and optionally fix) rows in rufus.db corrupted before the encoding fix.

    Returns the list of (video_id, column, before, after) that need repair.
    Dry run by default — this rewrites the record of every video ever made, so
    it shows its work before touching anything.
    """
    import db_manager

    findings: list[tuple[int, str, str, str]] = []
    cols = ", ".join(_TEXT_COLUMNS)
    with db_manager._conn() as c:
        existing = {r[1] for r in c.execute("PRAGMA table_info(videos)")}
        usable = [col for col in _TEXT_COLUMNS if col in existing]
        if not usable:
            return findings
        cols = ", ".join(usable)
        for row in c.execute(f"SELECT id, {cols} FROM videos").fetchall():
            vid = row[0]
            for col, value in zip(usable, row[1:]):
                if not isinstance(value, str) or not looks_corrupted(value):
                    continue
                fixed = repair_mojibake(value)
                if fixed != value:
                    findings.append((vid, col, value, fixed))

        if apply:
            for vid, col, _before, after in findings:
                c.execute(f"UPDATE videos SET {col} = ? WHERE id = ?", (after, vid))

    return findings


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the repairs (default: show them only)")
    args = ap.parse_args()

    findings = repair_database(apply=args.apply)
    if not findings:
        print("[text] no corrupted rows in rufus.db")
        return

    for vid, col, before, after in findings:
        print(f"  id={vid} {col}")
        print(f"    before: {before[:90]}")
        print(f"    after:  {after[:90]}")
    verb = "repaired" if args.apply else "would repair"
    print(f"[text] {verb} {len(findings)} field(s) across "
          f"{len({f[0] for f in findings})} video(s)")
    if not args.apply:
        print("[text] re-run with --apply to write them")


if __name__ == "__main__":
    _main()
