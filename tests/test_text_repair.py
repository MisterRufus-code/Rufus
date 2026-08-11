"""Text that was decoded with the wrong code page must never reach the voice.

The reported symptom: a finished English short whose narration audibly read a
Hebrew letter. The CTA in config/niches.json is

    "Save this — money was never what you think."

and the run printed

    [gpt] cta: Save this ╫עΓג¼Γא¥ money was never what you think.

That is the em-dash's three UTF-8 bytes decoded as cp1255 ("ג€”"), rendered
again through the console's OEM page. The text was already corrupt in memory,
so it went into the script, into the TTS call, and out of the speaker.

The read-side fix (utf-8 everywhere, PYTHONUTF8=1) stops new corruption. These
tests cover what that fix cannot reach: text already saved in rufus.db, and any
future mis-decode arriving from somewhere new.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import text_repair


# --------------------------------------------------------------- the reported bug

def test_the_reported_cta_is_repaired_exactly():
    original  = "Save this — money was never what you think."
    corrupted = original.encode("utf-8").decode("cp1255")

    assert corrupted != original                      # the bug reproduces
    assert text_repair.repair_mojibake(corrupted) == original


def test_corruption_is_detected_as_corruption():
    corrupted = "Save this — money was never what you think.".encode("utf-8").decode("cp1255")
    assert text_repair.looks_corrupted(corrupted)


def test_the_hebrew_letter_never_survives_to_the_voice():
    corrupted = "Save this — money was never what you think.".encode("utf-8").decode("cp1255")
    spoken = text_repair.clean_for_speech(corrupted)
    assert not text_repair.foreign_characters(spoken), (
        "a Hebrew letter reaching the TTS backend is read out loud in the video"
    )


@pytest.mark.parametrize("codepage", ["cp1255", "cp1252"])
def test_both_windows_locales_round_trip(codepage):
    original = "assets and liabilities — debits equalling credits."
    assert text_repair.repair_mojibake(
        original.encode("utf-8").decode(codepage)) == original


# ------------------------------------------------------------------ safety

def test_clean_text_is_untouched():
    for good in ("Save this for your next 4am.",
                 "In 1494, Luca Pacioli published a book in Venice.",
                 "assets and liabilities — debits equalling credits",
                 "café, £50, “smart quotes”"):
        assert text_repair.repair_mojibake(good) == good


def test_genuine_hebrew_is_not_mangled():
    """The round trip validates itself: real Hebrew is not valid UTF-8 once
    encoded to cp1255, so the decode raises and the original survives."""
    hebrew = "שלום עולם"
    assert text_repair.repair_mojibake(hebrew) == hebrew


def test_a_non_english_channel_keeps_its_script():
    """channels.json supports non-English channels. Majority-foreign text is
    the content, not debris — stripping it would silence the video."""
    hebrew_script = "שלום עולם, זהו סרטון על היסטוריה של הכסף בעולם המודרני."
    assert text_repair.clean_for_speech(hebrew_script) == hebrew_script


def test_repair_is_idempotent():
    corrupted = "Save this — money was never what you think.".encode("utf-8").decode("cp1255")
    once  = text_repair.repair_mojibake(corrupted)
    twice = text_repair.repair_mojibake(once)
    assert once == twice


def test_empty_and_plain_ascii_are_safe():
    assert text_repair.repair_mojibake("") == ""
    assert text_repair.clean_for_speech("") == ""
    assert text_repair.repair_mojibake("plain ascii") == "plain ascii"


# ------------------------------------------------------- wired into the pipeline

def test_tts_sanitizer_runs_the_repair():
    """_sanitize_for_speech is the single choke point every backend passes
    through — Kokoro, Edge, ElevenLabs and XTTS all receive its output."""
    import tts_engine

    corrupted = "Save this — money was never what you think.".encode("utf-8").decode("cp1255")
    out = tts_engine._sanitize_for_speech(corrupted)
    assert "—" in out
    assert not text_repair.foreign_characters(out)


def test_tts_sanitizer_still_strips_markdown():
    """The repair is added in front of the existing behaviour, not instead."""
    import tts_engine

    assert tts_engine._sanitize_for_speech("**bold** and [pause] gone") == "bold and gone"


def test_metadata_description_carries_a_clean_cta(monkeypatch):
    """The CTA also ships in the YouTube description and the pinned comment."""
    import metadata_writer

    monkeypatch.setattr(metadata_writer, "_load_key", lambda: "")   # force legacy path
    corrupted = "Save this — money was never what you think.".encode("utf-8").decode("cp1255")

    meta = metadata_writer.generate_metadata(
        "A script line.", "money_history", {"cta": corrupted}, ["#Shorts"])
    assert "Save this — money was never what you think." in meta["description"]


def test_metadata_repair_does_not_mutate_the_caller_config():
    """niche_cfg is shared config — generate_metadata must not write into it."""
    import metadata_writer

    cfg = {"cta": "Save this — money was never what you think."}
    metadata_writer.generate_metadata("A line.", "money_history", cfg, ["#Shorts"])
    assert cfg == {"cta": "Save this — money was never what you think."}


# ------------------------------------------------------------ database repair

def test_database_repair_finds_and_fixes_old_rows(tmp_path, monkeypatch):
    """Rows written before the encoding fix keep the corruption forever."""
    import db_manager

    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "rufus.db")
    db_manager.init_db()

    original  = "Accounting owes its survival to Luca's simple truths — really."
    corrupted = original.encode("utf-8").decode("cp1255")
    with db_manager._conn() as c:
        c.execute("INSERT INTO videos (niche, script_full) VALUES (?, ?)",
                  ("money_history", corrupted))
        c.execute("INSERT INTO videos (niche, script_full) VALUES (?, ?)",
                  ("money_history", "already clean"))

    found = text_repair.repair_database(apply=False)
    assert [(f[1], f[3]) for f in found] == [("script_full", original)]

    with db_manager._conn() as c:      # dry run wrote nothing
        rows = [r[0] for r in c.execute("SELECT script_full FROM videos")]
    assert corrupted in rows

    text_repair.repair_database(apply=True)
    with db_manager._conn() as c:
        rows = [r[0] for r in c.execute("SELECT script_full FROM videos")]
    assert rows == [original, "already clean"]
    assert text_repair.repair_database(apply=False) == []
