"""Every variable this pipeline reads, found by reading the code.

THE REPORT, in the owner's own account: seven `$env:` lines before every run,
one of them wrong in a way nothing surfaced until the video came out wrong.
RUFUS_STILS_ONLY sets cleanly, errors nowhere, and the run behaves as though
it were never typed — which it effectively was not.

124 of these exist and 49 were written down nowhere. A hand-maintained list of
124 things is wrong within a month, and the proof is that the ones missing from
the previous attempt are exactly the ones added most recently. So this is
generated, and the generator is what these tests are about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import env_doctor  # noqa: E402


def test_it_finds_the_variables_that_actually_matter():
    """Spot-check against ones known to be read, across several modules."""
    found = env_doctor.scan()
    for name in ("RUFUS_STILLS_ONLY", "RUFUS_BEAT_MOTION", "SD_CLIPS",
                 "RUFUS_INSERTS", "RUFUS_DISCORD_WEBHOOK", "RUFUS_STYLE",
                 "RUFUS_BUBBLE_GAIN", "RUFUS_SEED_TRIES"):
        assert name in found, name


def test_it_records_where_each_one_is_read():
    found = env_doctor.scan()
    assert "audio_gen.py" in found["RUFUS_BUBBLE_GAIN"]["files"]
    assert "comfy_client.py" in found["RUFUS_FRAMES_PER_BEAT"]["files"]


def test_it_records_the_default_the_code_falls_back_to():
    """"What happens if I don't set it" is the question people actually have."""
    assert env_doctor.scan()["RUFUS_SFX"]["default"] == "1"
    assert env_doctor.scan()["RUFUS_FRAMES_PER_BEAT"]["default"] == "1"


def test_it_claims_nothing_outside_this_pipeline():
    """PATH and HOME are not ours to document."""
    for name in env_doctor.scan():
        assert name.startswith("RUFUS_") or name in ("SD_CLIPS", "RENDER_TIMEOUT")


def test_it_does_not_count_its_own_examples():
    """This module names variables in its docstring and its regexes; counting
    those would make the reference describe the tool rather than the pipeline."""
    found = env_doctor.scan()
    assert "RUFUS_STILS_ONLY" not in found, "the typo from its own comment"


# ── the part that saves a run ───────────────────────────────────────────────

def test_a_mistyped_variable_is_reported():
    """The whole point. Nothing errors on RUFUS_STILS_ONLY; the run simply
    ignores it, and twenty-five minutes later the video is wrong."""
    stray = env_doctor.unread({"RUFUS_STILS_ONLY": "1",
                               "RUFUS_BEAT_MOTON": "cut"})
    assert stray == ["RUFUS_BEAT_MOTON", "RUFUS_STILS_ONLY"]


def test_a_correct_variable_is_not_reported():
    assert env_doctor.unread({"RUFUS_STILLS_ONLY": "1",
                              "RUFUS_BEAT_MOTION": "cut"}) == []


def test_other_peoples_variables_are_left_alone():
    assert env_doctor.unread({"PATH": "/usr/bin", "HOME": "/root",
                              "PYTHONUTF8": "1"}) == []


def test_the_run_checks_at_startup_before_spending_any_gpu():
    src = (Path(env_doctor.__file__).parent / "main.py").read_text(encoding="utf-8")
    entry = src.split('if __name__ == "__main__":')[1]
    assert "env_doctor.unread()" in entry
    assert entry.index("env_doctor.unread()") < entry.index("argparse.ArgumentParser")


def test_the_check_is_never_fatal():
    """A broken doctor must not stop a healthy run."""
    src = (Path(env_doctor.__file__).parent / "main.py").read_text(encoding="utf-8")
    block = src.split("import env_doctor")[1][:500]
    assert "except Exception" in block
    assert "non-fatal" in block


# ── the generated reference ─────────────────────────────────────────────────

def test_the_reference_is_generated_not_written():
    doc = Path(env_doctor.__file__).parent.parent / "docs" / "ENVIRONMENT.md"
    assert doc.is_file(), "run: python scripts/env_doctor.py --markdown"
    text = doc.read_text(encoding="utf-8")
    assert "Do not edit by hand" in text
    assert "--markdown" in text


def test_the_reference_is_current():
    """Committed generated output goes stale silently, which is the failure it
    was meant to prevent."""
    doc = Path(env_doctor.__file__).parent.parent / "docs" / "ENVIRONMENT.md"
    for name in env_doctor.scan():
        assert f"`{name}`" in doc.read_text(encoding="utf-8"), name
