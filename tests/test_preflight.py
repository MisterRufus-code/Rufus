"""Refuse to start a run that cannot possibly finish, and say why.

WHAT A FRESH INSTALL ACTUALLY DID, measured on a clean checkout with no
config/keys.json — the state every new machine is in. `python scripts/main.py`
fetched a dozen RSS feeds, queried Hacker News twice, tried Reddit, OpenAlex,
StackExchange, the Library of Congress and eight Wikipedia articles, put four
seeds through the supervisor, printed "Pre-analysis failed (non-fatal)" about
the very file it was about to die on, and stopped at step 4 with:

    ✗ Step 4 failed: [Errno 2] No such file or directory: '…/config/keys.json'

Minutes spent on work that could not be used, ending in a Python errno for a
file the person had never heard of.

The principle is already in this tree one level down — main.py runs the fact
gate before burning FLUX time on doomed images. These tests hold the same line
for the run itself, and hold the harder half too: a preflight that fires on
things a run could survive is one people learn to run past, so every check has
to be conditional on the configuration in front of it.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import preflight  # noqa: E402


@pytest.fixture
def keys(tmp_path, monkeypatch):
    """A config dir of our own, so these never read the real machine's keys."""
    monkeypatch.setattr(preflight, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(preflight, "KEYS_FILE", tmp_path / "keys.json")

    def write(doc):
        (tmp_path / "keys.json").write_text(
            doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return write


@pytest.fixture(autouse=True)
def ffmpeg_present(monkeypatch):
    """ffmpeg is checked separately below; elsewhere it must not be the reason
    a test passes or fails."""
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/" + name)


def _keys_of(found):
    return [b.what for b in found]


def test_a_fresh_install_is_stopped_before_it_spends_anything(keys):
    """The whole point. Nothing about the missing file required a single
    network call to discover."""
    found = preflight.blockers({}, probe_network=False)
    assert any("keys.json does not exist" in w for w in _keys_of(found))


def test_every_blocker_says_what_why_and_how_to_fix_it():
    """A message that stops at "what" is the errno again in a nicer font. The
    person reading it has to be able to act without opening the source."""
    found = preflight.blockers({}, probe_network=False)
    assert found
    for b in found:
        assert b.what and b.why and b.fix, b
        assert len(b.fix) > 25, f"{b.what}: the fix has to be actionable"


def test_a_broken_file_is_not_reported_as_a_missing_one(keys):
    """Different states, different fixes. On a hand-edited file a trailing
    comma is the usual cause, and "create this file" is unhelpful advice for a
    file that is sitting right there."""
    keys("{ \"openai\": \"sk-x\",, }")
    found = _keys_of(preflight.blockers({}, probe_network=False))
    assert any("not valid JSON" in w for w in found)
    assert not any("does not exist" in w for w in found)


def test_the_template_placeholder_is_not_a_key(keys):
    """It fails as a 401 partway through step 4 otherwise, which is the same
    class of bug one layer along."""
    keys({"openai": "YOUR_OPENAI_KEY_HERE"})
    assert any("no OpenAI key" in w
               for w in _keys_of(preflight.blockers({}, probe_network=False)))


def test_a_real_key_clears_it(keys):
    keys({"openai": "sk-realish-value"})
    assert not preflight.blockers({}, probe_network=False)


def test_a_local_model_needs_no_key_at_all(keys):
    """Demanding one would mean keeping a fake OpenAI key in config just to run
    offline — llm.py already refuses to require that, and the preflight must
    not reintroduce it one layer up."""
    assert not preflight.blockers({"RUFUS_LLM_BASE_URL": "http://localhost:1234"},
                                  probe_network=False)


def test_a_key_in_the_environment_counts(keys):
    assert not preflight.blockers({"OPENAI_API_KEY": "sk-env"},
                                  probe_network=False)


def test_a_missing_encoder_is_caught_before_the_video_is_made(keys,
                                                              monkeypatch):
    """Everything upstream of the render can succeed and still leave you with
    nothing: research, script, thirty-eight pictures and a voiceover, and then
    no ffmpeg to assemble them."""
    keys({"openai": "sk-x"})
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    found = _keys_of(preflight.blockers({}, probe_network=False))
    assert any("ffmpeg" in w for w in found)


# ── conditional on the configuration, which is what keeps it credible ───────

def test_pexels_matters_only_when_pexels_is_the_source(keys):
    keys({"openai": "sk-x"})
    off = preflight.blockers({}, probe_network=False)
    on = preflight.blockers({"RUFUS_VIDEO_SOURCE": "pexels"},
                            probe_network=False)
    assert not off
    assert any("Pexels" in b.what for b in on)


def test_comfyui_is_probed_only_when_comfyui_is_drawing(keys, monkeypatch):
    """A run started against a dead renderer gets an empty image back for every
    prompt — which is how a gallery once stopped at 9 of 38 and reported itself
    finished."""
    keys({"openai": "sk-x"})
    monkeypatch.setattr(preflight, "_comfy_up", lambda url, timeout=3.0: False)
    assert not preflight.blockers({}, probe_network=True)
    found = preflight.blockers({"RUFUS_VIDEO_SOURCE": "comfy"},
                               probe_network=True)
    assert any("ComfyUI" in b.what for b in found)


def test_a_live_renderer_is_not_a_blocker(keys, monkeypatch):
    keys({"openai": "sk-x"})
    monkeypatch.setattr(preflight, "_comfy_up", lambda url, timeout=3.0: True)
    assert not preflight.blockers({"RUFUS_VIDEO_SOURCE": "comfy"},
                                  probe_network=True)


def test_youtube_credentials_matter_only_when_this_run_would_publish(keys):
    """The review queue means most runs never upload. Firing on every one of
    them is how a preflight teaches people to ignore it."""
    keys({"openai": "sk-x"})
    queued = preflight.blockers({}, skip_upload=False, probe_network=False)
    auto = preflight.blockers({"RUFUS_AUTO_UPLOAD": "1"}, skip_upload=False,
                              probe_network=False)
    skipped = preflight.blockers({"RUFUS_AUTO_UPLOAD": "1"}, skip_upload=True,
                                 probe_network=False)
    assert not queued
    assert not skipped
    assert any("YouTube" in b.what for b in auto)


def test_nothing_fires_on_something_a_run_could_survive(keys, monkeypatch):
    """The line between this and health_check.py. A missing Reddit key costs
    you two seed sources out of nine; a missing OpenAI key costs you the video.
    Only the second belongs here."""
    keys({"openai": "sk-x"})
    monkeypatch.setattr(preflight, "_comfy_up", lambda url, timeout=3.0: True)
    assert preflight.blockers({"RUFUS_VIDEO_SOURCE": "comfy"},
                              probe_network=True) == []


def test_check_or_exit_stops_the_process_rather_than_returning(keys):
    with pytest.raises(SystemExit) as e:
        preflight.check_or_exit({}, probe_network=False)
    assert e.value.code == 2


def test_the_run_asks_before_the_first_byte_goes_out():
    """Placement is the feature. Called after the research had already
    happened, this would be a nicer-looking way to waste the same minutes."""
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text(
        encoding="utf-8")
    body = src[src.index("def run(skip_upload"):]
    assert "preflight.check_or_exit" in body[:2000], (
        "the preflight has to be the first thing run() does")
    assert body.index("preflight.check_or_exit") < body.index("_acquire_lock")


def test_the_writer_explains_a_missing_keys_file_rather_than_raising_an_errno(
        monkeypatch, tmp_path):
    """preflight stops a full run before this point, but anything importing
    script_writer directly still lands here, and FileNotFoundError is not an
    instruction."""
    import script_writer
    monkeypatch.setattr(script_writer, "KEYS_FILE", tmp_path / "nope.json")
    with pytest.raises(ValueError) as e:
        script_writer._load_key()
    assert "keys.json.template" in str(e.value)
