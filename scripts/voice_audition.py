#!/usr/bin/env python3
"""
voice_audition.py — the same line in every voice, so the narrator is chosen once.

WHY THIS IS NOT PER-VIDEO. /voice varies the TONE of one video's opening. This
varies the narrator, and a channel whose narrator changes every video has no
narrator — the voice is the single thing a returning viewer recognises before
they have read a word. So it is chosen deliberately, rarely, against one fixed
line, and then left alone. This page is the audio twin of /styles, which asks
the same question about the look for the same reason.

ONE LINE, EVERY VOICE, SAME WORDS. A voice compared against a different
sentence than the one before it is not being compared at all — half of what you
hear is the writing. The sample is fixed and short for the same reason /styles
renders one fixed scene through many looks.

WHAT IS ACTUALLY FREE AND LOCAL, since that is the question that leads people
here and the answer is narrower than the list of backends suggests:

    kokoro    LOCAL, free, Apache 2.0 — runs on CPU, safe to monetise.
    edge      free and good, but it is a Microsoft NETWORK service, not local,
              and its terms are written for a browser's read-aloud feature
              rather than for publishing.
    xtts      local and free to run, but Coqui XTTS v2 ships under CPML, which
              is NON-COMMERCIAL. Fine to experiment with, not for a channel
              that earns.
    elevenlabs cloud and paid.

Which leaves Kokoro as the one that is local, free AND commercially clear — so
the real question is which Kokoro voice, and that is what this exists to answer.

    python scripts/voice_audition.py
    python scripts/voice_audition.py --backend edge
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths

# The line every voice reads. Second person, one concrete noun, a full stop in
# the middle — long enough to hear a rhythm, short enough that comparing eight
# of them is a minute rather than ten.
SAMPLE = "You checked your balance today. In 1923 Berlin, that took a wheelbarrow."

# LABELS, NOT JUST IDS. "am_adam" tells you nothing you can choose between, and
# a page of eight of those is a page nobody uses twice.
KOKORO_VOICES = [
    ("am_adam",    "Adam — deep American male, the current default"),
    ("am_michael", "Michael — American male, lighter and quicker"),
    ("am_onyx",    "Onyx — American male, darker and slower"),
    ("af_heart",   "Heart — American female, warm"),
    ("af_bella",   "Bella — American female, brighter"),
    ("af_sky",     "Sky — American female, younger"),
    ("bm_george",  "George — British male, measured"),
    ("bf_emma",    "Emma — British female, crisp"),
]

EDGE_VOICES = [
    ("en-US-AndrewMultilingualNeural", "Andrew — American male, the default"),
    ("en-US-ChristopherNeural",        "Christopher — American male, deeper"),
    ("en-US-GuyNeural",                "Guy — American male, newsreader"),
    ("en-US-AriaNeural",               "Aria — American female"),
    ("en-GB-RyanNeural",               "Ryan — British male"),
    ("en-GB-SoniaNeural",              "Sonia — British female"),
]

BACKENDS = {"kokoro": KOKORO_VOICES, "edge": EDGE_VOICES}

# Which environment variable each backend reads its voice from. Kept here
# rather than at the call site so the dashboard can save the choice without
# knowing anything about tts_engine's internals.
VOICE_VAR = {"kokoro": "RUFUS_KOKORO_VOICE", "edge": "RUFUS_EDGE_VOICE"}


def audition_dir() -> Path:
    return paths.media_root() / "voice_audition"


def catalogue(backend: str) -> list[tuple[str, str]]:
    return BACKENDS.get(backend, [])


class _pinned_voice:
    """The backend and voice for one sample, then the caller's own back.

    Through the environment because that is where tts_engine reads both, and
    threading two more arguments down four call layers to reach the same two
    variables would be the same decision written twice.
    """

    def __init__(self, backend: str, voice: str):
        self.backend = backend
        self.voice = voice
        self.before: dict = {}

    def __enter__(self):
        var = VOICE_VAR.get(self.backend, "RUFUS_KOKORO_VOICE")
        for key, val in (("RUFUS_TTS", self.backend), (var, self.voice)):
            self.before[key] = os.environ.get(key)
            os.environ[key] = val
        # THE ENVIRONMENT IS NOT ENOUGH. tts_engine reads its voice into a
        # module constant at import time, so a process that auditions eight
        # voices in one run would record eight files of whichever voice was set
        # when it started. The constant has to be moved too — and moved back,
        # because this same process renders videos.
        #
        # The Kokoro pipeline itself is voice-agnostic: _kokoro passes
        # KOKORO_VOICE at call time, so the loaded model is reused across the
        # whole sheet rather than reloaded eight times.
        self._tts = None
        try:
            import tts_engine
            self._tts = tts_engine
            self._old = (tts_engine.KOKORO_VOICE, tts_engine.EDGE_VOICE)
            if self.backend == "kokoro":
                tts_engine.KOKORO_VOICE = self.voice
            else:
                tts_engine.EDGE_VOICE = self.voice
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        for key, val in self.before.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        if self._tts is not None:
            self._tts.KOKORO_VOICE, self._tts.EDGE_VOICE = self._old
        return False


def build(backend: str = "kokoro", sample: str = "",
          only: list[str] | None = None) -> list[dict]:
    """Read `sample` once per voice. Returns [{voice, label, path}].

    Fail-open per voice: one that this install cannot produce — a Kokoro voice
    file that was never downloaded, an Edge name that has been retired — leaves
    the rest of the sheet intact. A missing row is a voice you cannot pick,
    which is the truth about it.
    """
    import tts_engine

    voices = catalogue(backend)
    if not voices:
        print(f"[audition] no catalogue for backend {backend!r} — "
              f"try {' or '.join(BACKENDS)}")
        return []
    if only:
        voices = [(v, lbl) for v, lbl in voices if v in set(only)]

    text = (sample or SAMPLE).strip()
    out_dir = audition_dir() / backend
    out_dir.mkdir(parents=True, exist_ok=True)

    done: list[dict] = []
    for voice, label in voices:
        mp3 = out_dir / f"{voice}.mp3"
        try:
            with _pinned_voice(backend, voice):
                tts_engine.synthesize(text, mp3)
        except Exception as e:
            print(f"[audition] {voice}: no audio ({e})")
            continue
        if not mp3.exists() or mp3.stat().st_size < 1_000:
            print(f"[audition] {voice}: the file came back empty")
            continue
        done.append({"voice": voice, "label": label, "path": str(mp3),
                     "backend": backend})
        print(f"[audition] {voice} — {label}")

    print(f"[audition] {len(done)}/{len(voices)} voice(s) recorded in "
          f"{out_dir}")
    return done


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--backend", default="kokoro", choices=sorted(BACKENDS))
    ap.add_argument("--sample", default="")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    build(a.backend, a.sample, a.only)
