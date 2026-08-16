#!/usr/bin/env python3
"""
tts_engine.py — Pluggable text-to-speech for Rufus.

Four backends, selected via the RUFUS_TTS environment variable:

  RUFUS_TTS=edge        (default) — Microsoft Edge TTS. Free, fast, cloud, no
                                    GPU. Reliable but reads slightly flat.
  RUFUS_TTS=kokoro                — Kokoro-82M (Apache 2.0). Free, runs locally
                                    on CPU in real time. Voice quality between
                                    Edge and ElevenLabs — the best free local
                                    voice for narration with no GPU needed.
                                    Install: pip install kokoro soundfile
  RUFUS_TTS=xtts                  — Coqui XTTS v2. Free, local GPU (~3GB VRAM),
                                    near-ElevenLabs quality, voice cloning.
  RUFUS_TTS=elevenlabs            — ElevenLabs cloud. Most natural, ~$0.10/video.
                                    Needs "elevenlabs" key in config/keys.json.

Quality ranking: elevenlabs > kokoro > xtts > edge
Ease ranking:    edge > kokoro > elevenlabs > xtts

Every backend falls back to Edge TTS on any failure so renders never break.

Kokoro tuning (optional):
  RUFUS_KOKORO_VOICE=am_adam   # default: deep American male narration voice
                               # options: am_michael, bf_emma, af_heart, af_sky
  RUFUS_KOKORO_SPEED=1.0       # playback speed (applies to both kokoro + kokoro_api)
  Kokoro has no SSML/prosody control, so delivery comes from punctuation: the
  local backend inserts a silence after each line sized to its trailing
  punctuation (longest after em-dash/ellipsis "beats", shortest after commas) —
  pair with script_writer's punctuation-as-pacing guidance for the best result.

XTTS voice cloning (optional):
  RUFUS_TTS_VOICE=/path/to/reference.wav   # 6-30s clean speech sample to clone

ElevenLabs tuning (optional):
  RUFUS_ELEVEN_VOICE=<voice_id>   # default: James (lUTamkMw7gOzZbFIwmq4)
  RUFUS_ELEVEN_MODEL=<model_id>   # default: eleven_turbo_v2_5

All backends write to the exact output path requested (mp3).
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import text_repair

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"

# Edge TTS defaults
EDGE_VOICE = os.environ.get("RUFUS_EDGE_VOICE", "en-US-AndrewMultilingualNeural")
EDGE_RATE  = os.environ.get("RUFUS_EDGE_RATE", "+6%")

# Kokoro defaults — am_adam is the deep American male voice built for narration
KOKORO_VOICE = os.environ.get("RUFUS_KOKORO_VOICE", "am_adam")
# Kokoro-FastAPI (the Docker HTTP service) — used by the kokoro_api backend.
# Cross-platform: no native `kokoro` pip install needed (ideal on Windows 11).
KOKORO_API_URL = os.environ.get("KOKORO_API_URL", "http://localhost:8880").rstrip("/")
KOKORO_SPEED   = os.environ.get("RUFUS_KOKORO_SPEED", "1.0")

# XTTS defaults
XTTS_MODEL           = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE        = os.environ.get("RUFUS_TTS_LANG", "en")
XTTS_DEFAULT_SPEAKER = os.environ.get("RUFUS_XTTS_SPEAKER", "Damien Black")

# ElevenLabs defaults
ELEVEN_VOICE = os.environ.get("RUFUS_ELEVEN_VOICE", "lUTamkMw7gOzZbFIwmq4")  # James
ELEVEN_MODEL = os.environ.get("RUFUS_ELEVEN_MODEL", "eleven_turbo_v2_5")

_xtts_model   = None   # lazy singleton
_kokoro_pipe  = None   # lazy singleton


def _backend() -> str:
    explicit = os.environ.get("RUFUS_TTS", "").strip().lower()
    if explicit:
        return explicit
    # Auto-select best available, matching the documented quality ranking:
    # ElevenLabs (most natural, key configured = user opted in and pays for
    # it — use it) > Kokoro (human-quality, free) > Edge (robotic).
    if _eleven_key():
        return "elevenlabs"
    try:
        import kokoro  # noqa: F401
        return "kokoro"
    except ImportError:
        return "edge"


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def _pause_seconds(chunk_text: str) -> float:
    """How long a silence to insert AFTER this chunk, based on its trailing
    punctuation. Kokoro has no SSML/prosody control — punctuation is the only
    delivery cue it reads, so this is where "detailed direction" for a free
    local voice actually lives. Tuned for narration pace, not real speech:
    dramatic beats (em-dash/ellipsis) get the longest gap, full stops next,
    commas the shortest — anything unrecognized defaults to a light beat."""
    t = chunk_text.rstrip()
    if not t:
        return 0.15
    if t.endswith("...") or t.endswith("—") or t.endswith("–"):
        return 0.32
    if t[-1] in "?!":
        return 0.30
    if t[-1] == ".":
        return 0.26
    if t[-1] in ",;:":
        return 0.14
    return 0.15


def _tone_pause(tone: str) -> float:
    """Extra silence this beat's tone earns, on top of its punctuation.

    Imported lazily so tts_engine keeps working standalone if emotional_map is
    ever absent — the voice is not allowed to depend on the creative layer.
    """
    try:
        import emotional_map
        return emotional_map.pause_after(tone)
    except Exception:
        return 0.0


KOKORO_REQUIREMENTS = ("numpy", "soundfile", "kokoro")


def _missing_kokoro_deps() -> list[str]:
    """Which of Kokoro's imports are absent, ALL of them, in one pass.

    Python reports only the first missing import, and _kokoro's imports happen
    to run soundfile before kokoro — so a box missing both was told "No module
    named 'soundfile'", the owner installed exactly that, reran a whole
    pipeline, and got "No module named 'kokoro'" for their trouble. One round
    trip per missing package is a bad trade when listing them costs nothing."""
    import importlib.util
    return [m for m in KOKORO_REQUIREMENTS
            if importlib.util.find_spec(m) is None]


def _kokoro(script: str, out_path: Path,
            tones: list[str] | None = None) -> None:
    """Synthesize with Kokoro-82M (Apache 2.0, runs on CPU). Outputs mp3."""
    global _kokoro_pipe
    missing = _missing_kokoro_deps()
    if missing:
        raise RuntimeError(
            f"missing {', '.join(missing)} — install with: "
            f"pip install {' '.join(missing)}")
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    if _kokoro_pipe is None:
        _kokoro_pipe = KPipeline(lang_code="a")   # 'a' = American English
        print(f"[tts] Kokoro pipeline loaded (voice: {KOKORO_VOICE})")

    try:
        speed = float(KOKORO_SPEED)
    except ValueError:
        speed = 1.0

    def _as_numpy(audio):
        """Kokoro yields torch Tensors; everything below here is numpy.

        Without this, `np.zeros(gap, dtype=seg_audio.dtype)` is handed
        torch.float32 and raises "Cannot interpret 'torch.float32' as a data
        type" — which reads exactly like a numpy-2 incompatibility and was
        misdiagnosed as one for a long time. It is not: it reproduces on
        numpy 1.26.4, and it fires ONLY when the script splits into more than
        one chunk, because a single chunk never reaches the inter-chunk gap.
        That is why every one-sentence smoke test passed while every real
        script silently fell back to the flat Edge voice."""
        detach = getattr(audio, "detach", None)
        return detach().cpu().numpy() if detach is not None else np.asarray(audio)

    # Generator yields (graphemes, phonemes, audio_array) per chunk (split on
    # blank lines by default). Keep the source text alongside each chunk so we
    # can size the gap that follows it from its own punctuation.
    chunks = [(g, _as_numpy(audio)) for g, _, audio in
              _kokoro_pipe(script, voice=KOKORO_VOICE, speed=speed)]
    if not chunks:
        raise RuntimeError("Kokoro returned no audio segments")

    sample_rate = 24000   # Kokoro native sample rate
    pieces = []
    for i, (graphemes, seg_audio) in enumerate(chunks):
        pieces.append(seg_audio)
        if i < len(chunks) - 1:
            # Punctuation earns the base gap; the beat's tone adds to it. A
            # held beat before the turn is the only prosody a voice with no
            # SSML can be given, and it is free. Tones are positional against
            # Kokoro's own chunking, so a mismatch just means no bonus rather
            # than a pause landing in the wrong place.
            seconds = _pause_seconds(graphemes)
            if tones and i < len(tones):
                seconds += _tone_pause(tones[i])
            gap = int(seconds * sample_rate)
            if gap > 0:
                pieces.append(np.zeros(gap, dtype=seg_audio.dtype))

    audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    try:
        sf.write(wav_path, audio, sample_rate)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
             "-c:a", "libmp3lame", "-b:a", "192k", str(out_path)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 5_000:
            raise RuntimeError(f"Kokoro mp3 transcode failed: {r.stderr[-300:]}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ── Kokoro-FastAPI (HTTP) ───────────────────────────────────────────────────────

def _kokoro_api(script: str, out_path: Path) -> None:
    """Synthesize via the Kokoro-FastAPI Docker service (OpenAI-compatible route).

    POST /v1/audio/speech → mp3 bytes written to out_path. Just an HTTP call, so it
    sidesteps the fragile native `kokoro` pip install on Windows. Raises on failure
    so synthesize() falls back to Edge.
    """
    import requests

    try:
        speed = float(KOKORO_SPEED)
    except ValueError:
        speed = 1.0

    r = requests.post(
        f"{KOKORO_API_URL}/v1/audio/speech",
        json={"model": "kokoro", "input": script, "voice": KOKORO_VOICE,
              "response_format": "mp3", "speed": speed},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Kokoro-FastAPI HTTP {r.status_code}: {r.text[:200]}")
    out_path.write_bytes(r.content)
    if not out_path.exists() or out_path.stat().st_size < 5_000:
        raise RuntimeError("Kokoro-FastAPI returned an empty/too-small audio file")


# ── Edge TTS ──────────────────────────────────────────────────────────────────

async def _edge_async(script: str, out_path: Path) -> None:
    import edge_tts
    comm = edge_tts.Communicate(script, EDGE_VOICE, rate=EDGE_RATE)
    await comm.save(str(out_path))


def _edge(script: str, out_path: Path) -> None:
    asyncio.run(_edge_async(script, out_path))


# ── XTTS v2 (Coqui) ─────────────────────────────────────────────────────────────

def _load_xtts():
    """Load XTTS v2 once. Uses GPU if RUFUS_GPU is set and CUDA is available."""
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    from TTS.api import TTS  # heavy import — only when XTTS is actually used

    gpu = os.environ.get("RUFUS_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        import torch
        gpu = gpu and torch.cuda.is_available()
    except Exception:
        gpu = False

    _xtts_model = TTS(XTTS_MODEL, gpu=gpu)
    print(f"[tts] XTTS v2 loaded ({'GPU' if gpu else 'CPU'})")
    return _xtts_model


def _xtts(script: str, out_path: Path) -> None:
    """Synthesize with XTTS v2 → wav → transcode to the requested mp3 path."""
    model = _load_xtts()
    speaker_wav = os.environ.get("RUFUS_TTS_VOICE", "").strip()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        kwargs = {"text": script, "file_path": wav_path, "language": XTTS_LANGUAGE}
        if speaker_wav and Path(speaker_wav).exists():
            kwargs["speaker_wav"] = speaker_wav
        else:
            if speaker_wav:
                print(f"[tts] XTTS reference clip not found ({speaker_wav}) — using built-in speaker")
            kwargs["speaker"] = XTTS_DEFAULT_SPEAKER
        model.tts_to_file(**kwargs)

        # Transcode wav → mp3 at the requested path so downstream is unchanged.
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
             "-c:a", "libmp3lame", "-b:a", "192k", str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 5_000:
            raise RuntimeError(f"XTTS mp3 transcode failed: {r.stderr[-300:]}")
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ── ElevenLabs (cloud) ──────────────────────────────────────────────────────────

def _eleven_key() -> str:
    """Read the ElevenLabs key from config/keys.json. '' if unset/placeholder."""
    try:
        key = json.loads(KEYS_FILE.read_text(encoding="utf-8")).get("elevenlabs", "")
    except Exception:
        return ""
    if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
        return ""
    return key


# ElevenLabs refuses a single request past this many characters. A 40-second
# Short is ~650 and never came near it; a nine-minute script is ~7,500 and
# would have been rejected outright — the format change turning a working
# voice into a failed render, on the one stage with no fallback of its own.
ELEVEN_MAX_CHARS = 4800


def _paragraph_batches(script: str, limit: int) -> list[str]:
    """Split on paragraph breaks, packing as much into each request as fits.

    ON PARAGRAPHS, never mid-sentence. A join at a sentence boundary is
    inaudible; a join mid-clause is a stutter the listener hears and cannot
    explain. The long-form writer already separates its sections with blank
    lines, so the seams this cuts on are the ones the outline chose.
    """
    paras = [p.strip() for p in script.split("\n\n") if p.strip()]
    if not paras:
        return [script]
    out, cur = [], ""
    for para in paras:
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = para
        elif cur:
            cur = f"{cur}\n\n{para}"
        else:
            cur = para
        # A single paragraph over the limit still has to go somewhere; send it
        # and let the API decide rather than cutting a sentence in half.
        while len(cur) > limit and "\n\n" not in cur:
            out.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        out.append(cur)
    return out


def _elevenlabs(script: str, out_path: Path) -> None:
    """Synthesize with ElevenLabs → mp3 written directly to out_path.

    Voice settings tuned for narration with life: lower stability = more
    expressive delivery (pauses, emphasis), high similarity keeps the timbre
    consistent across a video, style adds intonation. speaker_boost adds presence.
    """
    import httpx

    key = _eleven_key()
    if not key:
        raise RuntimeError("no ElevenLabs key in config/keys.json")

    # LONGER THAN ONE REQUEST ALLOWS? Say it in pieces and glue the mp3s.
    # mp3 frames concatenate byte-wise, which is why every simple "cat *.mp3"
    # works — no re-encode, no quality loss, and the joins land on paragraph
    # breaks the writer already chose.
    batches = _paragraph_batches(script, ELEVEN_MAX_CHARS)
    if len(batches) > 1:
        print(f"[tts] {len(script)} characters — sending as {len(batches)} "
              f"requests, joined at paragraph breaks")
        parts = []
        try:
            for i, part in enumerate(batches):
                piece = out_path.with_suffix(f".part{i}.mp3")
                _elevenlabs_once(part, piece)
                parts.append(piece)
            with open(out_path, "wb") as f:
                for piece in parts:
                    f.write(piece.read_bytes())
        finally:
            for piece in parts:
                try:
                    piece.unlink()
                except OSError:
                    pass
        return
    _elevenlabs_once(script, out_path)


def _elevenlabs_once(script: str, out_path: Path) -> None:
    """One request. Raises on any non-200, with the cause distilled."""
    import httpx

    key = _eleven_key()
    if not key:
        raise RuntimeError("no ElevenLabs key in config/keys.json")

    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
           f"?output_format=mp3_44100_128")
    payload = {
        "text": script,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {
            "stability": 0.45,          # lower = more expressive, less monotone
            "similarity_boost": 0.8,
            "style": 0.35,              # intonation/emphasis for a human read
            "use_speaker_boost": True,
        },
    }
    headers = {"xi-api-key": key, "Content-Type": "application/json"}

    with httpx.stream("POST", url, json=payload, headers=headers, timeout=120) as r:
        if r.status_code != 200:
            body = r.read()[:300].decode("utf-8", "ignore")
            if r.status_code == 402 and "paid_plan_required" in body:
                # Distilled, actionable — the raw 300-char JSON dump buried
                # the actual cause every run: a "library" voice (a premade
                # ElevenLabs voice, not a cloned one) is blocked from the API
                # entirely on the free tier. No retry or code change fixes
                # this; it's an account-tier limit. Same UX pattern as the
                # Kokoro numpy-2 message below — name the fix, not the wire
                # protocol.
                raise RuntimeError(
                    f"ElevenLabs voice {ELEVEN_VOICE} is a library (premade) "
                    f"voice — free accounts cannot use those via the API "
                    f"(only a cloned voice, or a paid plan). Either upgrade "
                    f"ElevenLabs, or clone your own voice and set "
                    f"RUFUS_ELEVEN_VOICE to its id.")
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {body}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)

    if not out_path.exists() or out_path.stat().st_size < 5_000:
        raise RuntimeError("ElevenLabs returned an empty/too-small audio file")


# ── Public API ────────────────────────────────────────────────────────────────

def _sanitize_for_speech(script: str) -> str:
    """Strip text artifacts a TTS voice would read out loud. GPT occasionally
    leaks markdown emphasis (**word**), stray asterisks, or bracketed stage
    directions into a script — every backend received the text verbatim, so
    the voice would literally say 'asterisk' or read '[pause]'. Cheap, safe,
    and idempotent on clean scripts.

    Mis-decoded text is handled first and separately. A CTA read out of a UTF-8
    config under the wrong code page arrives here as Hebrew letters glued to
    punctuation debris, and a TTS backend pronounces exactly what it is given —
    that shipped in the audio of a finished English short while every gate
    downstream reported pass. text_repair reverses the decode where it can and
    drops what it cannot, saying so either way."""
    s = text_repair.clean_for_speech(script, label="script")
    s = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", s)   # **bold** / *italic* → bare text
    s = re.sub(r"[\[\(]\s*(pause|beat|sfx|music|silence)[^\]\)]*[\]\)]", "", s,
               flags=re.IGNORECASE)                    # [pause], (beat) stage directions
    s = s.replace("*", "").replace("#", "").replace("`", "")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def synthesize(script: str, out_path: Path,
               tones: list[str] | None = None) -> None:
    """Generate speech for `script` at `out_path` (mp3). Backend per RUFUS_TTS.

    Every backend falls back to Edge TTS on any failure so a render never breaks
    over the voice.
    """
    script   = _sanitize_for_speech(script)
    out_path = Path(out_path)
    backend  = _backend()

    if backend == "kokoro_api":
        try:
            print(f"[tts] backend: Kokoro-FastAPI ({KOKORO_VOICE} @ {KOKORO_API_URL})")
            _kokoro_api(script, out_path)
            return
        except Exception as e:
            print(f"[tts] Kokoro-FastAPI failed ({e}) — falling back to Edge TTS")
            _edge(script, out_path)
            return

    if backend == "elevenlabs":
        try:
            print(f"[tts] backend: ElevenLabs ({ELEVEN_MODEL})")
            _elevenlabs(script, out_path)
            return
        except Exception as e:
            print(f"[tts] ElevenLabs failed ({e}) — falling back to Kokoro")

    if backend in ("elevenlabs", "kokoro"):
        try:
            print(f"[tts] backend: Kokoro ({KOKORO_VOICE})")
            _kokoro(script, out_path, tones)
            return
        except Exception as e:
            print(f"[tts] Kokoro failed ({e}) — falling back to Edge TTS")
            if "cannot interpret" in str(e).lower() and "data type" in str(e).lower():
                # This USED to print "install numpy<2". It was wrong, and the
                # wrong hint cost two rounds of chasing the environment: the
                # real cause was Kokoro handing back torch Tensors where the
                # gap-padding expected numpy (fixed in _kokoro). If it shows up
                # again it is a NEW dtype leak, not a numpy version.
                print("[tts]   a torch dtype reached numpy — this is a code "
                      "bug in _kokoro, not a package version")

    if backend == "xtts":
        try:
            print("[tts] backend: XTTS v2 (local)")
            _xtts(script, out_path)
            return
        except Exception as e:
            print(f"[tts] XTTS failed ({e}) — falling back to Kokoro")
        try:
            print(f"[tts] backend: Kokoro ({KOKORO_VOICE})  [XTTS fallback]")
            _kokoro(script, out_path, tones)
            return
        except Exception as e:
            print(f"[tts] Kokoro failed ({e}) — falling back to Edge TTS")
            if "cannot interpret" in str(e).lower() and "data type" in str(e).lower():
                # This USED to print "install numpy<2". It was wrong, and the
                # wrong hint cost two rounds of chasing the environment: the
                # real cause was Kokoro handing back torch Tensors where the
                # gap-padding expected numpy (fixed in _kokoro). If it shows up
                # again it is a NEW dtype leak, not a numpy version.
                print("[tts]   a torch dtype reached numpy — this is a code "
                      "bug in _kokoro, not a package version")

    if backend not in ("elevenlabs", "kokoro", "xtts"):
        print(f"[tts] backend: Edge TTS ({EDGE_VOICE})")
    _edge(script, out_path)


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a Rufus voice test. One, two, three."
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tts_test.mp3")
    synthesize(text, out)
    print(f"OUTPUT={out}  ({out.stat().st_size} bytes)")
