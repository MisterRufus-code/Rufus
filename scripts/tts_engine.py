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

XTTS voice cloning (optional):
  RUFUS_TTS_VOICE=/path/to/reference.wav   # 6-30s clean speech sample to clone

ElevenLabs tuning (optional):
  RUFUS_ELEVEN_VOICE=<voice_id>   # default: Adam (pNInz6obpgDQGcFmaJgB)
  RUFUS_ELEVEN_MODEL=<model_id>   # default: eleven_turbo_v2_5

All backends write to the exact output path requested (mp3).
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"

# Edge TTS defaults
EDGE_VOICE = os.environ.get("RUFUS_EDGE_VOICE", "en-US-AndrewMultilingualNeural")
EDGE_RATE  = os.environ.get("RUFUS_EDGE_RATE", "+6%")

# Kokoro defaults — am_adam is the deep American male voice built for narration
KOKORO_VOICE = os.environ.get("RUFUS_KOKORO_VOICE", "am_adam")

# XTTS defaults
XTTS_MODEL           = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE        = os.environ.get("RUFUS_TTS_LANG", "en")
XTTS_DEFAULT_SPEAKER = os.environ.get("RUFUS_XTTS_SPEAKER", "Damien Black")

# ElevenLabs defaults
ELEVEN_VOICE = os.environ.get("RUFUS_ELEVEN_VOICE", "pNInz6obpgDQGcFmaJgB")  # Adam
ELEVEN_MODEL = os.environ.get("RUFUS_ELEVEN_MODEL", "eleven_turbo_v2_5")

_xtts_model   = None   # lazy singleton
_kokoro_pipe  = None   # lazy singleton


def _backend() -> str:
    return os.environ.get("RUFUS_TTS", "edge").strip().lower()


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def _kokoro(script: str, out_path: Path) -> None:
    """Synthesize with Kokoro-82M (Apache 2.0, runs on CPU). Outputs mp3."""
    global _kokoro_pipe
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    if _kokoro_pipe is None:
        _kokoro_pipe = KPipeline(lang_code="a")   # 'a' = American English
        print(f"[tts] Kokoro pipeline loaded (voice: {KOKORO_VOICE})")

    # Collect all audio segments (generator yields (graphemes, phonemes, audio_array))
    segments = [audio for _, _, audio in _kokoro_pipe(script, voice=KOKORO_VOICE)]
    if not segments:
        raise RuntimeError("Kokoro returned no audio segments")

    audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
    sample_rate = 24000   # Kokoro native sample rate

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
        key = json.loads(KEYS_FILE.read_text()).get("elevenlabs", "")
    except Exception:
        return ""
    if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
        return ""
    return key


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
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {body}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)

    if not out_path.exists() or out_path.stat().st_size < 5_000:
        raise RuntimeError("ElevenLabs returned an empty/too-small audio file")


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(script: str, out_path: Path) -> None:
    """Generate speech for `script` at `out_path` (mp3). Backend per RUFUS_TTS.

    Every backend falls back to Edge TTS on any failure so a render never breaks
    over the voice.
    """
    out_path = Path(out_path)
    backend  = _backend()

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
            _kokoro(script, out_path)
            return
        except Exception as e:
            print(f"[tts] Kokoro failed ({e}) — falling back to Edge TTS")

    if backend == "xtts":
        try:
            print("[tts] backend: XTTS v2 (local)")
            _xtts(script, out_path)
            return
        except Exception as e:
            print(f"[tts] XTTS failed ({e}) — falling back to Kokoro")
        try:
            print(f"[tts] backend: Kokoro ({KOKORO_VOICE})  [XTTS fallback]")
            _kokoro(script, out_path)
            return
        except Exception as e:
            print(f"[tts] Kokoro failed ({e}) — falling back to Edge TTS")

    if backend not in ("elevenlabs", "kokoro", "xtts"):
        print(f"[tts] backend: Edge TTS ({EDGE_VOICE})")
    _edge(script, out_path)


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a Rufus voice test. One, two, three."
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tts_test.mp3")
    synthesize(text, out)
    print(f"OUTPUT={out}  ({out.stat().st_size} bytes)")
