#!/usr/bin/env python3
"""
tts_engine.py — Pluggable text-to-speech for Rufus.

Three backends, selected via the RUFUS_TTS environment variable:

  RUFUS_TTS=edge        (default) — Microsoft Edge TTS. Free, fast, cloud, no
                                    GPU. Reliable but reads slightly flat.
  RUFUS_TTS=xtts                  — Coqui XTTS v2. Free forever, runs locally on
                                    a GTX 1060 6GB (~3GB VRAM), near-ElevenLabs
                                    quality, clones a voice from a 6s sample.
  RUFUS_TTS=elevenlabs            — ElevenLabs cloud. The most natural option;
                                    ~$0.10/video. Needs an "elevenlabs" key in
                                    config/keys.json. This is what most top
                                    faceless channels actually use.

Pick the trade-off you want: elevenlabs = best sound (paid), xtts = best free
(local GPU), edge = zero-setup fallback. Every backend degrades gracefully — any
failure (no key, API down, model missing) falls back to Edge TTS so a render
never breaks over a voice issue.

XTTS voice cloning (optional):
  RUFUS_TTS_VOICE=/path/to/reference.wav   # 6-30s clean speech sample to clone

ElevenLabs tuning (optional):
  RUFUS_ELEVEN_VOICE=<voice_id>   # default: Adam (deep narration)
  RUFUS_ELEVEN_MODEL=<model_id>   # default: eleven_turbo_v2_5 (fast + cheap)

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

# Edge TTS defaults. Andrew is the deep "documentary" multilingual-neural voice
# the community consistently rates most natural for narration; override with
# RUFUS_EDGE_VOICE (e.g. en-US-BrianMultilingualNeural, en-US-ChristopherNeural).
EDGE_VOICE = os.environ.get("RUFUS_EDGE_VOICE", "en-US-AndrewMultilingualNeural")
EDGE_RATE  = os.environ.get("RUFUS_EDGE_RATE", "+6%")

# XTTS defaults
XTTS_MODEL    = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = os.environ.get("RUFUS_TTS_LANG", "en")
# Built-in studio speaker used when no reference clip is provided.
XTTS_DEFAULT_SPEAKER = os.environ.get("RUFUS_XTTS_SPEAKER", "Damien Black")

# ElevenLabs defaults. Adam is the deep, confident narration preset most used
# for faceless content. eleven_turbo_v2_5 is the cheap, fast, high-quality model.
ELEVEN_VOICE = os.environ.get("RUFUS_ELEVEN_VOICE", "pNInz6obpgDQGcFmaJgB")  # Adam
ELEVEN_MODEL = os.environ.get("RUFUS_ELEVEN_MODEL", "eleven_turbo_v2_5")

_xtts_model = None   # lazy singleton — loading the model is expensive


def _backend() -> str:
    return os.environ.get("RUFUS_TTS", "edge").strip().lower()


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
            print(f"[tts] ElevenLabs failed ({e}) — falling back to Edge TTS")

    if backend == "xtts":
        try:
            print("[tts] backend: XTTS v2 (local)")
            _xtts(script, out_path)
            return
        except Exception as e:
            print(f"[tts] XTTS failed ({e}) — falling back to Edge TTS")

    if backend not in ("elevenlabs", "xtts"):
        print(f"[tts] backend: Edge TTS ({EDGE_VOICE})")
    _edge(script, out_path)


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "This is a Rufus voice test. One, two, three."
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tts_test.mp3")
    synthesize(text, out)
    print(f"OUTPUT={out}  ({out.stat().st_size} bytes)")
