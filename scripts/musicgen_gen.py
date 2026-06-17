#!/usr/bin/env python3
"""
musicgen_gen.py — Meta AudioCraft MusicGen for Rufus.

Generates AI music from a text description that's literally written for the
niche's emotional tone — so finance videos get cinematic tension strings, not
a random podcast track. Requires: pip install audiocraft

First run downloads the 'small' model (~300 MB, cached to ~/.cache/huggingface/).
Subsequent runs load from cache in under 10s.

CPU time: ~2 min / 65s clip.  GPU (CUDA) time: ~20s.

Usage:
    python scripts/musicgen_gen.py finance         # generate + print path
    RUFUS_GPU=1 python scripts/musicgen_gen.py motivation
"""

import os
import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SYNTH_DIR    = ROOT / "assets" / "music_synth"
MAX_VARIANTS = 3   # keep up to 3 distinct tracks per niche so every video sounds fresh

# One text description per niche — written as MusicGen prompts (natural language).
# Each prompt is engineered to produce instrumental, background-friendly music.
NICHE_PROMPTS: dict[str, str] = {
    "finance": (
        "cinematic dark ambient, no lyrics, tense but controlled, documentary strings, "
        "slow tempo, deep bass undertone, sophisticated, financial thriller atmosphere"
    ),
    "motivation": (
        "epic orchestral build, no lyrics, driving percussion, powerful strings, "
        "rising energy, heroic, cinematic, inspirational climax, trailer music"
    ),
    "mindset": (
        "contemplative solo piano, no lyrics, soft ambient pads behind, "
        "thoughtful, introspective, slow tempo, peaceful, meditative, Hans Zimmer style"
    ),
    "business": (
        "modern corporate ambient, no lyrics, clean minimal piano and strings, "
        "confident, forward motion, professional, subtle percussion, innovation feel"
    ),
    "personal_development": (
        "warm acoustic guitar fingerpicking, no lyrics, hopeful, gentle forward movement, "
        "soft string swells, morning energy, sunrise feel, optimistic, calm crescendo"
    ),
}
DEFAULT_PROMPT = "calm ambient instrumental, no lyrics, background music, neutral, peaceful"

_musicgen_model = None   # lazy singleton — loaded once per process


def _model():
    """Load MusicGen 'small' once. Prefers GPU if RUFUS_GPU is set."""
    global _musicgen_model
    if _musicgen_model is not None:
        return _musicgen_model

    from audiocraft.models import MusicGen

    gpu = os.environ.get("RUFUS_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    if gpu:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False

    device = "cuda" if gpu else "cpu"
    print(f"[musicgen] loading MusicGen-small ({device}) — first run downloads ~300MB…")
    _musicgen_model = MusicGen.get_pretrained("small", device=device)
    print("[musicgen] model ready")
    return _musicgen_model


def _variant_paths(niche: str) -> list[Path]:
    """Return all existing cached variant paths for this niche, sorted by slot."""
    return sorted(
        p for p in SYNTH_DIR.glob(f"musicgen_{niche}_*.wav")
        if p.stat().st_size > 100_000
    )


def generate_music(niche: str, duration: float = 65.0, force: bool = False) -> Path | None:
    """Generate or return cached AI music for `niche`.

    Up to MAX_VARIANTS distinct tracks are cached per niche
    (musicgen_{niche}_0.wav … musicgen_{niche}_2.wav). Returning a random
    variant means consecutive videos in the same niche get different music.
    A new variant is generated whenever fewer than MAX_VARIANTS exist.

    Args:
        niche:    Niche key — must be in NICHE_PROMPTS or falls back to default.
        duration: Target duration in seconds (MusicGen is approximate ±2s).
        force:    Force generation of a new variant even if MAX_VARIANTS cached.
    """
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    existing = _variant_paths(niche)

    # Return a random cached variant unless we still have room to grow the pool
    if existing and not force and len(existing) >= MAX_VARIANTS:
        chosen = random.choice(existing)
        print(f"[musicgen] cached ({len(existing)} variants): {chosen.name}")
        return chosen

    # Determine next slot (0-based, wrap around if somehow over cap)
    next_slot = len(existing) % MAX_VARIANTS
    out = SYNTH_DIR / f"musicgen_{niche}_{next_slot}.wav"

    prompt = NICHE_PROMPTS.get(niche, DEFAULT_PROMPT)
    print(f"[musicgen] generating variant {next_slot} ({duration:.0f}s) for niche='{niche}'")
    print(f"[musicgen] prompt: {prompt}")

    try:
        import torch

        model = _model()
        model.set_generation_params(duration=duration)

        tokens = model.generate([prompt], progress=True)
        # tokens shape: (1, channels, samples) — squeeze batch dim
        audio = tokens[0].cpu()

        # Save as WAV using torchaudio
        import torchaudio
        sr = model.sample_rate
        torchaudio.save(str(out), audio, sr)

        if not out.exists() or out.stat().st_size < 100_000:
            raise RuntimeError(f"output file too small ({out.stat().st_size if out.exists() else 0} bytes)")

        print(f"[musicgen] saved: {out}  ({out.stat().st_size // 1024}KB, {sr}Hz, variant {next_slot})")

        # Return a random variant from the updated pool so caller gets variety
        updated = _variant_paths(niche)
        return random.choice(updated) if updated else out

    except Exception as e:
        print(f"[musicgen] generation failed: {e}")
        out.unlink(missing_ok=True)
        # Fall back to any existing variant rather than returning None
        existing = _variant_paths(niche)
        if existing:
            return random.choice(existing)
        return None


if __name__ == "__main__":
    niche = sys.argv[1] if len(sys.argv) > 1 else "finance"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 65.0
    result = generate_music(niche, duration=duration)
    if result:
        print(f"MUSICGEN_PATH={result}")
    else:
        print("[musicgen] failed — check audiocraft install: pip install audiocraft")
        sys.exit(1)
