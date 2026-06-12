#!/usr/bin/env python3
"""
music_gen.py — Synthesized background music beds for Rufus. Zero APIs, zero keys.

When no streaming provider is available (Jamendo needs a client_id, archive.org
can be flaky), Rufus still ships a music bed: a per-niche ambient pad synthesized
locally with FFmpeg's aevalsrc — a 4-chord progression with per-chord swell
envelopes, a sub-octave drone, and (for energetic niches) a soft BPM pulse.

Under the renderer's sidechain ducking at bed volume this reads as a real
production bed, not dead air. CC-what? It's math — no license exists.

Beds are synthesized once per niche into assets/music_synth/ and cached.
ensure_music() returns None on any failure so callers degrade gracefully.
"""

import subprocess
from pathlib import Path

ROOT      = Path(__file__).parent.parent
MUSIC_DIR = ROOT / "assets" / "music_synth"

SAMPLE_RATE = 48000
BED_DUR     = 64.0    # > MAX_DUR(60s) so the renderer never hits a loop seam
CHORD_DUR   = 8.0     # seconds per chord
MIN_BYTES   = 500_000

# Note frequencies (Hz), equal temperament
_C2, _D2, _E2, _F2, _G2, _A2, _B2 = 65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.47
_C3, _D3, _E3, _F3, _G3, _A3, _B3 = 130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94
_BB2 = 116.54

# Per-niche bed design: 4 chords (root, mid, high), optional pulse BPM.
# Progressions chosen for mood: minor = tension, major = warmth/lift.
NICHE_BEDS = {
    "finance": {            # Am – F – C – G, dark/tense, slow pulse
        "chords": [[_A2, _C3, _E3], [_F2, _A2, _C3], [_C3, _E3, _G3], [_G2, _B2, _D3]],
        "bpm": 90,
    },
    "motivation": {         # C – G – Am – F power voicings, epic drive
        "chords": [[_C2, _G2, _C3], [_G2, _D3, _G3], [_A2, _E3, _A3], [_F2, _C3, _F3]],
        "bpm": 100,
    },
    "mindset": {            # C – Am – F – G soft, no pulse — contemplative
        "chords": [[_C3, _E3, _G3], [_A2, _C3, _E3], [_F2, _A2, _C3], [_G2, _B2, _D3]],
        "bpm": None,
    },
    "business": {           # F – C – Dm – Bb, neutral corporate
        "chords": [[_F2, _A2, _C3], [_C3, _E3, _G3], [_D3, _F3, _A3], [_BB2, _D3, _F3]],
        "bpm": 95,
    },
    "personal_development": {  # G – Em – C – D warm, no pulse
        "chords": [[_G2, _B2, _D3], [_E2, _G2, _B2], [_C3, _E3, _G3], [_D3, _F3 * 2**(1/6), _A3]],
        "bpm": None,
    },
}
DEFAULT_BED = NICHE_BEDS["mindset"]


def _pad_expr(chords: list[list[float]], seg: float = CHORD_DUR) -> str:
    """aevalsrc expression: chord progression pad + sub-octave drone.

    Each chord occupies one `seg`-second slot inside a repeating cycle; a
    clipped sine-arch envelope (zero at slot edges) swells each chord in and
    out so chord changes never click. between() gates each chord to its slot.
    """
    cycle = seg * len(chords)
    terms = []
    for i, (root, mid, high) in enumerate(chords):
        start = i * seg
        env = f"min(1,3*sin(PI*(mod(t,{cycle:g})-{start:g})/{seg:g}))"
        gate = f"between(mod(t,{cycle:g}),{start:g},{start + seg:g})"
        voice = (
            f"(0.9*sin(2*PI*{root / 2:.2f}*t)"      # sub-octave drone
            f"+sin(2*PI*{root:.2f}*t)"
            f"+0.8*sin(2*PI*{mid:.2f}*t)"
            f"+0.6*sin(2*PI*{high:.2f}*t))"
        )
        terms.append(f"{gate}*{env}*{voice}")
    return "0.11*(" + "+".join(terms) + ")"


def _music_cmd(niche: str, out_path: Path) -> list[str]:
    """Build the FFmpeg synthesis command for a niche bed."""
    bed    = NICHE_BEDS.get(niche, DEFAULT_BED)
    expr   = _pad_expr(bed["chords"])
    pad_in = f"aevalsrc=exprs='{expr}':d={BED_DUR:g}:s={SAMPLE_RATE}"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", pad_in]

    # Shared tail: gentle space (echo), roll off harsh highs, master volume.
    tail = (
        "lowpass=f=1800,"
        "aecho=0.6:0.3:46|92:0.35|0.25,"
        f"volume=0.85,aresample={SAMPLE_RATE}"
    )

    bpm = bed.get("bpm")
    if bpm:
        # Soft rhythmic pulse: lowpassed pink noise amplitude-modulated at the
        # beat rate. Reads as a muted heartbeat under the pad.
        cmd += ["-f", "lavfi", "-i", f"anoisesrc=colour=pink:duration={BED_DUR:g}:seed=42"]
        fc = (
            f"[0:a]{tail}[pad];"
            f"[1:a]lowpass=f=400,tremolo=f={bpm / 60:.3f}:d=0.85,volume=0.35,"
            f"aresample={SAMPLE_RATE}[pulse];"
            "[pad][pulse]amix=inputs=2:duration=first:normalize=0[a]"
        )
        cmd += ["-filter_complex", fc, "-map", "[a]"]
    else:
        cmd += ["-af", tail]

    cmd += ["-ac", "1", "-t", f"{BED_DUR:g}", str(out_path)]
    return cmd


def ensure_music(niche: str) -> Path | None:
    """Synthesize (or return cached) the niche music bed. None on failure."""
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    name = niche if niche in NICHE_BEDS else "default"
    path = MUSIC_DIR / f"{name}.wav"
    if path.exists() and path.stat().st_size >= MIN_BYTES:
        return path
    try:
        r = subprocess.run(_music_cmd(niche, path),
                           capture_output=True, text=True, timeout=90)
        if r.returncode == 0 and path.exists() and path.stat().st_size >= MIN_BYTES:
            print(f"[music] synthesized local bed → {path.name}")
            return path
        print(f"[music] bed synthesis failed: {r.stderr[-200:].strip()}")
        path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[music] bed synthesis error: {e}")
        path.unlink(missing_ok=True)
    return None


if __name__ == "__main__":
    import sys
    niche = sys.argv[1] if len(sys.argv) > 1 else "finance"
    p = ensure_music(niche)
    print(f"MUSIC_BED={p}" + (f" ({p.stat().st_size} bytes)" if p else ""))
