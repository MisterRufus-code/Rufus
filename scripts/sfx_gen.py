#!/usr/bin/env python3
"""
sfx_gen.py — Synthesized sound-effects library for Rufus. Zero APIs, zero keys.

Pro Shorts use three signature sounds the ear reads as "edited by a human":
  hit.wav    — sub-bass punch at t=0 (pattern interrupt under the hook)
  whoosh.wav — airy noise sweep on every cut (sells the transition)
  riser.wav  — tension swell into the final beat (pre-payoff lift)
  pop.wav    — short bright blip for a word-synced insert (see insert_director)

All three are synthesized once with FFmpeg lavfi (sine + shaped noise) into
assets/sfx/ and cached. No downloads, no licenses, works offline forever.

ensure_sfx() returns {} on any failure so the renderer degrades gracefully.
"""

import subprocess
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SFX_DIR = ROOT / "assets" / "sfx"

SAMPLE_RATE = 48000
MIN_BYTES   = 4_000   # sanity floor for a generated wav


def _sfx_cmd(name: str, out_path: Path) -> list[str] | None:
    """Build the FFmpeg synthesis command for one effect. None if unknown."""
    if name == "hit":
        # Sub-bass drop: 52 Hz sine with fast exponential decay + 160 Hz click
        # transient on top. Reads as a cinematic "boom" at low mix volume.
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=52:duration=0.45",
            "-f", "lavfi", "-i", "sine=frequency=160:duration=0.06",
            "-filter_complex",
            "[0:a]volume=0.9,afade=t=out:st=0.05:d=0.40:curve=exp[sub];"
            "[1:a]volume=0.35,afade=t=out:st=0.00:d=0.06[click];"
            f"[sub][click]amix=inputs=2:duration=longest:normalize=0,"
            f"aresample={SAMPLE_RATE}[a]",
            "-map", "[a]", "-ac", "1", str(out_path),
        ]
    if name == "whoosh":
        # Air sweep: band-limited pink noise with a fast swell and longer tail.
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "anoisesrc=colour=pink:duration=0.55:seed=7",
            "-af",
            "highpass=frequency=300,lowpass=frequency=2800,"
            "afade=t=in:st=0:d=0.20:curve=esin,"
            "afade=t=out:st=0.22:d=0.33,"
            f"volume=0.8,aresample={SAMPLE_RATE}",
            "-ac", "1", str(out_path),
        ]
    if name == "pop":
        # THE SOUND THE INSERT FORMAT NEEDS, and it is not the whoosh. A whoosh
        # is a TRANSITION — it says "we are moving from here to there", which is
        # right for a scene cut and wrong for an object appearing on top of a
        # scene that has not changed. An insert wants a bright, short blip: a
        # fast upward pitch bend, gone in a tenth of a second, so twenty of them
        # across forty seconds read as punctuation instead of traffic.
        #
        # Short on purpose. Anything with a tail overlaps the next insert at the
        # 0.45s minimum spacing and the two smear into mush.
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=680:duration=0.09",
            "-f", "lavfi", "-i", "sine=frequency=1180:duration=0.05",
            "-filter_complex",
            "[0:a]volume=0.8,afade=t=out:st=0.02:d=0.07:curve=exp[low];"
            "[1:a]volume=0.5,afade=t=in:st=0:d=0.01,"
            "afade=t=out:st=0.01:d=0.04:curve=exp[high];"
            f"[low][high]amix=inputs=2:duration=longest:normalize=0,"
            f"aresample={SAMPLE_RATE}[a]",
            "-map", "[a]", "-ac", "1", str(out_path),
        ]
    if name == "riser":
        # Tension swell: filtered white noise rising over ~1.2s, hard stop.
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "anoisesrc=colour=white:duration=1.30:seed=11",
            "-af",
            "highpass=frequency=600,lowpass=frequency=4000,"
            "afade=t=in:st=0:d=1.05:curve=tri,"
            "afade=t=out:st=1.10:d=0.20,"
            f"volume=0.55,aresample={SAMPLE_RATE}",
            "-ac", "1", str(out_path),
        ]
    return None


def ensure_sfx() -> dict[str, Path]:
    """Synthesize the SFX set if missing. Returns {name: path}, or {} on failure."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in ("hit", "whoosh", "riser", "pop"):
        path = SFX_DIR / f"{name}.wav"
        if path.exists() and path.stat().st_size >= MIN_BYTES:
            out[name] = path
            continue
        cmd = _sfx_cmd(name, path)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and path.exists() and path.stat().st_size >= MIN_BYTES:
                print(f"[sfx] synthesized {name}.wav")
                out[name] = path
            else:
                print(f"[sfx] {name} synthesis failed: {r.stderr[-150:].strip()}")
                path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[sfx] {name} synthesis error: {e}")
            path.unlink(missing_ok=True)
    # THE THREE ORIGINALS ARE THE ALL-OR-NOTHING SET; "pop" is not. It serves
    # one optional layer (word-synced inserts) and a box that cannot synthesize
    # it should still get the hit/whoosh/riser it always had, rather than
    # losing the whole sound design to a feature it is not using.
    core = {k: v for k, v in out.items() if k in ("hit", "whoosh", "riser")}
    if len(core) < 3:
        return {}
    return out


if __name__ == "__main__":
    paths = ensure_sfx()
    for k, v in paths.items():
        print(f"{k}: {v} ({v.stat().st_size} bytes)")
    if not paths:
        print("SFX generation failed — renderer will skip the SFX layer")
