#!/usr/bin/env python3
"""
emotional_map.py — one per-beat tone that the whole render reads from.

THE PROBLEM THIS SOLVES. Three systems already decide how a beat should feel,
and none of them can see the other two:

    storyboard.py     picks the shot
    edit_director.py  picks the camera move  ("peak at beat 7 → push_in, rise…")
    audio_gen.py      places the SFX and grades the picture

Each is reasonable alone. Together they are three separate opinions about the
same video that never compare notes — the grade is one global `ffmpeg_eq` from
niches.json applied identically to every beat, the SFX are placed by position
rather than by meaning, and the camera move is chosen by a model that no other
stage can hear. That is what reads as flat: not a missing agent, a missing
shared artifact.

WHAT THIS IS. A tone per beat, and the pure translation from a tone into the
concrete numbers the renderer already accepts. No model call, no network, no
new pipeline stage — edit_director already asks a model to read the script, so
the tone rides along in that same reply at no extra cost, and everything here
is a lookup on the result.

DELIBERATELY NOT A GATE. This codebase has been bitten by stacked deterministic
gates rejecting work for stylistic reasons (see CLAUDE.md on the wasted-
generation rejection ladder). Nothing here can reject anything. An unknown tone
degrades to NEUTRAL, a missing plan degrades to the niche's own grade, and the
render is identical to today's. The worst case is no effect.

WHY THE GRADES ARE SMALL. These are ±0.12 nudges around the niche's base look,
not a filter. A grade you notice is a grade that is too strong — the eye should
read "this beat feels colder" without being able to say why. Beat-to-beat
contrast is what carries feeling; absolute values are what break a channel's
consistent look.
"""

from __future__ import annotations

# The vocabulary. Small on purpose: every tone here must map to a render
# difference a viewer can actually perceive. A longer list would produce tones
# that grade identically, which is vocabulary without meaning.
TONES = (
    "tension",      # something is wrong / about to go wrong — cold, hard, drained
    "curiosity",    # the question is open — cool and clean, slightly lifted
    "revelation",   # the turn, the number, the reveal — warm and saturated
    "weight",       # consequence, aftermath, cost — dark and desaturated
    "resolution",   # the line that lets it sit — warm, soft, low contrast
    "neutral",      # narration carrying information; the base look
)

NEUTRAL = "neutral"

# tone → (contrast, saturation, brightness, gamma, red shift, blue shift)
#
# Contrast/saturation/brightness/gamma feed ffmpeg's `eq`; the two shifts feed
# `colorbalance` midtones and are what actually carry warm-vs-cold, which is
# the axis a viewer feels most and names least.
_GRADE: dict[str, tuple[float, float, float, float, float, float]] = {
    "tension":    (1.12, 0.88, -0.02, 0.98, -0.04,  0.06),
    "curiosity":  (1.04, 0.97,  0.01, 1.00, -0.02,  0.03),
    "revelation": (1.08, 1.12,  0.02, 1.02,  0.05, -0.04),
    "weight":     (1.06, 0.82, -0.04, 0.96, -0.01,  0.02),
    "resolution": (0.98, 1.04,  0.01, 1.02,  0.04, -0.02),
    "neutral":    (1.00, 1.00,  0.00, 1.00,  0.00,  0.00),
}

# tone → how loud this beat's SFX should sit, relative to the mix's own level.
# The riser into a revelation should be audible; a resolution beat should not
# have a whoosh competing with the closing line.
_SFX_WEIGHT: dict[str, float] = {
    "tension":    1.15,
    "curiosity":  0.95,
    "revelation": 1.25,
    "weight":     1.05,
    "resolution": 0.70,
    "neutral":    1.00,
}

# tone → extra silence AFTER the beat, in seconds, on top of whatever the
# trailing punctuation already earns.
#
# This is the only prosody control a free local voice has. Kokoro has no SSML
# — tts_engine._pause_seconds sizes a gap from punctuation because that is
# literally the only delivery cue it reads. A held beat before the turn is
# what makes the turn land, and it costs nothing but silence.
#
# Small numbers on purpose: this is added to a gap that already exists, and a
# Short cannot afford dead air. 0.30s reads as a held breath; 1s reads as a
# broken file.
_PAUSE_AFTER: dict[str, float] = {
    "tension":    0.30,   # let the wrongness sit before resolving it
    "curiosity":  0.16,
    "revelation": 0.34,   # the longest hold in the video, right after the turn
    "weight":     0.22,
    "resolution": 0.00,   # the closing line is followed by the CTA, not a gap
    "neutral":    0.00,
}

# tone → film-grain strength for ffmpeg's `noise` filter (alls value).
#
# Grain is what stops an AI still from reading as plastic — the "warm, not
# sterile" lever. It must be TEMPORAL (allf=t): static grain looks like a dirty
# lens, grain that changes per frame looks like film. Values are low; grain you
# can consciously see is grain that is too strong, and it also costs bitrate on
# a platform that re-encodes everything.
_GRAIN: dict[str, int] = {
    "tension":    10,
    "curiosity":   6,
    "revelation":  5,   # least grain on the clearest moment
    "weight":     11,
    "resolution":  6,
    "neutral":     7,
}

# Clamps. A model that returns a tone is trusted; arithmetic on top of a
# niche's own base grade is not — a niche could ship ffmpeg_eq=contrast=1.4 and
# a tension beat on top of it would crush the picture.
_CONTRAST_RANGE   = (0.70, 1.60)
_SATURATION_RANGE = (0.30, 1.80)
_BRIGHTNESS_RANGE = (-0.20, 0.20)
_GAMMA_RANGE      = (0.70, 1.40)


def normalise(tone: object) -> str:
    """Any input → a tone this module knows. Never raises, never rejects."""
    if not isinstance(tone, str):
        return NEUTRAL
    t = tone.strip().lower()
    return t if t in TONES else NEUTRAL


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


def sfx_weight(tone: object) -> float:
    """Relative SFX gain for a beat of this tone. 1.0 is today's behaviour."""
    return _SFX_WEIGHT[normalise(tone)]


# tone → how long this beat's PICTURE should hold, relative to an even share.
#
# WHY THE SHOTS SHOULD NOT BE EQUAL LENGTH. The cut planner divided the
# timeline evenly and snapped to the nearest pause, which gives every beat the
# same duration whatever it carries — so the number, the turn and the line that
# lets it sit all flash past at the same rate as "and then this happened". A
# viewer reads an even rhythm as a slideshow no matter how good the pictures
# are. These are deliberately mild: this is a fast-cut channel, and the point
# is that the reveal breathes and the connective tissue does not, not that the
# edit lurches.
_HOLD_WEIGHT: dict[str, float] = {
    "revelation": 1.45,   # the turn. Let it land before moving on.
    "weight":     1.30,   # consequence. The beat a viewer feels.
    "resolution": 1.25,   # the closing line, given room to sit.
    "tension":    1.05,   # slightly held — dread is a held frame.
    "curiosity":  0.85,   # the question is asked quickly, then answered.
    "neutral":    0.85,   # information. Say it and move.
}


def hold_weight(tone: object) -> float:
    """How long a beat of this tone should hold, relative to an even share.

    1.0 is exactly the even grid the planner used before. Everything else is
    time borrowed from the connective beats and given to the ones that carry
    the story.
    """
    return _HOLD_WEIGHT[normalise(tone)]


def pause_after(tone: object) -> float:
    """Extra silence after a beat of this tone, in seconds. 0.0 for neutral."""
    return _PAUSE_AFTER[normalise(tone)]


def grain_enabled() -> bool:
    """RUFUS_FILM_GRAIN=0 turns the texture layer off. On by default."""
    import os
    return os.environ.get("RUFUS_FILM_GRAIN", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def grain_filter(tone: object) -> str:
    """ffmpeg `noise` fragment for this tone, or "" when grain is off.

    Temporal and uniform: allf=t makes the pattern change every frame, which is
    the entire difference between film grain and a smudged lens.
    """
    if not grain_enabled():
        return ""
    strength = _GRAIN[normalise(tone)]
    scale = _grain_scale()
    value = max(0, min(40, round(strength * scale)))
    return f"noise=alls={value}:allf=t+u" if value > 0 else ""


def _grain_scale() -> float:
    """RUFUS_FILM_GRAIN_SCALE multiplies every grain value (default 1.0), so the
    look is tunable in one place without editing the table."""
    import os
    try:
        return max(0.0, min(3.0, float(os.environ.get("RUFUS_FILM_GRAIN_SCALE", "1.0"))))
    except ValueError:
        return 1.0


def grade_filter(tone: object, base_contrast: float = 1.1,
                 base_saturation: float = 1.0) -> str:
    """An ffmpeg filter fragment grading one clip for `tone`.

    Multiplies onto the niche's own base look rather than replacing it, so a
    channel keeps its identity and the tone only bends it. Returns an `eq`
    alone when the tone is neutral — no colorbalance node for zero shift, since
    a no-op filter still costs a pass over every frame of every clip.
    """
    contrast, saturation, brightness, gamma, r_shift, b_shift = _GRADE[normalise(tone)]

    c = _clamp(base_contrast * contrast, _CONTRAST_RANGE)
    s = _clamp(base_saturation * saturation, _SATURATION_RANGE)
    b = _clamp(brightness, _BRIGHTNESS_RANGE)
    g = _clamp(gamma, _GAMMA_RANGE)

    eq = f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f}:gamma={g:.3f}"
    if r_shift != 0.0 or b_shift != 0.0:
        eq = f"{eq},colorbalance=rm={r_shift:.3f}:bm={b_shift:.3f}"

    grain = grain_filter(tone)
    return f"{eq},{grain}" if grain else eq



# tone → (rate %, pitch Hz, volume %) as DELTAS on the voice's base settings.
#
# WHY THIS ARRIVED LAST. The tone already reached the picture, the music and
# the silence between beats. It never reached the VOICE, and the reason is a
# sentence in this file that was true when it was written: "this is the only
# prosody control a free local voice has". That is true of Kokoro, which has
# no SSML — and it quietly governed the default backend, which is Edge, and
# which takes a rate, a pitch and a volume on every call. Six tones were being
# computed and the narration was reading all of them the same way.
#
# SMALL, for the same reason the grades are small. Prosody you can consciously
# hear is prosody that is too strong; the ear should register "he slowed down
# there" without being able to say by how much. These are a few percent, not a
# performance.
#
# NEUTRAL IS EXACTLY ZERO on all three, which is what makes this safe to ship:
# a script whose tones are all neutral — the fail-open state of
# tones_from_plan — synthesizes byte-for-byte what the pipeline produces
# today.
_VOICE: dict[str, tuple[int, int, int]] = {
    # something is wrong: tighter and lower, the way people talk when careful
    "tension":    (-4, -6,  -4),
    # the question is open: a touch quicker and lifted, leaning in
    "curiosity":  (+3, +4,   0),
    # the turn, the number: slower and up, because this is the line that pays
    "revelation": (-6, +8,  +6),
    # consequence and cost: the slowest and lowest thing in the video
    "weight":     (-8, -10, -3),
    # let it sit: slow, soft, coming to rest
    "resolution": (-5, -3,  -8),
    "neutral":    (0,   0,   0),
}

# Clamped for the same reason grade_filter clamps: RUFUS_EDGE_RATE is an owner
# setting, a tone delta lands on top of it, and "+6%" plus an unbounded delta
# is how a voice ends up chipmunked or unintelligibly slow.
_RATE_RANGE   = (-40, 40)
_PITCH_RANGE  = (-40, 40)
_VOLUME_RANGE = (-40, 40)


def voice(tone: object, base_rate_pct: int = 0) -> dict[str, str]:
    """Edge TTS rate/pitch/volume strings for one beat's tone.

    Returned in the exact shape `edge_tts.Communicate` wants — "+6%", "-10Hz" —
    because the alternative is every caller formatting them and one of them
    getting the unit wrong. Pitch is Hz for Edge; rate and volume are percent.

    An unknown tone degrades to NEUTRAL, which is a zero delta, which is the
    voice the pipeline already ships.
    """
    rate, pitch, volume = _VOICE.get(normalise(tone), _VOICE[NEUTRAL])
    rate = int(_clamp(base_rate_pct + rate, _RATE_RANGE))
    pitch = int(_clamp(pitch, _PITCH_RANGE))
    volume = int(_clamp(volume, _VOLUME_RANGE))
    return {"rate": f"{rate:+d}%", "pitch": f"{pitch:+d}Hz",
            "volume": f"{volume:+d}%"}


def voice_is_neutral(tones: list[str] | None) -> bool:
    """True when varying the voice would change nothing.

    The caller uses this to take the single-request path it has always taken.
    Synthesizing beat by beat costs one network round trip per beat, and
    paying that to apply a delta of zero would be a cost with no picture to
    show for it.
    """
    if not tones:
        return True
    return all(normalise(t) == NEUTRAL for t in tones)


def tones_from_plan(plan: dict | None, n_beats: int) -> list[str]:
    """The per-beat tone list from an edit_director plan.

    Fail-open at every step: no plan, a short plan, a plan with junk in it —
    all produce a full-length list of NEUTRAL, which grades exactly as the
    pipeline does today.
    """
    tones = [NEUTRAL] * max(0, n_beats)
    if not isinstance(plan, dict):
        return tones

    beats = plan.get("beats")
    if not isinstance(beats, list):
        return tones

    for i, entry in enumerate(beats):
        if i >= n_beats:
            break
        if isinstance(entry, dict):
            tones[i] = normalise(entry.get("tone"))
    return tones


def describe(tones: list[str]) -> str:
    """One log line. The map is invisible in the output by design, so the log
    is the only place its work is legible — and a run where every beat came
    back NEUTRAL should be obvious at a glance, not something to infer from a
    video that looks unchanged."""
    if not tones:
        return "no beats"
    shown = ", ".join(tones)
    if all(t == NEUTRAL for t in tones):
        return f"{shown}  (no tones in the edit plan — grading unchanged)"
    return shown
