"""Tests for the v4.0 cinematic render upgrade — cut planning, audio graph,
brand accents, and SFX synthesis commands. All pure functions, no FFmpeg run."""

from types import SimpleNamespace

import pytest

from audio_gen import (
    XFADE_DUR,
    FIRST_CUT_MIN,
    FIRST_CUT_MAX,
    GREEN,
    _audio_filter_complex,
    _audio_filter_simple,
    _concat_input_lengths,
    _hex_to_ass,
    _is_highlight,
    _plan_cuts,
    _sentence_ends,
    _video_filter_complex,
    _video_filter_complex_concat,
    _xfade_input_lengths,
)
from sfx_gen import _sfx_cmd


def _word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(words):
    return SimpleNamespace(words=words)


# ── Sentence detection ───────────────────────────────────────────────────────────

def test_sentence_ends_detects_terminal_punctuation():
    seg = _segment([
        _word("Money", 0.0, 0.4),
        _word("lies.", 0.4, 1.0),
        _word("Banks", 1.2, 1.6),
        _word("win!", 1.6, 2.2),
        _word("Why?", 2.5, 3.0),
        _word("Because", 3.1, 3.5),
    ])
    assert _sentence_ends([seg]) == [1.0, 2.2, 3.0]


def test_sentence_ends_handles_trailing_quotes():
    seg = _segment([_word('rich."', 0.0, 1.5)])
    assert _sentence_ends([seg]) == [1.5]


# ── Cut planning ─────────────────────────────────────────────────────────────────

def test_plan_cuts_single_clip_returns_empty():
    assert _plan_cuts([5.0, 10.0], 40.0, 1) == []


def test_plan_cuts_first_cut_in_hook_window():
    ends = [2.8, 11.0, 21.0, 31.0]
    cuts = _plan_cuts(ends, 40.0, 4)
    assert FIRST_CUT_MIN <= cuts[0] <= FIRST_CUT_MAX
    assert cuts[0] == 2.8     # snapped to the real sentence end


def test_plan_cuts_snaps_to_sentence_ends():
    # Hook cut at 3.0, rest re-spread over [3, 40] → targets ≈ 15.3, 27.7;
    # 16.1 is within the snap window, 30.5 is not (2.8s away > 2.0).
    ends = [3.0, 16.1, 30.5]
    cuts = _plan_cuts(ends, 40.0, 4)
    assert cuts[0] == 3.0
    assert cuts[1] == 16.1                          # snapped to sentence end
    assert cuts[2] == pytest.approx(27.667, abs=0.01)  # grid fallback


def test_plan_cuts_rebalances_after_early_hook_cut():
    """Hook cut pulls to ~3s — remaining cuts must spread over [3s, end], not
    stay on the original n-grid (which would leave a 17s second clip)."""
    cuts = _plan_cuts([2.6], 40.0, 4)
    assert cuts[0] == 2.6
    gaps = [b - a for a, b in zip([0.0] + cuts, cuts + [40.0])]
    assert max(gaps[1:]) - min(gaps[1:]) < 1.0      # body clips near-equal


def test_plan_cuts_falls_back_to_grid_without_sentences():
    cuts = _plan_cuts([], 40.0, 4)
    assert len(cuts) == 3
    assert FIRST_CUT_MIN <= cuts[0] <= FIRST_CUT_MAX
    assert cuts == sorted(cuts)


def test_plan_cuts_monotonic_with_min_spacing():
    # Pathological: sentence ends bunched together
    ends = [2.5, 2.6, 2.7, 2.8]
    cuts = _plan_cuts(ends, 30.0, 4)
    assert cuts == sorted(cuts)
    assert all(b - a >= 1.0 for a, b in zip(cuts, cuts[1:]))


# ── Input-length math ────────────────────────────────────────────────────────────

def test_xfade_lengths_telescope_to_total():
    boundaries = [3.0, 18.5, 30.2]
    total      = 44.0
    lengths    = _xfade_input_lengths(boundaries, total)
    assert len(lengths) == 4
    # Chained xfade output: L1 + Σ(Lk - XFADE) == total
    chained = lengths[0] + sum(l - XFADE_DUR for l in lengths[1:])
    assert chained == pytest.approx(total, abs=0.01)


def test_concat_lengths_sum_to_total():
    boundaries = [3.0, 18.5, 30.2]
    total      = 44.0
    lengths    = _concat_input_lengths(boundaries, total)
    assert sum(lengths) == pytest.approx(total, abs=0.01)


def test_lengths_single_clip():
    assert _xfade_input_lengths([], 44.0)  == [44.0]
    assert _concat_input_lengths([], 44.0) == [44.0]


# ── Brand accent ─────────────────────────────────────────────────────────────────

def test_hex_to_ass_swaps_to_bgr():
    assert _hex_to_ass("#FFC53D") == "&H003DC5FF"


def test_hex_to_ass_invalid_falls_back_to_green():
    assert _hex_to_ass("oops") == GREEN


def test_highlight_fires_on_numbers_and_opinion_words():
    assert _is_highlight("$500")
    assert _is_highlight("90%")
    assert _is_highlight("WORST")     # opinion_pool word
    assert _is_highlight("SECRET.")   # punctuation stripped before match
    assert not _is_highlight("THE")


# ── Video filter graph ───────────────────────────────────────────────────────────

def _video_args():
    return dict(over_w=1188, over_h=2112, pad_y=96,
                eq_filter="eq=contrast=1.1", ass_esc="/tmp/x.ass",
                fonts_dir_esc="/tmp/fonts", accent_hex="#FFC53D")


def test_video_fc_xfade_offsets_end_at_boundaries():
    boundaries = [3.0, 20.0]
    lengths    = _xfade_input_lengths(boundaries, 40.0)
    fc = _video_filter_complex(lengths, boundaries, 40.0, **_video_args())
    assert f"offset={3.0 - XFADE_DUR:.3f}"  in fc
    assert f"offset={20.0 - XFADE_DUR:.3f}" in fc
    assert "xfade=transition=dissolve" in fc


def test_video_fc_includes_progress_bar_and_captions():
    fc = _video_filter_complex([40.0], [], 40.0, **_video_args())
    assert "color=c=0xFFC53D" in fc          # accent-colored bar
    assert "overlay=x='-1080+1080*t/" in fc  # slides left → right over duration
    assert "ass='/tmp/x.ass'" in fc


def test_video_fc_no_fade_in():
    """FADE_IN=0 means the filter graph must NOT contain a fade-in filter."""
    from audio_gen import FADE_IN
    fc = _video_filter_complex([40.0], [], 40.0, **_video_args())
    if FADE_IN == 0:
        assert "fade=type=in" not in fc


def test_video_fc_concat_has_no_xfade():
    lengths = _concat_input_lengths([3.0, 20.0], 40.0)
    fc = _video_filter_complex_concat(lengths, 40.0, **_video_args())
    assert "concat=n=3" in fc
    assert "xfade" not in fc


# ── Audio filter graph ───────────────────────────────────────────────────────────

def test_audio_fc_full_mix_has_ducking_and_mastering():
    fc = _audio_filter_complex(n=3, audio_dur=40.0, has_music=True,
                               sfx_events=[(0.03, 0.9), (2.82, 0.65)])
    assert "sidechaincompress" in fc          # dynamic ducking, not static volume
    assert "loudnorm=I=-14" in fc             # YouTube -14 LUFS master
    assert "acompressor" in fc                # voice chain
    assert "aformat=sample_rates=48000:channel_layouts=stereo" in fc
    assert "[3:a]" in fc and "[4:a]" in fc    # voice + music inputs
    assert "[5:a]" in fc and "[6:a]" in fc    # two sfx inputs after music
    assert "adelay=2820|2820" in fc
    assert fc.strip().endswith("[aout]")


def test_audio_fc_voice_only_still_masters_to_aout():
    fc = _audio_filter_complex(n=2, audio_dur=35.0, has_music=False, sfx_events=[])
    assert "sidechaincompress" not in fc
    assert "loudnorm=I=-14" in fc
    assert fc.strip().endswith("[aout]")


def test_audio_fc_sfx_without_music_indexes_correctly():
    fc = _audio_filter_complex(n=2, audio_dur=35.0, has_music=False,
                               sfx_events=[(0.03, 0.9)])
    assert "[3:a]" in fc                       # first sfx right after voice [2:a]
    assert "[4:a]" not in fc


def test_audio_simple_voice_only_maps_aout():
    assert _audio_filter_simple(2, 30.0, has_music=False) == "[2:a]anull[aout]"


def test_audio_simple_with_music_uses_static_volume():
    fc = _audio_filter_simple(2, 30.0, has_music=True)
    assert "volume=0.14" in fc
    assert "amix=inputs=2" in fc
    assert "[aout]" in fc


# ── SFX synthesis commands ───────────────────────────────────────────────────────

def test_sfx_cmds_use_lavfi_synthesis():
    from pathlib import Path
    hit    = _sfx_cmd("hit",    Path("/tmp/hit.wav"))
    whoosh = _sfx_cmd("whoosh", Path("/tmp/whoosh.wav"))
    riser  = _sfx_cmd("riser",  Path("/tmp/riser.wav"))
    assert "sine=frequency=52:duration=0.45" in " ".join(hit)
    assert "anoisesrc=colour=pink" in " ".join(whoosh)
    assert "anoisesrc=colour=white" in " ".join(riser)
    assert _sfx_cmd("nope", Path("/tmp/x.wav")) is None
