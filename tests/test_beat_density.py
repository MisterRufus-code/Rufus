"""How many pictures a script becomes, and where each one is cut.

THE COMPLAINT THIS ANSWERS, in the owner's words: "the code sends only 10
requests to the z image model, I want a lot more in one video, and perfect
matching the text to scene."

Both halves matter and they are different problems. More pictures is the beat
count. Matching them to the words is where the beats are cut and where the
renderer changes picture. Rendering one prompt four times — which is what
RUFUS_FRAMES_PER_BEAT does — answers the first and not the second: four
variations of one description are still one description.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main  # noqa: E402

_REAL = ("In 1849, James Marshall knelt by the American River, California, "
         "sifting gravel for gold flecks. The news sparked a frenzy; miners "
         "flooded in, each dreaming of riches. But most found the goldfields "
         "far harsher than promised. Within a few years, gold supplies "
         "dwindled. Marshall and countless others left empty-handed, their "
         "dreams crushed. The free-for-all feeling turned out to be a lie. "
         "The real winners? Merchants who sold shovels and supplies. The "
         "truth hurts: chasing shiny dreams often leaves you poorer. Will you "
         "ignore the lessons from those broken miners?")


# ── how many ────────────────────────────────────────────────────────────────

def test_a_real_script_becomes_about_twenty_pictures(monkeypatch):
    """Ten sentences used to mean ten pictures, whatever anyone set."""
    monkeypatch.delenv("SD_CLIPS", raising=False)
    target = main._target_beats(_REAL)
    assert 18 <= target <= 26, target
    assert len(main._split_beats(_REAL, max_scenes=target, grow=True)) == target


def test_sd_clips_still_overrides(monkeypatch):
    monkeypatch.setenv("SD_CLIPS", "12")
    assert main._target_beats(_REAL) == 12


def test_a_junk_sd_clips_is_loud_and_ignored(monkeypatch, capsys):
    monkeypatch.setenv("SD_CLIPS", "lots")
    assert main._target_beats(_REAL) >= 10
    assert "not a number" in capsys.readouterr().out


def test_a_tiny_script_does_not_become_three_pictures(monkeypatch):
    monkeypatch.delenv("SD_CLIPS", raising=False)
    assert main._target_beats("Short one. Very short.") == 10


def test_a_long_script_is_capped(monkeypatch):
    """Past thirty the storyboard call loses the thread and the GPU bill stops
    being worth it."""
    monkeypatch.delenv("SD_CLIPS", raising=False)
    assert main._target_beats("word " * 400) == 30


# ── growth is opt-in ────────────────────────────────────────────────────────

def test_max_scenes_is_still_only_a_ceiling_by_default():
    """Every existing caller passes max_scenes meaning "no more than this".
    Turning it into a target by default would split a three-sentence script
    into ten fragments for all of them."""
    script = ("First sentence here now. Second sentence follows it. "
              "Third one closes the whole thing.")
    assert len(main._split_beats(script, max_scenes=10)) == 3
    assert len(main._split_beats(script, max_scenes=10, grow=True)) > 3


# ── where the cut lands ─────────────────────────────────────────────────────

def test_a_place_name_is_never_torn_in_half():
    """"knelt by the American River," / "California, sifting gravel" splits a
    single place into two beats, and the second one is set nowhere."""
    beats = main._split_beats(_REAL, max_scenes=26, grow=True)
    for b in beats:
        assert not b.rstrip().endswith("American River,"), b
        assert not b.startswith("California"), b
        assert not b.startswith("Reading "), b


def test_a_beat_never_ends_on_a_dangling_article():
    """"But most found the" points at nothing. Splitting before a phrase
    starter rather than merely near the middle is what fixes it."""
    for b in main._split_beats(_REAL, max_scenes=26, grow=True):
        assert b.split()[-1].lower() not in ("the", "a", "an", "of", "and",
                                             "to", "in", "on", "for"), b


def test_every_beat_is_long_enough_to_be_a_picture():
    for b in main._split_beats(_REAL, max_scenes=26, grow=True):
        assert len(b.split()) >= main._MIN_BEAT_WORDS, b


def test_beats_still_reconstruct_the_script():
    """Order is what lets clip[i] line up with beat[i] at render time. A
    dropped or reordered word there narrates every later picture wrong."""
    beats = main._split_beats(_REAL, max_scenes=26, grow=True)
    assert " ".join(" ".join(beats).split()) == " ".join(_REAL.split())


# ── the renderer changes picture where the voice pauses ─────────────────────

def test_cuts_can_snap_to_clause_ends_when_there_are_not_enough_sentences():
    """With a picture every four words there are twice as many cuts as
    sentences, so cuts fell back to an even grid and landed mid-phrase — which
    is the "image doesn't match what he's saying" half of the complaint."""
    import audio_gen

    class _W:
        def __init__(self, word, end):
            self.word, self.end, self.start = word, end, end - 0.3

    class _Seg:
        def __init__(self, words):
            self.words = words

    seg = _Seg([_W("gold,", 2.0), _W("silver;", 4.0), _W("done.", 6.0),
                _W("next", 8.0), _W("line—", 10.0)])
    assert audio_gen._sentence_ends([seg]) == [6.0]
    assert audio_gen._clause_ends([seg]) == [2.0, 4.0, 10.0]


def test_the_renderer_only_reaches_for_clause_ends_when_it_needs_them():
    """Sentence ends are the better boundary; clauses are the top-up."""
    src = Path(__import__("audio_gen").__file__).read_text(encoding="utf-8")
    assert "if n > len(_snap_points) + 1:" in src
