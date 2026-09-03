"""Animate the one beat that is actually a scene.

A video model is good at "render this one concrete action" and bad at
"illustrate this abstraction". Handed "two percent of Europe's GDP" it produces
a slow drift over generic scenery; handed "in 1523 a hand slides a sealed letter
across an oak counter" it produces a shot.

THE SCENE is, by construction, the only beat in a script that is already a
motion prompt — every other beat is evidence. So hero mode spends motion time
on that beat and leaves the rest as cut stills: ~5 minutes instead of ~3 hours
on this hardware, and one moving shot among stills reads as a deliberate accent
rather than wallpaper the viewer stops noticing by beat three.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client as cc


SCENE = "In 1523 Jakob Fugger sent Charles V a letter demanding repayment of the loan."

PROMPTS = [
    "A wide view of a bustling market square at dawn, stalls being opened.",
    "A merchant's hands seal a letter with wax on an oak counter, then slide "
    "it across to a waiting courier.",
    "A vast empty landscape under a grey sky.",
]


def test_the_beat_carrying_the_scene_is_chosen():
    assert cc._hero_beat(PROMPTS, SCENE) == 1


def test_no_scene_means_no_hero():
    """The architect returns NONE when a source has no filmable moment. The run
    is then exactly the stills run the pipeline already produces."""
    assert cc._hero_beat(PROMPTS, "") is None


def test_no_prompts_is_not_an_error():
    assert cc._hero_beat([], SCENE) is None


def test_a_scene_matching_nothing_declines():
    assert cc._hero_beat(
        ["A vast empty landscape.", "A grey sky over water."],
        "In 1602 the company chartered its first fleet of spice ships.") is None


def test_a_tie_declines_rather_than_guessing():
    """What separates signal from coincidence is whether ONE beat leads. A word
    in every shot is not discriminating; picking arbitrarily would animate the
    wrong beat and cost the whole motion budget on it."""
    assert cc._hero_beat(
        ["Hands counting coins on a counter.",
         "Hands counting coins by a window."],
        "A clerk sat counting coins all afternoon.") is None


def test_one_shared_word_is_enough_when_only_one_beat_has_it():
    """A scene and its shot describe one moment in different words — "sent a
    letter" against "seals a letter, then slides it across" shares exactly one.
    Demanding two would decline the correct beat."""
    assert cc._hero_beat(
        ["A market square at dawn.",
         "A hand seals a letter with wax.",
         "A grey sky over water."],
        "In 1523 he sent Charles V a letter.") == 1


def test_abstractions_shared_by_both_do_not_count_as_a_match():
    """'power' and 'history' in both is two abstractions, not a beat about the
    moment — the same filtering storyboard._is_a_thing already applies."""
    assert cc._hero_beat(
        ["A landscape suggesting power and history and economy."],
        "The power and history of the economy over time.") is None


def test_hero_is_a_recognised_mode():
    assert "hero" in cc.BEAT_MOTION_MODES


def test_hero_mode_resolves_the_motion_chain(monkeypatch):
    """The other modes treat frames_per_beat > 1 as 'motion off'. Hero is the
    one mode where both are true at once: cuts on eight beats, a motion clip on
    the ninth."""
    monkeypatch.setenv("RUFUS_BEAT_MOTION", "hero")
    assert cc._beat_motion() == "hero"


def test_hero_defaults_to_the_shorter_clip():
    """61 frames rather than 121 is what puts one clip near 5 minutes. Every
    engine freeze-extends to fill the beat, so this shortens the GENERATION,
    not the beat."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert 'setdefault("RUFUS_HUNYUAN_FRAMES", "61")' in src
    assert (61 - 1) % 4 == 0, "Hunyuan's 3D causal VAE requires 4n+1 frames"


def test_an_owner_override_of_frames_wins():
    """setdefault, not assignment — an explicit RUFUS_HUNYUAN_FRAMES must not
    be silently overwritten by the mode."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert 'os.environ["RUFUS_HUNYUAN_FRAMES"]' not in src


def test_only_the_hero_beat_runs_the_chain():
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert "beat_engines = motion_engines if i == hero_i else []" in src
    assert "for eng_name, animate in beat_engines:" in src


def test_the_hero_beat_is_prompted_with_the_scene():
    """THE SCENE names the action; the image prompt names the composition. The
    action is the better motion prompt, and is why the beat was chosen."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert "motion_text = hero_scene" in src


def test_the_clip_duration_is_printed():
    """Measured into motion_log since forever but never shown, so nobody could
    tell a 5-minute clip from a 20-minute one without opening the run report."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert "motion_secs" in src
    assert "{took}" in src or "took" in src


def test_a_ready_but_switched_off_engine_says_so(capsys):
    """'off — disabled (RUFUS_STILLS_ONLY=1)' reads the same whether an engine
    is unusable or one env var from working. On this box Hunyuan's template is
    committed and its models are installed, and that line printed for weeks."""
    class _Ready:
        @staticmethod
        def ready():
            return True, "Hunyuan template loaded"

    cc._say_if_ready_but_switched_off("hunyuan 1.5", _Ready)
    out = capsys.readouterr().out
    assert "template IS exported" in out
    assert "RUFUS_STILLS_ONLY=0" in out


def test_a_genuinely_unavailable_engine_stays_quiet(capsys):
    class _NotReady:
        @staticmethod
        def ready():
            return False, "no API export"

    cc._say_if_ready_but_switched_off("hunyuan 1.5", _NotReady)
    assert capsys.readouterr().out == ""


def test_a_raising_client_stays_quiet(capsys):
    class _Broken:
        @staticmethod
        def ready():
            raise RuntimeError("ComfyUI down")

    cc._say_if_ready_but_switched_off("hunyuan 1.5", _Broken)
    assert capsys.readouterr().out == ""


# ── how many stills the eight non-hero beats get ─────────────────────────────
#
# Once the hero beat is the only motion clip, the stills phase IS the run. The
# owner's ComfyUI queue shows 12-14 seconds per still, so 9 beats x 3 = 27
# stills is about six minutes before any motion starts.

def test_the_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("RUFUS_HERO_OTHER_FRAMES", raising=False)
    assert cc._hero_other_frames() == cc.HERO_OTHER_FRAMES == 3


def test_one_still_per_beat_can_be_asked_for(monkeypatch):
    """27 stills -> 9. Costs the hard-cut inside each narration line, which is
    the trade the owner should get to make per run rather than per release."""
    monkeypatch.setenv("RUFUS_HERO_OTHER_FRAMES", "1")
    assert cc._hero_other_frames() == 1


def test_zero_and_negative_cannot_produce_a_beat_with_no_picture(monkeypatch):
    """A beat still needs one image to animate or hold. Clamping beats raising:
    a silly value should cost a little quality, never a whole run."""
    for bad in ("0", "-4"):
        monkeypatch.setenv("RUFUS_HERO_OTHER_FRAMES", bad)
        assert cc._hero_other_frames() == 1


def test_junk_falls_back_loudly(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_HERO_OTHER_FRAMES", "three")
    assert cc._hero_other_frames() == cc.HERO_OTHER_FRAMES
    assert "not a number" in capsys.readouterr().out


def test_the_hero_branch_reads_the_override_not_the_constant():
    """The constant was the value in use for as long as hero mode existed; the
    env var only helps if the branch actually calls the resolver."""
    import inspect
    src = inspect.getsource(cc.generate_clips)
    assert "_hero_other_frames()" in src
    assert "= HERO_OTHER_FRAMES" not in src
