#!/usr/bin/env python3
"""
gallery_variants.py — two complete galleries for one script, so a person picks.

WHY A WHOLE GALLERY AND NOT A SAMPLE. What is being chosen is the set of
pictures this video ships with, sixteen or so of them, in beat order. A
three-image probe answers "which look" — a different question, answered rarely,
and config/styles.json already owns it. This answers "which draws came out",
which has to be asked of the real prompts because that is where the empty faces
and the blank sheets actually happen.

WHY THE SHOT IS THE UNIT OF CHOICE. A gallery of sixteen pictures is sixteen
independent draws, not one artefact. Variant A comes back best on shot 3 and
worst on shot 9; variant B the other way round. Picking a whole bundle throws
away the good half of the other one. So a set is chosen as a BASE in one click
and then corrected shot by shot — one judgement plus a handful, instead of
sixteen.

WHY TWO AND NOT THREE. At a defect rate around one in five, the chance both
variants fail the same shot is about four per cent — a sixteen-shot set expects
well under one unfixable shot. A third variant takes that under a sixth of a
shot and costs another thirteen minutes of the 3090. Two draws with a swap
already kill the great majority of what a third would.

THE PROMPTS ARE BUILT ONCE AND STORED. The storyboard is a model call and does
not repeat itself, so rebuilding the prompts at render time would mean the
pictures a person chose were drawn for a different set of beats than the ones
that ship. Every image row carries the prompt it came from, and the run that
uses the set reads those rather than planning again.

    RUFUS_GALLERY_VARIANTS  2   how many complete galleries to draw
"""

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paths

DEFAULT_VARIANTS = 2


def how_many() -> int:
    try:
        return max(1, int(os.environ.get("RUFUS_GALLERY_VARIANTS",
                                         DEFAULT_VARIANTS)))
    except ValueError:
        return DEFAULT_VARIANTS


def gallery_dir(set_id: int) -> Path:
    return paths.media_root() / "galleries" / str(set_id)


def build(script_file: str, *, candidate_id: int | None = None,
          niche: str | None = None, channel: str | None = None,
          topic: str = "", n_variants: int | None = None,
          with_voice: bool = True) -> int | None:
    """Draw `n_variants` complete galleries for the script. Returns the set id.

    Fail-open per image, like every other render loop here: a beat that will
    not draw in one variant leaves that variant short and the other one still
    covers the shot. A set with a hole in BOTH variants is the caller's problem
    to see, and chosen_gallery refuses to hand back a short list for exactly
    that reason.
    """
    import comfy_client
    import db_manager
    import main as rufus_main

    script = Path(script_file).read_text(encoding="utf-8")
    if niche is None:
        try:
            import research
            niche = research._load_niche()[1]
        except Exception:
            niche = "money_history"
    if channel is None:
        try:
            from channel_config import load_channel
            channel = load_channel().id
        except Exception:
            channel = "main_en"

    n_variants = n_variants or how_many()
    beats = rufus_main._target_beats(script)

    set_id = db_manager.save_gallery_set(
        candidate_id=candidate_id, channel=channel, niche=niche,
        topic=topic or (script.strip().split("\n")[0][:120]),
        script_file=str(script_file), n_variants=n_variants)
    out_dir = gallery_dir(set_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # THE VOICE FIRST, AND THE PAGE ALREADY SAID SO. Shot lengths come from
    # Whisper reading real audio; until a take exists there is nothing to
    # measure, and the gallery stage was promising "shot 3, 4.2s" beside every
    # picture while nothing had been recorded to measure. The takes are cheap
    # and quick next to forty minutes of drawing, and doing them here rather
    # than at the voice stage is the whole reason the ordering was changed.
    #
    # Fail-open: a TTS backend that will not start costs the shot lengths, not
    # the pictures. The voice stage will record them later and the render still
    # works — you simply choose images without knowing how long each is up.
    spoken: list[dict] = []
    if with_voice:
        try:
            import voice_takes
            print(f"[galleries] recording the voice first, so the shot "
                  f"lengths are measured rather than guessed")
            takes = voice_takes.build(str(script_file), set_id=set_id,
                                      topic=topic, n_beats=beats)
            # THE WORDS UNDER EACH SHOT, from the take we just recorded.
            # Prompts planned from a split of the script text describe beat i
            # of the TEXT; the renderer cuts beat i of the AUDIO. Nothing made
            # those agree, so a picture could be drawn for a sentence that is
            # not the one playing over it. Reading the words back off the
            # audio removes the guess entirely.
            if takes:
                import beat_timing
                spoken = beat_timing.spoken_shots(takes[0]["path"], beats)
                # HOLD RATHER THAN CUT, where two shots are about the same
                # thing. "Many images repeat one after another — not the same
                # image, but a new one very similar to the previous": of
                # course they did. Two consecutive sentences about the same
                # bank were two prompts about the same bank, drawn from noise
                # twice, with a cut between them the narration never asked
                # for. One picture across both stops the near-duplicates, gives
                # the rest more time to be read, and draws fewer of them.
                spoken = beat_timing.merge_same_subject(spoken)
        except Exception as e:
            print(f"[galleries] no voice takes ({e}) — the pictures will be "
                  f"drawn from a split of the script instead, and without "
                  f"shot lengths beside them")

    said = [r["text"] for r in spoken if r.get("text")]
    if spoken and len(said) == len(spoken):
        held = sum(1 for r in spoken if int(r.get("held", 1)) > 1)
        print(f"[galleries] planning {len(spoken)} shot(s) from the words "
              f"actually spoken over each one"
              + (f", {held} of them held across more than one sentence"
                 if held else ""))
        # max_scenes follows the MERGED count. Leaving it at the original beat
        # count would let the prompt builder pad back to a number the audio no
        # longer has cuts for, which is the duplicate problem reappearing one
        # layer down.
        prompts = rufus_main._build_sd_prompts(script, niche,
                                               max_scenes=len(said),
                                               beats=said)
    else:
        # A shot sitting in a pause has no words to draw from, and a partial
        # list would silently slide every later picture onto the wrong window
        # — worse than the text split this falls back to.
        if spoken:
            print(f"[galleries] {len(spoken) - len(said)} shot(s) have no "
                  f"words under them — falling back to the script split")
        prompts = rufus_main._build_sd_prompts(script, niche, max_scenes=beats)
    if not prompts:
        print("[galleries] no prompts were planned — nothing to draw")
        return None

    # The take's stored spans were measured against the UNMERGED count. Once
    # shots are held, the page would show a shot list one length and a picture
    # list another — so the merged spans replace them, and the seconds beside
    # each picture describe the picture that is actually there.
    if spoken and len(said) == len(spoken):
        try:
            import json as _json
            import db_manager as _db
            with _db._conn() as _c:
                _c.execute("UPDATE voice_takes SET spans=? WHERE set_id=?",
                           (_json.dumps(spoken), int(set_id)))
        except Exception as e:
            print(f"[galleries] could not store the merged shot lengths ({e})")

    # The target, recorded before the first picture: the progress panel reads
    # it rather than counting what has arrived, so a set with nothing drawn yet
    # reads as 0 of 32 instead of as finished.
    db_manager.set_gallery_beats(set_id, len(prompts))

    # THE LOOK, APPENDED HERE BECAUSE render_one_beat DOES NOT APPEND IT.
    # The ordinary render loop styles its prompts (`prompts = [_with_detail(p)
    # for p in prompts]`) and then renders them; render_one_beat deliberately
    # does not, because its other caller is the regenerate button, which is fed
    # a sidecar that ALREADY carries the style — styling there would give that
    # prompt a second helping of it.
    #
    # This stage was reading prompts straight out of _build_sd_prompts, which
    # are unstyled, and handing them to render_one_beat, which adds nothing. So
    # every picture in every gallery was drawn with no style block at all: the
    # checkpoint's own look, which is photographic, on a channel whose style is
    # stickman_micro. Built, wired, tested against a mocked renderer that never
    # looks at the prompt — and never actually compared against what the real
    # loop sends.
    #
    # Styled ONCE for both variants: _detail_suffix re-reads config/styles.json
    # on every call, and the block is identical across variants by definition —
    # a look that differed between A and B would make the comparison a
    # comparison of two styles rather than two draws.
    styled = [comfy_client._with_detail(p) for p in prompts]

    print(f"[galleries] set #{set_id}: {n_variants} × {len(prompts)} "
          f"picture(s) → {out_dir}")

    drawn = 0
    for variant in range(n_variants):
        # ONE BASE SEED PER VARIANT, offset by beat. Two variants that shared a
        # seed would draw the same picture twice and the choice would be
        # between a thing and itself; a seed re-rolled per image would make the
        # variant meaningless as a unit, which is what the base click selects.
        base_seed = random.randint(1, 2**31 - 1)
        print(f"[galleries] variant {variant} — base_seed={base_seed}")
        for i, prompt in enumerate(prompts):
            png = out_dir / f"v{variant}_{i:02d}.png"
            seed = (base_seed + i) % (2**31 - 1)
            ok = comfy_client.render_one_beat(styled[i], png, seed=seed,
                                              niche=niche)
            if not ok:
                print(f"[galleries] variant {variant} beat {i}: no image — "
                      f"the other variant still covers this shot")
                continue
            # THE SHOT IS STORED, NOT THE STYLED PROMPT. This row is read
            # in two places and neither wants the style block: the page prints
            # it under the picture, where six hundred identical words would
            # push the one line that differs off the end; and prompts_of feeds
            # the captions and the image-prompt history, where a constant tail
            # on every entry defeats the de-duplication it exists for.
            db_manager.save_gallery_image(
                set_id=set_id, variant=variant, beat_index=i, path=str(png),
                prompt=prompt, seed=seed)
            drawn += 1

    print(f"[galleries] set #{set_id}: {drawn} picture(s) drawn — "
          f"choose at /galleries")
    return set_id


def clips_from(set_id: int, clip_duration: float = 8.0) -> list[Path]:
    """The chosen picture per beat, animated into clips the renderer can use.

    THE HANDOFF BACK INTO THE ORDINARY PIPELINE. Everything downstream of
    generate_clips expects a list of mp4s where clip[i] belongs to beat[i], so
    that is what a chosen set becomes — via the same Ken Burns step the normal
    path uses for a still, rather than a second animation route that would
    drift from it.

    Returns [] when the set has a beat with nothing chosen. A short list would
    silently slide every later picture onto the wrong sentence, which is the
    one failure this whole stage exists to prevent.
    """
    import db_manager
    from sd_client import _animate_to_clip

    rows = db_manager.chosen_gallery(set_id)
    if not rows:
        print(f"[galleries] set #{set_id} has no complete choice — every beat "
              f"needs a picked picture before it can be rendered")
        return []

    out_dir = gallery_dir(set_id) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for row in rows:
        png = Path(row["path"])
        if not png.exists():
            print(f"[galleries] beat {row['beat_index']}: {png} is gone")
            return []
        clip = out_dir / f"{row['beat_index']:02d}.mp4"
        if not _animate_to_clip(png, clip, duration=clip_duration,
                                idx=row["beat_index"]):
            print(f"[galleries] beat {row['beat_index']}: could not animate")
            return []
        clips.append(clip)
    print(f"[galleries] set #{set_id}: {len(clips)} chosen clip(s) ready")
    return clips


def prompts_of(set_id: int) -> list[str]:
    """The prompts the chosen pictures were actually drawn from.

    The storyboard is a model call and does not repeat itself. A run that
    rebuilt its prompts would be describing different beats than the ones a
    person looked at, so it reads them back instead of planning again.
    """
    import db_manager
    rows = db_manager.chosen_gallery(set_id)
    return [r["prompt"] or "" for r in rows]


if __name__ == "__main__":
    # THE SCHEMA, BEFORE ANYTHING TRIES TO WRITE TO IT. The dashboard calls
    # init_db at startup and every test fixture calls it too, so every path
    # that had ever been exercised already had the tables — and the one path
    # nobody had run, the command line, died on "no such table" after paying
    # for a script. Built, tested, and never actually run, which is this
    # repo's oldest bug wearing a new hat.
    import argparse
    import db_manager
    db_manager.init_db()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("script_file")
    ap.add_argument("--candidate", type=int, default=None)
    ap.add_argument("--topic", default="")
    ap.add_argument("--variants", type=int, default=None)
    ap.add_argument("--no-voice", action="store_true",
                    help="skip the takes; the pictures get no shot lengths")
    a = ap.parse_args()
    build(a.script_file, candidate_id=a.candidate, topic=a.topic,
          n_variants=a.variants, with_voice=not a.no_voice)
