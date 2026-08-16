#!/usr/bin/env python3
"""
storyboard.py — plan the pictures WITH the script, as one continuous scene.

THE PROBLEM THIS SOLVES. Today the two halves are strangers. script_writer
finishes a script; main._split_beats chops it into sentences; then a separate
model reads those ten sentences cold and illustrates each one on its own. It
has never seen the story — only a list of lines — so it decorates each line
independently and the result is ten unrelated pictures that happen to share a
colour palette.

That is exactly what the owner reported, and the last run shows it plainly.
The script's beat 2 was about the denarius holding 4.5 grams of silver. The
image generated for it: "a family gathered around a modest dinner table,
sharing a simple meal of bread and vegetables". Not wrong, not connected. Beat
8 became "a concerned modern-day person at a kitchen table with financial
documents" — the stock-photo of an idea rather than a moment in this story.

A storyboard fixes it by construction. One pass sees the WHOLE script and
plans the pictures as a sequence, where each shot may carry something forward
from the one before it — the same coin, thinner; the same table, emptier; the
same hand, now closing on nothing. That continuity is what makes a Short feel
authored instead of assembled, and it cannot be produced one sentence at a
time by a model that has not seen the others.

CONTRACT: fail-open. No key, no reply, bad JSON, wrong shot count — all return
None, and main falls back to the per-beat prompt writer it uses today. A
storyboard is a better way to get the pictures, never a prerequisite for
getting them.
"""

import json
import os
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"

MODEL_DEFAULT = "gpt-4o"          # the arc is the point; a mini model loses it
MIN_VISUAL_CHARS = 40


def enabled() -> bool:
    return os.environ.get("RUFUS_STORYBOARD", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _model() -> str:
    return os.environ.get("RUFUS_STORYBOARD_MODEL", MODEL_DEFAULT).strip() \
        or MODEL_DEFAULT


def _direction_clause() -> str:
    """The owner's standing direction, for the pictures.

    Half of what the owner writes as direction is about images — smooth, related
    to the line, one shot that moves — so limiting it to the script writer would
    silently drop that half. Loaded from script_writer so there is exactly one
    reader of those files and one place the layering rules live; any failure
    yields "" and this prompt is what it was.
    """
    try:
        import script_writer
        text, _note = script_writer.load_direction()
    except Exception:
        return ""
    return (f"\nCHANNEL DIRECTION (the owner's standing instructions — they "
            f"outrank the numbered rules above where they disagree, except on "
            f"anything the rules call a previous run's mistake):\n{text}\n"
            if text else "")


def _prompt(script: str, beats: list[str], era_tags: list[str],
            character_clause: str = "", scene: str = "") -> str:
    numbered = "\n".join(
        f"{i + 1}. [{era_tags[i] if i < len(era_tags) else 'present day'}] {b}"
        for i, b in enumerate(beats))
    return (
        "You are the storyboard artist for a 40-second vertical documentary "
        "Short. You are given the FINISHED narration. Your job is to plan the "
        "pictures as ONE CONTINUOUS SEQUENCE, not to illustrate each sentence "
        "on its own.\n\n"
        f"FULL SCRIPT (read it all before you draw anything):\n{script}\n\n"
        + (f"THE MOMENT THIS SCRIPT WAS BUILT ON:\n{scene}\n"
           "That moment is the anchor for the whole sequence. Whichever beat "
           "lands on it gets the strongest, most literal shot in the video, and "
           "the `setting` below should be the place it happens in. Everything "
           "else is that place before, during or after.\n\n" if scene else "")
        + f"THE {len(beats)} SHOTS, with the era each one is set in:\n{numbered}\n\n"
        "WHAT MAKES THIS A STORYBOARD AND NOT A SLIDESHOW:\n"
        "1. SHOW THE LITERAL THING THE LINE NAMES. If the line says the coin "
        "held four and a half grams of silver, the shot is THAT COIN — not a "
        "family at dinner, not a person looking thoughtful. A picture of the "
        "general topic instead of the specific sentence is the single most "
        "common way this goes wrong.\n"
        "2. CARRY A PHYSICAL OBJECT FORWARD — NEVER A MOOD. `carries_over` "
        "must name a THING you could point a camera at: the same coin now "
        "thinner, the same lantern, the same wooden table, the same coat. "
        "It must NOT be a feeling or an idea. These are all real answers a "
        "previous run gave, and every one of them is WRONG: \"emptiness and "
        "desolation\", \"emptiness and chaos\", \"sense of despair and loss\", "
        "\"unresolved financial burden\", \"ongoing neglect\", \"threat of "
        "repeating past mistakes\". Carrying a mood forward does not connect "
        "the shots — it REPEATS them. \"Emptiness\" four beats running renders "
        "four empty rooms, and the sequence looks like the same picture over "
        "and over. Objects can travel through a story; feelings can only be "
        "restated. Use null when nothing physical genuinely carries over — "
        "that is far better than naming a mood.\n"
        "3. ONE SUBJECT PER SHOT. A frame with a coin AND a merchant AND a "
        "market AND a ledger has no subject. Choose the one thing the line is "
        "actually about and fill the frame with it.\n"
        "3a. THE SUBJECT MUST CHANGE ACROSS THE SEQUENCE. Rule 1 says draw the "
        "thing the line names, and a script about money names money in most of "
        "its lines — so obeying rule 1 alone produces ten pictures of a coin, "
        "which is the single most common complaint about this channel's "
        "videos. No object may be the subject of more than about a THIRD of "
        "the shots. What a line gives you besides its object: the PERSON doing "
        "it, their hands mid-action, their face, the place around them, what "
        "they were doing a second before, the thing that happened next, the "
        "person it happened TO. \"A hand pushes a coin across the counter\" is "
        "about the hand. \"The queue outside has not moved\" is about the "
        "queue.\n"
        "3b. WHEN THE SAME OBJECT MUST APPEAR TWICE, SHOW A DIFFERENT VERB. "
        "The picture is the ACTION, not the noun: minted, weighed, bitten, "
        "hidden, spent, refused, dropped, swept up. A coin lying on a surface, "
        "then lying on a different surface, is one picture twice however "
        "different the surfaces are — nothing is happening in either. If you "
        "cannot say what is being DONE in a shot, it is not a shot yet.\n"
        "3c. PUT PEOPLE IN IT. At least half the shots must show a person "
        "DOING something — hands counting coins, a man locking a door, a woman "
        "carrying a crate, a queue that does not move. The same previous run "
        "returned eight shots out of ten with no one in them (\"devoid of "
        "people\", \"absence of workers\", \"abandoned\", \"unused\", "
        "\"barren\") for a script about one in four people losing their job. "
        "Empty rooms are the cheapest way to say a thing is bad, and they make "
        "a viewer feel nothing. A face or a pair of hands is what makes a "
        "number land.\n"
        "4. LET THE FRAME CARRY THE FEELING, not a described emotion. Not "
        "'revealing the anguish of misplaced trust' — a fist of coins held out "
        "and no one reaching. A feeling named as a feeling gives the image "
        "model nothing to draw.\n"
        "4b. DRAW THE FACE AS GEOMETRY, IN THESE EXACT WORDS. A face is the "
        "fastest way a viewer feels a line, and the renderer can only draw "
        "five of them. Every shot with a person picks ONE and writes it out:\n"
        "    - brows angled down and inward over a flat or down-curved mouth "
        " (anger, defiance, resolve)\n"
        "    - brows raised high over a small open oval mouth  (shock, alarm, "
        "disbelief)\n"
        "    - brows slanted up at their inner ends over a downward mouth "
        "curve  (worry, grief, resignation)\n"
        "    - a wide upturned curve with the eyes squeezed to short arcs  "
        "(relief, triumph, delight)\n"
        "    - brows flat and mouth a short straight line  (cold "
        "indifference, official detachment)\n"
        "Never 'looking determined', 'a face showing resignation', 'a look of "
        "loss on their face' — those name the feeling instead of drawing it, "
        "which is rule 4 in a costume, and the renderer answers every one of "
        "them with the SAME neutral worried face. A gallery of sixty stills "
        "from this channel had that face on nearly every figure.\n"
        "4c. THE FACES MUST DIFFER ACROSS THE SEQUENCE. Use at least three of "
        "the five in any sequence with people in it, and put them where the "
        "narration turns: the shock belongs on the reveal, the flat official "
        "face on the announcement, the grief after the loss. A sequence where "
        "every face does the same thing is a sequence with no story in it, "
        "whatever the pictures behind the figures are doing. The posture "
        "carries it too — shoulders thrown up, back hunched, head dropped, "
        "one arm reaching, one figure leaning away from another.\n"
        "5. OBEY THE ERA TAG on each shot. [present day] means an ordinary "
        "scene of today; a year means every visible detail belongs to it.\n"
        "6. NEVER NAME WORDS THAT WOULD BE PRINTED IN FRAME. No headline text, "
        "no inscriptions, no dates on objects — the image model garbles "
        "lettering and it is the clearest sign a machine made this. Write the "
        "object as a blank physical thing instead.\n"
        "6b. THE LAST LINE IS A SIGN-OFF, NOT A SCENE. The narration ends on a "
        "fixed channel line — \"Part of a series on how money really began\", "
        "\"Follow for the real history of money\". It is spoken, so it needs a "
        "picture, but it is ABOUT the channel and not about the story, and a "
        "run that took it literally returned \"a banner reading 'Part of a "
        "series'\" and \"a book titled 'How Money Really Began'\" — two frames "
        "of garbled lettering to close on. Give those beats the last image of "
        "the STORY instead: the object one final time, the place empty, the "
        "hand closing. Never a banner, a book cover, a title card or a logo.\n"
        "7. DECIDE THE PLACE BEFORE YOU DRAW ANY SHOT. `setting` names the ONE "
        "location this sequence lives in, described by its physical anchors — "
        "the surface, the walls, the light and where it comes from. \"A low "
        "stone counting-house: oak counter, lime-washed walls, one high window "
        "throwing hard light from the left\" is right. \"Renaissance Europe\" "
        "and \"a place of financial power\" are wrong: they are labels, not "
        "rooms, and every shot invents its own building from them. The point "
        "of naming the anchors is that they RECUR — the same counter, the same "
        "wall, the same light direction — so consecutive shots read as one "
        "place seen from different angles rather than ten unrelated buildings. "
        "Shots that leave it (a street, the present day) are fine and expected; "
        "they just have to be a deliberate departure from somewhere, and the "
        "sequence should come back.\n"
        "8. THE LIGHT IS PART OF THE PLACE. Whatever direction and quality you "
        "give the setting, keep it in every shot set there. Light that jumps "
        "from window-left to firelit-right between two shots of the same room "
        "is the fastest way to make one location look like two.\n"
        f"{character_clause}"
        + _direction_clause()
        + "\nEach `visual` is 2-3 plain sentences describing only what the camera "
        "sees: the subject, what it is doing or how it sits, and the setting. "
        "No camera bodies, no lens specs, no style words — the renderer adds "
        "the house style itself.\n\n"
        "Reply with ONLY this JSON:\n"
        '{"setting": "<the one place, named by its physical anchors: surface, '
        'walls, light source and direction. A room a camera could stand in, '
        'not a period or a theme>",\n'
        ' "through_line": "<the ONE physical object the sequence returns to — '
        'a thing, not a topic. \\"one coin, thinning\\" is right; \\"rising '
        'unemployment and its impact on society\\" is an essay title and is '
        'wrong>",\n'
        ' "shots": [{"n": 1, "visual": "...", "carries_over": null, '
        '"in_setting": true}, ...]}\n'
        f"Exactly {len(beats)} shots, n from 1 to {len(beats)}. Set "
        f"`in_setting` false only for a shot that deliberately leaves the "
        f"place."
    )


# Words that name a FEELING or an IDEA rather than a thing you could point a
# camera at. A carries_over made only of these is the failure this list exists
# to catch — see _is_a_thing.
_ABSTRACT = {
    "emptiness", "empty", "desolation", "desolate", "despair", "hope",
    "hopelessness", "chaos", "silence", "loss", "neglect", "burden", "threat",
    "disregard", "fear", "anxiety", "anguish", "grief", "sorrow", "tension",
    "uncertainty", "instability", "collapse", "decline", "crisis", "ruin",
    "poverty", "wealth", "struggle", "suffering", "hardship", "mood",
    "atmosphere", "sense", "feeling", "tone", "theme", "idea", "impact",
    "consequence", "aftermath", "legacy", "history", "past", "future",
    "unresolved", "ongoing", "looming", "foreboding", "repeating", "mistakes",
    "issues", "problems", "lessons", "change", "shift", "power", "greed",
    "value", "trust", "belief", "doubt", "time", "society", "economy",
    # Domain adjectives: they qualify a thing but never name one, so
    # "unresolved financial burden" must not read as concrete.
    "financial", "economic", "monetary", "social", "political",
    "historical", "personal", "public", "general", "widespread",
}
_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "from",
    "with", "for", "same", "previous", "shot", "still", "now", "its", "his",
    "her", "their", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "it", "as", "by", "more", "less", "one",
}


def _is_a_thing(carries: str) -> bool:
    """Whether `carries_over` names something a camera could point at.

    THE LIVE FAILURE, in full. Every carries_over in the Great Depression run
    was a mood: "emptiness and desolation", "emptiness and chaos", "sense of
    despair and loss", "unresolved financial burden", "ongoing neglect",
    "threat of repeating past mistakes". Not one named a physical object.

    That is worse than no continuity at all. An image model handed "carry
    emptiness forward" four beats running renders four empty rooms — the
    instruction to connect the shots becomes the instruction to repeat them,
    which is exactly the repetitiveness the owner reported. A thread has to be
    a THING: the same coin, the same table, the same lantern. Objects can move
    through a story; feelings can only be restated.

    Fail-open in spirit: a carries_over that fails this is dropped, and the
    shot itself is kept. A shot with no stated thread still renders fine."""
    words = [w for w in re.findall(r"[a-z]+", carries.lower())
             if w not in _STOPWORDS]
    concrete = [w for w in words if w not in _ABSTRACT]
    return bool(concrete)


# A setting has to be a room, not a label. "Renaissance Europe" and "a place of
# financial power" are the failure this guards: they name a period or a theme,
# so every shot invents its own building and the sequence looks like ten
# unrelated locations — the same mistake _is_a_thing catches for the object
# thread, one level up at the level of place.
_SETTING_MIN_WORDS = 6

# Physical anchors. A usable setting names at least one surface or structure and
# at least one light source, because those are what actually recur between
# shots — a room described only as "grand" gives the model nothing to repeat.
_SETTING_STRUCTURE = {
    "wall", "walls", "floor", "floors", "ceiling", "counter", "counters",
    "table", "tables", "desk", "desks", "bench", "benches",
    "door", "doors", "doorway", "window", "windows", "arch", "column", "beam",
    "stair", "stairs", "shelf", "shelves", "crate", "crates", "stall", "roof",
    "vault", "corridor", "hall", "room", "chamber", "yard", "courtyard",
    "quay", "dock", "street", "workshop", "foundry", "office", "warehouse",
    "cellar", "kitchen", "screen", "screens", "pillar", "railing", "gate",
    "stone", "oak", "timber", "brick", "bricks", "plaster", "tile", "marble",
    "glass", "concrete", "cobblestone", "cobblestones",
    # OUTDOORS. The first version of this set was entirely interiors and
    # masonry, and it threw away a perfectly good place on a live run:
    #
    #   "A rugged riverside area with dusty ground, scattered rocks, and a
    #    slow-moving section of the American River. Tall trees line the
    #    background, and sunlight filters through the branches, casting
    #    dappled light across the scene."
    #
    # Ground, rocks, river, trees, dappled light — that is a room a camera can
    # stand in by every standard this check was written to enforce, and it was
    # rejected for containing no wall. A money channel that covers gold
    # rushes, ports, mines and fields cannot have an indoor-only vocabulary.
    "ground", "earth", "dirt", "dust", "mud", "sand", "gravel", "grass",
    "rock", "rocks", "boulder", "boulders", "riverbank", "bank", "river",
    "stream", "creek", "water", "shore", "shoreline", "beach", "hill",
    "hillside", "ridge", "slope", "cliff", "valley", "mountain", "field",
    "fields", "path", "track", "tracks", "trail", "road", "bridge", "pier",
    "jetty", "wharf", "deck", "mast", "sail", "hull", "well", "pit", "mine",
    "shaft", "quarry", "furnace", "forge", "kiln", "market", "square",
    "platform", "steps", "porch", "awning", "canopy", "tent", "tents",
    "wagon", "wagons", "cart", "fence", "post", "posts", "tree", "trees",
    "trunk", "branch", "branches", "snow", "ice", "camp",
}
_SETTING_LIGHT = {
    "light", "lights", "lighting", "lit", "sunlight", "daylight", "lamplight",
    "candlelight", "firelight", "lantern", "lanterns", "candle", "candles",
    "lamp", "lamps", "window", "windows", "shadow", "shadows", "glow",
    "glowing", "dim", "bright", "overcast", "dusk", "dawn", "gloom", "sunlit",
    "backlit", "reflections", "gaslight", "torchlight", "moonlight", "neon",
}


def _in_setting_flags(raw: dict, n: int) -> list[bool]:
    """Per-shot `in_setting`, defaulting to True.

    Missing or malformed means "yes, this shot is in the room" — the common
    case by a wide margin, and the safe one: restating a place the shot was
    already in costs nothing, while omitting it is the failure this exists to
    stop.
    """
    shots = raw.get("shots")
    flags = [True] * n
    if not isinstance(shots, list):
        return flags
    for i, entry in enumerate(shots[:n]):
        if isinstance(entry, dict) and entry.get("in_setting") is False:
            flags[i] = False
    return flags


def _is_a_place(setting: str) -> bool:
    """Whether `setting` describes a room a camera could stand in.

    Two requirements, both learned from what goes wrong without them: a
    structure word, so there is a surface to return to, and a light word, so
    consecutive shots of the same place are lit the same way. A period name
    passes neither.

    Fail-open in spirit, exactly like _is_a_thing: a setting that fails this is
    dropped and the shots are kept. Shots with no shared place still render.
    """
    words = set(re.findall(r"[a-z]+", (setting or "").lower()))
    if len(re.findall(r"[a-z]+", (setting or "").lower())) < _SETTING_MIN_WORDS:
        return False
    return bool(words & _SETTING_STRUCTURE) and bool(words & _SETTING_LIGHT)


def _pin_setting(visual: str, setting: str) -> str:
    """Append the place to a shot that is meant to be set in it.

    Same reasoning as _pin_character: an image model renders each beat from
    noise with no memory of the others, so a shot that assumes the room without
    describing it gets a different room. Naming the place is not describing it.
    """
    if not setting:
        return visual

    # DEDUPED. The list form counted "street" twice in "A cobblestone street…
    # across the street", which raised the half-of-anchors bar to three and made
    # shots that clearly established the place fail it.
    anchors = {w for w in re.findall(r"[a-z]+", setting.lower())
               if w in _SETTING_STRUCTURE}
    light = {w for w in re.findall(r"[a-z]+", setting.lower())
             if w in _SETTING_LIGHT}
    # WHOLE WORDS. Substring containment made "oak" match inside "cloak", and
    # _pin_character runs first — so it injects "cloak" into exactly the
    # recurring-character shots, and every one of them then looked like it had
    # already established the room. Also "stone" inside "cobblestone", "arch"
    # inside "march".
    text = set(re.findall(r"[a-z]+", visual.lower()))

    # ANY structural anchor means the shot has already built the room. The
    # earlier rule wanted half of them and got the arithmetic wrong on top, so a
    # shot opening "workers huddle outside the Philadelphia and Reading Railroad
    # office… the closed wooden doors" still had the entire 40-word setting
    # appended — five times across one sequence, competing with each beat's own
    # description and the style suffix for the model's attention.
    if anchors & text:
        return visual

    # Otherwise pin it, compactly: two structural anchors and one light word,
    # taken in the order the SETTING names them so the phrasing tracks how the
    # room was described rather than the alphabet ("brick, office, light" for a
    # setting that opens on a cobblestone street reads like a different place).
    ordered = [w for w in re.findall(r"[a-z]+", setting.lower())]
    seen: set[str] = set()
    picked_anchor, picked_light = [], []
    for w in ordered:
        if w in seen:
            continue
        seen.add(w)
        if w in anchors and len(picked_anchor) < 2:
            picked_anchor.append(w)
        elif w in light and not picked_light:
            picked_light.append(w)
    parts = picked_anchor + picked_light
    if not parts:
        return visual
    return (f"{visual.rstrip().rstrip('.')}. Same place as the rest of the "
            f"sequence: {', '.join(parts)}.")


# How much of the character's look a shot must already restate before we accept
# it as self-sufficient. Calibrated on the Great Depression run's three
# Chronicler shots: 0.67 for the one that described him, 0.33 and 0.27 for the
# two that only named him — and those two are the ones that came back wrong.
_CHARACTER_ECHO_MIN = 0.5


def _restates_the_look(visual: str, short_ref: str) -> bool:
    """Whether this shot repeats enough of the character's appearance."""
    words = {w for w in re.findall(r"[a-z]+", short_ref.lower())
             if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return True
    seen = set(re.findall(r"[a-z]+", visual.lower()))
    return len(words & seen) / len(words) >= _CHARACTER_ECHO_MIN


def _pin_character(visual: str, name: str, short_ref: str) -> str:
    """Restate the character's LOOK in any shot that only names them.

    The image model renders each beat from noise with no memory of the others,
    so "The hooded figure, the Chronicler, appears again" gives it nothing to
    match and it invents an appearance.

    Live proof, from the Great Depression run's own images. Beat 1 said
    "weathered sepia-and-antique-gold cloak" and rendered a tan-gold cloak.
    Beat 10 said "sepia-and-gold cloak" and rendered brown. Beat 5 said only
    "The hooded figure, the Chronicler, appears again" — and rendered him in a
    BLACK cloak. One character, named three times, three different colours on
    screen, in a channel whose whole point is a recurring figure.

    The storyboard is the right place to fix it: it is the only stage that
    knows a later shot is the SAME person as an earlier one.
    """
    if not name or not short_ref:
        return visual
    bare = re.sub(r"(?i)^the\s+", "", name).strip()
    if not bare or bare.lower() not in visual.lower():
        return visual
    if _restates_the_look(visual, short_ref):
        return visual
    return (f"{visual.rstrip().rstrip('.')}. {name} is {short_ref} — the same "
            f"figure, identical in every appearance.")


# A trailing participial clause that explains what the shot MEANS. It gives an
# image model nothing to draw and actively pulls it toward generic stock
# imagery, because a phrase like "embodying the control and power within
# Europe" describes no object, no person and no place.
#
# All four of these are verbatim from one run's storyboard — four of nine shots:
#   "...silhouetted against a stormy sky, suggesting the onset of the Reformation."
#   "...exchanging handfuls of coins, indicating shrewd trade practices."
#   "...with territories marked, showing the extent of economic influence."
#   "A large medieval castle on a hill, embodying the control and power within Europe."
# The castle is what the model drew from the last one. The script it belonged to
# never mentions a castle.
_ABSTRACTION_TAIL = re.compile(
    r",\s*(?:"
    # Participial: "..., embodying the control and power within Europe."
    # A participle after a comma is a comment on the shot, whatever follows it,
    # so the tail may run over further commas to the end of the sentence.
    r"(?:embodying|symbolizing|symbolising|signifying|representing|"
    r"suggesting|indicating|reflecting|evoking|illustrating|highlighting|"
    r"underscoring|conveying|hinting at|showing the|emphasizing|emphasising)"
    r"\b[^.]*"
    r"|"
    # Noun-appositive: "..., a symbol of forgotten prosperity." Same defect,
    # different grammar, and the participial list alone missed it live.
    #
    # THE TAIL STOPS AT THE NEXT COMMA, unlike the participial branch, because
    # this form is indistinguishable from an ordinary list item until you see
    # what follows it. "A cluttered desk holds a ledger, a picture of the
    # founder, and a brass lamp." — the middle item is a real object in a real
    # list, and `[^.]*` deleted the rest of the sentence with it. Requiring the
    # phrase to END the sentence keeps the abstraction case (which always
    # trails) and leaves lists alone.
    r"an?\s+(?:symbol|emblem|reminder|testament|metaphor|image|picture|"
    r"sign|marker|echo|monument)\s+(?:of|to|for)\b[^.,]*(?=\s*\.|\s*$)"
    r")",
    re.IGNORECASE,
)


# ── faces ────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS, from a gallery of sixty stills: every figure had the same
# worried face — brows up at the inner ends, small frown — whatever the line
# was about. Not a model limitation. The storyboard was writing
#
#     "their face showing resignation"
#     "a look of loss on their face"
#     "Roosevelt's face, determined and focused"
#
# which is rule 4 ("a feeling named as a feeling gives the image model nothing
# to draw") surviving in the one place it does the most damage. The renderer's
# style block defines five faces in GEOMETRY — where the brows sit, what the
# mouth does — and an emotion word matches none of them, so the model falls
# back to the same neutral-sad default on every frame.
#
# The fix is a translation, not a rejection: the shot meant something, and the
# geometry is the same sentence in a language the model can draw. Phrased to
# match config/styles.json word for word, so the prompt and the style block
# are not two descriptions of a face competing with each other.
_FACE_GEOMETRY = {
    "anger": "brows angled down and inward over a flat or down-curved mouth",
    "shock": "brows raised high over a small open oval mouth",
    "grief": "brows slanted up at their inner ends over a downward mouth curve",
    "delight": "a wide upturned curve with the eyes squeezed to short arcs",
    "cold": "brows flat and mouth a short straight line",
}

# Emotion words → which of the five faces draws it. Deliberately small: these
# are the words that actually turned up in this channel's storyboards, and a
# word not in the list keeps its own phrasing rather than being forced into an
# approximation.
_FACE_WORDS = {
    "anger": ("anger", "angry", "fury", "furious", "rage", "outrage",
              "indignation", "defiance", "defiant", "determined",
              "determination", "resolve", "resolute", "frustration",
              "frustrated"),
    "shock": ("shock", "shocked", "surprise", "surprised", "disbelief",
              "astonishment", "astonished", "alarm", "alarmed", "panic",
              "fear", "afraid", "horror", "horrified"),
    "grief": ("grief", "sorrow", "sadness", "sad", "loss", "despair",
              "resignation", "resigned", "defeat", "defeated", "worry",
              "worried", "anxiety", "anxious", "unease", "dismay",
              "exhaustion", "exhausted", "weary"),
    "delight": ("delight", "delighted", "joy", "joyful", "relief", "relieved",
                "triumph", "triumphant", "excitement", "excited", "hope",
                "hopeful", "pride", "proud"),
    "cold": ("indifference", "indifferent", "detachment", "detached",
             "impassive", "blank", "unmoved", "stern", "cold", "calm",
             "composure", "composed", "focused", "concentration"),
}
_WORD_TO_FACE = {w: face for face, words in _FACE_WORDS.items() for w in words}

_EMOTION_ALT = "|".join(sorted(_WORD_TO_FACE, key=len, reverse=True))

# The three shapes this defect actually takes. Each captures the emotion word
# so it can be swapped for geometry rather than deleted — a shot with a face
# described badly is still a shot that wanted a face.
_FACE_PATTERNS = (
    # "a look of loss on their face", "an expression of relief on his face"
    re.compile(r"\b an?\s+(?:look|expression|air)\s+of\s+"
               rf"(?P<word>{_EMOTION_ALT})\b"
               r"(?:\s+(?:on|across|over)\s+(?:their|his|her|its|the)\s+face)?",
               re.IGNORECASE | re.VERBOSE),
    # "their face showing resignation", "her face full of worry"
    re.compile(r"\b(?:their|his|her|its|the)\s+face[,]?\s+"
               r"(?:showing|shows|full\s+of|filled\s+with|set\s+in|"
               r"twisted\s+in|lit\s+with|betraying)\s+"
               rf"(?P<word>{_EMOTION_ALT})\b",
               re.IGNORECASE),
    # "looking determined", "appearing anxious", "eyes wide with fear"
    re.compile(r"\b(?:looking|appearing|seeming|visibly)\s+"
               rf"(?P<word>{_EMOTION_ALT})\b", re.IGNORECASE),
)


def _pin_expression(visual: str) -> str:
    """Swap a named emotion for the face geometry that draws it."""
    def _swap(m: "re.Match") -> str:
        face = _WORD_TO_FACE.get(m.group("word").lower())
        return f" {_FACE_GEOMETRY[face]}" if face else m.group(0)

    out = visual
    for pat in _FACE_PATTERNS:
        out = pat.sub(_swap, out)
    return re.sub(r"\s{2,}", " ", out).replace(" ,", ",").strip()


def _face_variety(visuals: list[str]) -> dict:
    """Which of the five faces the sequence uses, and how often.

    A sequence where every face does the same thing is a sequence with no
    story in it — the prompt says so, and until this counted it, nothing
    checked whether the sequence had obeyed.
    """
    from collections import Counter
    counts: Counter = Counter()
    for v in visuals:
        for face, geom in _FACE_GEOMETRY.items():
            if geom.split(" over ")[0] in v or geom in v:
                counts[face] += 1
    return dict(counts)


def _strip_abstraction(visual: str) -> str:
    """Cut the "…, embodying X" tail off a shot description.

    A repair, not a rejection — same principle as script_writer._repair_banned:
    if the fix is mechanical and we would accept it anyway, spending a whole
    generation to arrive at it is waste.
    """
    return _ABSTRACTION_TAIL.sub("", visual).rstrip(" ,;") or visual


# Endings that mark a word as an action, a quality or a concept rather than an
# object a camera can point at. Without this the check reported "shocked",
# "vanished", "failed", "overnight" and "then" as things the pictures failed to
# show, which is the same noise problem it was written to remove.
_UNFILMABLE_SUFFIXES = ("ed", "ing", "ly", "tion", "sion", "ness", "ment",
                        "ity", "ance", "ence", "ism", "hood", "ship")

# Concrete nouns the suffix rule would eat. "ship" ends with "ship", "city"
# with "ity", "building" and "ceiling" with "ing", "bread" with "ed", "fence"
# with "ence". These are exactly the kind of thing a shot should contain, and
# the motivating case — a script naming a thing no picture shows — would have
# gone unreported if the noun had been "ship" instead of "copper".
_FILMABLE_DESPITE_SUFFIX = {
    "ship", "ships", "city", "cities", "fence", "fences", "monument",
    "monuments", "building", "buildings", "ceiling", "ceilings", "bread",
    "shed", "sheds", "bed", "beds", "thread", "threads", "field", "fields",
    "shield", "road", "roads", "seed", "seeds", "wood", "child", "gold",
    "king", "kings", "ring", "rings", "string", "strings", "spring", "wing",
    "wings", "coffin", "engine", "engines", "machine", "machines", "mine",
    "mines", "line", "lines", "vessel", "vessels", "parity", "charity",
}

# IRREGULAR PAST TENSES. The suffix rule catches "vanished" and "collapsed"
# because they end in -ed; it cannot catch "took", "gave" or "came", which are
# the commonest verbs in a narration and look exactly like nouns to it. A code
# review flagged this months ago and nothing acted on it until insert_director
# chose "took" as a thing to draw — "as receivers took over" — which is the
# cost of a false keep going from a warning line to a GPU render.
_IRREGULAR_PAST = {
    "took", "gave", "came", "went", "made", "saw", "said", "told", "found",
    "left", "held", "brought", "began", "drew", "fell", "rose", "grew",
    "sold", "bought", "paid", "sent", "kept", "lost", "meant", "felt",
    "knew", "thought", "became", "chose", "broke", "spoke", "wrote", "drove",
    "rode", "stood", "understood", "built", "sought", "taught", "caught",
    "swept", "threw", "flew", "blew", "hung", "sank", "shook", "struck",
    "wore", "tore", "swore", "bore", "beat", "burnt", "dealt", "dug",
    "fought", "forgot", "froze", "hid", "lay", "lent", "meant", "rang",
    "sang", "shot", "shut", "slept", "spent", "spread", "stole", "swam",
    "took", "wept", "wound", "gone", "seen", "done", "given", "taken",
    "written", "spoken", "broken", "chosen", "driven", "risen", "fallen",
}

# Connectives and time words long enough to clear the 4-letter floor.
_NON_OBJECT_WORDS = {
    "then", "over", "under", "after", "before", "again", "still", "also",
    "once", "when", "while", "since", "until", "about", "into", "onto",
    "every", "each", "most", "many", "much", "some", "such", "than", "them",
    "they", "their", "there", "here", "what", "which", "would", "could",
    "night", "overnight", "today", "yesterday", "tomorrow", "year", "years",
}


def _content_words(text: str) -> set[str]:
    """Words in `text` naming something a camera could point at.

    A heuristic, not a parser — this repo has no POS tagger and does not need
    one for a warning. Filters stopwords, the abstract vocabulary _is_a_thing
    already uses, obvious verb and concept endings, and the connectives that
    survive the length floor.
    """
    out = set()
    # Only words that appear LOWERCASE in the source. A proper noun — Europe,
    # America, the Habsburgs, Philadelphia — is not a thing a camera can be
    # pointed at, and reporting "Europe is shown in no shot" is noise. This
    # loses sentence-initial words too, which are stopwords often enough that
    # the trade is worth it for a warning.
    for w in re.findall(r"\b[a-z]{4,}\b", text or ""):
        if w in _STOPWORDS or w in _ABSTRACT or w in _NON_OBJECT_WORDS:
            continue
        if w in _IRREGULAR_PAST:
            continue
        if w in _FILMABLE_DESPITE_SUFFIX:
            out.add(w)
            continue
        if w.endswith(_UNFILMABLE_SUFFIXES):
            continue
        out.add(w)
    return out


def _unshown_nouns(visuals: list[str], beats: list[str]) -> list[str]:
    """Concrete things the script names that NO shot in the sequence shows.

    THE LEVEL THIS WORKS AT, AND WHY. The first version of this compared each
    shot to its own beat and reported a mismatch. Live, it fired on 7 of 10
    shots and was wrong about most of them:

        beat 3: Their jobs vanished overnight. Then, panic spread.
        shot 3: The same single bronze coin is held tightly in a worker's
                hand, their knuckles white with tension.

    That is an excellent shot for that line. It shares no word with it because
    "jobs vanished" and "a worker's hand" are one idea in two vocabularies.
    Word overlap at the SENTENCE level is the wrong instrument, and a warning
    that fires seven times out of ten is not a signal — it is something people
    learn to scroll past, which is worse than no warning at all.

    The failure it was built for is rarer and lives one level up: the Fugger
    script said the family "controlled Europe's copper" and not one of nine
    shots contained copper. So ask the sequence question instead — which
    concrete nouns does the narration name that the pictures never show — and
    a beat answered by a neighbouring shot no longer counts as a miss.

    THE SECOND ROUND OF NOISE, and the fix. Moving up a level stopped the
    seven-of-ten false alarms but the report itself then ran long: a live run
    named "costs, often, private, sells, quite, takes (+23 more)". Half of
    those are adverbs and verbs — _content_words is tuned for VISUALS, where
    the cost of keeping a doubtful word is one extra noun in a prompt, and
    narration is full of words that pass it and could never be drawn.

    insert_director._is_drawable is the strict half of the same vocabulary,
    tuned on this channel's scripts for exactly this question — is there a
    picture of this word — so ask it rather than keeping a second list here.
    Imported lazily because insert_director imports THIS module; fail-open to
    the looser test if it is unavailable, which is the report as it read
    before, noise and all, rather than no report.
    """
    shown = set()
    for v in visuals:
        shown |= _content_words(v)

    try:
        from insert_director import _is_drawable as drawable
    except Exception:
        def drawable(w: str) -> bool:      # type: ignore[misc]
            return True

    named: list[str] = []
    for beat in beats:
        for w in sorted(_content_words(beat)):
            if w not in shown and w not in named and drawable(w):
                named.append(w)
    return named


# At most this share of a sequence may carry the restated thread. See
# _already_shows for the run that set it.
THREAD_SHARE = 0.35

# And at most this share may carry the restated SETTING. Lower than the thread
# because the setting clause is longer and says less: a thread names a new
# object the shot should contain, while the setting repeats a room the shot is
# already in.
SETTING_SHARE = 0.25


def _already_shows(visual: str, carries: str) -> bool:
    """Whether the shot already names the thing the thread is carrying.

    THE FAILURE, from a real run. The thread was "the lemonade stand" and the
    line `Continuing from the previous shot: the lemonade stand.` was appended
    to NINE of ten shots — including shot 2, which opened "A child stands
    behind the wooden lemonade stand". A Danegeld run did the same with a
    stack of silver coins in six of ten. The owner's report was the obvious
    consequence: every picture had coins in it.

    Two bugs in one. The thread was restated where the shot said it already —
    pure duplication, and duplication in a prompt reads as emphasis, so the
    coins grew until they were the subject of every frame. And it was restated
    EVERYWHERE, which is the same mistake as carrying a mood: a thread named in
    every shot stops connecting them and starts making them the same picture.

    So this answers the first half — does the visual already contain the
    thread's own nouns — and THREAD_SHARE answers the second. A thread is a
    thing the sequence RETURNS to; returning is only visible against shots that
    do something else."""
    words = _content_words(carries)
    if not words:
        return False
    seen = _content_words(visual)
    return bool(words & seen)


def _clean(plan: dict, n_beats: int,
           beats: list[str] | None = None) -> list[str] | None:
    """The visuals in beat order, or None if the reply can't be trusted.

    Beat i must line up with shot i — the renderer cuts on the assumption that
    clip[i] belongs to beat[i], so a short or mis-numbered list would narrate
    every later picture against the wrong sentence."""
    if not isinstance(plan, dict):
        return None
    shots = plan.get("shots")
    if not isinstance(shots, list) or len(shots) != n_beats:
        return None
    out: list[str] = []
    # HOW OFTEN THE THREAD MAY BE RESTATED. See _already_shows for the failure.
    thread_budget = max(2, round(n_beats * THREAD_SHARE))
    restated = 0
    for entry in shots:
        if not isinstance(entry, dict):
            return None
        visual = str(entry.get("visual", "")).strip()
        if len(visual) < MIN_VISUAL_CHARS:
            return None
        visual = _pin_expression(_strip_abstraction(visual))
        carries = entry.get("carries_over")
        if isinstance(carries, str) and carries.strip():
            carries = carries.strip()
            if not _is_a_thing(carries):
                print(f"[storyboard] dropped mood-only thread: {carries!r}")
            elif _already_shows(visual, carries):
                pass                       # the shot names it already
            elif restated < thread_budget:
                # Stated as continuity so the image model has the thread too,
                # not only the storyboard's own notes.
                visual = (f"{visual.rstrip('.')}. Continuing from the previous "
                          f"shot: {carries}.")
                restated += 1
        out.append(visual)
    if restated:
        print(f"[storyboard] carried the thread into {restated} of "
              f"{n_beats} shot(s)")

    # THE FACES, counted. The complaint was "always the same face expressions"
    # and nothing in this pipeline could have noticed: the shots said
    # "resignation" and "a look of loss", the renderer drew its default, and
    # every stage downstream saw text that looked fine. One line, only when
    # there are people to have faces.
    faces = _face_variety(out)
    if faces:
        used = ", ".join(f"{k}×{v}" for k, v in sorted(faces.items(),
                                                       key=lambda kv: -kv[1]))
        print(f"[storyboard] faces: {used}")
        if len(faces) == 1 and sum(faces.values()) >= 3:
            print(f"[storyboard] ⚠ every face in this sequence is the same "
                  f"one — the arc has no turn in it")

    # Name the concrete things the script mentions that no picture shows.
    # WARNING, never rejection: not every noun deserves a frame, and this repo
    # has real wasted-generation bugs from stacking hard gates (AGENTS.md).
    # Capped at a handful — a long list is the same noise problem the per-beat
    # version had, in a different shape.
    if beats:
        missing = _unshown_nouns(out, beats)
        if missing:
            shown = ", ".join(missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            print(f"[storyboard] ⚠ named in the script, shown in no shot: "
                  f"{shown}{more}")
    return out


def dominant_subject(visuals: list[str]) -> dict:
    """The object present in the most shots, and in how many.

    ONE DEFINITION, MEASURED IN ONE PLACE. run_review already answers exactly
    this question about finished runs — it is the "why all the images with
    coin" report as a number — so this asks run_review rather than growing a
    second opinion beside it. A pipeline that repairs by one definition while
    the analyzer reports by another produces a run that fixes itself and is
    then told it did not, which is the failure mode this repo has hit more
    than once.

    Fail-open to the local vocabulary if run_review will not import: a rougher
    count is still worth having, and no storyboard should die over a metric.
    """
    if not visuals:
        return {"word": "", "shots": 0, "share": 0.0}
    try:
        from run_review import _subject_words as subject_words
    except Exception:
        subject_words = _content_words           # type: ignore[assignment]
    from collections import Counter
    counts: Counter = Counter()
    for v in visuals:
        counts.update(subject_words(v))
    if not counts:
        return {"word": "", "shots": 0, "share": 0.0}
    word, hits = counts.most_common(1)[0]
    return {"word": word, "shots": hits, "share": round(hits / len(visuals), 3)}


def _dominant_share_limit() -> float:
    """The share at which one object stops being a thread and becomes the
    subject of the video. run_review owns the number for the same reason
    dominant_subject does."""
    try:
        from run_review import DOMINANT_SHARE
        return float(DOMINANT_SHARE)
    except Exception:
        return 0.55


def repair_enabled() -> bool:
    return os.environ.get("RUFUS_STORYBOARD_REPAIR", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _revision_prompt(script: str, beats: list[str], visuals: list[str],
                     word: str, n_hits: int, targets: list[int]) -> str:
    """Ask for new shots for the ones that repeat the dominant object."""
    lines = "\n".join(
        f"{i + 1}. LINE: {beats[i] if i < len(beats) else ''}\n"
        f"   CURRENT SHOT: {visuals[i]}" for i in targets)
    return (
        f"You planned a {len(visuals)}-shot sequence for this narration, and "
        f"\"{word}\" is the subject of {n_hits} of the {len(visuals)} shots. "
        f"That is the sequence's main defect: a viewer sees the same picture "
        f"over and over and swipes away.\n\n"
        f"FULL NARRATION:\n{script}\n\n"
        f"REPLACE ONLY THESE SHOTS. Each must still be a picture of ITS OWN "
        f"LINE — do not drift to a different part of the story — but it must "
        f"be about something other than the {word}:\n{lines}\n\n"
        f"WHAT TO SHOW INSTEAD, in order of preference: the person acting, "
        f"their hands mid-action, their face and posture, the person it "
        f"happens to, the place around them, the moment just before or just "
        f"after. Every replacement must have a VERB in it — something being "
        f"done, not something lying somewhere. The {word} may still be "
        f"visible; it must not be what the shot is about.\n"
        f"Keep the same house rules: no lettering, no words printed in frame, "
        f"no camera or lens language, no named emotions — draw the face and "
        f"the posture instead. Two or three plain sentences each.\n\n"
        f'Reply with ONLY this JSON: {{"shots": [{{"n": <shot number>, '
        f'"visual": "..."}}, ...]}}')


def _revary(client, model: str, script: str, beats: list[str],
            visuals: list[str]) -> list[str]:
    """Re-plan the shots that repeat the dominant object.

    WHY A SECOND CALL AND NOT A HARDER RULE. Rule 3a asks for variety and the
    model mostly obeys it, but when the narration names one thing in nine of
    its ten lines, rule 1 (draw the literal thing the line names) and rule 3a
    point in opposite directions, and rule 1 wins — correctly, shot by shot,
    and wrongly for the sequence. Nothing written into a single prompt
    reliably resolves that, because the conflict only becomes visible once all
    the shots exist. Measuring the finished plan and asking again for the
    offending shots is the cheapest thing that actually sees the problem.

    Costs one extra call, on runs that need it. Fail-open in every direction:
    any error, any malformed reply, any shot that comes back too short, and
    the original plan stands.
    """
    if not repair_enabled() or len(visuals) < 5:
        return visuals
    d = dominant_subject(visuals)
    limit = _dominant_share_limit()
    if not d["word"] or d["share"] <= limit:
        return visuals

    word = d["word"]
    # Keep the shots where it belongs most — the earliest ones, which is where
    # the object is introduced — and revise the surplus. Budget is the share
    # rule 3a asks for, so the repair aims past the threshold rather than at
    # it; landing exactly on the limit leaves the next measurement reporting a
    # defect the run just fixed.
    budget = max(1, round(len(visuals) / 3.0))
    hits = [i for i, v in enumerate(visuals) if word in _subject_of(v)]
    targets = hits[budget:]
    if not targets:
        return visuals
    print(f"[storyboard] \"{word}\" is the subject of {d['shots']} of "
          f"{len(visuals)} shots — re-planning {len(targets)} of them")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": _revision_prompt(script, beats, visuals,
                                                   word, d["shots"],
                                                   targets)}],
            temperature=0.9, max_tokens=1500, timeout=90,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        print(f"[storyboard] re-plan skipped (non-fatal): {e}")
        return visuals

    out = list(visuals)
    changed = 0
    for entry in (raw.get("shots") or []):
        if not isinstance(entry, dict):
            continue
        try:
            i = int(entry.get("n", 0)) - 1
        except (TypeError, ValueError):
            continue
        new = _strip_abstraction(str(entry.get("visual", "")).strip())
        if i not in targets or len(new) < MIN_VISUAL_CHARS:
            continue
        # A replacement that names the same object again is not a replacement.
        # Without this the pass reports success on a reply that changed the
        # words and not the picture — the shape of failure this whole file
        # keeps running into.
        if word in _subject_of(new):
            continue
        out[i] = new
        changed += 1
    after = dominant_subject(out)
    if changed:
        print(f"[storyboard] re-planned {changed} shot(s); \"{after['word']}\" "
              f"is now the widest subject at {after['shots']} of {len(out)}")
    else:
        print("[storyboard] re-plan returned nothing usable — keeping the "
              "original shots")
    return out


def _subject_of(visual: str) -> set[str]:
    try:
        from run_review import _subject_words
        return _subject_words(visual)
    except Exception:
        return _content_words(visual)


def _character(niche: str | None) -> tuple[str, str]:
    """(name, short_ref) for the niche's recurring character, or ("", "")."""
    if not niche:
        return "", ""
    try:
        import character_engine
        if not character_engine.enabled(niche):
            return "", ""
        cfg = character_engine.niche_character(niche) or {}
        return str(cfg.get("name", "")).strip(), character_engine.short_ref(niche)
    except Exception as e:
        print(f"[storyboard] character look-up skipped (non-fatal): {e}")
        return "", ""


def plan(script: str, beats: list[str], era_tags: list[str] | None = None,
         character_clause: str = "", niche: str | None = None,
         scene: str = "") -> list[str] | None:
    """One visual per beat, planned as a sequence. None to use the old path.

    `scene` is the story architect's THE SCENE — the one filmable moment the
    script was built to land on. Passing it here anchors the pictures to the
    same moment the words turn on, instead of leaving the storyboard to infer a
    location from aggregate sentences and invent a castle.
    """
    if not enabled() or not beats:
        return None
    era_tags = era_tags or []
    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        key = ""
        if keys_file.exists():
            key = json.loads(keys_file.read_text(encoding="utf-8")).get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            return None
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user",
                       "content": _prompt(script, beats, era_tags,
                                          character_clause, scene)}],  # noqa: E501
            temperature=0.8, max_tokens=2000, timeout=120,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        visuals = _clean(raw, len(beats), beats)
        if visuals:
            # BEFORE the thread and the setting are pinned in. Those clauses
            # are text this pipeline appends, not pictures the storyboard
            # chose, and counting them would measure our own writing — the
            # mistake that once made a run look like it was about "market,
            # cobblestone, bright".
            visuals = _revary(client, _model(), script, beats, visuals)
            name, short = _character(niche)
            pinned = [_pin_character(v, name, short) for v in visuals]
            n_fixed = sum(1 for a, b in zip(visuals, pinned) if a != b)
            if n_fixed:
                print(f"[storyboard] restated {name}'s look in {n_fixed} shot(s) "
                      f"that only named him")
            visuals = pinned

            # The place, restated into the shots that belong to it. A shot the
            # model marked as leaving the setting is left alone — a deliberate
            # departure is part of the sequence, and forcing the room back into
            # a present-day street would just make that shot wrong.
            setting = str(raw.get("setting", "")).strip()
            if setting and not _is_a_place(setting):
                print(f"[storyboard] setting \"{setting}\" is a label, not a "
                      f"room — dropping it (shots keep their own places)")
                setting = ""
            if setting:
                # CAPPED, for the same reason the thread is. At ten shots this
                # fired three or four times and read as a reminder; at
                # twenty-four it fired THIRTEEN times, and "Same place as the
                # rest of the sequence: market, cobblestone, bright" appended
                # to half the sequence is no longer a reminder — it is a
                # second description competing with each shot's own, on every
                # other frame. The pin exists for the shot that forgot where
                # it was, not for the sequence.
                flags = _in_setting_flags(raw, len(visuals))
                budget = max(2, round(len(visuals) * SETTING_SHARE))
                placed, used = [], 0
                for v, keep in zip(visuals, flags):
                    pinned_v = _pin_setting(v, setting) if keep else v
                    if pinned_v != v and used >= budget:
                        pinned_v = v
                    elif pinned_v != v:
                        used += 1
                    placed.append(pinned_v)
                n_placed = sum(1 for a, b in zip(visuals, placed) if a != b)
                print(f"[storyboard] setting: {setting}")
                if n_placed:
                    print(f"[storyboard] restated the location in {n_placed} "
                          f"shot(s) that assumed it")
                visuals = placed
    except Exception as e:
        print(f"[storyboard] skipped (non-fatal): {e}")
        return None

    if visuals is None:
        print("[storyboard] reply didn't validate — using per-beat prompts")
        return None
    through = str(raw.get("through_line", "")).strip()
    if through:
        print(f"[storyboard] through-line: {through}")
    print(f"[storyboard] {len(visuals)} shots planned as one sequence")
    return visuals
