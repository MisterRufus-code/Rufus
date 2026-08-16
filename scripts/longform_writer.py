#!/usr/bin/env python3
"""
longform_writer.py — a nine-minute script, written in sections.

WHY THE SHORTS WRITER CANNOT DO THIS BY RAISING A NUMBER. script_writer is
built around one hook, one 105-word body and one set of checks over the whole
thing, and every part of that shape is load-bearing at forty seconds. Ask it
for 1,300 words and you do not get an explainer, you get a padded Short: one
idea stretched, the same figure restated to fill the space, and a cadence
check averaging over thirteen paragraphs it cannot see separately.

Length is not the difficulty. STRUCTURE is. A viewer stays nine minutes
because the script keeps making and paying small promises, and that is
something you plan before you write and verify afterwards — neither of which
a single generation call does.

THE SHAPE, read off a channel that holds an audience for eleven minutes:

  COLD OPEN     second person, present tense, sensory, no preamble. "Imagine
                you haven't eaten in two days." Ends on the turn that creates
                the problem: "And then it starts raining."
  THESIS        negation-then-correction plus a counted promise. "Rain wasn't
                an inconvenience. It was a crisis" … "could kill you in three
                different ways."
  SECTIONS      each pays one piece of the promise, opens with a hinge from
                the last one, carries its own fact from the source, and ends
                by owing the next.
  TURN          the section that reframes everything before it. "But here's
                where the story takes a turn nobody expects."
  CLOSE         recap as crescendo — the same shape repeated, then one line
                that lands it against the viewer's own life.

WHAT THIS FILE ADDS THAT LENGTH ALONE WOULD NOT:

  - An OUTLINE checked against the source before a word of prose is paid for.
    At 1,300 words there is ten times the surface area to invent on, and the
    fact gate is the most expensive place to discover that.
  - Each section written with the previous section's ending in hand, so the
    hinges connect and nothing is said twice.
  - A repetition check ACROSS sections — the long-form form of the defect the
    owner reported as "why is everything coins".

CONTRACT: fail-open, like every other writer here. No key, no outline, a
section that will not come back — all fall through to a shorter script or to
None, and main falls back to the Shorts writer. A long script is a better
video, never a prerequisite for one.
"""

from __future__ import annotations

import json
import os
import re

# The outline is cheap and decides everything; the prose is expensive and
# decides only its own paragraph. Spend the better model on the outline.
OUTLINE_MODEL_DEFAULT = "gpt-4o"
SECTION_MODEL_DEFAULT = "gpt-4o"

# Sections, not paragraphs. Below four there is no arc to speak of; above
# eight each one is too thin to pay a real promise and the video becomes a
# list. Six is the shape the reference scripts settle on.
MIN_SECTIONS = 4
MAX_SECTIONS = 8

# A section shorter than this is a paragraph pretending to be a chapter.
MIN_SECTION_WORDS = 90


def enabled() -> bool:
    """Long-form is chosen by the format, not by its own switch.

    A second flag would let the two disagree — RUFUS_FORMAT=long with the
    long writer off is a nine-minute request answered by a forty-second
    writer, which is precisely the confusion the format profile exists to
    prevent.
    """
    import video_format
    return video_format.is_long()


# ── the outline ──────────────────────────────────────────────────────────────

def _outline_prompt(seed_block: str, analysis: str, niche: str,
                    words: int, sections: int) -> str:
    return (
        f"You are planning a {words}-word narrated explainer video for a "
        f"{niche} channel. Plan it; do not write it.\n\n"
        f"{seed_block}\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"WHAT MAKES SOMEBODY WATCH FOR NINE MINUTES. Not the topic — the "
        f"promises. The script makes a small promise, pays it, and opens "
        f"another before the payment lands. Plan those promises now, because "
        f"they cannot be added to finished prose.\n\n"
        f"THE SHAPE:\n"
        f"1. COLD OPEN — second person, present tense, a situation the viewer "
        f"is standing inside. No 'in this video', no greeting, no throat "
        f"clearing. It ends on the TURN: the single event that breaks the "
        f"situation open.\n"
        f"2. THESIS — one negation-then-correction ('X wasn't A. It was B.') "
        f"followed by a COUNTED promise the rest of the video pays: 'three "
        f"ways', 'two things that had to be true', 'the four days that "
        f"followed'.\n"
        f"3. {sections} SECTIONS. Each one: pays a piece of the promise, "
        f"carries ONE fact taken from the source above, and ends owing the "
        f"next section something.\n"
        f"4. One of the middle sections is the TURN — it reframes everything "
        f"before it rather than adding to it.\n"
        f"5. CLOSE — restate the sections as a short list in one shape, then "
        f"one line that lands the whole thing against the viewer's own life.\n\n"
        f"EVERY `fact` MUST BE COPIED FROM THE SOURCE ABOVE, not recalled. A "
        f"figure, a date, a name or a documented event, quoted closely enough "
        f"that a fact-checker comparing the two would agree. If the source "
        f"cannot support {sections} sections of real material, plan fewer and "
        f"say so in `note` — a short honest video beats a long invented one, "
        f"and inventing is the single largest cause of rejected scripts on "
        f"this channel.\n\n"
        f"Reply with ONLY this JSON:\n"
        f'{{"cold_open": "<the situation, one sentence, second person>",\n'
        f' "turn_line": "<the event that breaks it, one short sentence>",\n'
        f' "thesis": "<the negation-then-correction, one or two sentences>",\n'
        f' "promise": "<the counted promise, one sentence>",\n'
        f' "sections": [{{"title": "<3-6 words, for you not the viewer>",\n'
        f'   "pays": "<which piece of the promise this pays>",\n'
        f'   "fact": "<the fact from the source, quoted closely>",\n'
        f'   "hinge": "<the line that opens this section from the last>",\n'
        f'   "is_turn": true|false}}, ...],\n'
        f' "close": "<what the final lines land on>",\n'
        f' "note": "<empty, or why you planned fewer sections>"}}'
    )


def _clean_outline(raw: dict) -> dict | None:
    """Validate the plan before any prose is bought."""
    if not isinstance(raw, dict):
        return None
    sections = raw.get("sections")
    if not isinstance(sections, list) or not sections:
        return None
    out_sections = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        fact = str(s.get("fact") or "").strip()
        pays = str(s.get("pays") or "").strip()
        if len(fact) < 12 or not pays:
            continue                     # a section with no fact is filler
        out_sections.append({
            "title": str(s.get("title") or "").strip()[:80],
            "pays": pays,
            "fact": fact,
            "hinge": str(s.get("hinge") or "").strip(),
            "is_turn": bool(s.get("is_turn")),
        })
    if len(out_sections) < MIN_SECTIONS:
        return None
    out_sections = out_sections[:MAX_SECTIONS]
    # EXACTLY ONE TURN. Two reframings cancel each other out and none leaves a
    # list of facts, which is the shape of every explainer nobody finishes.
    turns = [s for s in out_sections if s["is_turn"]]
    if not turns:
        out_sections[max(0, len(out_sections) // 2)]["is_turn"] = True
    elif len(turns) > 1:
        for s in turns[1:]:
            s["is_turn"] = False
    for key in ("cold_open", "thesis", "promise"):
        if len(str(raw.get(key) or "").strip()) < 10:
            return None
    return {
        "cold_open": str(raw["cold_open"]).strip(),
        "turn_line": str(raw.get("turn_line") or "").strip(),
        "thesis": str(raw["thesis"]).strip(),
        "promise": str(raw["promise"]).strip(),
        "sections": out_sections,
        "close": str(raw.get("close") or "").strip(),
        "note": str(raw.get("note") or "").strip(),
    }


def ungrounded_facts(outline: dict, source: str) -> list[str]:
    """Outline facts whose numbers are nowhere in the source.

    THE CHEAPEST PLACE TO CATCH AN INVENTION. The fact gate finds these too,
    after 1,300 words have been written and scored — and at long-form length
    that is minutes of generation and the largest single cost in the run. The
    same arithmetic one step earlier costs nothing.

    Deliberately checks NUMBERS only. Names and events need a judge; digits
    need a set membership, and digits are what get invented.
    """
    import script_writer as sw
    out = []
    for s in outline.get("sections", []):
        bad = sw._ungrounded_number(s.get("fact", ""), source)
        if bad is not None:
            out.append(f"{s.get('title') or s.get('pays')}: '{bad}'")
    return out


# ── the prose ────────────────────────────────────────────────────────────────

def _section_prompt(outline: dict, index: int, words: int,
                    previous_tail: str, used_facts: list[str],
                    seed_block: str) -> str:
    s = outline["sections"][index]
    first = index == 0
    last = index == len(outline["sections"]) - 1
    said = ("\n".join(f"  - {f}" for f in used_facts[-6:]) or "  (nothing yet)")
    return (
        f"{seed_block}\n"
        f"You are writing ONE section of a narrated explainer, roughly "
        f"{words} words. Write only this section's narration — no heading, no "
        f"label, no stage direction, nothing but the words that are spoken.\n\n"
        f"THE VIDEO'S PROMISE: {outline['promise']}\n"
        f"THIS SECTION PAYS: {s['pays']}\n"
        f"THE FACT IT CARRIES (from the source — use it, do not embellish it):"
        f"\n{s['fact']}\n\n"
        + (f"HOW IT OPENS: {s['hinge']}\n" if s.get("hinge") and not first else "")
        + (f"THE LAST WORDS BEFORE THIS SECTION:\n…{previous_tail}\n\n"
           if previous_tail else "\n")
        + f"ALREADY SAID — do not repeat any of it:\n{said}\n\n"
        + ("THIS IS THE TURN. It does not add another fact to the pile — it "
           "reframes everything already said. Open by admitting what the "
           "viewer is expecting, then break it.\n\n" if s.get("is_turn") else "")
        + ("THIS IS THE LAST SECTION. Restate what came before as a short "
           "list in ONE repeated shape — 'They kept the fire alive. They made "
           "the tools. They told the stories.' — then land one final line "
           "against the viewer's own life. Do not add new facts here.\n\n"
           if last else
           "END THIS SECTION OWING THE NEXT ONE SOMETHING. A short line that "
           "opens a question rather than closing one.\n\n")
        + "HOW IT MUST BE WRITTEN:\n"
        "- Concrete over abstract, always. A thing somebody dug up beats a "
        "claim about the past.\n"
        "- Vary the rhythm: at least one sentence under six words and one "
        "over fifteen. A paragraph of same-length sentences reads as "
        "monotone however good the facts are.\n"
        "- Numbers are SPOKEN. 'four point two trillion', not "
        "'4,210,500,000,000'.\n"
        "- Never state what anyone felt, feared or intended unless the source "
        "says so. Attribute to the outcome, not to the mind.\n"
        "- Where the evidence stops, SAY so — 'we are not sure', 'no source "
        "records', 'what we do know is'. A narrator who admits a limit is "
        "trusted on everything else. Never 'maybe', 'perhaps', 'possibly', "
        "'probably', 'kind of', 'sort of' — those soften the claim without "
        "saying why, and they are rejected.\n"
        "- No 'in this video', no 'let's dive in', no addressing the "
        "algorithm, no asking for likes.\n"
    )


def _tail(text: str, words: int = 25) -> str:
    return " ".join(text.split()[-words:])


def _section_words(total: int, n_sections: int) -> int:
    """Prose budget per section.

    The cold open, thesis and close take roughly a fifth of the script between
    them; the sections divide what is left. Asking for total/n would overshoot
    the format's ceiling by exactly that fifth, and a script over its ceiling
    is a render over its length.
    """
    return max(MIN_SECTION_WORDS, int((total * 0.8) / max(1, n_sections)))


def repeated_across_sections(sections: list[str]) -> str | None:
    """A content word carrying most of the sections.

    THE LONG-FORM SHAPE OF "why is everything coins". At forty seconds a
    repeated noun is the storyboard's problem; at 1,300 words it is the
    script's, and it is what makes a long video feel like a short one said
    four times. Reuses the storyboard's own subject vocabulary so the two
    cannot disagree about what counts as a subject.
    """
    if len(sections) < 4:
        return None
    try:
        from run_review import _subject_words
    except Exception:
        return None
    from collections import Counter
    counts: Counter = Counter()
    for s in sections:
        counts.update(_subject_words(s))
    if not counts:
        return None
    word, hits = counts.most_common(1)[0]
    if hits / len(sections) > 0.8 and hits >= 4:
        return (f"'{word}' is the subject of {hits} of {len(sections)} "
                f"sections — the long version of one picture repeated")
    return None


# ── putting it together ──────────────────────────────────────────────────────

def _json_call(client, model: str, prompt: str, *, max_tokens: int,
               temperature: float, as_json: bool = True):
    kw = dict(model=model, messages=[{"role": "user", "content": prompt}],
              max_tokens=max_tokens, temperature=temperature, timeout=180)
    if as_json:
        kw["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kw)
    return (resp.choices[0].message.content or "").strip()


def plan(seed: dict | None, analysis: str, niche: str,
         client=None) -> dict | None:
    """The outline, validated and grounded. None if it cannot be trusted."""
    import llm
    import script_writer as sw
    import video_format

    prof = video_format.profile("long")
    words = int((prof["words_min"] + prof["words_max"]) / 2)
    n = 6
    seed_block = sw._seed_block(seed) if seed else ""
    prompt = _outline_prompt(seed_block, analysis or "", niche, words, n)

    client = client or llm.client()
    try:
        raw = json.loads(_json_call(
            client, llm.model_for("longform_outline", OUTLINE_MODEL_DEFAULT),
            prompt, max_tokens=1800, temperature=0.8))
    except Exception as e:
        print(f"[longform] outline failed ({e})")
        return None

    outline = _clean_outline(raw)
    if not outline:
        print("[longform] outline did not validate — falling back")
        return None

    source = sw.grounding_corpus(seed, analysis or "")
    bad = ungrounded_facts(outline, source)
    if bad:
        # LOUD, and not fatal. The fact gate will still see the finished
        # script; this is the cheap early warning, and dropping a section is
        # better than writing two hundred words on an invented figure.
        print(f"[longform] ⚠ {len(bad)} outline fact(s) cite a number the "
              f"source does not: {'; '.join(bad[:3])}")
        keep = [s for s in outline["sections"]
                if sw._ungrounded_number(s["fact"], source) is None]
        if len(keep) >= MIN_SECTIONS:
            outline["sections"] = keep
            print(f"[longform]   dropped them — {len(keep)} sections remain")
        else:
            print(f"[longform]   too few would remain; keeping the plan and "
                  f"letting the fact gate judge the prose")

    if outline["note"]:
        print(f"[longform] planner's note: {outline['note']}")
    print(f"[longform] {len(outline['sections'])} sections planned; "
          f"turn at #{1 + next(i for i, s in enumerate(outline['sections']) if s['is_turn'])}")
    return outline


def _opening(outline: dict) -> str:
    """Cold open, turn, thesis and promise — the first thirty seconds.

    Assembled rather than generated: every part of it is already a finished
    sentence in the outline, and asking a model to rewrite four sentences it
    just wrote is how the counted promise stops matching the sections that
    pay it.
    """
    bits = [outline["cold_open"]]
    if outline["turn_line"]:
        bits.append(outline["turn_line"])
    bits.append(outline["thesis"])
    bits.append(outline["promise"])
    return " ".join(b.rstrip() for b in bits if b)


def write(seed: dict | None, analysis: str, niche: str,
          run_id: str = None) -> dict | None:
    """A long-form script. None to fall back to the Shorts writer."""
    import llm
    import script_writer as sw
    import video_format

    if not enabled():
        return None
    if not llm.usable():
        print("[longform] no model available — falling back")
        return None

    llm.announce()
    client = llm.client()
    outline = plan(seed, analysis, niche, client=client)
    if not outline:
        return None

    prof = video_format.profile("long")
    target = int((prof["words_min"] + prof["words_max"]) / 2)
    per = _section_words(target, len(outline["sections"]))
    seed_block = sw._seed_block(seed) if seed else ""
    model = llm.model_for("longform_section", SECTION_MODEL_DEFAULT)

    body: list[str] = []
    used_facts: list[str] = []
    for i, s in enumerate(outline["sections"]):
        prompt = _section_prompt(outline, i, per, _tail(body[-1]) if body else "",
                                 used_facts, seed_block)
        try:
            text = _json_call(client, model, prompt, max_tokens=per * 3,
                              temperature=0.85, as_json=False)
        except Exception as e:
            print(f"[longform] section {i + 1} failed ({e}) — stopping here")
            break
        text = _strip_headings(text)
        if len(text.split()) < MIN_SECTION_WORDS // 2:
            print(f"[longform] section {i + 1} came back too short — skipped")
            continue
        # THE DETERMINISTIC GATES, PER SECTION. Running them on the finished
        # 1,300 words would name one banned phrase and throw away twelve good
        # paragraphs with the one that has it. Here a failure costs one
        # section, and the log says which.
        for label, found in (("banned phrase", sw._find_banned(text)),
                             ("hedging", sw._find_hedging(text))):
            if found:
                print(f"[longform] section {i + 1}: {label} '{found}' — "
                      f"rewriting once")
                try:
                    text = _strip_headings(_json_call(
                        client, model,
                        f"{prompt}\n\nYour last attempt used '{found}', which "
                        f"is banned. Rewrite it without that word or any "
                        f"variation of it.",
                        max_tokens=per * 3, temperature=0.9, as_json=False))
                except Exception:
                    pass
                break
        body.append(text.strip())
        used_facts.append(s["fact"])
        print(f"[longform] section {i + 1}/{len(outline['sections'])}: "
              f"{len(text.split())} words")

    if len(body) < MIN_SECTIONS:
        print(f"[longform] only {len(body)} sections survived — falling back")
        return None

    script = "\n".join([_opening(outline)] + body)
    words = len(script.split())

    dup = repeated_across_sections(body)
    if dup:
        print(f"[longform] ⚠ {dup}")

    print(f"[longform] {words} words in {len(body)} sections "
          f"(target {prof['words_min']}–{prof['words_max']})")
    return {
        "script": script,
        "hook": outline["cold_open"],
        "outline": outline,
        "words": words,
        "sections": len(body),
        "format": "long",
    }


_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+.*|\*\*[^*]{0,80}\*\*\s*|SECTION\s*\d+\s*[:.]?.*|"
    r"\[[^\]]{0,60}\])\s*$", re.IGNORECASE)


def _strip_headings(text: str) -> str:
    """Drop the labels a model adds when asked for one section of something.

    "**Section 3: The Fire**" is not narration, and a voice engine reads it
    out loud. Asking the prompt not to do it works most of the time, which is
    exactly the reliability that needs a deterministic backstop.
    """
    lines = [ln for ln in (text or "").splitlines()
             if not _HEADING_RE.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
