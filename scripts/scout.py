#!/usr/bin/env python3
"""
scout.py — the agent that watches the neighbours and proposes what to make.

WHAT "RESEARCHING FOR DAYS" ACTUALLY MEANS HERE, because the tempting reading
is wrong. It is not a model thinking for hours: that costs a great deal, has no
ceiling, and does not improve with time because there is nothing new to think
about between one hour and the next. It is a small, cheap pass on a schedule
that WRITES DOWN WHAT IT SAW. Days of research is then a growing table — the
same competitor video seen on Monday at 4,000 views and on Thursday at 40,000
is a fact no single pass could ever have, and `scout_observations` is the only
reason it survives.

ONE PASS, IN SIX STEPS:

    observe    competitors.observe() — every watched channel's recent uploads,
               each scored against ITS OWN channel's median
    remember   db_manager.record_observations() — appended, never updated
    choose     the strongest outperformer this channel has NOT already covered
    ground     research.get_seed(topic=...) resolves it to a real Wikipedia
               article, so the fact gate has something to check against
    write      script_writer.write_script_until_good() — already scored,
               fact-gated, cost-capped
    propose    a row in `proposals`, with the evidence that chose it

IT NEVER RENDERS. A script is cents and seconds; a render is hours of the 3090.
The human approves a proposal and a normal run makes the video — which is what
makes it affordable for this to be wrong sometimes.

    RUFUS_SCOUT_MAX_PENDING   6      stop proposing when this many are waiting
    RUFUS_SCOUT_MAX_COST      1.00   USD per day across every pass
    python scripts/scout.py --once --dry-run   choose and explain, spend nothing

THE THING THIS CANNOT DO YET, said plainly: nothing published through this
pipeline has view counts, so `feedback_analyzer` has never run and there are no
winning hooks to learn from. The scout is therefore a well-informed guesser. It
gets genuinely better the day three videos have metrics — see /tracking.

CONTRACT: fail-open, bounded, and never raises into a scheduled task.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_MAX_PENDING = 6
DEFAULT_MAX_COST = 1.00

# How strong an outperformance has to be before it is worth building on. The
# same number competitors.py uses, imported rather than repeated.
from competitors import OUTPERFORMANCE  # noqa: E402


def max_pending() -> int:
    try:
        return max(1, int(os.environ.get("RUFUS_SCOUT_MAX_PENDING",
                                         DEFAULT_MAX_PENDING)))
    except ValueError:
        return DEFAULT_MAX_PENDING


def max_cost() -> float:
    try:
        return max(0.0, float(os.environ.get("RUFUS_SCOUT_MAX_COST",
                                             DEFAULT_MAX_COST)))
    except ValueError:
        return DEFAULT_MAX_COST


# ── choosing ─────────────────────────────────────────────────────────────────

_STOP = {"the", "a", "an", "of", "and", "that", "this", "how", "why", "what",
         "who", "when", "was", "is", "are", "were", "in", "on", "for", "to",
         "it", "its", "with", "from", "you", "your", "their", "his", "her"}


def _keywords(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (title or "").lower())
            if len(w) > 3 and w not in _STOP}


def already_covered(title: str, made: list[str]) -> bool:
    """Has this channel already made something about this?

    Word overlap and not exact matching, because two videos about the same
    event never share a title. Half the candidate's content words appearing in
    one thing already made is enough — the cost of a false positive is one
    skipped idea, and the cost of a false negative is a duplicate video.
    """
    keys = _keywords(title)
    if not keys:
        return True
    for other in made:
        shared = keys & _keywords(other)
        if len(shared) >= max(2, len(keys) // 2):
            return True
    return False


def choose(candidates: list[dict], made: list[str]) -> dict | None:
    """The strongest candidate this channel has not covered. None if none."""
    for c in candidates:
        if not already_covered(c.get("title", ""), made):
            return c
    return None


def _made_titles(limit: int = 200) -> list[str]:
    """Everything this channel has already made or proposed.

    Both, deliberately: a proposal that is still waiting for approval is
    exactly as much of a duplicate as a finished video, and a scout that
    forgets its own pending queue proposes the same idea every four hours.
    """
    out: list[str] = []
    try:
        import db_manager
        out += db_manager.recent_titles(limit=limit)
        out += [p.get("topic") or ""
                for p in db_manager.proposals(status=None, limit=limit)]
    except Exception as e:
        # LOUD, because failing open here means "nothing has been made", and
        # that is not a degraded answer — it is the wrong one. The scout would
        # propose the same idea every pass and each one would look correct.
        print(f"[scout] ⚠ could not read what has already been made ({e}) — "
              f"every candidate will look new, so this pass may well propose "
              f"a duplicate")
    return [t for t in out if t]


def subject_of(title: str, niche: str) -> str:
    """The researchable subject behind a competitor's title.

    A title is written to be clicked, not to be researched — "The Man Who
    Bought Manhattan For $24" is a headline whose subject is the Dutch
    purchase of Manhattan. Handing the headline itself to a Wikipedia lookup
    finds nothing, so one cheap call turns it into something groundable.

    Falls back to the title itself, which is what a lookup would have got
    anyway, so a model outage costs precision rather than the whole pass.
    """
    try:
        import llm
        if not llm.usable():
            return title
        resp = llm.client().chat.completions.create(
            model=llm.model_for("scout_subject", "gpt-4o-mini"),
            messages=[{"role": "user", "content": (
                f"A {niche} video is titled:\n\"{title}\"\n\n"
                f"Name the real historical subject it is about, as a phrase "
                f"that would be a Wikipedia article title. Not the headline, "
                f"not a question, not a sentence — the SUBJECT. If you cannot "
                f"tell, answer with the single word NONE.\n\n"
                f"Reply with the phrase and nothing else.")}],
            temperature=0, max_tokens=30, timeout=30,
        )
        got = (resp.choices[0].message.content or "").strip().strip('"')
        if not got or got.upper() == "NONE" or len(got) > 90:
            return title
        return got
    except Exception as e:
        print(f"[scout] subject lookup failed ({e}) — using the title")
        return title


def evidence_of(candidate: dict, trend_note: str = "") -> str:
    """Why this proposal exists, in the row itself.

    A proposal without its evidence is an instruction, and an instruction from
    an agent is exactly the thing a person cannot audit. This is what makes
    approving one a judgement rather than a coin toss.
    """
    bits = [
        f"{candidate.get('channel_title', '?')} published "
        f"\"{candidate.get('title', '')}\"",
        f"{candidate.get('views', 0):,} views — "
        f"{candidate.get('outperformance', 0):.1f}x that channel's own median",
    ]
    if candidate.get("sightings", 0) > 1:
        bits.append(f"seen on {candidate['sightings']} passes, so this is not "
                    f"one day's spike")
    if trend_note:
        bits.append(f"trending now: {trend_note}")
    return " · ".join(bits)


# ── the pass ─────────────────────────────────────────────────────────────────

def blocked() -> str:
    """Why this pass should not run, or "". Checked before anything is spent."""
    try:
        import db_manager
        db_manager.init_db()
        pending = db_manager.pending_proposal_count()
        if pending >= max_pending():
            return (f"{pending} proposal(s) already waiting and the ceiling is "
                    f"{max_pending()} — an agent that fills a queue nobody has "
                    f"read is generating work, not doing it")
        spent = db_manager.proposal_cost_today()
        if spent >= max_cost():
            return (f"${spent:.2f} spent on proposals today, ceiling "
                    f"${max_cost():.2f}")
    except Exception as e:
        return f"the database could not be read ({e})"
    try:
        import run_review
        busy = run_review._gpu_is_busy()
        if busy:
            return (f"a render is using the GPU ({busy}) — the writer competes "
                    f"for it when a local model serves it")
    except Exception:
        pass
    return ""


def observe_and_remember() -> int:
    """One competitor pass, recorded. Returns how many observations landed."""
    try:
        import competitors
        import db_manager
        seen = competitors.observe()
        if not seen:
            return 0
        n = db_manager.record_observations(seen)
        print(competitors.describe(seen))
        return n
    except Exception as e:
        print(f"[scout] observing failed (non-fatal): {e}")
        return 0


def pass_once(dry_run: bool = False) -> dict:
    """One full scout pass. Always returns a dict; never raises."""
    import db_manager
    out: dict = {"observed": 0, "candidate": None, "proposal_id": None,
                 "skipped": ""}

    why = blocked()
    if why:
        print(f"[scout] standing down: {why}")
        out["skipped"] = why
        return out

    out["observed"] = observe_and_remember()

    candidates = db_manager.rising(min_outperformance=OUTPERFORMANCE)
    if not candidates:
        # TWO DIFFERENT FACTS, AND THEY WERE ONE MESSAGE. A pass that observed
        # nothing has learned nothing, and calling that "a quiet week" reports
        # a conclusion about the competitors when the truth is about the
        # configuration. On a four-hourly schedule that writes the same false
        # sentence into the log six times a day while the owner reasonably
        # concludes their competitors are not publishing anything good.
        #
        # This is research.trending_queries_with_reason's bug, one layer up:
        # four situations, one empty list, and a page whose whole job is to
        # tell you something offering three guesses and a shrug.
        if not out["observed"]:
            out["skipped"] = ("nothing was observed at all, so there is "
                              "nothing to say about what is rising — the "
                              "[competitors] lines above say why")
        else:
            out["skipped"] = (f"{out['observed']} video(s) observed and none "
                              f"beat its own channel's median — a quiet week "
                              f"is a real answer")
        print(f"[scout] {out['skipped']}")
        return out

    pick = choose(candidates, _made_titles())
    if not pick:
        out["skipped"] = (f"all {len(candidates)} candidate(s) are things this "
                          f"channel has already made or already proposed")
        print(f"[scout] {out['skipped']}")
        return out
    out["candidate"] = pick

    try:
        import research
        niche = research._load_niche()[1]
        trend, _reason = research.trending_queries_with_reason(niche)
    except Exception:
        niche, trend = "money_history", []
    evidence = evidence_of(pick, ", ".join(trend[:3]))
    print(f"[scout] candidate: {pick['title']}\n[scout] because: {evidence}")

    if dry_run:
        out["skipped"] = "dry run — chose an angle and spent nothing"
        print(f"[scout] {out['skipped']}")
        return out

    subject = subject_of(pick["title"], niche)
    print(f"[scout] subject: {subject}")

    try:
        import script_writer
        seed = research.get_seed(niche, topic=subject)
        analysis, run_id, cost = script_writer.preanalyze(
            seed, seed.get("content", "")[:400])
        result = script_writer.write_script_until_good(
            seed.get("content", "")[:400], seed=seed,
            precomputed_analysis=analysis, run_id=run_id)
    except Exception as e:
        out["skipped"] = f"could not write a script for {subject!r}: {e}"
        print(f"[scout] {out['skipped']}")
        return out

    try:
        from channel_config import load_channel
        channel_id = load_channel().id
    except Exception:
        channel_id = "main_en"

    # THE HOOK IS THE SCRIPT'S FIRST LINE, and is derived rather than read from
    # the result: write_script_until_good's documented return shape has no
    # "hook" key at all (script, run_id, score, criterion_scores,
    # attempts_used, final_temperature, reasoning, cost_usd). Asking for one
    # returns "" forever, and an empty column nobody displays is the kind of
    # wrong that survives for months. This is how metadata_writer and the
    # uploader's legacy path both get it.
    script_text = result.get("script", "") or ""
    hook = script_text.strip().split("\n")[0][:300]

    out["proposal_id"] = db_manager.save_proposal(
        channel=channel_id, niche=niche, topic=subject,
        hook=hook, script=script_text,
        score=int(result.get("score", 0) or 0),
        evidence=evidence,
        cost_usd=float(cost or 0) + float(result.get("cost_usd", 0) or 0))
    print(f"[scout] proposal #{out['proposal_id']} — "
          f"{result.get('score', 0)}/10 — waiting for approval at /scout")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    # --once is accepted and does nothing, on purpose: one pass IS the only
    # behaviour, the schedule does the repeating, and the flag is already
    # written into schedule_scout.ps1's help and this module's docstring. A
    # flag that errors on someone who followed the instructions is worse than
    # one that is redundant.
    unknown = [a for a in args if a not in ("--dry-run", "--once")]
    if unknown:
        print(f"[scout] unknown option(s): {' '.join(unknown)} — "
              f"only --once and --dry-run are understood")
        raise SystemExit(2)
    print(json.dumps(pass_once(dry_run="--dry-run" in args), indent=2,
                     default=str))
