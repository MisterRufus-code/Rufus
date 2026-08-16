#!/usr/bin/env python3
"""
advisor.py — turn the measurements into what to change before the next run.

WHY THIS IS NOT A MODEL. The dashboard was asked to be smart, and the tempting
reading of that is a language model reading the logs and offering opinions. But
almost everything worth saying about this pipeline right now is already in the
numbers, and a number carries something an opinion cannot: it is checkable, it
is the same tomorrow, and it names the exact lever. "The setting clause was on
half the shots in four of your last six runs, and SETTING_SHARE caps it" is
worth more than a paragraph of plausible advice, and costs nothing to produce.

So this reads run_review's findings, the score and hold history, and the
settings currently in force, and returns a short list of concrete changes, each
with the setting that makes it and the value to set. A model can be layered on
top later for the things that genuinely need judgement — whether a picture
matches its line, whether a face reads as fear — but it should never be the
first answer to a question arithmetic already settles.

WHAT MAKES A GOOD SUGGESTION HERE, learned from the warnings this repo has
already had to walk back: it must be rare, specific, and actionable. A check
that fires on most runs is one people scroll past, which is worse than no check
at all — that exact mistake shipped twice in this codebase (the drift warning
at seven of ten, the unshown-noun list at twenty-nine items).

CONTRACT: pure and read-only. Takes data, returns advice. No files, no network,
no model, and nothing it says changes anything until a human clicks.
"""

from __future__ import annotations

# How many of the recent runs must show a finding before it is worth raising as
# a systematic problem rather than one bad seed.
SYSTEMATIC_SHARE = 0.4

# Below this average score the writing is the bottleneck and no amount of
# picture tuning will help.
WEAK_SCORE = 6.5

# The fix for each recurring finding: what to change, and to what. `setting` is
# None when the change is a code-level constant rather than something the owner
# can set — worth saying anyway, because knowing a knob does NOT exist stops
# somebody hunting for it.
_REMEDIES = {
    "setting_clause_everywhere": {
        "title": "The location is restated in too many shots",
        "why": ("Past a third of the sequence the clause stops being a "
                "reminder and becomes a second description competing with "
                "each shot's own, on frame after frame."),
        "setting": None,
        "action": ("storyboard.SETTING_SHARE caps this at a quarter of the "
                   "shots. If it is still firing above that, the storyboard is "
                   "marking shots as in-setting that are not."),
    },
    "thread_everywhere": {
        "title": "The through-line is restated in too many shots",
        "why": ("A thread named in every shot stops connecting them and "
                "starts repeating them — the same failure as carrying a mood "
                "forward."),
        "setting": None,
        "action": ("storyboard.THREAD_SHARE caps it, and a shot that already "
                   "names the object is skipped."),
    },
    "one_object_dominates": {
        "title": "One object is in most of the frames",
        "why": ("This is the \"why is everything coins\" report as a number. "
                "A sequence built around one prop has nothing to cut to."),
        "setting": None,
        "action": ("The storyboard now measures its own plan and re-plans the "
                   "surplus shots before anything is rendered, so a run made "
                   "since that change should not show this. If it does, the "
                   "re-plan is either switched off (RUFUS_STORYBOARD_REPAIR) "
                   "or the replacements are coming back naming the same "
                   "object — the run's log says which. Older runs keep "
                   "reporting it until newer ones are measured."),
        # NOT "the script's fault, check the Insights page", which is what
        # this said while the pipeline had no answer for it. Advice that
        # explains a defect instead of naming the lever is advice nobody can
        # act on, and it aged into advice that contradicted the fix.
    },
    "pictures_held_too_long": {
        "title": "Pictures are held too long",
        "why": ("A still on screen past five seconds stops reading as "
                "emphasis and starts reading as a stall — the defect a viewer "
                "feels without being able to name, and answers by swiping."),
        "setting": None,
        "action": ("The beat count is derived from the script and the cut "
                   "planner now weights each shot by its tone, so a long hold "
                   "is usually deliberate. A hold past five seconds is not: "
                   "check whether the narration has a long stretch with no "
                   "pause in it for a cut to land on."),
    },
    "repeated_images": {
        "title": "Several keyframes are near-identical",
        "why": ("The freshness gate rejects identical bytes, so these got "
                "through by a pixel — on screen they read as the same "
                "picture twice."),
        "setting": None,
        "action": ("Usually one object dominating, or a beat whose prompt is "
                   "a paraphrase of its neighbour."),
    },
    "text_props_everywhere": {
        "title": "The storyboard keeps asking for signs and screens",
        "why": ("Lettering is the hardest thing for an image model to draw "
                "without garbling, and garbled lettering is the clearest "
                "sign a machine made the video."),
        "setting": None,
        "action": ("The defusal clause catches them, but a shot built around "
                   "a document is a weak shot even with blank surfaces."),
    },
    "few_pictures": {
        "title": "Too few pictures for the length",
        "why": ("A picture per five spoken words is roughly a two-second "
                "shot; far fewer and a still sits on screen long enough for a "
                "viewer to feel it."),
        # NO LONGER "SET SD_CLIPS=24". It was, and the owner applied it — an
        # hour after the beat count became adaptive and the cut planner
        # started weighting shots by tone. A flat 24 overrides both, and on a
        # ninety-word script that is more cuts than the narration has pauses
        # to put them on, which is exactly the machine-gun run that prompted
        # the rhythm work. Advice that undoes the last fix is worse than none.
        "clears": "SD_CLIPS",
        "setting": None,
        "action": ("The beat count is computed from the script — about one "
                   "picture per five spoken words — and the splitter stops "
                   "before it cuts mid-phrase, because a cut with no pause "
                   "under it has nowhere real to land. If SD_CLIPS is set, "
                   "clearing it lets the script decide."),
    },
}


def _pct(x: float) -> int:
    return int(round(x * 100))


def advise(patterns: dict, stats: dict | None = None,
           settings: dict | None = None,
           rejections: list[dict] | None = None) -> list[dict]:
    """The changes worth making, most important first.

    Each entry is {title, why, action, severity, setting, value, evidence}.
    `setting`/`value` are present only when the dashboard can apply the change
    with one click; everything else is advice a human acts on.
    """
    stats = stats or {}
    settings = settings or {}
    out: list[dict] = []

    runs = patterns.get("runs_reviewed", 0)

    # 1. What the measurements say keeps happening.
    for r in patterns.get("recurring", []):
        if r.get("share", 0) < SYSTEMATIC_SHARE:
            continue
        rem = _REMEDIES.get(r["id"])
        if not rem:
            continue
        item = {
            "id": r["id"],
            "title": rem["title"],
            "why": rem["why"],
            "action": rem["action"],
            "severity": "high" if r["share"] >= 0.6 else "medium",
            "evidence": f"{r['runs']} of {runs} measured runs ({_pct(r['share'])}%)",
            "setting": rem.get("setting"),
            "value": rem.get("value"),
        }
        # A REMEDY CAN BE TO UNSET SOMETHING. The pipeline's own defaults are
        # now better than most fixed values a page could suggest, so "stop
        # overriding this" is a real fix and needs to be as clickable as
        # "set it to 24" was — offered only when the override is actually
        # there, since telling someone to clear a setting they never set is
        # the same noise as telling them to set one they already did.
        clears = rem.get("clears")
        if clears and settings.get(clears):
            item["setting"] = clears
            item["value"] = ""
            item["clear_label"] = f"Clear {clears} (currently {settings[clears]})"
        # ADVICE ALREADY FOLLOWED IS HISTORY, NOT ADVICE. Removing the button
        # was not enough: a live page showed "Too few pictures for the length"
        # as the top HIGH finding with "Already set to 24" tacked on the end,
        # and drove the readiness line with it. The measurements behind it are
        # of runs that PREDATE the change — the fix is in, and what is left is
        # a record of the runs made before it. Saying so, quietly, at the
        # bottom, is the honest shape; leading with it tells the owner to do
        # something they have already done.
        current = settings.get(item["setting"]) if item["setting"] else None
        if item["setting"] and current == item["value"]:
            item["title"] += " (already fixed)"
            item["action"] = (f"{item['setting']} is already {current}. These "
                              f"measurements are of runs made before that, so "
                              f"this clears once newer runs are measured.")
            item["severity"] = "low"
            item["done"] = True
            item["setting"] = item["value"] = None
        out.append(item)

    # 2. The writing, which no picture setting can rescue.
    avg = stats.get("avg_score") or 0
    if stats.get("total", 0) >= 5 and 0 < avg < WEAK_SCORE:
        out.append({
            "id": "weak_scripts",
            "title": f"Scripts are averaging {avg}/10",
            "why": ("Below about 7 the bottleneck is the writing, and the "
                    "pictures are illustrating a weak script faithfully. The "
                    "seed is upstream of everything: a source with no event "
                    "in it cannot produce a story."),
            "action": ("Check the Failures page for which gate is rejecting "
                       "most often. If it is the seed supervisor, raise "
                       "RUFUS_SEED_TRIES so a rejected source is replaced "
                       "rather than used."),
            "severity": "high",
            "evidence": f"average over the last {stats['total']} videos",
            "setting": "RUFUS_SEED_TRIES",
            "value": "6",
        })

    # 3. Held videos piling up is a review problem, not a pipeline one.
    held = stats.get("held", 0)
    if held >= 10:
        out.append({
            "id": "review_backlog",
            "title": f"{held} videos are waiting for review",
            "why": ("Nothing publishes itself, so a backlog means the channel "
                    "is not shipping. It also starves the learning loop: with "
                    "nothing published there are no view counts to tell good "
                    "hooks from bad ones."),
            "action": "Approve or reject from the front page.",
            "severity": "medium",
            "evidence": f"{held} pending of {stats.get('total', 0)}",
            "setting": None,
            "value": None,
        })

    # 4. Nothing is measured yet.
    if not runs:
        out.append({
            "id": "no_measurements",
            "title": "No runs measured yet",
            "why": ("Everything on this page is computed from run_review's "
                    "output, and there is none."),
            "action": ("Run `python scripts/run_review.py --all` once to "
                       "measure the runs already on disk. After that every "
                       "finished run measures itself."),
            "severity": "low",
            "evidence": "",
            "setting": None,
            "value": None,
        })

    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda d: order.get(d["severity"], 3))
    return out


def readiness(patterns: dict, stats: dict | None = None,
              settings: dict | None = None) -> dict:
    """A one-line answer to "is this channel in shape today".

    Deliberately blunt and deliberately not a score out of a hundred: a made-up
    composite number invites tuning the number. Three states, each with the one
    thing standing between here and the next one.
    """
    stats = stats or {}
    # Anything already acted on is excluded: a readiness line that reads "needs
    # work — too few pictures" when the beat count was raised an hour ago is
    # reporting the past as the present.
    items = [i for i in advise(patterns, stats, settings) if not i.get("done")]
    high = [i for i in items if i["severity"] == "high"]
    if high:
        return {"state": "needs work", "detail": high[0]["title"]}
    if items:
        return {"state": "workable", "detail": items[0]["title"]}
    if not patterns.get("runs_reviewed"):
        return {"state": "unmeasured", "detail": "no runs measured yet"}
    return {"state": "good", "detail": "every measured run is inside its thresholds"}
