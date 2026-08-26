#!/usr/bin/env python3
"""
script_candidates.py — three scripts on one topic, for a person to choose between.

WHY THREE AND NOT ONE. Nothing published through this pipeline has view counts
yet, so feedback_analyzer has never run and config/learnings.json does not
exist: the writer is scoring its own homework against thresholds, and the only
judgement in the loop is a number it assigned itself. A person choosing between
three finished scripts produces something the score cannot — a labelled
preference pair, on the day it is clicked rather than the week the analytics
arrive. That is what script_candidates rows are for, and why the two that lose
are kept rather than deleted.

WHY THREE AND NOT FIVE. The limit is not what it costs to write them, it is
what it costs to read them. Three hundred-word scripts is about a minute of
attention; five is the point where a list stops being read and starts being
skimmed, and a skimmed choice is a coin flip wearing a lab coat.

ONE STYLE EACH. Three samples from one prompt are three versions of one script
— the model reaches for its favourite opening whatever the temperature. Each
candidate is therefore pinned to one of the niche's declared hook_styles
(money_history: counterintuitive / shocking_stat / warning) via
RUFUS_HOOK_STYLE, so the set is a real choice about how to open rather than
three haircuts.

THE SEED IS FETCHED AND ANALYSED ONCE. All three candidates are about the same
topic and check against the same source, so the research call and the
pre-analysis are shared. Only the prose is paid for three times.

ONE CYCLE EACH, NOT THREE. write_script_until_good escalates: a cycle that
scores below the bar or fails the fact gate is thrown away and a COMPLETELY
fresh attempt runs on a different angle, up to RUFUS_SCRIPT_CYCLES times. That
is the right shape when one script is being written and nobody will look at it
until it is finished. It is the wrong shape here twice over — three styles
times three cycles is nine full attempts for three cards, and more importantly
the retry is trying to do the job the person is about to do. The different
angle IS the other two candidates.

So the gates stop rejecting and start labelling. The score and the fact gate
still run, and both land on the card where a reviewer can weigh them; what
stops happening is a threshold quietly binning a script nobody was shown. Good
scripts being blocked is the complaint this whole flow exists to answer.

THE FACT GATE IS REPORTED, NOT REMOVED, and that is a deliberate asymmetry. A
person reading three scripts can tell which is better written. They cannot tell
whether the denarius really lost ninety per cent of its silver by 250 AD — the
source is the only thing that knows, the gate is the only thing that reads it,
and a channel about history that ships invented figures has nothing left to
sell. So a candidate that fails it is still shown, still choosable, and carries
a warning that says which claim.

    RUFUS_CANDIDATE_STYLES    3      how many candidates to write
    RUFUS_CANDIDATE_MAX_COST  0.60   hard USD ceiling across the whole set
    RUFUS_CANDIDATE_CYCLES    1      retries per candidate (the person is the retry)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_N = 3
DEFAULT_MAX_COST = 0.60


def how_many() -> int:
    try:
        return max(1, int(os.environ.get("RUFUS_CANDIDATE_STYLES", DEFAULT_N)))
    except ValueError:
        return DEFAULT_N


def max_cost() -> float:
    try:
        return max(0.0, float(os.environ.get("RUFUS_CANDIDATE_MAX_COST",
                                             DEFAULT_MAX_COST)))
    except ValueError:
        return DEFAULT_MAX_COST


def styles_for(niche_cfg: dict, n: int) -> list[str]:
    """The hook styles to write one candidate each in.

    Falls back to unpinned candidates rather than refusing: a niche with no
    hook_styles declared still deserves a choice of three, it just gets three
    from the same distribution — which is what every script in this repo got
    before hook_styles was wired at all.
    """
    declared = [str(x).strip()
                for x in (niche_cfg.get("hook_styles") or []) if str(x).strip()]
    if not declared:
        return [""] * n
    out = []
    while len(out) < n:
        out.extend(declared)
    return out[:n]


class _relaxed_gates:
    """RUFUS_SCRIPT_CYCLES pinned low for the duration of a candidate set.

    Set through the environment rather than a parameter because that is where
    write_script_until_good reads it, and threading a "how hard should you try"
    argument down through three call layers to reach the same variable would be
    the same decision written twice. An explicit RUFUS_CANDIDATE_CYCLES wins if
    someone wants the old escalation back for a set.
    """

    def __init__(self):
        self.before = None

    def __enter__(self):
        self.before = os.environ.get("RUFUS_SCRIPT_CYCLES")
        os.environ["RUFUS_SCRIPT_CYCLES"] = str(
            os.environ.get("RUFUS_CANDIDATE_CYCLES", "1"))
        return self

    def __exit__(self, *exc):
        if self.before is None:
            os.environ.pop("RUFUS_SCRIPT_CYCLES", None)
        else:
            os.environ["RUFUS_SCRIPT_CYCLES"] = self.before
        return False


class _pinned_style:
    """RUFUS_HOOK_STYLE for the duration of one candidate, then put back.

    Restoring the previous value rather than deleting it: the dashboard runs
    this in a process that may have been started with the variable already set,
    and a helper that silently clears its caller's environment is a bug that
    only shows up on the second call.
    """

    def __init__(self, style: str):
        self.style = style
        self.before = None

    def __enter__(self):
        self.before = os.environ.get("RUFUS_HOOK_STYLE")
        if self.style:
            os.environ["RUFUS_HOOK_STYLE"] = self.style
        else:
            os.environ.pop("RUFUS_HOOK_STYLE", None)
        return self

    def __exit__(self, *exc):
        if self.before is None:
            os.environ.pop("RUFUS_HOOK_STYLE", None)
        else:
            os.environ["RUFUS_HOOK_STYLE"] = self.before
        return False


def write_for(topic: str, *, proposal_id: int | None = None,
              niche: str | None = None, channel: str | None = None,
              n: int | None = None) -> list[dict]:
    """Write and store the candidate set for `topic`. Returns the rows saved.

    Fail-open per candidate, like the rest of the pipeline: one style that
    cannot produce a factual script does not cost the other two. Returning two
    candidates with a printed reason is a smaller loss than returning none.
    """
    import db_manager
    import research
    import script_writer

    n = n or how_many()
    if niche is None:
        try:
            niche = research._load_niche()[1]
        except Exception:
            niche = "money_history"
    try:
        niche_cfg = script_writer._load_niche()[0]
    except Exception:
        niche_cfg = {}
    if channel is None:
        try:
            from channel_config import load_channel
            channel = load_channel().id
        except Exception:
            channel = "main_en"

    seed = research.get_seed(niche, topic=topic)
    scene = (seed.get("content", "") or "")[:400]
    analysis, run_id, seed_cost = script_writer.preanalyze(seed, scene)

    ceiling = max_cost()
    spent = float(seed_cost or 0)
    saved: list[dict] = []

    for style in styles_for(niche_cfg, n):
        if spent >= ceiling:
            print(f"[candidates] ${spent:.2f} spent and the ceiling is "
                  f"${ceiling:.2f} — stopping at {len(saved)} candidate(s)")
            break
        try:
            with _pinned_style(style), _relaxed_gates():
                result = script_writer.write_script_until_good(
                    scene, seed=seed, precomputed_analysis=analysis,
                    run_id=run_id)
        except Exception as e:
            print(f"[candidates] {style or 'unpinned'}: no script ({e})")
            continue

        script_text = (result.get("script", "") or "").strip()
        if not script_text:
            print(f"[candidates] {style or 'unpinned'}: writer returned nothing")
            continue
        cost = float(result.get("cost_usd", 0) or 0)
        spent += cost
        # The hook is the script's first line, derived rather than read: the
        # writer's documented return shape has no "hook" key, and asking for
        # one returns "" forever. scout.py makes the same derivation for the
        # same reason.
        fact_ok = bool(result.get("fact_ok", True))
        fact_reason = str(result.get("fact_reason") or "")
        row_id = db_manager.save_candidate(
            proposal_id=proposal_id, channel=channel, niche=niche, topic=topic,
            hook_style=style, hook=script_text.split("\n")[0][:300],
            script=script_text, score=int(result.get("score", 0) or 0),
            run_id=run_id, cost_usd=cost,
            fact_ok=fact_ok, fact_reason=fact_reason)
        saved.append({"id": row_id, "hook_style": style,
                      "score": int(result.get("score", 0) or 0),
                      "fact_ok": fact_ok, "cost_usd": cost})
        flag = "" if fact_ok else f" ⚠ {fact_reason[:80]}"
        print(f"[candidates] #{row_id} {style or 'unpinned'} — "
              f"{result.get('score', 0)}/10 — ${cost:.3f}{flag}")

    print(f"[candidates] {len(saved)} candidate(s) for {topic!r}, "
          f"${spent:.2f} total")
    return saved


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("topic")
    ap.add_argument("--proposal", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    write_for(a.topic, proposal_id=a.proposal, n=a.n)
