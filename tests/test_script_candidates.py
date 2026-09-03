"""Three scripts on one topic, and the record of which one a person preferred.

WHY THIS LAYER EXISTS AT ALL. Nothing published through this pipeline has view
counts, so feedback_analyzer has never run and config/learnings.json does not
exist — the writer scores its own homework against thresholds it also owns. A
person ruling between three finished scripts produces the one thing the score
cannot: a labelled preference pair, on the day it is clicked. Which is why the
two that lose are kept, and why choose_candidate marks both sides in one call.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager  # noqa: E402
import script_candidates as sc  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    return db_manager


# ── one style each ───────────────────────────────────────────────────────────

def test_each_candidate_is_pinned_to_a_different_style():
    """Three samples from one prompt are three versions of one script — the
    model reaches for its favourite opening whatever the temperature. The set
    is only a choice if the candidates differ in shape."""
    cfg = {"hook_styles": ["counterintuitive", "shocking_stat", "warning"]}
    assert sc.styles_for(cfg, 3) == ["counterintuitive", "shocking_stat",
                                     "warning"]


def test_more_candidates_than_styles_cycles_rather_than_truncating():
    cfg = {"hook_styles": ["a", "b"]}
    assert sc.styles_for(cfg, 3) == ["a", "b", "a"]


def test_a_niche_with_no_styles_still_gets_a_choice():
    """Fail-open: three from one distribution is what every script in this repo
    got before hook_styles was wired at all. Refusing to write any is worse."""
    assert sc.styles_for({}, 3) == ["", "", ""]


def test_the_pin_is_put_back_and_not_deleted():
    """The dashboard may run this in a process that already had the variable
    set. A helper that silently clears its caller's environment is a bug that
    only appears on the second call."""
    os.environ["RUFUS_HOOK_STYLE"] = "warning"
    try:
        with sc._pinned_style("shocking_stat"):
            assert os.environ["RUFUS_HOOK_STYLE"] == "shocking_stat"
        assert os.environ["RUFUS_HOOK_STYLE"] == "warning"
    finally:
        os.environ.pop("RUFUS_HOOK_STYLE", None)


def test_an_unset_pin_is_left_unset():
    os.environ.pop("RUFUS_HOOK_STYLE", None)
    with sc._pinned_style("warning"):
        assert os.environ["RUFUS_HOOK_STYLE"] == "warning"
    assert "RUFUS_HOOK_STYLE" not in os.environ


def test_the_style_block_narrows_to_the_pinned_one():
    """The pin has to reach the prompt, not just the environment. money_history
    declares three styles; a pinned candidate must be shown one."""
    import script_writer
    cfg = {"hook_styles": ["counterintuitive", "shocking_stat", "warning"]}
    with sc._pinned_style("warning"):
        block = script_writer._hook_styles_block(cfg)
    assert "warning" in block
    assert "shocking_stat" not in block
    assert "Cover more than one" not in block, (
        "the multi-style instruction is the opposite of what a pinned "
        "candidate is for")


# ── writing the set ──────────────────────────────────────────────────────────

def _writer(monkeypatch, results):
    """Feed write_for a queue of writer results, one per candidate."""
    import research
    import script_writer
    monkeypatch.setattr(research, "get_seed",
                        lambda niche, topic=None: {"content": "a real source"})
    monkeypatch.setattr(research, "_load_niche", lambda: ({}, "money_history"))
    monkeypatch.setattr(script_writer, "_load_niche",
                        lambda: ({"hook_styles": ["a", "b", "c"]}, "n"))
    monkeypatch.setattr(script_writer, "preanalyze",
                        lambda seed, scene="": ("analysis", "run1", 0.01))
    queue = list(results)

    def _write(*a, **k):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(script_writer, "write_script_until_good", _write)


def test_the_seed_is_researched_once_for_the_whole_set(db, monkeypatch):
    """All three are about the same topic and check against the same source.
    Paying for the research and the pre-analysis three times buys nothing."""
    import research
    import script_writer
    calls = {"seed": 0, "pre": 0}
    monkeypatch.setattr(script_writer, "_load_niche",
                        lambda: ({"hook_styles": ["a", "b", "c"]}, "n"))
    monkeypatch.setattr(research, "_load_niche", lambda: ({}, "money_history"))

    def _seed(niche, topic=None):
        calls["seed"] += 1
        return {"content": "src"}

    def _pre(seed, scene=""):
        calls["pre"] += 1
        return ("a", "r1", 0.0)
    monkeypatch.setattr(research, "get_seed", _seed)
    monkeypatch.setattr(script_writer, "preanalyze", _pre)
    monkeypatch.setattr(script_writer, "write_script_until_good",
                        lambda *a, **k: {"script": "Line one.\nRest.",
                                         "score": 8, "cost_usd": 0.02})

    sc.write_for("Panic of 1893", proposal_id=1)
    assert calls == {"seed": 1, "pre": 1}
    assert len(db.candidates(proposal_id=1)) == 3


def test_one_style_that_fails_does_not_cost_the_other_two(db, monkeypatch):
    """Fail-open per candidate, like the rest of the pipeline. Two candidates
    with a printed reason is a smaller loss than none."""
    _writer(monkeypatch, [
        {"script": "First line.\nBody.", "score": 8, "cost_usd": 0.02},
        RuntimeError("the fact gate rejected every cycle"),
        {"script": "Third line.\nBody.", "score": 7, "cost_usd": 0.02},
    ])
    saved = sc.write_for("T", proposal_id=1)
    assert len(saved) == 2
    assert len(db.candidates(proposal_id=1)) == 2


def test_an_empty_script_is_not_saved_as_a_candidate(db, monkeypatch):
    """A blank card in a choose-one page is a choice a person cannot make and
    a row that scores zero forever."""
    _writer(monkeypatch, [
        {"script": "   ", "score": 9, "cost_usd": 0.02},
        {"script": "Real.\nBody.", "score": 7, "cost_usd": 0.02},
        {"script": "Also real.\nBody.", "score": 6, "cost_usd": 0.02},
    ])
    assert len(sc.write_for("T", proposal_id=1)) == 2


def test_the_set_stops_at_its_cost_ceiling(db, monkeypatch):
    """An agent with a model and no ceiling is a runaway bill. Three scripts
    are cheap; three scripts on every proposal in a queue nobody read is not."""
    monkeypatch.setenv("RUFUS_CANDIDATE_MAX_COST", "0.03")
    _writer(monkeypatch, [
        {"script": "One.\nB.", "score": 8, "cost_usd": 0.02},
        {"script": "Two.\nB.", "score": 8, "cost_usd": 0.02},
        {"script": "Three.\nB.", "score": 8, "cost_usd": 0.02},
    ])
    assert len(sc.write_for("T", proposal_id=1)) < 3


def test_the_hook_is_the_scripts_first_line(db, monkeypatch):
    """write_script_until_good's documented return shape has no "hook" key —
    script, run_id, score, criterion_scores, attempts_used, final_temperature,
    reasoning, cost_usd. Asking for one returns "" forever, and an empty column
    nobody displays is the kind of wrong that survives for months. This is how
    metadata_writer and the uploader's legacy path both get it."""
    _writer(monkeypatch, [
        {"script": "You checked your portfolio today.\nThat is the problem.",
         "run_id": "r1", "score": 9, "criterion_scores": {},
         "attempts_used": 1, "final_temperature": 0.9, "reasoning": "",
         "cost_usd": 0.03}] * 3)
    sc.write_for("T", proposal_id=1)
    row = db.candidates(proposal_id=1)[0]
    assert row["hook"] == "You checked your portfolio today."


# ── the choice, and what it was chosen over ──────────────────────────────────

def _three(db, proposal_id=1):
    ids = []
    for i, style in enumerate(["counterintuitive", "shocking_stat", "warning"]):
        ids.append(db.save_candidate(
            proposal_id=proposal_id, channel="main_en", niche="money_history",
            topic="T", hook_style=style, hook=f"h{i}", script=f"s{i}",
            score=7 + i, cost_usd=0.02))
    return ids


def test_choosing_one_rejects_its_siblings(db):
    """The value is the PAIR. A chosen row alone says a script was made; a
    chosen row beside the two it beat says a person compared three."""
    ids = _three(db)
    got = db.choose_candidate(ids[1])
    assert got["id"] == ids[1] and got["status"] == "chosen"
    rows = {r["id"]: r["status"] for r in db.candidates(proposal_id=1)}
    assert rows[ids[1]] == "chosen"
    assert rows[ids[0]] == rows[ids[2]] == "rejected"


def test_the_losers_are_kept_not_deleted(db):
    """They are half of every preference pair, and the only training signal
    this channel can collect before it has view counts."""
    ids = _three(db)
    db.choose_candidate(ids[0])
    assert len(db.candidates(proposal_id=1)) == 3


def test_a_second_click_cannot_re_decide_a_set(db):
    """A slow page invites a double click. Without this the second one flips
    which sibling lost, and the pair records the wrong preference."""
    ids = _three(db)
    assert db.choose_candidate(ids[0])
    assert db.choose_candidate(ids[1]) is None
    rows = {r["id"]: r["status"] for r in db.candidates(proposal_id=1)}
    assert rows[ids[0]] == "chosen" and rows[ids[1]] == "rejected"


def test_choosing_an_unknown_candidate_is_a_none_not_a_crash(db):
    assert db.choose_candidate(9999) is None


def test_a_candidate_with_no_proposal_has_no_siblings_to_reject(db):
    """A manually requested topic writes candidates with proposal_id NULL, and
    NULL = NULL is not true in SQL — so the sibling UPDATE would match nothing
    and quietly call that a set of one. Two manual sets must not collide."""
    a = db.save_candidate(proposal_id=None, channel="c", niche="n", topic="A",
                          hook_style="", hook="h", script="s", score=8)
    b = db.save_candidate(proposal_id=None, channel="c", niche="n", topic="B",
                          hook_style="", hook="h", script="s", score=8)
    db.choose_candidate(a)
    rows = {r["id"]: r["status"] for r in db.candidates()}
    assert rows[a] == "chosen"
    assert rows[b] == "pending", "another topic's candidate is not a sibling"


def test_a_set_reads_best_first(db):
    """Three scripts in generation order buries the likeliest pick under two
    that were not."""
    _three(db)
    scores = [r["score"] for r in db.candidates(proposal_id=1)]
    assert scores == sorted(scores, reverse=True)


# ── the gates label, they do not reject ──────────────────────────────────────
#
# "Why do all the good scripts get blocked" was the complaint that started this
# whole flow. write_script_until_good escalates: a cycle below the score bar or
# failing the fact gate is binned and a completely fresh attempt runs on a
# different angle. That is right when one script is being written and nobody
# will look at it until it is done. Here it is wrong twice — three styles times
# three cycles is nine attempts for three cards, and the retry is trying to do
# the job the person is about to do. The different angle IS the other two.

def test_the_score_stops_deciding_and_the_fact_gate_keeps_deciding(monkeypatch):
    """TWO GATES, AND ONLY ONE OF THEM IS THE READER'S JOB.

    A score below the bar is a label — ruling on it is exactly why a person is
    here. An unsupported claim is not: they can tell which of three is better
    written and cannot tell whether the source backs "the U.S. dictated the
    rules"."""
    monkeypatch.delenv("RUFUS_CANDIDATE_CYCLES", raising=False)
    monkeypatch.setenv("RUFUS_SCRIPT_CYCLES", "3")
    with sc._relaxed_gates():
        assert os.environ["RUFUS_ACCEPT_ON_FACTS_ALONE"] == "1"
        assert os.environ["RUFUS_SCRIPT_CYCLES"] == "2"
    assert os.environ["RUFUS_SCRIPT_CYCLES"] == "3", "the caller's value is put back"
    assert "RUFUS_ACCEPT_ON_FACTS_ALONE" not in os.environ


def test_a_fact_gate_failure_gets_one_retry(monkeypatch):
    """THE RUN THAT PROMPTED THIS. Attempt 1 scored 9/10, tripped MIND-READ on
    a phrase, was capped to 4, and with a single cycle there was no second try
    — the relaxation meant to stop good scripts being blocked had made one
    phrasing flag fatal. A clean first draft still costs exactly one cycle,
    because the score no longer ends one."""
    import script_writer
    monkeypatch.delenv("RUFUS_CANDIDATE_CYCLES", raising=False)
    with sc._relaxed_gates():
        assert int(os.environ["RUFUS_SCRIPT_CYCLES"]) >= 2
        assert script_writer._accept_on_facts_alone()
    assert not script_writer._accept_on_facts_alone(), "off everywhere else"


def test_the_cycle_count_can_be_asked_for(monkeypatch):
    monkeypatch.setenv("RUFUS_CANDIDATE_CYCLES", "4")
    with sc._relaxed_gates():
        assert os.environ["RUFUS_SCRIPT_CYCLES"] == "4"


def test_a_low_scoring_script_is_still_offered(db, monkeypatch):
    """The score is shown, not enforced. A 5/10 a person likes beats an 8/10
    they do not, and this page is where that gets decided."""
    _writer(monkeypatch, [{"script": "Weak.\nBut interesting.", "score": 4,
                           "cost_usd": 0.02}] * 3)
    sc.write_for("T", proposal_id=1)
    rows = db.candidates(proposal_id=1)
    assert len(rows) == 3
    assert all(r["score"] == 4 for r in rows)


def test_a_script_the_fact_gate_failed_is_shown_with_its_reason(db, monkeypatch):
    """REPORTED, NOT REMOVED, and the asymmetry is deliberate. A reviewer can
    tell which of three is better written. They cannot tell whether the
    denarius really lost ninety per cent of its silver by 250 AD — the source
    is the only thing that knows and the gate is the only thing that reads it.
    So it is still offered, and it says which claim."""
    _writer(monkeypatch, [{
        "script": "A claim.\nBody.", "score": 8, "cost_usd": 0.02,
        "fact_ok": False,
        "fact_reason": "the source does not give a silver percentage"}] * 3)
    sc.write_for("T", proposal_id=1)
    row = db.candidates(proposal_id=1)[0]
    assert row["fact_ok"] == 0
    assert "silver percentage" in row["fact_reason"]


def test_a_clean_script_is_recorded_as_clean(db, monkeypatch):
    """The default has to be "fine" rather than "unknown": write_script's
    documented return shape has no fact_ok key when nothing checked, and a
    missing key that reads as a failure would put a red warning on every card
    the moment the gate is skipped."""
    _writer(monkeypatch, [{"script": "A.\nB.", "score": 8, "cost_usd": 0.02}] * 3)
    sc.write_for("T", proposal_id=1)
    assert db.candidates(proposal_id=1)[0]["fact_ok"] == 1


def test_the_command_line_creates_the_schema_before_it_writes(tmp_path,
                                                              monkeypatch):
    """"no such table: script_candidates", after paying for a script.

    The dashboard calls init_db at startup and every test fixture calls it, so
    every path that had ever been exercised already had the tables. The one
    path nobody had run was the command line — built, tested, never actually
    run, which is this repo's oldest bug wearing a new hat."""
    for mod in ("script_candidates", "gallery_variants", "voice_takes"):
        src = (Path(__file__).parent.parent / "scripts" / f"{mod}.py"
               ).read_text(encoding="utf-8")
        main = src.split('if __name__ == "__main__":', 1)[1]
        assert "db_manager.init_db()" in main, mod
