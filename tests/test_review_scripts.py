"""Why a script scored what it scored.

The rubric records its reasoning and its per-criterion marks at scoring time.
Nothing printed them. So when four consecutive scripts came back 3 and 4
against a target of 7, the answer was sitting in the database and the tool for
reviewing scripts showed the total and the text and nothing in between.

A score with no reasoning beside it is a number to argue with. With the
reasoning it is a note about what to change.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import review_scripts  # noqa: E402


def _row(**over):
    base = {
        "id": 7, "upload_date": "2026-08-18", "niche": "money_history",
        "score": 3, "youtube_id": "", "video_file": "", "script_full": "BODY",
        "script_hook": "HOOK", "seed_type": "", "seed_source": "",
        "seed_content": "",
        "score_specificity": 1, "score_hook": 1, "score_compression": 0,
        "score_loop": 1, "score_human": 0,
        "attempts_used": 3, "final_temperature": 1.1,
        "score_reasoning": "SPECIFICITY: 1 — no dates, no names, no figures.",
        "hold_reason": "",
    }
    base.update(over)
    return base


def _render(row) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        review_scripts._print_video(row)
    return buf.getvalue()


def test_the_reasoning_is_printed():
    out = _render(_row())
    assert "WHY THIS SCORE" in out
    assert "no dates, no names, no figures" in out


def test_the_per_criterion_marks_are_printed():
    """The total says 3. The breakdown says WHICH three, and compression 0 and
    specificity 0 are different problems with different fixes."""
    out = _render(_row())
    assert "specificity 1/3" in out
    assert "compression 0/3" in out


def test_how_many_attempts_it_took_is_printed():
    """Three attempts ending at 1.1 means the writer escalated all the way and
    still could not clear the bar — which is a fact about the topic, not the
    prompt."""
    out = _render(_row())
    assert "3 attempt(s)" in out
    assert "1.1" in out


def test_a_hold_reason_is_printed_when_there_is_one():
    out = _render(_row(hold_reason="below score threshold"))
    assert "below score threshold" in out


def test_a_row_with_no_rubric_columns_still_prints():
    """Rows predate these columns. A reviewer that raises on old history is
    worse than one that shows less of it."""
    row = _row()
    for k in ("score_specificity", "score_hook", "score_compression",
              "score_loop", "score_human", "attempts_used",
              "final_temperature", "score_reasoning", "hold_reason"):
        row.pop(k)
    out = _render(row)
    assert "BODY" in out
    assert "WHY THIS SCORE" not in out


def test_the_script_itself_is_still_the_last_thing_printed():
    """The reasoning goes above the script, not after it — the script is the
    long part and anything printed under it is not read."""
    out = _render(_row())
    assert out.index("WHY THIS SCORE") < out.index("BODY")
