"""The examples in the pre-analysis prompt, and the run they wrecked.

A few-shot example is not inert. The generic branch of this prompt carried
three of them, all left over from when this pipeline was seeded by Reddit:

    Right: '$2.4M by 38. Still scared to retire.'
    for a frugal savings story: 'hardware store tools, leaky faucet repair…'
    'the envelope from the IRS, unopened on the kitchen counter…'

All three point at modern American personal finance, and on a money-HISTORY
channel they pulled every analysis in that direction. Two consecutive live
runs prove it: a Wikipedia article on the Hungarian pengő and a page of a
1901 Iowa newspaper both came back with

    2. HOOK ANGLE: '$2.4M by 38. Still scared to retire.'

verbatim — the example returned as the answer. The newspaper run then wrote a
whole video about retirement anxiety and a patent-medicine tonic, which the
owner reviewed as having no financial history in it at all. They were right,
and the cause was four words in a prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer as sw  # noqa: E402

# The prompt is built by string concatenation, so a phrase that spans two
# source lines reads as `must " "not appear`. Joining the fragments back gives
# the text the model is actually handed, which is what these assert on.
SRC = " ".join(Path(sw.__file__).read_text(encoding="utf-8").split()).replace('" "', "")


def test_the_leaked_hook_example_is_gone():
    """It came back as the HOOK ANGLE on two unrelated historical sources."""
    assert "Still scared to retire" not in SRC


def test_no_personal_finance_examples_remain_in_the_generic_branch():
    """The channel is money HISTORY. An example about a leaky faucet or an
    IRS envelope is a different channel, and the model follows the example
    rather than the instruction when the two disagree."""
    for leftover in ("leaky faucet repair", "envelope from the IRS",
                     "A Reddit user saved"):
        assert leftover not in SRC, leftover


def test_the_examples_are_marked_as_shapes_not_content():
    """Naming the failure in the prompt is what stops the next model from
    repeating it — the same reason every other rule in this file quotes the
    run that produced it."""
    assert "THE EXAMPLES ABOVE ARE SHAPES" in SRC
    assert "must not appear in your answer" in SRC
    assert "Every one of the 8 items must come from THIS source." in SRC


def test_the_replacement_examples_belong_to_this_channel():
    assert "Spain's silver made Spain poorer" in SRC
    assert "silver coins, mint press, market stall" in SRC
