# Changelog

What changed in each build, in the words of what it does for the person using
it rather than the diff. Versions are bumped by hand in `scripts/version.py`,
in the same commit as the change that earns one — a version that moves on its
own says a release happened when one did not.

Every video records the build that made it, so the "Did the code get better?"
panel on the Measure page compares these against real scores instead of memory.

## 0.5.0

The build that made Rufus something you could hand to someone else.

**One video at a time.** A project that is open or rendering blocks starting
another, refused at the request rather than by hiding a button. "Rendering"
counts as unfinished, so the block no longer lifts while the GPU is still
cutting the video together. A render that lands closes its project by itself;
one that crashed can be closed by hand, because a block with no way out is
worse than no block. The four queue pages show the options belonging to the
video in flight and nothing else — they had become inventories of every
half-made video ever started.

**Nothing is spent until the run can finish.** A fresh install with no API key
used to research for minutes and then stop with a Python errno naming a file
the owner had never heard of. `preflight.py` now runs first and answers in
milliseconds, naming what is missing, why the run needs it, and the command
that fixes it. Every check is conditional on what you actually selected.

**Rufus knows what it is made of.** `licensing.py` separates two questions that
were being collapsed — may I sell copies of the software, and may I make money
from the videos — works out which models and services the current
configuration switches on, and names every one whose terms nobody has read. It
ships with nothing cleared on purpose. Also at `/licence` in the dashboard.

**Subtitle styles are a choice, not a constant.** Six presets picked on the
render page. The default overrides nothing, so existing renders are unchanged.

**A render tells you how it ended.** Six outcomes were possible and one of them
made a sound; a video held by QC finished in silence. Every ending is now
recorded and announced — the video to Discord, a push to your phone — when the
render page asks for it.

**A room to watch the drawing.** `/drawing/<set>` fills in as each picture
lands, and says so when the renderer has stopped rather than quoting an ETA at
a dead run. The gallery loop gives up after four empty draws instead of burning
every remaining prompt against a ComfyUI that is not there.

**Backups.** `rufus.db` holds every judgement anybody has made on this channel
and nothing copied it. One verified snapshot a day, taken by whichever of the
dashboard or a run opens the database first; twelve kept; restore from the
command line or from `/system`, with the displaced database moved aside rather
than deleted.

**Versions.** This file, `scripts/version.py`, a build line at the foot of
every dashboard page, and the build stamped onto every video row.

## Before 0.5.0

Untracked. The project ran as one person's pipeline and its history is the git
log.
