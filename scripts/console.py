#!/usr/bin/env python3
"""
console.py — make printing incapable of killing a run.

THE CRASH THIS EXISTS FOR, reported from a real run:

    File "...\\Lib\\encodings\\cp1255.py", line 19, in encode
      return codecs.charmap_encode(input, self.errors, encoding_table)[0]
    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2717'

\\u2717 is ✗. A finished render — the script written and judged, the voice
recorded, every picture drawn, an hour of the 3090 — died at the line that
prints the QC verdict, because the tick mark in it has no cp1255 code point.
The video was fine. The report about the video is what fell over.

WHY cp1255 AT ALL. Windows picks stdout's encoding from the system ANSI code
page, which on this owner's Hebrew-locale box is cp1255, and it does that
whenever PYTHONIOENCODING/PYTHONUTF8 are not set. The .bat launchers all set
them — AGENTS.md documents that fix — but a process started any other way does
not get them: a run launched from the dashboard inherits whatever started the
dashboard, and its stdout is a redirected FILE, where Python has no console to
ask and falls straight back to the locale.

So the launcher fix is necessary and not sufficient. The only place that
cannot be forgotten is inside the program: reconfigure both streams to UTF-8
at startup, with errors="replace" so that even a stream that refuses UTF-8
degrades to a question mark instead of an exception.

THIS IS NOT text_repair. That module repairs mojibake in DATA that was read
wrong. This one stops output crashing on the way out. They are the two ends of
the same code-page problem and they fail differently — one gives you "×" where
an em-dash should be, the other gives you no run at all.

CONTRACT: idempotent, safe to call from every entry point, and never raises.
"""

from __future__ import annotations

import sys

_DONE = False


def force_utf8() -> bool:
    """Make stdout and stderr UTF-8. True if anything changed.

    Safe to call repeatedly and from anywhere: the second call is a no-op, and
    a stream that cannot be reconfigured (a pytest capture object, a pipe
    already wrapped by something else) is left exactly as it was.
    """
    global _DONE
    if _DONE:
        return False
    _DONE = True
    changed = False
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except Exception:
            # A stream that will not be reconfigured is not a reason to fail
            # the run — which is the entire point of this module.
            pass
    return changed
