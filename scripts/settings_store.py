#!/usr/bin/env python3
"""
settings_store.py — the settings the dashboard saves, read by every launch path.

THE GAP THIS CLOSES, and it is a bad one. config/dashboard_settings.json was
written by the settings page and read by exactly one caller: the dashboard's
own _launch_run. So the owner could set the style, the beat count, the Discord
webhook and the sound levels in a form built for the purpose, then start a
video with run.bat — the way every video on this channel has actually been
started — and none of it applied. The dashboard's own help text admitted as
much in a parenthesis, which is not a fix, it is a confession.

The instruction was "the whole thing is managed from the dashboard, not run
every time from the software". A settings page that only three of five launch
paths obey does not do that.

PRECEDENCE, and why this way round. A variable already in the environment
wins; saved settings fill in what is missing. So:

    the dashboard form   = the channel's defaults, set once, obeyed everywhere
    $env:X="Y" at a prompt = an override for THIS run

which is the way round a person expects: what you type now beats what you
saved last week. It also means run_dashboard.bat's own `set RUFUS_STILLS_ONLY`
still works, and a settings file cannot silently countermand a deliberate
one-off experiment.

CONTRACT: never raises, never overwrites, and says what it did. A settings
file that quietly changed a run would be worse than no settings file — this
prints the keys it applied, so a surprising run has its cause in the log.
"""

import json
import os
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent.parent / "config" / "dashboard_settings.json"

# Never let a saved setting reach into the process environment for anything
# outside this pipeline's own namespace. The file is written by a form, and a
# form that could set PATH would be a form that could run anything.
_ALLOWED_PREFIXES = ("RUFUS_", "SD_CLIPS", "RENDER_TIMEOUT")


# Settings whose VALUE is a credential and must never be echoed into a log.
# WEBHOOK and TOKEN were obvious. NTFY_TOPIC is the one that got missed: ntfy
# has no accounts, so the topic string is the entire authentication — anyone
# holding it can read every alert and push fake ones — and notify.py's own
# header says so. It was being printed in full on every run, into logs that
# get pasted into chats when something goes wrong. A secret is defined by what
# holding it lets you do, not by whether the word "token" is in its name.
_SECRET_MARKS = ("WEBHOOK", "TOKEN", "NTFY_TOPIC", "SECRET", "PASSWORD", "KEY")


def _is_secret(key: str) -> bool:
    return any(mark in key.upper() for mark in _SECRET_MARKS)


def load() -> dict:
    """The saved settings, or {} if there are none or the file is unreadable.

    utf-8-SIG, NOT utf-8, and this is not pedantry. Windows PowerShell 5.1's
    `Set-Content -Encoding utf8` writes a BYTE ORDER MARK, so a settings file
    edited from a PowerShell prompt — which is how the owner was told to edit
    it — starts with three bytes that json.loads rejects outright:

        Unexpected UTF-8 BOM (decode using utf-8-sig)

    utf-8-sig reads both, so nothing is lost by preferring it.

    AND IT SAYS SO WHEN IT CANNOT READ. The silent return was the real damage:
    one BOM made EVERY saved setting vanish — the style, the renderer, the
    dashboard URL, the ntfy topic — with no message anywhere, and the pipeline
    ran on its built-in defaults as though the file had never existed. A
    missing file is normal and stays quiet; a file that exists and will not
    parse is a person's configuration being ignored, and they have to be told.
    """
    try:
        raw = SETTINGS_FILE.read_text(encoding="utf-8-sig")
    except OSError:
        return {}                       # no file yet: the ordinary case
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[settings] {SETTINGS_FILE.name} exists but is not valid JSON "
              f"({e}) — EVERY saved setting is being ignored for this run. "
              f"Fix it on the dashboard's Settings page, or delete the file "
              f"to start over.")
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()
            if v not in (None, "") and str(k).startswith(_ALLOWED_PREFIXES)}


def apply(env: dict | None = None, *, announce: bool = True) -> list[str]:
    """Fill in the settings the environment does not already set.

    Returns the keys applied. `env` defaults to os.environ, so the ordinary
    call mutates the real process environment — which is the point: every
    module downstream reads its configuration from there, and this is the one
    place that has to know a settings file exists at all.
    """
    target = os.environ if env is None else env
    saved = load()
    if not saved:
        return []
    applied = []
    for key, value in sorted(saved.items()):
        if target.get(key) in (None, ""):
            target[key] = value
            applied.append(key)
    if applied and announce:
        shown = ", ".join(f"{k}={saved[k]}" for k in applied
                          if not _is_secret(k))
        hidden = sum(1 for k in applied if _is_secret(k))
        note = shown + (f" (+{hidden} secret)" if hidden else "")
        print(f"[settings] from the dashboard: {note}")
    skipped = [k for k in saved if k not in applied]
    if skipped and announce:
        print(f"[settings] the environment already sets {', '.join(sorted(skipped))} "
              f"— those win for this run")
    return applied
