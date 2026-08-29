#!/usr/bin/env python3
"""Which build is this, and which build made that video.

TWO USES, AND THE SECOND IS THE ONE THAT PAYS FOR THIS FILE.

The first is support. "Which version are you running?" is the opening question
of every conversation about software somebody paid for, and until now there was
no answer anywhere in this tree — no version string, no tag, nothing on the
dashboard. A bug report and a fix could be about different code and neither
side would know.

The second is the reason this repository exists in the state it is in. Its
standing complaint, written into DIRECTION.md and half the module docstrings,
is code running ahead of evidence: changes made on judgement with no way to
tell afterwards whether they helped. The measure pages compare videos against
each other and cannot see the one variable that changed most between them —
which build produced each. Stamping the version onto every video row makes
"did that change help" a question the database can be asked, instead of one
answered from memory.

MEASURED, NOT DECLARED. The version below is written by a person because only a
person knows whether a change is worth calling a release. Everything else here
is read off the machine at the moment it is asked: the commit from git, whether
the tree is dirty, the Python and the platform. A build fingerprint that a
stale constant could get wrong is worse than none, because it is believed.

A git checkout is the normal case; a copy with no .git (a zip, an installer
payload) falls back to a BUILD file written when that copy was made, and to
"unknown" when there is neither. Never to a guess.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUILD_FILE = ROOT / "BUILD"

# THE PRODUCT VERSION. Bumped by hand, in the same commit as the change that
# earns it, and recorded in CHANGELOG.md. Nothing derives it from tags or dates
# — a version that moves on its own says a release happened when one did not.
VERSION = "0.5.0"

# Cache: git is a subprocess and this is read on every dashboard page render.
_CACHE: dict = {}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def commit() -> str:
    """The short commit this copy is at, or "" when that cannot be known.

    Empty rather than "unknown" or "dev": a caller deciding what to display
    wants a falsy value it can branch on, and a placeholder that reads like a
    commit is exactly the thing a support conversation must not be given.
    """
    if "commit" in _CACHE:
        return _CACHE["commit"]
    sha = _git("rev-parse", "--short", "HEAD")
    if not sha and BUILD_FILE.exists():
        # A copy taken without .git — a zip, an installer payload. Whoever made
        # it wrote down what it was made from.
        for line in BUILD_FILE.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("commit:"):
                sha = line.split(":", 1)[1].strip()
                break
    _CACHE["commit"] = sha
    return sha


def dirty() -> bool:
    """Whether this copy has uncommitted changes.

    Worth reporting on its own: "0.5.0 at a1b2c3d" and "0.5.0 at a1b2c3d with
    local edits" are different claims, and a support conversation that mistakes
    the second for the first wastes everybody's afternoon.
    """
    if "dirty" in _CACHE:
        return _CACHE["dirty"]
    status = _git("status", "--porcelain")
    _CACHE["dirty"] = bool(status.strip())
    return _CACHE["dirty"]


def stamp() -> str:
    """One line identifying this build, for a log header or a footer."""
    sha = commit()
    if not sha:
        return f"Rufus {VERSION}"
    return f"Rufus {VERSION} ({sha}{'+local' if dirty() else ''})"


def build() -> dict:
    """Everything a support conversation needs, in one call."""
    return {
        "version": VERSION,
        "commit": commit(),
        "dirty": dirty(),
        "stamp": stamp(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
    }


def _cli() -> int:
    b = build()
    print(f"\n  {b['stamp']}")
    print(f"  Python {b['python']} on {b['platform']}")
    if not b["commit"]:
        print("  No commit recorded — this copy has no .git and no BUILD file, "
              "so which code it holds cannot be established.")
    elif b["dirty"]:
        print("  Working tree has uncommitted changes: what is running is not "
              "exactly this commit.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
