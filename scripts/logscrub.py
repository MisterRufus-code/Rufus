#!/usr/bin/env python3
"""Find and remove credentials that ended up in the log files.

HOW THEY GOT THERE. `auth.py add` prints a sign-in link containing that user's
token, because the token IS the credential and the link is how you hand it
over. serve.ps1 sends the dashboard's stderr to logs/dashboard.log, and
Werkzeug writes one line per request containing the full request target — so
opening `https://…/?token=…` once records an owner credential in plaintext, in
a file that a backup copies, a support bundle would collect, and a bug report
gets pasted into.

The dashboard now redacts both at the source (the sign-in redirects to a clean
URL) and at the logger (see _RedactSecrets). Neither of those touches a line
written last month, which is what this is for.

SCANNING IS THE DEFAULT AND REWRITING IS OPT-IN, because a log is evidence.
The scan says which files hold what, so the decision to alter them is made by a
person who can see the cost.

AND REMOVING IT IS NOT THE END. A secret that has been sitting in a file for a
month may already have been copied — into a backup, a support bundle, a
screenshot, an old disk. The only action that actually revokes it is rotating
it, so this says so every time it finds one rather than letting a clean scan
read as safety.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ONE COPY OF THIS PATTERN. dashboard.py imports it rather than keeping its
# own — two regexes for the same job drift, and the one that drifts is the one
# nobody notices has stopped matching.
#
# Matched on the VALUE's position rather than on how sensitive the name sounds:
# what makes `code=4/0Ab` dangerous is that it is a credential, not that it is
# called "code".
SECRET_QUERY_RE = re.compile(
    r"((?:token|code|state|key|secret|password|access_token)=)[^&\s\"'<>]+",
    re.IGNORECASE)

REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Replace the value of every credential-shaped query parameter."""
    return SECRET_QUERY_RE.sub(r"\1" + REDACTED, text)


def _hits(text: str) -> int:
    """How many real secrets are in this text — already-redacted ones do not
    count, or a scrubbed file would report itself as still leaking forever."""
    return sum(1 for m in SECRET_QUERY_RE.finditer(text)
               if m.group(0).split("=", 1)[1] != REDACTED)


def scan(directory: Path | None = None) -> list[dict]:
    """Every log file holding a credential, and how many. Newest first."""
    import paths
    directory = Path(directory) if directory else paths.log_dir()
    if not directory.exists():
        return []
    found = []
    for path in sorted(directory.glob("*.log"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = _hits(text)
        if n:
            found.append({"path": path, "hits": n})
    return found


def scrub(directory: Path | None = None) -> list[dict]:
    """Rewrite the files in place, leaving everything that is not a secret.

    In place and with no backup copy, deliberately: a .bak beside the file
    would keep the credential on the same disk, which is the thing being
    undone. Written to a temp file and moved over the original so a crash
    mid-write cannot leave a half-scrubbed log.
    """
    cleaned = []
    for row in scan(directory):
        path: Path = row["path"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tmp = path.with_suffix(path.suffix + ".scrubbing")
            tmp.write_text(redact(text), encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            print(f"[logscrub] could not rewrite {path.name}: {e}")
            continue
        cleaned.append(row)
    return cleaned


def _cli() -> int:
    import sys

    fix = "--fix" in sys.argv[1:]
    found = scan()

    if not found:
        print("\n  No credentials found in the logs.\n")
        return 0

    total = sum(r["hits"] for r in found)
    print(f"\n  {total} credential(s) in {len(found)} log file(s):\n")
    for row in found:
        print(f"    {row['path'].name:44} {row['hits']}")

    if not fix:
        print("\n  Run again with --fix to replace the values in place.")
        print("  Nothing has been changed.\n")
        return 1

    scrub()
    print(f"\n  Rewritten. The values are gone from these files.")
    print("  ROTATE THEM ANYWAY. A secret that sat in a file may already have")
    print("  been copied — into a backup, a support bundle, a screenshot, an")
    print("  old disk. Rewriting the file does not revoke anything:")
    print("      python scripts/auth.py rotate <name>\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
