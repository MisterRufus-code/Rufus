#!/usr/bin/env python3
"""
resolve_channels.py — turn @handles into the UC ids competitors.json wants.

WHY THIS EXISTS. config/competitors.json is the one file standing between a
built, tested, documented scout and its first observation, and it wants channel
ids — not handles. The example file says so and then leaves you to it:

    "open a channel, view source, search for channelId"

Fifteen channels is fifteen rounds of view-source, and one mistyped character
is a channel that silently never reports. The API answers this in one unit per
handle, so there is no reason for a person to do it by hand.

    python scripts/resolve_channels.py @EconomicsExplained @MoneyMacro
    python scripts/resolve_channels.py --from handles.txt
    python scripts/resolve_channels.py @Foo --write     # merge into the config

DRY BY DEFAULT. It prints what it found and changes nothing until --write,
because the whole point of naming the channels yourself is that you look at the
list — a resolver that silently rewrote your config would put channels you
never saw into the thing that chooses your next video's topic.

QUOTA. channels().list(forHandle=...) is 1 unit. The search fallback is 100,
which is why it only runs when the handle lookup finds nothing and why it says
so out loud when it does.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CONFIG_DIR = Path(__file__).parent.parent / "config"
COMPETITORS_FILE = CONFIG_DIR / "competitors.json"
EXAMPLE_FILE = CONFIG_DIR / "competitors.json.example"

# Accepts what a person actually copies: a handle, a bare name, or any of the
# URL shapes YouTube hands out. The id form is passed straight through — asking
# someone to strip a URL before pasting it is the kind of friction that makes a
# tool go unused.
_URL_HANDLE_RE = re.compile(r"youtube\.com/@([A-Za-z0-9._-]+)", re.I)
_URL_ID_RE = re.compile(r"youtube\.com/channel/(UC[A-Za-z0-9_-]{22})", re.I)
_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


def normalize(raw: str) -> tuple[str, str]:
    """(kind, value) where kind is "id" or "handle"."""
    raw = raw.strip()
    m = _URL_ID_RE.search(raw)
    if m:
        return ("id", m.group(1))
    if _ID_RE.match(raw):
        return ("id", raw)
    m = _URL_HANDLE_RE.search(raw)
    if m:
        return ("handle", m.group(1))
    return ("handle", raw.lstrip("@"))


def _service():
    """The same authenticated client competitors.py uses, or None with a why.

    Imported rather than rebuilt: two auth paths against one token file is how
    a scheduled task ends up sitting on an interactive OAuth prompt at 2am, and
    competitors.py's docstring already made that decision.
    """
    try:
        import competitors
        return competitors._service()
    except Exception as e:
        print(f"[resolve] no YouTube access ({e})")
        return None


def _by_handle(yt, handle: str) -> dict | None:
    resp = yt.channels().list(part="id,snippet,statistics",
                              forHandle=handle).execute()
    items = resp.get("items") or []
    return items[0] if items else None


def _by_search(yt, name: str) -> dict | None:
    """100 units. Only when the handle lookup came back empty."""
    print(f"[resolve] {name}: no channel with that handle — falling back to "
          f"search (100 quota units)")
    resp = yt.search().list(part="snippet", type="channel",
                            q=name, maxResults=1).execute()
    items = resp.get("items") or []
    if not items:
        return None
    cid = items[0]["snippet"].get("channelId") or items[0]["id"].get("channelId")
    if not cid:
        return None
    full = yt.channels().list(part="id,snippet,statistics", id=cid).execute()
    got = full.get("items") or []
    return got[0] if got else None


def _by_id(yt, channel_id: str) -> dict | None:
    resp = yt.channels().list(part="id,snippet,statistics",
                              id=channel_id).execute()
    items = resp.get("items") or []
    return items[0] if items else None


def resolve(yt, raw: str) -> dict | None:
    """One input to {"id", "title", "subs"}, or None with a printed reason."""
    kind, value = normalize(raw)
    try:
        item = _by_id(yt, value) if kind == "id" else _by_handle(yt, value)
        if item is None and kind == "handle":
            item = _by_search(yt, value)
    except Exception as e:
        print(f"[resolve] {raw}: lookup failed ({e})")
        return None
    if item is None:
        print(f"[resolve] {raw}: nothing found")
        return None
    stats = item.get("statistics") or {}
    return {
        "id": item["id"],
        "title": (item.get("snippet") or {}).get("title", "?"),
        "subs": stats.get("subscriberCount", "?"),
    }


def merge(found: list[dict]) -> tuple[list[str], list[str]]:
    """(the ids the file will hold, the ones that are new).

    Existing entries are kept and order is preserved: the file is the owner's
    list of channels they chose to watch, and a resolver is a way to add to it,
    not a thing that decides what belongs in it.
    """
    existing: list[str] = []
    if COMPETITORS_FILE.exists():
        try:
            raw = json.loads(COMPETITORS_FILE.read_text(encoding="utf-8"))
            existing = [str(c).strip()
                        for c in (raw.get("channels") or []) if str(c).strip()]
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"[resolve] {COMPETITORS_FILE.name} is unreadable ({e}) — "
                  f"refusing to overwrite it")
            raise SystemExit(1)
    try:
        import competitors
        existing = [c for c in existing if not competitors._is_placeholder(c)]
    except Exception:
        pass
    out = list(existing)
    added = []
    for f in found:
        if f["id"] not in out:
            out.append(f["id"])
            added.append(f["id"])
    return out, added


def write(ids: list[str]) -> None:
    doc = {}
    if EXAMPLE_FILE.exists():
        try:
            doc = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            doc = {}
    if COMPETITORS_FILE.exists():
        try:
            doc = json.loads(COMPETITORS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    doc["channels"] = ids
    COMPETITORS_FILE.write_text(json.dumps(doc, indent=2, ensure_ascii=False)
                                + "\n", encoding="utf-8")
    print(f"[resolve] wrote {len(ids)} channel(s) → {COMPETITORS_FILE}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("handles", nargs="*",
                    help="@handle, channel URL, or UC id")
    ap.add_argument("--from", dest="from_file",
                    help="file with one handle/URL/id per line (# comments ok)")
    ap.add_argument("--write", action="store_true",
                    help="merge the results into config/competitors.json")
    args = ap.parse_args(argv)

    wanted = list(args.handles)
    if args.from_file:
        for line in Path(args.from_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                wanted.append(line)
    if not wanted:
        ap.error("give at least one handle, URL or id (or --from a file)")

    yt = _service()
    if yt is None:
        return 1

    found = []
    for raw in wanted:
        got = resolve(yt, raw)
        if got:
            print(f"  {got['id']}  {got['title']}  ({got['subs']} subs)")
            found.append(got)

    if not found:
        print("[resolve] nothing resolved — config untouched")
        return 1

    ids, added = merge(found)
    if not args.write:
        print(f"\n[resolve] dry run. {len(added)} new, {len(ids)} total. "
              f"Re-run with --write to save.")
        return 0
    write(ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
