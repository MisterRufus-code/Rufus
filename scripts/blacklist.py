#!/usr/bin/env python3
"""
Blacklist manager – read/write topics already used.
Used by the n8n AI Agent tool via SSH.

Usage:
    python blacklist.py check  "Bitcoin hits new high"
    python blacklist.py add    "Bitcoin hits new high"
    python blacklist.py list
"""

import json
import sys
from pathlib import Path

BLACKLIST_FILE = Path(__file__).parent.parent / "config" / "blacklist.json"


def _load() -> list[str]:
    if BLACKLIST_FILE.exists():
        return json.loads(BLACKLIST_FILE.read_text(encoding="utf-8"))
    return []


def _save(items: list[str]) -> None:
    BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    BLACKLIST_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def check(topic: str) -> bool:
    """Return True if topic already exists (case-insensitive substring match)."""
    needle = topic.lower()
    return any(needle in item.lower() for item in _load())


def add(topic: str) -> None:
    items = _load()
    items.append(topic)
    # Keep last 500 entries to avoid unbounded growth
    _save(items[-500:])


def main():
    if len(sys.argv) < 2:
        print("Usage: blacklist.py <check|add|list> [topic]")
        sys.exit(1)

    cmd = sys.argv[1]
    topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    if cmd == "check":
        result = check(topic)
        print("EXISTS" if result else "NEW")
    elif cmd == "add":
        add(topic)
        print(f"Added: {topic}")
    elif cmd == "list":
        for t in _load():
            print(t)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
