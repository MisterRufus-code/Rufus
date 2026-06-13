#!/usr/bin/env python3
"""
switch_niche.py – Change the active niche in one command.

Usage:
    python switch_niche.py motivation
    python switch_niche.py list
"""

import json
import sys
from pathlib import Path

NICHES_FILE = Path(__file__).parent.parent / "config" / "niches.json"


def main():
    data = json.loads(NICHES_FILE.read_text())

    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print(f"Active niche: {data['active']}\n")
        print("Available niches:")
        for key, val in data["niches"].items():
            marker = "→" if key == data["active"] else " "
            print(f"  {marker} {key:20s}  {val['display_name']}")
        return

    target = sys.argv[1].lower().replace("-", "_")
    if target not in data["niches"]:
        print(f"Unknown niche: '{target}'")
        print("Available:", ", ".join(data["niches"].keys()))
        sys.exit(1)

    data["active"] = target
    NICHES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Switched to: {target}  ({data['niches'][target]['display_name']})")


if __name__ == "__main__":
    main()
