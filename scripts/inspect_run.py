#!/usr/bin/env python3
"""
inspect_run.py
AI quality-inspection for a Rufus run. Reads the keyframes saved by a
RUFUS_DEBUG=1 run (media_library/debug/<stamp>/), sends each image to GPT-4o
Vision for an honest critique (photorealism 1-10, does it match the spoken
beat, every visible flaw), pulls the matching script from the DB, and writes a
single shareable critique.md you can read or hand back for diagnosis.

Usage:
    RUFUS_DEBUG=1 python scripts/main.py --skip-upload   # 1. produce a debug folder
    python scripts/inspect_run.py                        # 2. critique the newest one
    python scripts/inspect_run.py media_library/debug/1781357443   # or a specific one
"""

import base64
import json
import sqlite3
import sys
from pathlib import Path

import paths

from openai import OpenAI

ROOT      = Path(__file__).parent.parent
DEBUG_DIR = paths.debug_root()
DB_FILE   = ROOT / "rufus.db"
CONFIG    = ROOT / "config"


def _client() -> OpenAI:
    keys = json.loads((CONFIG / "keys.json").read_text(encoding="utf-8"))
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_"):
        raise SystemExit("OpenAI key not set in config/keys.json")
    return OpenAI(api_key=key)


def _newest_run() -> Path:
    if not DEBUG_DIR.exists():
        raise SystemExit(
            "No debug folder yet. Produce one first:\n"
            "  RUFUS_DEBUG=1 python scripts/main.py --skip-upload")
    runs = sorted([d for d in DEBUG_DIR.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"{DEBUG_DIR} is empty — run with RUFUS_DEBUG=1 first.")
    return runs[0]


def _latest_script() -> dict:
    """Newest row from the videos table (script + score), or {} if unavailable."""
    if not DB_FILE.exists():
        return {}
    try:
        c = sqlite3.connect(str(DB_FILE))
        c.row_factory = sqlite3.Row
        row = c.execute(
            "SELECT script_hook, script_full, title, score, "
            "score_specificity, score_hook, score_compression, score_loop, "
            "score_human, score_reasoning "
            "FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        c.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def _critique_image(client: OpenAI, png: Path, beat: str) -> str:
    b64 = base64.b64encode(png.read_bytes()).decode("utf-8")
    prompt = (
        "You are a brutally honest photo-realism critic for an AI-generated "
        "YouTube Shorts pipeline. The narrator says this line while this image "
        f"is on screen:\n\n\"{beat}\"\n\n"
        "Judge ONLY what you see. Respond in exactly this format:\n"
        "PHOTOREALISM: <1-10> (10 = indistinguishable from a real photo)\n"
        "MATCHES_BEAT: <yes/partly/no> — <one line why>\n"
        "FLAWS: <comma-separated list of every visible problem: bad hands, "
        "extra fingers, warped faces, plastic skin, garbled text, AI-slop look, "
        "wrong subject, flat lighting, etc. Say 'none' only if truly clean>\n"
        "FIX: <one concrete prompt or setting change that would most improve it>"
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}", "detail": "high"}},
        ]}],
        max_tokens=300, temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _newest_run()
    pngs    = sorted(run_dir.glob("[0-9]*.png"))
    if not pngs:
        raise SystemExit(f"No keyframes in {run_dir}")

    print(f"[inspect] critiquing {len(pngs)} image(s) in {run_dir}")
    client = _client()
    script = _latest_script()

    lines = [f"# Rufus run critique — {run_dir.name}\n"]

    if script:
        lines += [
            "## Script\n",
            f"**Hook:** {script.get('script_hook','?')}\n",
            f"**Title:** {script.get('title','—')}\n",
            f"**Score:** {script.get('score','?')}/10  "
            f"(spec={script.get('score_specificity','?')}, "
            f"hook={script.get('score_hook','?')}, "
            f"comp={script.get('score_compression','?')}, "
            f"loop={script.get('score_loop','?')}, "
            f"human={script.get('score_human','?')})\n",
            f"\n```\n{script.get('script_full','(not in DB)')}\n```\n",
            f"\n_Score note:_ {script.get('score_reasoning','—')}\n",
        ]

    lines.append("\n## Images\n")
    scores = []
    for png in pngs:
        beat_file = png.with_suffix(".txt")
        beat = "(unknown)"
        if beat_file.exists():
            txt  = beat_file.read_text(encoding="utf-8")
            beat = txt.split("FULL SD PROMPT:")[0].replace("BEAT QUERY:", "").strip()
        print(f"[inspect] {png.name} …")
        try:
            verdict = _critique_image(client, png, beat)
        except Exception as e:
            verdict = f"(critique failed: {e})"
        for ln in verdict.splitlines():
            if ln.upper().startswith("PHOTOREALISM:"):
                digits = "".join(ch for ch in ln if ch.isdigit())
                if digits:
                    scores.append(int(digits[:2]) if int(digits[:2]) <= 10 else int(digits[0]))
        lines += [f"### {png.name}\n", f"_Beat:_ {beat}\n", f"\n{verdict}\n"]

    if scores:
        avg = sum(scores) / len(scores)
        lines.insert(1, f"\n**Average photorealism: {avg:.1f}/10** "
                        f"across {len(scores)} image(s)\n")

    out = run_dir / "critique.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[inspect] wrote {out}")
    print(f"[inspect] images: {run_dir}/*.png   →  upload these + critique.md for diagnosis")
    if scores:
        print(f"[inspect] average photorealism {sum(scores)/len(scores):.1f}/10")


if __name__ == "__main__":
    main()
