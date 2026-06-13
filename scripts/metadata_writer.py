#!/usr/bin/env python3
"""
metadata_writer.py — GPT-optimized YouTube title/description/tags for Rufus.

Why: the old title was the script's first line truncated to 80 chars — the
viewer read the punchline before hearing it, and search keywords were ignored.
This module writes a title that *complements* the spoken hook (curiosity gap,
≤70 chars, one niche keyword) and a keyword-rich description.

One gpt-4o-mini call (~$0.001/video). HARD FALLBACK: any failure (no key, API
down, bad JSON) returns exactly the legacy first-line metadata, so uploads
never depend on this call succeeding.
"""

import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"

TITLE_MAX = 70
MODEL     = "gpt-4o-mini"

# Per-niche search keywords the title should work in naturally.
NICHE_KEYWORDS = {
    "finance":              ["money", "investing", "wealth", "rich"],
    "motivation":           ["discipline", "success", "mindset"],
    "mindset":              ["mindset", "psychology", "habits"],
    "business":             ["business", "entrepreneur", "startup"],
    "personal_development": ["habits", "self improvement", "growth"],
}


def _legacy_metadata(script: str, niche_name: str, niche_cfg: dict,
                     hashtags: list[str]) -> dict:
    """The original behavior — first script line as title. Always works."""
    first_line = script.strip().split("\n")[0][:80]
    return {
        "title":       first_line if first_line else "Daily Short",
        "description": f"{script}\n\n{' '.join(hashtags)}\n\n{niche_cfg.get('cta', '')}",
        "tags":        [t.lstrip("#") for t in hashtags],
    }


def _load_key() -> str:
    try:
        key = json.loads(KEYS_FILE.read_text()).get("openai", "")
        if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
            return key
    except Exception:
        pass
    return ""


def _clamp_title(title: str, hook: str) -> str:
    """Enforce length and reject titles that just repeat the spoken hook."""
    title = re.sub(r"\s+", " ", title).strip().strip('"').strip("'")
    if len(title) > TITLE_MAX:
        cut = title[:TITLE_MAX]
        title = cut[:cut.rfind(" ")] if " " in cut else cut
    # If the model ignored the "don't repeat the hook" rule, signal fallback.
    t_norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    h_norm = re.sub(r"[^a-z0-9 ]", "", hook.lower())
    if t_norm and h_norm and (t_norm in h_norm or h_norm in t_norm):
        return ""
    return title


def generate_metadata(script: str, niche_name: str, niche_cfg: dict,
                      hashtags: list[str] | None = None,
                      language: str = "en") -> dict:
    """Return {"title", "description", "tags"} — GPT-optimized, legacy on failure."""
    hashtags = hashtags or ["#Shorts"]
    legacy   = _legacy_metadata(script, niche_name, niche_cfg, hashtags)
    hook     = script.strip().split("\n")[0]

    key = _load_key()
    if not key:
        return legacy

    keywords = ", ".join(NICHE_KEYWORDS.get(niche_name, [niche_name]))
    lang_note = ("" if language == "en"
                 else f"\nWrite the title and description in this language: {language}.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "You write YouTube Shorts metadata that maximizes click-through "
                    "without clickbait.\n\n"
                    f"NICHE: {niche_name} (search keywords: {keywords})\n"
                    f"SCRIPT (the video's narration):\n{script}\n\n"
                    "Rules:\n"
                    f"- TITLE: max {TITLE_MAX} chars. Open a curiosity gap that the video "
                    "answers. It must NOT repeat or paraphrase the script's first line — "
                    "the viewer hears that line immediately; the title must add a second "
                    "angle. Include exactly one of the search keywords naturally. "
                    "No ALL-CAPS words, no emoji, no '...'.\n"
                    "- DESCRIPTION_LEAD: 2 short lines, keyword-rich, that tease the "
                    "payoff. No hashtags here.\n"
                    "- TAGS: 12 comma-separated search tags (single words or short "
                    "phrases, no # symbol)."
                    f"{lang_note}\n\n"
                    'Reply ONLY with JSON: {"title": "...", "description_lead": "...", '
                    '"tags": ["...", ...]}'
                ),
            }],
            max_tokens=300,
            temperature=0.8,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        title = _clamp_title(str(data.get("title", "")), hook)
        if not title:
            return legacy

        lead = str(data.get("description_lead", "")).strip()
        tags = [str(t).strip().lstrip("#") for t in data.get("tags", []) if str(t).strip()]
        tags = (tags or legacy["tags"])[:15]

        description = (
            (f"{lead}\n\n" if lead else "")
            + f"{script}\n\n{' '.join(hashtags)}\n\n{niche_cfg.get('cta', '')}"
        )
        return {"title": title, "description": description, "tags": tags}
    except Exception as e:
        print(f"[metadata] GPT metadata failed ({e}) — using legacy title")
        return legacy


if __name__ == "__main__":
    import sys
    demo = sys.argv[1] if len(sys.argv) > 1 else (
        "Banks made $176B while savers lost money.\n"
        "Here is how the spread actually works.\nSave this."
    )
    print(json.dumps(generate_metadata(demo, "finance", {"cta": "Follow."}), indent=2))
