"""AI-powered video idea and script generator using Ollama (100% free, local)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

DEFAULT_MODEL = "mistral"   # change to "llama3.1" or "phi3" if preferred


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    """Return True if Ollama is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _list_models() -> list[str]:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def _clean_str(value: object) -> str:
    """Convert to string, strip emojis and excess whitespace."""
    return _strip_emojis(str(value)).strip()


def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL, system: str = "", json_mode: bool = False) -> str:
    """Send a prompt to the local Ollama server and return the response text."""
    if not _check_ollama():
        raise RuntimeError(
            "Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Install from:  https://ollama.ai"
        )

    import httpx
    from src.task_lock import task_lock
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        # num_predict=-1 unlimited output; num_ctx=8192 prevents prompt eating
        # all context leaving no room for the response (default is 4096)
        "options": {"temperature": 0.85, "num_predict": -1, "num_ctx": 8192},
    }
    # JSON mode: model is constrained to output only valid JSON — eliminates
    # all truncation and bracket-recovery issues for structured generation calls
    if json_mode:
        payload["format"] = "json"

    with task_lock("ollama_generate"):
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=180,
        )
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    # Strip emojis that the model sneaks into JSON strings — they don't break
    # JSON but cause issues downstream and are unwanted in titles/hooks
    return _strip_emojis(raw)


def _clean_json_str(s: str) -> str:
    """Fix common LLM JSON issues: unescaped newlines/tabs inside strings."""
    def fix_string(m: re.Match) -> str:
        return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = re.sub(r'"(?:[^"\\]|\\.)*"', fix_string, s, flags=re.DOTALL)
    # Strip trailing commas before ] or } (invalid JSON that LLMs often emit)
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s


def _complete_truncated_json(s: str) -> str:
    """
    Walk the string tracking bracket/brace depth and whether we're inside a string.
    If the JSON is truncated, close any open strings and structures.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in "[{":
                stack.append("]" if ch == "[" else "}")
            elif ch in "]}":
                if stack and stack[-1] == ch:
                    stack.pop()

    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return s + suffix


def _parse_json_response(raw: str) -> list | dict:
    """Extract JSON from model output — handles markdown fences, emojis, and truncation."""
    raw = raw.strip()

    def try_parse(s: str) -> list | dict | None:
        cleaned = _clean_json_str(s)
        for candidate in (s, cleaned):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Try completing a truncated string/structure, then re-clean so any
        # literal newlines inside the newly-closed string get escaped.
        completed = _complete_truncated_json(cleaned)
        if completed != cleaned:
            for variant in (completed, _clean_json_str(completed)):
                try:
                    return json.loads(variant)
                except json.JSONDecodeError:
                    pass
        # Truncation may have happened mid-key-name (e.g. "descripti instead of
        # "description": "...") making completion produce invalid JSON.  Fall back
        # to salvaging all complete objects that appear before the cut.
        bracket = cleaned.find("[")
        last_brace = cleaned.rfind("}")
        if bracket != -1 and last_brace > bracket:
            salvaged = cleaned[bracket:last_brace + 1].rstrip().rstrip(",") + "]"
            for variant in (salvaged, _clean_json_str(salvaged)):
                try:
                    result = json.loads(variant)
                    if result:
                        return result
                except json.JSONDecodeError:
                    pass
        return None

    # Strip markdown code fences
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            result = try_parse(part)
            if result is not None:
                return result

    result = try_parse(raw)
    if result is not None:
        return result

    # Slice from first [ or {
    for start_char in ("[", "{"):
        s = raw.find(start_char)
        if s != -1:
            result = try_parse(raw[s:])
            if result is not None:
                return result

    # Last-resort: extract all complete top-level {...} objects (handles flat nested)
    objects: list[dict] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(_clean_json_str(raw[start:i + 1]))
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    if objects:
        return objects

    # Dump the full response so we can diagnose truncation/malformation
    try:
        import pathlib
        pathlib.Path("/tmp/rufus_json_fail.txt").write_text(raw, encoding="utf-8")
    except Exception:
        pass
    raise ValueError(f"Could not parse JSON from model response (first 200 chars): {raw[:200]!r}")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VideoIdea:
    title: str
    hook: str
    description: str
    tags: list[str]
    estimated_virality: str


@dataclass
class VideoScript:
    title: str
    hook: str
    sections: list[dict]
    call_to_action: str
    description: str
    tags: list[str]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_search_keywords(
    topic: str,
    niche: str,
    model: str = DEFAULT_MODEL,
    count: int = 8,
    ml_context: str = "",
) -> list[str]:
    """Generate stock footage search queries for a topic using Ollama."""
    prompt = f"""Generate {count} specific search queries to find stock footage for a YouTube video.
Topic: {topic}
Niche: {niche}
{('Context from past videos: ' + ml_context) if ml_context else ''}

Return a JSON array of strings. Each string must be 2-4 words, visual, and searchable on stock footage sites.
Good examples: "people working laptop", "robot automation factory", "data charts screen", "city skyline night"
Bad examples: "introduction", "section 1", "conclusion"

Return ONLY the JSON array, no explanation."""
    try:
        raw = _ollama_generate(prompt, model=model, json_mode=True,
                               system="You are a stock footage researcher. Return only valid JSON arrays of strings.")
        data = _parse_json_response(raw)
        if isinstance(data, list):
            return [str(k).strip() for k in data if k][:count]
    except Exception:
        pass
    return [topic]


def generate_ideas_from_media(
    topic: str,
    niche: str,
    media_captions: list[str],
    count: int = 3,
    trending_context: str = "",
    model: str = DEFAULT_MODEL,
    ml_prefix: str = "",
) -> list[VideoIdea]:
    """Generate video ideas based on available footage captions."""
    captions_block = "\n".join(f"- {c}" for c in media_captions[:20])
    trending_block = f"\nTrending context: {trending_context}" if trending_context else ""
    prompt = f"""{ml_prefix}
Generate {count} viral YouTube video ideas.
Topic: {topic}
Niche: {niche}{trending_block}

Available footage we actually have:
{captions_block}

Create ideas that can be made using ONLY the footage described above.

Return a JSON array. Each object must have exactly these keys:
- title: compelling title under 70 characters
- hook: first 3 seconds attention-grabbing line
- description: 2 sentences about the video
- tags: list of 10 SEO tags
- estimated_virality: one of "Low", "Medium", "High", "Viral"

Return ONLY the JSON array, no explanation."""
    raw = _ollama_generate(prompt, model=model, json_mode=True,
                           system="You are a YouTube viral content strategist. Always respond with valid JSON only.")
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    return [
        VideoIdea(
            title=_clean_str(d.get("title", "")),
            hook=_clean_str(d.get("hook", "")),
            description=_clean_str(d.get("description", "")),
            tags=[str(t) for t in d.get("tags", [])],
            estimated_virality=d.get("estimated_virality", "Medium"),
        )
        for d in data
    ]


def _viral_script_instructions(virality: str) -> str:
    """Return virality-level-specific writing instructions for the script prompt."""
    base = (
        "Structure: hook (shock the viewer in 5 words) → problem/tension → "
        "revelation → proof → call to action. Every sentence must pull the viewer forward."
    )
    extras = {
        "Low": "The topic needs extra energy — start with the most surprising stat or fact you can invent from context.",
        "Medium": "Use storytelling — open with a specific moment, not a generic statement.",
        "High": (
            "Open with the most shocking sentence possible. Use present tense for urgency. "
            "Every section must end with a teaser for the next one (open loop)."
        ),
        "Viral": (
            "Write like MrBeast meets a financial thriller. Open with the single most jaw-dropping "
            "fact. Use short punchy sentences. Build tension in every paragraph. "
            "Reveal information in layers — never give it all at once. "
            "End every section on a cliffhanger. The CTA must feel urgent, not generic."
        ),
    }
    return f"{base}\n{extras.get(virality, extras['Medium'])}"


def generate_script_from_media(
    idea: VideoIdea,
    media_captions: list[str],
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
    model: str = DEFAULT_MODEL,
    ml_prefix: str = "",
) -> "VideoScript":
    """Generate a viral script with sections that map to the available media clips."""
    captions_block = "\n".join(f"footage_{i+1}: {c}" for i, c in enumerate(media_captions[:15]))
    virality = getattr(idea, "estimated_virality", "Medium")
    viral_instructions = _viral_script_instructions(virality)
    prompt = f"""{ml_prefix}
Write a high-retention YouTube video script designed to go viral.

Title: {idea.title}
Hook: {idea.hook}
Virality target: {virality}
Style: {style}, high-energy, no fluff
Target duration: {duration_minutes} minutes

Viral writing rules:
{viral_instructions}

Available footage clips (MUST use these):
{captions_block}

Each section MUST reference footage that visually matches one of the clips above.
Each section script must be punchy, specific, and keep the viewer watching.

Return a JSON object with exactly these keys:
- title: viral video title (use power words, numbers, or shock value)
- hook: spoken opening — most shocking sentence first, 2-3 sentences max
- sections: array of objects with "heading", "script", and "clip_hint" keys
  ("script" = what is spoken, "clip_hint" = visual description matching available footage)
- call_to_action: urgent, specific CTA (not "like and subscribe" — give them a reason)
- description: YouTube description with keywords
- tags: list of 15 SEO tags

Return ONLY the JSON object, no explanation."""
    raw = _ollama_generate(prompt, model=model, json_mode=True,
                           system="You are a viral YouTube scriptwriter. Write to maximise watch time and shares. Always respond with valid JSON only.")
    data = _parse_json_response(raw)
    script = VideoScript(
        title=_clean_str(data.get("title", idea.title)),
        hook=_clean_str(data.get("hook", "")),
        sections=data.get("sections", []),
        call_to_action=_clean_str(data.get("call_to_action", "")),
        description=_clean_str(data.get("description", "")),
        tags=data.get("tags", []),
    )
    try:
        from src.viral_intelligence.retention_engine import RetentionEngine
        engine = RetentionEngine()
        score  = engine.score(script.sections, hook=script.hook)
        if score.total < 55:
            script.sections = _inject_retention_patterns(script.sections)
            score = engine.score(script.sections, hook=script.hook)
        engine.print_report(score)
    except Exception:
        pass
    return script


def generate_video_ideas(
    topic: str,
    niche: str,
    count: int = 5,
    trending_context: str = "",
    model: str = DEFAULT_MODEL,
) -> list[VideoIdea]:
    """Generate viral YouTube video ideas using a local Ollama model."""
    trending_block = (
        f"\nCurrently trending context: {trending_context}" if trending_context else ""
    )

    prompt = f"""Generate {count} viral YouTube video ideas.
Topic: {topic}
Niche: {niche}{trending_block}

Return a JSON array. Each object must have exactly these keys:
- title: compelling title under 70 characters
- hook: first 3 seconds attention-grabbing line
- description: 2 sentences about the video
- tags: list of 10 SEO tags
- estimated_virality: one of "Low", "Medium", "High", "Viral"

Return ONLY the JSON array, no explanation."""

    raw = _ollama_generate(
        prompt,
        model=model,
        json_mode=True,
        system="You are a YouTube viral content strategist. Always respond with valid JSON only.",
    )
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    return [
        VideoIdea(
            title=_clean_str(d.get("title", "")),
            hook=_clean_str(d.get("hook", "")),
            description=_clean_str(d.get("description", "")),
            tags=[str(t) for t in d.get("tags", [])],
            estimated_virality=d.get("estimated_virality", "Medium"),
        )
        for d in data
    ]


def generate_video_script(
    idea: VideoIdea,
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
    model: str = DEFAULT_MODEL,
    niche: str = "general",
) -> VideoScript:
    """Generate a full video script using a local Ollama model.

    Automatically injects ViralDNA patterns and RetentionEngine requirements
    into the prompt so every script is built for maximum watch time.
    """
    # ── 1. Load ViralDNA context ────────────────────────────────────────
    dna_context = ""
    try:
        from src.viral_intelligence.dna_extractor import ViralDNA
        dna = ViralDNA(niche=niche)
        profile = dna.extract()
        dna_context = "\n\n" + profile.to_ollama_context()
    except Exception:
        pass

    # ── 2. Build retention-aware prompt ────────────────────────────────
    prompt = f"""You are the world's best YouTube scriptwriter. Your videos average 70%+ retention.

Title: {idea.title}
Hook: {idea.hook}
Style: {style}
Target duration: {duration_minutes} minutes
Niche: {niche}
{dna_context}

RETENTION ARCHITECTURE RULES (mandatory):
- First 30 seconds: create 3 open loops (questions viewer must stay to see answered)
- Every section: end with a bridge "But here's where it gets interesting..."
- Every section: deliver one specific reward (number, insight, revelation)
- Use exact dollar amounts, percentages, timeframes — never vague
- Close ALL open loops before the final section
- Final section: "If this surprised you, wait until you see [related topic]"

Return a JSON object with exactly these keys:
- title: the video title (10 words max, must include a number or dollar amount)
- hook: spoken opening (first 15 seconds, must use loss aversion + open loop)
- sections: array of objects with "heading" and "script" keys (4-6 sections)
- call_to_action: final 30-second spoken CTA
- description: YouTube description with timestamps
- tags: list of 15 SEO tags

Return ONLY valid JSON, no explanation."""

    raw = _ollama_generate(
        prompt,
        model=model,
        json_mode=True,
        system="You are a professional YouTube scriptwriter. Always respond with valid JSON only.",
    )
    data = _parse_json_response(raw)

    script = VideoScript(
        title=_clean_str(data.get("title", idea.title)),
        hook=_clean_str(data.get("hook", "")),
        sections=data.get("sections", []),
        call_to_action=_clean_str(data.get("call_to_action", "")),
        description=_clean_str(data.get("description", "")),
        tags=data.get("tags", []),
    )

    # ── 3. Score retention and auto-enhance if below threshold ─────────
    try:
        from src.viral_intelligence.retention_engine import RetentionEngine
        engine = RetentionEngine()
        score  = engine.score(script.sections, hook=script.hook)

        if score.total < 55:
            script.sections = _inject_retention_patterns(script.sections)
            score = engine.score(script.sections, hook=script.hook)

        engine.print_report(score)
    except Exception:
        pass

    return script


def _inject_retention_patterns(sections: list[dict]) -> list[dict]:
    """
    Programmatically inject retention patterns into script sections when
    Ollama produces a low-quality script. Ensures every section has:
      - An open-loop question near the start
      - A pattern interrupt in the middle
      - A bridge to the next section at the end
    """
    _BRIDGES = [
        "But here's where it gets really interesting...",
        "Now, this is the part most people never hear about.",
        "Wait — there's something even more important you need to know.",
        "And this next part changes everything.",
        "But before I reveal the answer, you need to understand this first.",
    ]
    _INTERRUPTS = [
        "But here's the thing — ",
        "Now here's what nobody talks about: ",
        "Stop and think about this for a second. ",
        "Here's the shocking part: ",
        "This is where it gets wild — ",
    ]
    _OPEN_LOOPS = [
        "But what actually causes this? I'll tell you in just a moment.",
        "You're probably wondering why — and the answer is going to surprise you.",
        "Stay with me, because what comes next is the key to everything.",
        "I'll reveal the exact reason in just a second.",
        "And the reason why will completely change how you think about this.",
    ]

    import random as _rnd
    enhanced = []
    for i, section in enumerate(sections):
        text: str = section.get("script", "")
        if not text:
            enhanced.append(section)
            continue

        sentences = [s.strip() for s in text.replace("  ", " ").split(". ") if s.strip()]

        # Inject open loop after 1st sentence (if not already present)
        if i < len(sections) - 1 and len(sentences) >= 2:
            loop_words = ["wondering", "reveal", "moment", "answer", "reason", "stay with"]
            has_loop = any(w in text.lower() for w in loop_words)
            if not has_loop:
                sentences.insert(1, _rnd.choice(_OPEN_LOOPS))

        # Inject pattern interrupt before the middle sentence
        interrupt_words = ["here's the thing", "nobody talks", "shocking", "wild", "stop and"]
        has_interrupt = any(w in text.lower() for w in interrupt_words)
        if not has_interrupt and len(sentences) >= 3:
            mid = max(1, len(sentences) // 2)
            sentences[mid] = _rnd.choice(_INTERRUPTS) + sentences[mid][0].lower() + sentences[mid][1:]

        # Add bridge at end (if not last section)
        bridge_words = ["interesting", "important", "need to know", "changes everything", "before i reveal"]
        has_bridge = any(w in text.lower() for w in bridge_words)
        if i < len(sections) - 1 and not has_bridge:
            sentences.append(_rnd.choice(_BRIDGES))

        enhanced.append({**section, "script": ". ".join(sentences)})

    return enhanced


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_ideas(ideas: list[VideoIdea]) -> None:
    virality_colors = {
        "Low": "dim", "Medium": "yellow", "High": "green", "Viral": "bold magenta"
    }
    for i, idea in enumerate(ideas, 1):
        color = virality_colors.get(idea.estimated_virality, "white")
        console.print(
            Panel(
                f"[bold]{idea.title}[/bold]\n\n"
                f"[cyan]Hook:[/cyan] {idea.hook}\n\n"
                f"[blue]Description:[/blue] {idea.description}\n\n"
                f"[yellow]Tags:[/yellow] {', '.join(idea.tags[:6])}...\n\n"
                f"[{color}]Virality: {idea.estimated_virality}[/{color}]",
                title=f"Idea #{i}",
                border_style=color,
            )
        )


def print_script(script: VideoScript) -> None:
    md_parts = [f"# {script.title}\n", f"## Hook\n{script.hook}\n"]
    for section in script.sections:
        md_parts.append(f"## {section.get('heading','')}\n{section.get('script','')}\n")
    md_parts.append(f"## Call to Action\n{script.call_to_action}\n")
    md_parts.append(f"---\n**YouTube Description:**\n{script.description}")
    console.print(Markdown("\n".join(md_parts)))


def generate_shorts_script(
    idea: VideoIdea,
    model: str = "mistral",
    ml_prefix: str = "",
) -> VideoScript:
    """Generate a viral ≤60-second script for YouTube Shorts scaled to the idea's virality."""
    virality = getattr(idea, "estimated_virality", "Medium")
    viral_instructions = {
        "Low":    "Make it bold — lead with the most surprising fact you can.",
        "Medium": "Start with curiosity, end with a cliffhanger that makes them follow.",
        "High":   "Every sentence must be a mini-hook. No sentence is throwaway.",
        "Viral":  (
            "Write like it's the most important 60 seconds on the internet today. "
            "Open with the single most shocking stat or fact. "
            "Use present tense. Short punchy sentences only. "
            "Create tension in every section. End on a reveal that makes them want to share it."
        ),
    }.get(virality, "")

    prompt = f"""{ml_prefix}
You are a viral YouTube Shorts scriptwriter. Write a punchy 45-60 second script.

Topic: {idea.title}
Hook idea: {idea.hook}
Virality target: {virality}
Viral writing rule: {viral_instructions}

Script rules:
- Hook: ONE sentence, max 10 words, use shock or a curiosity gap
- Exactly 3 sections, each 15-25 words — every word must earn its place
- Call to action: specific and urgent, max 10 words
- NO filler words, NO generic phrases like "in today's video"
- Total word count: 80-120 words

Return ONLY valid JSON, no markdown:
{{
  "title": "...",
  "hook": "...",
  "sections": [
    {{"heading": "Point 1", "script": "..."}},
    {{"heading": "Point 2", "script": "..."}},
    {{"heading": "Point 3", "script": "..."}}
  ],
  "call_to_action": "...",
  "description": "...",
  "tags": ["tag1", "tag2", "Shorts"]
}}"""

    raw = _ollama_generate(prompt.strip(), model, json_mode=True)
    data = _parse_json_response(raw)
    return VideoScript(
        title=_clean_str(data.get("title", idea.title)),
        hook=_clean_str(data.get("hook", idea.hook)),
        sections=data.get("sections", []),
        call_to_action=_clean_str(data.get("call_to_action", "Follow for more!")),
        description=_clean_str(data.get("description", "")),
        tags=data.get("tags", ["Shorts"]),
    )


def check_ollama_status() -> None:
    """Print Ollama status and available models."""
    if _check_ollama():
        models = _list_models()
        console.print(f"[green]Ollama running[/green] — models: {', '.join(models) or 'none pulled yet'}")
        if not models:
            console.print("[yellow]Pull a model: ollama pull mistral[/yellow]")
    else:
        console.print("[red]Ollama not running.[/red] Start with: [bold]ollama serve[/bold]")
        console.print("Install from: https://ollama.ai")
