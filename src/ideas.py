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

# Phrases the model loves to write that kill credibility and waste airtime.
# Applied as a post-processing strip after every script generation.
_FILLER_PHRASES: list[tuple[str, str]] = [
    # "stay with me" family
    ("Stay with me, because what comes next is the key to everything.", ""),
    ("Stay with me.", ""),
    ("Stay with me —", ""),
    ("Stay with me,", ""),
    # "don't go anywhere" family
    ("Don't go anywhere.", ""),
    ("Don't miss this.", ""),
    ("Don't skip this part.", ""),
    ("Don't skip this.", ""),
    # "I'll tell you in a moment" family
    ("I'll tell you in just a moment.", ""),
    ("I'll tell you right after this.", ""),
    ("I'll reveal the exact reason right after this.", ""),
    ("I'll get to that in a moment.", ""),
    ("We'll get to that in a moment.", ""),
    # "but first" padding
    ("But before I reveal the answer, you need to understand this first.", ""),
    ("But first, let me explain something.", ""),
    ("But first —", ""),
    # "here's the thing" empty bridges
    ("But here's where it gets really interesting...", ""),
    ("But here's where it gets interesting —", ""),
    ("Here's where it gets really interesting.", ""),
    ("This is where it gets interesting —", ""),
    # "you're watching" ego-strokes
    ("If you're watching this,", ""),
    ("Since you're watching this,", ""),
    ("The fact that you're watching this", ""),
    # hollow loops
    ("You're probably wondering why — and the answer is going to surprise you.", ""),
    ("The reason why will completely change how you think about this.", ""),
    ("What comes next will change how you see everything.", ""),
    ("And that changes everything.", ""),
    ("This changes everything.", ""),
    # hollow closures
    ("Now you have the full picture — and most people never get this far.", ""),
    ("Now you understand exactly why this happens.", ""),
]


def _strip_filler(text: str) -> str:
    """Remove known filler/stalling phrases from generated script text."""
    for phrase, replacement in _FILLER_PHRASES:
        text = text.replace(phrase, replacement)
    # Collapse double spaces and leading/trailing whitespace per sentence
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\. \.', '.', text)
    return text.strip()


def _log_script(script: "VideoScript", niche: str, topic: str) -> None:
    """Save generated script to logs/scripts/ so it can be inspected and improved."""
    import pathlib, datetime
    logs_dir = pathlib.Path("logs/scripts")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_niche = re.sub(r'[^a-z0-9_]', '_', niche.lower())
    log_path = logs_dir / f"{ts}_{safe_niche}.txt"
    lines = [
        f"NICHE:  {niche}",
        f"TOPIC:  {topic}",
        f"TITLE:  {script.title}",
        f"HOOK:   {script.hook}",
        "",
    ]
    for i, sec in enumerate(script.sections, 1):
        lines.append(f"── SECTION {i}: {sec.get('heading', '(no heading)')}")
        lines.append(sec.get('script', '(empty)'))
        lines.append("")
    lines.append("── CALL TO ACTION")
    lines.append(script.call_to_action)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]Script logged → {log_path}[/dim]")


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


def _parse_and_validate(
    raw: str,
    schema_cls,
    prompt: str,
    model: str,
    system: str = "",
    max_retries: int = 2,
):
    """
    Parse Ollama JSON output and validate against a Pydantic schema.
    Auto-retries up to max_retries times on validation failure.
    Returns validated model instance or raises on exhausted retries.
    """
    from pydantic import ValidationError
    last_exc: Exception | None = None
    current_raw = raw
    for attempt in range(max_retries + 1):
        try:
            data = _parse_json_response(current_raw)
            if isinstance(data, list) and data:
                data = data[0]  # take first element for single-object schemas
            return schema_cls.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            last_exc = exc
            if attempt < max_retries:
                console.print(f"[yellow]JSON validation attempt {attempt+1} failed ({exc.__class__.__name__}) — retrying…[/yellow]")
                try:
                    current_raw = _ollama_generate(
                        prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching the schema exactly.",
                        model=model,
                        json_mode=True,
                        system=system or "You must respond with valid JSON only.",
                    )
                except Exception:
                    pass
    raise ValueError(f"Pydantic validation failed after {max_retries} retries: {last_exc}")


def _parse_json_response(raw: str) -> list | dict:
    """Extract JSON from model output — handles markdown fences, emojis, and truncation."""
    raw = raw.strip()

    def try_parse(s: str) -> list | dict | None:
        cleaned = _clean_json_str(s)
        for candidate in (s, cleaned):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as _jde:
                # "Extra data" means JSON ends before the string does — parse just the valid prefix
                if "Extra data" in str(_jde) and _jde.pos > 0:
                    try:
                        return json.loads(candidate[:_jde.pos])
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
    niche_visual_keywords: list[str] | None = None,
) -> list[str]:
    """Generate stock footage search queries for a topic using Ollama."""
    visual_hint = ""
    if niche_visual_keywords:
        visual_hint = f"\nNiche visual style keywords (prefer these themes): {', '.join(niche_visual_keywords[:8])}"
    prompt = f"""Generate {count} specific search queries to find stock footage for a YouTube video.
Topic: {topic}
Niche: {niche}{visual_hint}
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
            keywords = [str(k).strip() for k in data if k and str(k).strip() != topic][:count]
            if len(keywords) >= 3:
                return keywords
    except Exception:
        pass
    # Fallback: derive visual search terms from topic words + niche context
    return _fallback_keywords(topic, niche, count)


def _fallback_keywords(topic: str, niche: str, count: int = 8) -> list[str]:
    """Generate visual stock footage search terms when Ollama fails."""
    _NICHE_VISUALS = {
        "finance":  ["money cash stack", "stock market chart", "business meeting",
                     "laptop finance work", "bank building", "coins investment",
                     "person counting money", "financial graph screen"],
        "tech":     ["computer code screen", "robot automation", "data center servers",
                     "smartphone technology", "circuit board closeup", "ai neural network",
                     "programmer working laptop", "futuristic technology"],
        "health":   ["healthy food vegetables", "exercise gym workout", "meditation calm",
                     "doctor hospital", "running outdoors", "vitamins supplements",
                     "sleep rest bedroom", "healthy lifestyle"],
        "business": ["team meeting office", "entrepreneur working", "startup whiteboard",
                     "handshake deal", "growth chart success", "coffee laptop work",
                     "presentation audience", "business success"],
    }
    base = _NICHE_VISUALS.get(niche, _NICHE_VISUALS["business"])
    # Mix niche visuals with topic-derived terms
    words = [w for w in topic.lower().split() if len(w) > 3]
    topic_terms = [f"{words[i]} {words[i+1]}" for i in range(min(len(words)-1, 3))] if len(words) > 1 else []
    combined = topic_terms + base
    return combined[:count]


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
    system_msg = "You are a YouTube viral content strategist. Always respond with valid JSON only."
    raw = _ollama_generate(prompt, model=model, json_mode=True, system=system_msg)
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    ideas = []
    from src.models import VideoIdeaModel
    for d in data:
        try:
            m = VideoIdeaModel.from_dict(d, fallback_title=topic)
        except Exception:
            m = VideoIdeaModel(title=topic[:65], hook=f"You need to know the truth about {topic}")
        ideas.append(VideoIdea(
            title=m.title,
            hook=m.hook or f"You need to know the truth about {topic}",
            description=m.description,
            tags=m.tags or [niche, topic],
            estimated_virality=m.estimated_virality,
        ))
    return ideas


_BANNED_PHRASES_BLOCK = """
BANNED PHRASES — never write any of these:
- "Stay with me" / "Don't go anywhere" / "Don't miss this"
- "I'll tell you in a moment" / "I'll reveal this later" / "We'll get to that"
- "But first let me explain" / "Before we continue"
- "Here's where it gets interesting" / "This is where it gets really interesting"
- "If you're watching this" / "Since you're watching this"
- "You're probably wondering" / "The answer might surprise you"
- "This changes everything" / "And that changes everything"
- "Stay tuned" / "Keep watching" / "Let that sink in"
Any sentence that delays the point rather than making it is FORBIDDEN.
"""


def _viral_script_instructions(virality: str) -> str:
    """Return virality-level-specific writing instructions for the script prompt."""
    base = (
        "Structure: hook (one shocking specific fact) → problem with real numbers → "
        "historical parallel → who controls the outcome today → what the viewer can do. "
        "Every single sentence must deliver information, a number, a name, or a revelation. "
        "Never stall. Never delay. Get to the point immediately."
    )
    extras = {
        "Low": "Open with the most surprising verified statistic. Every sentence = one concrete fact.",
        "Medium": "Open with a real event and its real consequence. Name the institution. Give the year.",
        "High": (
            "First sentence: one jaw-dropping verified number. No preamble. "
            "Each section ends by revealing the next piece of the puzzle — not by teasing it."
        ),
        "Viral": (
            "First sentence is a verified fact that reframes everything the viewer thought they knew. "
            "Short sentences. Real names. Real numbers. Real years. "
            "Each section answers ONE question completely, then raises the next question by "
            "revealing a new fact — never by saying 'I'll tell you later'. "
            "The CTA must tell the viewer exactly what awareness or action this calls for."
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
    niche_system_prompt: str = "",
) -> "VideoScript":
    """Generate a viral script with sections that map to the available media clips."""
    captions_block = "\n".join(f"footage_{i+1}: {c}" for i, c in enumerate(media_captions[:15]))
    virality = getattr(idea, "estimated_virality", "Medium")
    viral_instructions = _viral_script_instructions(virality)
    prompt = f"""{ml_prefix}
Write a factual, high-retention YouTube video script on this topic.

Title: {idea.title}
Hook: {idea.hook}
Virality target: {virality}
Style: {style}
Target duration: {duration_minutes} minutes

Content rules:
{viral_instructions}

{_BANNED_PHRASES_BLOCK}

Available footage clips (MUST use these):
{captions_block}

Each section MUST reference footage that visually matches one of the clips above.

Return a JSON object with exactly these keys:
- title: video title under 70 characters, include a specific number or year
- hook: first spoken sentence — one verified fact that immediately reframes the topic, 2-3 sentences max
- sections: array of 5-7 objects with "heading", "script", and "clip_hint" keys.
  CRITICAL: each "script" MUST be at least 120 words of full spoken narration.
  Write real paragraphs with specific events, institutions, numbers, and consequences.
  "clip_hint" = short visual description matching available footage above.
- call_to_action: specific action the viewer can take right now, minimum 60 words spoken
- description: YouTube description with keywords
- tags: list of 15 SEO tags

Return ONLY the JSON object, no explanation."""
    base_sys = "You are a viral YouTube scriptwriter. Write to maximise watch time and shares. Always respond with valid JSON only."
    system_msg = f"{niche_system_prompt}\n\n{base_sys}".strip() if niche_system_prompt else base_sys
    raw = _ollama_generate(prompt, model=model, json_mode=True, system=system_msg)
    from src.models import VideoScriptModel
    try:
        validated = _parse_and_validate(raw, VideoScriptModel, prompt, model, system_msg)
        data = validated.model_dump()
    except Exception:
        data = _parse_json_response(raw)
    raw_sections = data.get("sections", []) or []
    clean_sections = [
        {**s, "script": _strip_filler(s.get("script", ""))} for s in raw_sections
    ]
    script = VideoScript(
        title=_clean_str(data.get("title", idea.title)) or idea.title,
        hook=_strip_filler(_clean_str(data.get("hook", idea.hook))),
        sections=clean_sections,
        call_to_action=_strip_filler(_clean_str(data.get("call_to_action", ""))),
        description=_clean_str(data.get("description", "")),
        tags=data.get("tags", []) or [],
    )
    _log_script(script, niche=getattr(idea, "niche", "unknown"), topic=idea.title)
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

    system_msg = "You are a YouTube viral content strategist. Always respond with valid JSON only."
    raw = _ollama_generate(prompt, model=model, json_mode=True, system=system_msg)
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    ideas = []
    from src.models import VideoIdeaModel
    for d in data:
        try:
            m = VideoIdeaModel.from_dict(d, fallback_title=topic)
        except Exception:
            m = VideoIdeaModel(title=topic[:65], hook=f"You need to know the truth about {topic}")
        ideas.append(VideoIdea(
            title=m.title,
            hook=m.hook or f"You need to know the truth about {topic}",
            description=m.description,
            tags=m.tags or [niche, topic],
            estimated_virality=m.estimated_virality,
        ))
    return ideas


def generate_video_script(
    idea: VideoIdea,
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
    model: str = DEFAULT_MODEL,
    niche: str = "general",
    niche_system_prompt: str = "",
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
    virality = getattr(idea, "estimated_virality", "High")
    viral_instructions = _viral_script_instructions(virality)
    prompt = f"""Write a factual, high-retention YouTube video script on this topic.

Title: {idea.title}
Hook: {idea.hook}
Style: {style}
Target duration: {duration_minutes} minutes
Niche: {niche}
{dna_context}

CONTENT RULES (mandatory):
{viral_instructions}

{_BANNED_PHRASES_BLOCK}

Return a JSON object with exactly these keys:
- title: the video title (under 70 characters, include a specific number or year)
- hook: first spoken sentence — one verified fact that reframes the topic, 2-3 sentences max
- sections: array of 5-7 objects with "heading" and "script" keys.
  CRITICAL: each "script" value MUST be at least 120 words of spoken narration.
  Write full paragraphs — specific events, real institutions, real numbers, real consequences.
  Do NOT write bullet points. Do NOT write filler. Every sentence must add information.
- call_to_action: final spoken CTA — specific action the viewer can take right now, minimum 60 words
- description: YouTube description with keywords
- tags: list of 15 SEO tags

Return ONLY valid JSON, no explanation."""

    base_sys = "You are a professional YouTube scriptwriter. Always respond with valid JSON only."
    system_msg = f"{niche_system_prompt}\n\n{base_sys}".strip() if niche_system_prompt else base_sys
    raw = _ollama_generate(prompt, model=model, json_mode=True, system=system_msg)
    from src.models import VideoScriptModel
    try:
        validated = _parse_and_validate(raw, VideoScriptModel, prompt, model, system_msg)
        data = validated.model_dump()
    except Exception:
        data = _parse_json_response(raw)

    raw_sections = data.get("sections", []) or []
    clean_sections = [
        {**s, "script": _strip_filler(s.get("script", ""))} for s in raw_sections
    ]
    script = VideoScript(
        title=_clean_str(data.get("title", idea.title)) or idea.title,
        hook=_strip_filler(_clean_str(data.get("hook", idea.hook))),
        sections=clean_sections,
        call_to_action=_strip_filler(_clean_str(data.get("call_to_action", ""))),
        description=_clean_str(data.get("description", "")),
        tags=data.get("tags", []) or [],
    )

    # ── 3. Score retention (report only — no filler injection) ─────────
    try:
        from src.viral_intelligence.retention_engine import RetentionEngine
        engine = RetentionEngine()
        score = engine.score(script.sections, hook=script.hook)
        engine.print_report(score)
    except Exception:
        pass

    _log_script(script, niche=niche, topic=idea.title)
    return script


def _inject_retention_patterns(sections: list[dict]) -> list[dict]:
    """
    Fallback enhancement when the model produces thin content.
    Strips filler and ensures each section has at least some substance.
    Does NOT inject filler phrases — only cleans what's there.
    """
    enhanced = []
    for section in sections:
        text: str = str(section.get("script") or section.get("text") or "")
        cleaned = _strip_filler(text)
        enhanced.append({**section, "script": cleaned})
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
