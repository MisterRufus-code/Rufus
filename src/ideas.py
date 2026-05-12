"""AI-powered video idea and script generator using Ollama (100% free, local)."""

from __future__ import annotations

import json
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


def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL, system: str = "") -> str:
    """Send a prompt to the local Ollama server and return the response text."""
    if not _check_ollama():
        raise RuntimeError(
            "Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Install from:  https://ollama.ai"
        )

    import httpx
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.85, "num_predict": 2048},
    }
    r = httpx.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _clean_json_str(s: str) -> str:
    """Fix common LLM JSON issues: unescaped newlines/tabs inside strings."""
    import re
    # Replace literal newlines inside JSON string values with \n
    def fix_string(m: re.Match) -> str:
        return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_string, s, flags=re.DOTALL)


def _parse_json_response(raw: str) -> list | dict:
    """Extract JSON from model output — handles markdown fences and LLM quirks."""
    raw = raw.strip()

    def try_parse(s: str) -> list | dict | None:
        for candidate in (s, _clean_json_str(s)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

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

    # Last resort: slice from first [ or {
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char)
        if s != -1 and e != -1:
            result = try_parse(raw[s : e + 1])
            if result is not None:
                return result

    raise ValueError(f"Could not parse JSON from model response:\n{raw[:300]}")


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
        raw = _ollama_generate(prompt, model=model,
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
    raw = _ollama_generate(prompt, model=model,
                           system="You are a YouTube viral content strategist. Always respond with valid JSON only.")
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    return [
        VideoIdea(
            title=d.get("title", ""),
            hook=d.get("hook", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            estimated_virality=d.get("estimated_virality", "Medium"),
        )
        for d in data
    ]


def generate_script_from_media(
    idea: VideoIdea,
    media_captions: list[str],
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
    model: str = DEFAULT_MODEL,
    ml_prefix: str = "",
) -> "VideoScript":
    """Generate a script with sections that map to the available media clips."""
    captions_block = "\n".join(f"[CLIP {i+1}]: {c}" for i, c in enumerate(media_captions[:15]))
    prompt = f"""{ml_prefix}
Write a YouTube video script using the available clips listed below.

Title: {idea.title}
Hook: {idea.hook}
Style: {style}
Target duration: {duration_minutes} minutes

Available clips:
{captions_block}

Each section MUST reference footage that visually matches one of the clips above.

Return a JSON object with exactly these keys:
- title: the video title
- hook: spoken opening (first 15 seconds)
- sections: array of objects with "heading", "script", and "clip_hint" keys
  (clip_hint = short phrase matching the visual content of one available clip)
- call_to_action: final 30-second spoken CTA
- description: full YouTube description
- tags: list of 15 SEO tags

Return ONLY the JSON object, no explanation."""
    raw = _ollama_generate(prompt, model=model,
                           system="You are a professional YouTube scriptwriter. Always respond with valid JSON only.")
    data = _parse_json_response(raw)
    return VideoScript(
        title=data.get("title", idea.title),
        hook=data.get("hook", ""),
        sections=data.get("sections", []),
        call_to_action=data.get("call_to_action", ""),
        description=data.get("description", ""),
        tags=data.get("tags", []),
    )


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
        system="You are a YouTube viral content strategist. Always respond with valid JSON only.",
    )
    data = _parse_json_response(raw)
    if isinstance(data, dict):
        data = [data]
    return [
        VideoIdea(
            title=d.get("title", ""),
            hook=d.get("hook", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            estimated_virality=d.get("estimated_virality", "Medium"),
        )
        for d in data
    ]


def generate_video_script(
    idea: VideoIdea,
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
    model: str = DEFAULT_MODEL,
) -> VideoScript:
    """Generate a full video script using a local Ollama model."""
    prompt = f"""Write a YouTube video script.
Title: {idea.title}
Hook: {idea.hook}
Style: {style}
Target duration: {duration_minutes} minutes

Return a JSON object with exactly these keys:
- title: the video title
- hook: spoken opening (first 15 seconds)
- sections: array of objects with "heading" and "script" keys
- call_to_action: final 30-second spoken CTA for likes/subscribe
- description: full YouTube description with timestamp placeholders
- tags: list of 15 SEO tags

Return ONLY the JSON object, no explanation."""

    raw = _ollama_generate(
        prompt,
        model=model,
        system="You are a professional YouTube scriptwriter. Always respond with valid JSON only.",
    )
    data = _parse_json_response(raw)
    return VideoScript(
        title=data.get("title", idea.title),
        hook=data.get("hook", ""),
        sections=data.get("sections", []),
        call_to_action=data.get("call_to_action", ""),
        description=data.get("description", ""),
        tags=data.get("tags", []),
    )


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
