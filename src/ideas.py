"""AI-powered video idea and script generator using Claude."""

from __future__ import annotations

import anthropic
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import config

console = Console()


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


def _get_client() -> anthropic.Anthropic:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def generate_video_ideas(
    topic: str,
    niche: str,
    count: int = 5,
    trending_context: str = "",
) -> list[VideoIdea]:
    """Generate viral video ideas for a given topic and niche."""
    client = _get_client()

    trending_block = (
        f"\n\nCurrently trending context:\n{trending_context}"
        if trending_context
        else ""
    )

    prompt = f"""You are a YouTube viral content strategist. Generate {count} viral video ideas for:
Topic: {topic}
Niche/Channel: {niche}{trending_block}

For each idea return a JSON array with objects containing:
- title: compelling, clickbait-worthy title (under 70 chars)
- hook: first 3 seconds hook line that grabs attention
- description: 2-sentence video description
- tags: list of 10 relevant SEO tags
- estimated_virality: "Low" | "Medium" | "High" | "Viral"

Return ONLY valid JSON, no markdown, no explanation."""

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system="You are an expert YouTube growth strategist who specializes in creating viral content. Always return valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    ideas_data = json.loads(raw)
    return [
        VideoIdea(
            title=d["title"],
            hook=d["hook"],
            description=d["description"],
            tags=d.get("tags", []),
            estimated_virality=d.get("estimated_virality", "Medium"),
        )
        for d in ideas_data
    ]


def generate_video_script(
    idea: VideoIdea,
    duration_minutes: int = 10,
    style: str = "engaging and conversational",
) -> VideoScript:
    """Generate a full video script from a VideoIdea."""
    client = _get_client()

    prompt = f"""Write a full YouTube video script for:

Title: {idea.title}
Hook: {idea.hook}
Style: {style}
Target duration: ~{duration_minutes} minutes

Return a JSON object with:
- title: the video title
- hook: attention-grabbing opening (first 15 seconds, spoken word)
- sections: array of objects with "heading" and "script" (the spoken words for that section)
- call_to_action: final 30-second CTA encouraging likes/subscribe/comment
- description: full YouTube description (with timestamps placeholder)
- tags: list of 15 SEO-optimized tags

Return ONLY valid JSON, no markdown wrapper."""

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system="You are a professional YouTube scriptwriter. Write engaging, retention-optimized scripts. Always return valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return VideoScript(
        title=data["title"],
        hook=data["hook"],
        sections=data.get("sections", []),
        call_to_action=data["call_to_action"],
        description=data["description"],
        tags=data.get("tags", []),
    )


def print_ideas(ideas: list[VideoIdea]) -> None:
    virality_colors = {
        "Low": "dim",
        "Medium": "yellow",
        "High": "green",
        "Viral": "bold magenta",
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
        md_parts.append(f"## {section['heading']}\n{section['script']}\n")
    md_parts.append(f"## Call to Action\n{script.call_to_action}\n")
    md_parts.append(f"---\n**Description:**\n{script.description}")
    console.print(Markdown("\n".join(md_parts)))
