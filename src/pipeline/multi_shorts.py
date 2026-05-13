"""
Multi-angle Shorts generator.

Given one long-form script, generates 4 distinct Shorts angles automatically:
  Angle 1 — "The Aha Moment"   : the most surprising/counterintuitive section
  Angle 2 — "The Mistake"      : loss-aversion framing of the main point
  Angle 3 — "The Question"     : hook reframed as an open question
  Angle 4 — "The Trailer"      : hook + CTA only (60s teaser for long form)

Each angle gets its own script, render, and optionally its own upload.
4x upload volume, near-zero extra cost per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule

console = Console()


@dataclass
class ShortsAngle:
    label: str
    hook: str
    sections: list[dict]
    call_to_action: str
    tags: list[str] = field(default_factory=list)


def _ollama(prompt: str, model: str) -> str:
    import httpx
    from src.task_lock import task_lock
    with task_lock("multi_shorts_ollama"):
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.9, "num_predict": 800}},
            timeout=90,
        )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def generate_angles(
    topic: str,
    niche: str,
    script_sections: list[dict],
    original_hook: str,
    original_cta: str,
    model: str = "mistral",
) -> list[ShortsAngle]:
    """
    Derive 4 Shorts angles from the long-form script.
    Returns angles sorted by estimated virality (best first).
    """
    sections_text = "\n".join(
        f"- {s.get('heading', '')}: {s.get('script', '')[:120]}"
        for s in script_sections
    )

    prompt = f"""You are a YouTube Shorts strategist. Given this long-form video script, create 4 distinct 60-second Shorts angles.

Topic: {topic}
Niche: {niche}
Original hook: {original_hook}

Script sections:
{sections_text}

Create these 4 angles. Each must have a hook (8-15 words) and 3 short script sections (20 words each max):

ANGLE 1 — "The Aha Moment": Pick the most surprising/counterintuitive point. Hook should create a curiosity gap.
ANGLE 2 — "The Mistake": Reframe the main point as a mistake most people make. Hook starts with "Stop" or "You're".
ANGLE 3 — "The Question": Turn the main insight into a question the viewer asks themselves. Start with "What if" or "Did you know".
ANGLE 4 — "The Teaser": Use only the hook and conclusion as a trailer for the full video. Include "Watch the full video" in CTA.

Return ONLY this JSON (no markdown, no explanation):
[
  {{
    "label": "The Aha Moment",
    "hook": "...",
    "sections": [
      {{"heading": "Setup", "script": "..."}},
      {{"heading": "The Twist", "script": "..."}},
      {{"heading": "The Payoff", "script": "..."}}
    ],
    "call_to_action": "...",
    "tags": ["tag1", "tag2", "Shorts"]
  }},
  ... (3 more)
]"""

    try:
        raw = _ollama(prompt, model)
        # Strip markdown fences
        import re
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(raw)
        angles = []
        for item in data:
            angles.append(ShortsAngle(
                label=item.get("label", "Angle"),
                hook=item.get("hook", original_hook),
                sections=item.get("sections", []),
                call_to_action=item.get("call_to_action", original_cta),
                tags=item.get("tags", ["Shorts", niche]),
            ))
        return angles
    except Exception as exc:
        console.print(f"[yellow]Angle generation failed ({exc}) — using fallback angles.[/yellow]")
        return _fallback_angles(topic, niche, original_hook, original_cta, script_sections)


def _fallback_angles(
    topic: str,
    niche: str,
    hook: str,
    cta: str,
    sections: list[dict],
) -> list[ShortsAngle]:
    """Deterministic fallback when Ollama fails."""
    first_section = sections[0] if sections else {"heading": "Main Point", "script": f"The key insight about {topic}."}
    mid_section = sections[len(sections) // 2] if sections else first_section

    return [
        ShortsAngle(
            label="The Aha Moment",
            hook=f"The surprising truth about {topic} nobody talks about",
            sections=[first_section, mid_section, {"heading": "The Insight", "script": f"Now you know what changes with {topic}."}],
            call_to_action=cta,
            tags=["Shorts", niche, topic],
        ),
        ShortsAngle(
            label="The Mistake",
            hook=f"Stop making this {topic} mistake — it's costing you",
            sections=[{"heading": "The Problem", "script": f"Most people approach {topic} completely wrong."}, first_section, {"heading": "The Fix", "script": "Here's exactly what to do instead."}],
            call_to_action=cta,
            tags=["Shorts", niche, topic],
        ),
        ShortsAngle(
            label="The Question",
            hook=f"What if everything you know about {topic} is wrong?",
            sections=[{"heading": "The Question", "script": f"What if {topic} works the opposite way you think?"}, mid_section, {"heading": "The Answer", "script": "The answer changes how you approach everything."}],
            call_to_action=cta,
            tags=["Shorts", niche, topic],
        ),
        ShortsAngle(
            label="The Teaser",
            hook=hook,
            sections=[{"heading": "Preview", "script": f"In this video you'll learn the most important thing about {topic}."}, {"heading": "Tease", "script": "I break down exactly what works and what doesn't."}, {"heading": "CTA", "script": "Watch the full video for the complete breakdown."}],
            call_to_action="Watch the full video — link in bio",
            tags=["Shorts", niche, topic, "trailer"],
        ),
    ]


def render_all_angles(
    angles: list[ShortsAngle],
    clips: list,          # list[TimelineClip]
    audio_dir: Path,
    output_dir: Path,
    tts_voice: str = "af_heart",
    ollama_model: str = "mistral",
    upload: bool = False,
    privacy: str = "private",
) -> list[dict]:
    """
    Render each angle as a separate Shorts MP4.
    Returns a list of result dicts with paths and status.
    """
    from src.pipeline.shorts_renderer import render_shorts
    results = []

    for i, angle in enumerate(angles):
        label_slug = angle.label.lower().replace(" ", "_").replace("/", "")
        angle_dir = output_dir / f"short_{label_slug}"
        angle_dir.mkdir(parents=True, exist_ok=True)

        console.print(Rule(f"[bold cyan]Rendering Shorts Angle {i+1}: {angle.label}[/bold cyan]"))

        # Generate TTS for this angle
        audio_path: Optional[Path] = None
        try:
            from src.tts.kokoro import synthesize_sections, merge_audio_files
            tts_sections = [{"script": angle.hook}] + angle.sections + [{"script": angle.call_to_action}]
            wav_paths = synthesize_sections(tts_sections, angle_dir / "audio", voice=tts_voice)
            if wav_paths:
                audio_path = angle_dir / "voiceover.wav"
                merge_audio_files(wav_paths, audio_path)
        except Exception as exc:
            console.print(f"[yellow]TTS skipped for {angle.label} ({exc})[/yellow]")

        # Render
        output_path = angle_dir / f"short_{label_slug}.mp4"
        try:
            render_shorts(
                clips=clips,
                audio_path=audio_path,
                output_path=output_path,
                script_sections=angle.sections,
                hook=angle.hook,
                tmp_dir=angle_dir / "_tmp",
            )
            result = {"angle": angle.label, "path": str(output_path), "success": True}
        except Exception as exc:
            console.print(f"[red]Render failed for {angle.label}: {exc}[/red]")
            result = {"angle": angle.label, "path": None, "success": False, "error": str(exc)}
            results.append(result)
            continue

        # Optional upload
        if upload and output_path.exists():
            try:
                from src.uploader import upload_video
                vid_id = upload_video(
                    video_path=str(output_path),
                    title=f"{angle.hook} #Shorts",
                    description=f"{angle.hook}\n\n#{' #'.join(angle.tags)}",
                    tags=angle.tags,
                    privacy_status=privacy,
                )
                result["youtube_video_id"] = vid_id
                console.print(f"[green]Uploaded {angle.label}:[/green] {vid_id}")
            except Exception as exc:
                console.print(f"[yellow]Upload failed for {angle.label}: {exc}[/yellow]")

        results.append(result)
        console.print(f"[green]✓ {angle.label}:[/green] {output_path.name}")

    return results
