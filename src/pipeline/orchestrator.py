"""
Full pipeline orchestrator — 100% free, local execution.

Flow:
  topic → trends → ideas (Ollama) → script (Ollama) → footage download
       → semantic match → entropy check → TTS voiceover → FFmpeg render → upload
       → record feedback → ML optimize
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule

import config
from src.database.models import TimelineClip

console = Console()


@dataclass
class PipelineResult:
    topic: str
    niche: str
    video_path: Optional[Path] = None
    youtube_video_id: Optional[str] = None
    script_path: Optional[Path] = None
    ideas_path: Optional[Path] = None
    entropy_score: float = 0.0
    success: bool = False
    errors: list[str] = field(default_factory=list)


def run_pipeline(
    topic: str,
    niche: str = "general",
    ollama_model: str = "mistral",
    tts_voice: str = "af_heart",
    ideas_count: int = 3,
    geo: str = "US",
    download_footage: bool = True,
    videos_per_scene: int = 3,
    output_dir: Optional[Path] = None,
    upload: bool = False,
    privacy: str = "private",
    music_path: Optional[Path] = None,
) -> PipelineResult:
    result = PipelineResult(topic=topic, niche=niche)
    out = Path(output_dir or config.OUTPUT_PATH) / topic.replace(" ", "_")[:40]
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Trend research
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 1 — Trend Research[/bold]"))
    trending_ctx = ""
    try:
        from src.trends import fetch_google_trends
        topics = fetch_google_trends([topic], geo=geo)
        trending_ctx = ", ".join(
            [t.keyword] + t.related_queries[:3] for t in topics
        )
        console.print(f"[green]Trending context:[/green] {trending_ctx[:120]}")
    except Exception as exc:
        console.print(f"[yellow]Trends unavailable ({exc}), continuing...[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 2 — Generate ideas (Ollama, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 2 — Idea Generation (Ollama)[/bold]"))
    from src.ideas import generate_video_ideas, generate_video_script
    try:
        ideas = generate_video_ideas(
            topic, niche, count=ideas_count,
            trending_context=trending_ctx, model=ollama_model
        )
    except Exception as exc:
        result.errors.append(f"Idea generation failed: {exc}")
        console.print(f"[red]{exc}[/red]")
        return result

    ideas_file = out / "ideas.json"
    ideas_file.write_text(
        json.dumps([{"title": i.title, "hook": i.hook, "tags": i.tags,
                     "estimated_virality": i.estimated_virality} for i in ideas], indent=2)
    )
    result.ideas_path = ideas_file
    best_idea = max(
        ideas,
        key=lambda i: {"Low": 1, "Medium": 2, "High": 3, "Viral": 4}.get(i.estimated_virality, 2)
    )
    console.print(f"[cyan]Top idea:[/cyan] {best_idea.title} [{best_idea.estimated_virality}]")

    # ------------------------------------------------------------------ #
    # Step 3 — Generate script (Ollama, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 3 — Script Generation (Ollama)[/bold]"))
    try:
        from src.ml.optimizer import get_optimized_prompt_prefix
        _ = get_optimized_prompt_prefix(niche)   # warms up learned context
        script = generate_video_script(best_idea, model=ollama_model)
    except Exception as exc:
        result.errors.append(f"Script generation failed: {exc}")
        console.print(f"[red]{exc}[/red]")
        return result

    script_file = out / "script.json"
    script_file.write_text(
        json.dumps({
            "title": script.title, "hook": script.hook,
            "sections": script.sections, "call_to_action": script.call_to_action,
            "description": script.description, "tags": script.tags,
        }, indent=2)
    )
    result.script_path = script_file

    # ------------------------------------------------------------------ #
    # Step 4 — Download free footage (Pexels + Pixabay)
    # ------------------------------------------------------------------ #
    if download_footage:
        console.print(Rule("[bold]Step 4 — Footage Download (Pexels/Pixabay)[/bold]"))
        scene_queries = [topic] + [s.get("heading", topic) for s in script.sections[:5]]

        try:
            from src.media_fetch.pexels import download_videos as pexels_dl
            pexels_dl(scene_queries, videos_per_query=videos_per_scene)
        except Exception as exc:
            console.print(f"[yellow]Pexels: {exc}[/yellow]")

        try:
            from src.media_fetch.pixabay import download_videos as pixabay_dl
            pixabay_dl(scene_queries, videos_per_query=videos_per_scene)
        except Exception as exc:
            console.print(f"[yellow]Pixabay: {exc}[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 5 — Index media & semantic matching
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 5 — Semantic Media Matching[/bold]"))
    try:
        from src.ingestion.indexer import index_library
        index_result = index_library()
        console.print(f"[green]Indexed:[/green] {index_result}")

        from src.matching.semantic import scenes_from_script, match_all_scenes
        scenes = scenes_from_script(script.sections)
        match_results = match_all_scenes(scenes)
        matched_clips: list[TimelineClip] = []
        for mr in match_results:
            if mr.selected:
                asset = mr.selected
                duration = min(asset.duration_seconds or 5.0, 8.0)
                matched_clips.append(TimelineClip(
                    asset=asset,
                    start_time=0,
                    end_time=duration,
                    scene_index=mr.scene.scene_index,
                    in_point=0.0,
                    out_point=duration,
                ))
        console.print(f"[green]Matched {len(matched_clips)} clips for {len(scenes)} scenes[/green]")
    except Exception as exc:
        console.print(f"[yellow]Matching skipped ({exc})[/yellow]")
        matched_clips = []

    # ------------------------------------------------------------------ #
    # Step 6 — Visual entropy check
    # ------------------------------------------------------------------ #
    if matched_clips:
        console.print(Rule("[bold]Step 6 — Visual Entropy Check[/bold]"))
        try:
            from src.entropy.engine import score_timeline, print_entropy_report
            report = score_timeline(matched_clips)
            print_entropy_report(report)
            result.entropy_score = report.overall_score
            if not report.passed:
                console.print("[yellow]Entropy below threshold — using available clips anyway.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Entropy check skipped ({exc})[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 7 — TTS voiceover (Kokoro, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 7 — Voiceover (Kokoro TTS)[/bold]"))
    audio_path: Optional[Path] = None
    try:
        from src.tts.kokoro import synthesize_sections, merge_audio_files
        audio_dir = out / "audio"
        all_sections = [{"script": script.hook}] + script.sections + [{"script": script.call_to_action}]
        wav_paths = synthesize_sections(all_sections, audio_dir, voice=tts_voice)
        if wav_paths:
            audio_path = out / "voiceover.wav"
            merge_audio_files(wav_paths, audio_path)
    except Exception as exc:
        console.print(f"[yellow]TTS skipped ({exc})[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 8 — Render video (FFmpeg, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 8 — Video Render (FFmpeg)[/bold]"))
    video_path = out / "final_video.mp4"
    if matched_clips:
        try:
            from src.pipeline.renderer import render_video
            render_video(matched_clips, audio_path, video_path, music_path=music_path)
            result.video_path = video_path
        except Exception as exc:
            result.errors.append(f"Render failed: {exc}")
            console.print(f"[red]Render error: {exc}[/red]")
    else:
        console.print("[yellow]No clips matched — render skipped. Add media to media_library/ first.[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 9 — Upload to YouTube (free API)
    # ------------------------------------------------------------------ #
    if upload and result.video_path and result.video_path.exists():
        console.print(Rule("[bold]Step 9 — YouTube Upload[/bold]"))
        try:
            from src.uploader import upload_video
            vid_id = upload_video(
                video_path=str(result.video_path),
                title=script.title,
                description=script.description,
                tags=script.tags,
                privacy_status=privacy,
            )
            result.youtube_video_id = vid_id
        except Exception as exc:
            result.errors.append(f"Upload failed: {exc}")
            console.print(f"[red]Upload error: {exc}[/red]")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    result.success = result.video_path is not None and result.video_path.exists()
    console.print(Rule("[bold green]Pipeline Complete[/bold green]"))
    console.print(f"Output directory: [cyan]{out.resolve()}[/cyan]")
    if result.youtube_video_id:
        console.print(f"YouTube: [cyan]https://www.youtube.com/watch?v={result.youtube_video_id}[/cyan]")
    if result.errors:
        for err in result.errors:
            console.print(f"[red]• {err}[/red]")
    return result
