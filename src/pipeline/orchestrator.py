"""
Full pipeline orchestrator — 100% free, local execution.

Media-first flow:
  topic → trends → AI keywords → footage download → index & analyze media
       → ideas from media → script from media → semantic match (global dedup)
       → entropy check → TTS voiceover → FFmpeg render
       → record asset usage (ML) → upload
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
    keywords_used: list[str] = field(default_factory=list)
    asset_ids_used: list[str] = field(default_factory=list)
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
            item for t in topics for item in ([t.keyword] + t.related_queries[:3])
        )
        console.print(f"[green]Trending context:[/green] {trending_ctx[:120]}")
    except Exception as exc:
        console.print(f"[yellow]Trends unavailable ({exc}), continuing...[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 2 — AI keyword generation (footage search terms)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 2 — AI Keyword Generation (Ollama)[/bold]"))
    from src.ideas import generate_search_keywords
    from src.ml.optimizer import get_keyword_context
    try:
        ml_kw_context = get_keyword_context(niche, topic)
        keywords = generate_search_keywords(
            topic, niche, model=ollama_model,
            count=8, ml_context=ml_kw_context,
        )
        result.keywords_used = keywords
        console.print(f"[green]Search keywords:[/green] {', '.join(keywords)}")
    except Exception as exc:
        console.print(f"[yellow]Keyword generation failed ({exc}), using topic directly.[/yellow]")
        keywords = [topic]
        result.keywords_used = keywords

    # ------------------------------------------------------------------ #
    # Step 3 — Download footage FIRST (media-first approach)
    # ------------------------------------------------------------------ #
    if download_footage:
        console.print(Rule("[bold]Step 3 — Footage Download (Pexels/Pixabay)[/bold]"))
        try:
            from src.media_fetch.pexels import download_videos as pexels_dl
            pexels_dl(keywords, videos_per_query=videos_per_scene)
        except Exception as exc:
            console.print(f"[yellow]Pexels: {exc}[/yellow]")

        try:
            from src.media_fetch.pixabay import download_videos as pixabay_dl
            pixabay_dl(keywords, videos_per_query=videos_per_scene)
        except Exception as exc:
            console.print(f"[yellow]Pixabay: {exc}[/yellow]")
    else:
        console.print(Rule("[bold]Step 3 — Footage Download (skipped)[/bold]"))

    # ------------------------------------------------------------------ #
    # Step 4 — Index media & extract captions
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 4 — Index & Analyze Media[/bold]"))
    media_captions: list[str] = []
    try:
        from src.ingestion.indexer import index_library
        index_result = index_library()
        console.print(f"[green]Indexed:[/green] {index_result}")

        from src.database.vector_store import get_vector_store
        store = get_vector_store()
        all_assets = store.get_all_assets()
        media_captions = [
            a.caption for a in all_assets
            if a.caption and a.caption.strip() and a.caption != "media asset"
        ]
        # Deduplicate captions while preserving order
        seen: set[str] = set()
        unique_captions: list[str] = []
        for c in media_captions:
            if c not in seen:
                seen.add(c)
                unique_captions.append(c)
        media_captions = unique_captions
        console.print(f"[green]{len(media_captions)} unique captions available for script context[/green]")
    except Exception as exc:
        console.print(f"[yellow]Media analysis skipped ({exc})[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 5 — Generate ideas from media context
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 5 — Idea Generation from Media (Ollama)[/bold]"))
    from src.ml.optimizer import get_optimized_prompt_prefix
    ml_prefix = get_optimized_prompt_prefix(niche)

    try:
        if media_captions:
            from src.ideas import generate_ideas_from_media
            ideas = generate_ideas_from_media(
                topic, niche,
                media_captions=media_captions,
                count=ideas_count,
                trending_context=trending_ctx,
                model=ollama_model,
                ml_prefix=ml_prefix,
            )
        else:
            from src.ideas import generate_video_ideas
            ideas = generate_video_ideas(
                topic, niche, count=ideas_count,
                trending_context=trending_ctx, model=ollama_model,
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
    # Step 6 — Generate script from media context
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 6 — Script Generation from Media (Ollama)[/bold]"))
    try:
        if media_captions:
            from src.ideas import generate_script_from_media
            script = generate_script_from_media(
                best_idea,
                media_captions=media_captions,
                model=ollama_model,
                ml_prefix=ml_prefix,
            )
        else:
            from src.ideas import generate_video_script
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
    console.print(f"[green]Script ready:[/green] {len(script.sections)} sections")

    # ------------------------------------------------------------------ #
    # Step 7 — Semantic matching with global deduplication
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 7 — Semantic Matching (global dedup)[/bold]"))
    matched_clips: list[TimelineClip] = []
    try:
        from src.matching.semantic import scenes_from_script, match_all_scenes
        scenes = scenes_from_script(script.sections)
        match_results = match_all_scenes(scenes, global_dedup_videos=5)
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

    # ------------------------------------------------------------------ #
    # Step 8 — Visual entropy check
    # ------------------------------------------------------------------ #
    if matched_clips:
        console.print(Rule("[bold]Step 8 — Visual Entropy Check[/bold]"))
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
    # Step 9 — TTS voiceover (Kokoro, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 9 — Voiceover (Kokoro TTS)[/bold]"))
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
    # Step 10 — Render video (FFmpeg, free)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 10 — Video Render (FFmpeg)[/bold]"))
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
    # Step 11 — Record asset usage for ML deduplication
    # ------------------------------------------------------------------ #
    if matched_clips:
        result.asset_ids_used = [c.asset.asset_id for c in matched_clips]
        try:
            from src.ml.feedback import record_render
            record_render(
                topic=topic,
                niche=niche,
                asset_ids=result.asset_ids_used,
                model=ollama_model,
                entropy_score=result.entropy_score,
                script_style=ml_prefix[:80] if ml_prefix else "",
                keywords_used=result.keywords_used,
            )
        except Exception as exc:
            console.print(f"[yellow]ML record skipped ({exc})[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 12 — Upload to YouTube (free API, optional)
    # ------------------------------------------------------------------ #
    if upload and result.video_path and result.video_path.exists():
        console.print(Rule("[bold]Step 12 — YouTube Upload[/bold]"))
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
