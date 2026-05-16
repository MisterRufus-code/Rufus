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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule

import config
from src.database.models import TimelineClip
from src.logging_config import get_logger
from src.utils.retry import CircuitBreaker
from src.pipeline.checkpoint import CheckpointManager, PipelineStage

_ollama_cb = CircuitBreaker(threshold=5, reset_timeout=120.0)


def _run_with_retry(fn, *args, max_attempts=3, **kwargs):
    """Run fn(*args, **kwargs) with exponential backoff. Used for Ollama/network steps."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                raise
            import time, random
            wait = min(2 ** (attempt - 1) + random.uniform(0, 1), 30)
            time.sleep(wait)

console = Console()
log = get_logger("pipeline.orchestrator")


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
    thumbnail_path: Optional[Path] = None
    success: bool = False
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


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
    shorts: bool = False,
    low_power: bool = False,
    multi_shorts: bool = False,
    dry_run: bool = False,  # skip TTS, FFmpeg, upload — validate pipeline logic only
    duration_minutes: int = 5,  # target long-form video length
) -> PipelineResult:
    import time

    if low_power:
        from src.ingestion.extractor import set_low_power
        set_low_power(True)
        console.print("[dim]Low-power mode: CPU threads limited to 2[/dim]")

    def breathe(seconds: float = 2.0) -> None:
        if low_power:
            time.sleep(seconds)

    result = PipelineResult(topic=topic, niche=niche, dry_run=dry_run)
    safe_topic = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:40] or "video"
    out = Path(output_dir or config.OUTPUT_PATH) / safe_topic
    out.mkdir(parents=True, exist_ok=True)

    cp = CheckpointManager(out)
    _cp = cp.load()
    if _cp and not dry_run:
        console.print(f"[bold cyan]Resuming from checkpoint: stage={_cp.stage}[/bold cyan]")
        log.info("pipeline resuming from checkpoint", stage=_cp.stage, topic=topic)
    else:
        cp.save(PipelineStage.STARTED, payload={"topic": topic, "niche": niche})

    log.info("pipeline started", topic=topic, niche=niche, model=ollama_model,
             shorts=shorts, dry_run=dry_run, output_dir=str(out))

    if dry_run:
        console.print("[bold yellow]DRY RUN — TTS, FFmpeg render, and upload are skipped.[/bold yellow]")

    # ------------------------------------------------------------------ #
    # Pre-flight — fail fast before any expensive work
    # ------------------------------------------------------------------ #
    try:
        from src.preflight import run_preflight
        run_preflight(output_dir=out, abort_on_failure=True)
    except SystemExit:
        result.errors.append("Pre-flight check failed — see output above.")
        return result

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
        with _ollama_cb:
            keywords = _run_with_retry(
                generate_search_keywords,
                topic, niche, model=ollama_model,
                count=8, ml_context=ml_kw_context,
            )
        result.keywords_used = keywords
        console.print(f"[green]Search keywords:[/green] {', '.join(keywords)}")
    except Exception as exc:
        console.print(f"[yellow]Keyword generation failed ({exc}), using topic directly.[/yellow]")
        keywords = [topic]
        result.keywords_used = keywords
    cp.advance(PipelineStage.KEYWORDS, keywords=keywords)

    breathe(2)
    # ------------------------------------------------------------------ #
    # Step 3 — Download footage FIRST (media-first approach)
    # ------------------------------------------------------------------ #
    if download_footage:
        console.print(Rule("[bold]Step 3 — Footage Download (Pexels/Pixabay)[/bold]"))
        if not cp.completed_stage(PipelineStage.FOOTAGE_DOWNLOADED):
            try:
                from src.media_fetch.pexels import download_videos as pexels_dl
                pexels_dl(keywords, videos_per_query=videos_per_scene)
            except Exception as exc:
                console.print(f"[yellow]Pexels: {exc}[/yellow]")

            try:
                from src.media_fetch.ytcc import download_videos as ytcc_dl
                ytcc_dl(keywords, videos_per_query=videos_per_scene)
            except Exception as exc:
                console.print(f"[yellow]YouTube CC: {exc}[/yellow]")
            cp.advance(PipelineStage.FOOTAGE_DOWNLOADED)
        else:
            console.print("[dim]  Footage already downloaded (checkpoint) — skipping.[/dim]")
    else:
        console.print(Rule("[bold]Step 3 — Footage Download (skipped)[/bold]"))

    # ------------------------------------------------------------------ #
    # Step 4 — Index media & extract captions
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 4 — Index & Analyze Media[/bold]"))
    media_captions: list[str] = []
    try:
        from src.ingestion.indexer import index_library
        index_result = index_library(low_power=low_power)
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
        cp.advance(PipelineStage.MEDIA_INDEXED)
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
            with _ollama_cb:
                ideas = _run_with_retry(
                    generate_ideas_from_media,
                    topic, niche,
                    media_captions=media_captions,
                    count=ideas_count,
                    trending_context=trending_ctx,
                    model=ollama_model,
                    ml_prefix=ml_prefix,
                )
        else:
            from src.ideas import generate_video_ideas
            with _ollama_cb:
                ideas = _run_with_retry(
                    generate_video_ideas,
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
    if not ideas:
        result.errors.append("Idea generation returned empty list")
        console.print("[red]No ideas generated — aborting.[/red]")
        return result
    best_idea = max(
        ideas,
        key=lambda i: {"Low": 1, "Medium": 2, "High": 3, "Viral": 4}.get(i.estimated_virality, 2)
    )
    cp.advance(PipelineStage.IDEAS_GENERATED, best_idea_title=best_idea.title)
    console.print(f"[cyan]Top idea:[/cyan] {best_idea.title} [{best_idea.estimated_virality}]")

    breathe(2)
    # ------------------------------------------------------------------ #
    # Step 6 — Generate script from media context (or Shorts script)
    # ------------------------------------------------------------------ #
    mode_label = "Shorts (60s)" if shorts else "Long Form"
    console.print(Rule(f"[bold]Step 6 — Script Generation ({mode_label}, Ollama)[/bold]"))
    try:
        if shorts:
            from src.ideas import generate_shorts_script
            with _ollama_cb:
                script = _run_with_retry(
                    generate_shorts_script,
                    best_idea, model=ollama_model, ml_prefix=ml_prefix,
                )
        elif media_captions:
            from src.ideas import generate_script_from_media
            with _ollama_cb:
                script = _run_with_retry(
                    generate_script_from_media,
                    best_idea,
                    media_captions=media_captions,
                    duration_minutes=duration_minutes,
                    model=ollama_model,
                    ml_prefix=ml_prefix,
                )
        else:
            from src.ideas import generate_video_script
            with _ollama_cb:
                script = _run_with_retry(
                    generate_video_script,
                    best_idea, duration_minutes=duration_minutes, model=ollama_model,
                )
    except Exception as exc:
        result.errors.append(f"Script generation failed: {exc}")
        console.print(f"[red]{exc}[/red]")
        return result

    # ------------------------------------------------------------------ #
    # Step 6b — Psychology hook optimisation (experiment-driven style)
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 6b — Psychology Hook Optimisation[/bold]"))
    hook_exp_variant = None
    try:
        from src.experiments.engine import get_engine as _get_exp_engine
        hook_exp_variant = _get_exp_engine().assign_variant("hook_style")
        if hook_exp_variant:
            console.print(f"[dim]Experiment: hook_style variant → {hook_exp_variant.name}[/dim]")
    except Exception:
        pass
    try:
        from src.psychology.hooks import generate_hooks, print_hook_report
        hook_scores = generate_hooks(topic, niche, model=ollama_model, count=8)
        print_hook_report(hook_scores)
        if hook_scores and hook_scores[0].total > 5.0:
            old_hook = script.hook
            script.hook = hook_scores[0].hook
            console.print(f"[green]Hook upgraded:[/green] {old_hook[:50]}… → {script.hook[:50]}…")
        else:
            console.print("[dim]Hook score below threshold — keeping AI-generated hook.[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Hook optimisation skipped ({exc})[/yellow]")

    script_file = out / "script.json"
    script_file.write_text(
        json.dumps({
            "title": script.title, "hook": script.hook,
            "sections": script.sections, "call_to_action": script.call_to_action,
            "description": script.description, "tags": script.tags,
        }, indent=2)
    )
    result.script_path = script_file
    cp.advance(PipelineStage.SCRIPT_GENERATED, script_hook=script.hook[:80])
    console.print(f"[green]Script ready:[/green] {len(script.sections)} sections")

    # ------------------------------------------------------------------ #
    # Step 7 — Semantic matching with global deduplication
    # ------------------------------------------------------------------ #
    console.print(Rule("[bold]Step 7 — Semantic Matching (global dedup)[/bold]"))
    matched_clips: list[TimelineClip] = []
    try:
        from src.matching.semantic import scenes_from_script, match_all_scenes
        scenes = scenes_from_script(script.sections)
        match_results = match_all_scenes(scenes, global_dedup_videos=5, niche=niche)
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
        cp.advance(PipelineStage.SCENES_MATCHED, clip_count=len(matched_clips))
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
    _section_durations: list[float] = []
    if dry_run:
        console.print("[dim]dry-run: TTS skipped[/dim]")
    else:
        try:
            from src.tts.kokoro import synthesize_sections, merge_audio_files, pick_voice
            from src.humanize import random_tts_speed
            audio_dir = out / "audio"
            normalized = [
                {"script": s.get("script") or s.get("heading") or s.get("text") or ""}
                for s in script.sections
                if s.get("script") or s.get("heading") or s.get("text")
            ]
            all_sections = [{"script": script.hook}] + normalized + [{"script": script.call_to_action}]
            # Use niche-aware voice unless caller explicitly overrode the default
            effective_voice = tts_voice if tts_voice != "af_heart" else pick_voice(niche)
            effective_speed = random_tts_speed()
            console.print(f"[dim]TTS: voice={effective_voice}, speed={effective_speed}[/dim]")
            wav_paths = synthesize_sections(
                all_sections, audio_dir, voice=effective_voice, speed=effective_speed
            )
            if wav_paths:
                audio_path = out / "voiceover.wav"
                merge_audio_files(wav_paths, audio_path)
                cp.advance(PipelineStage.TTS_COMPLETED, audio_path=str(audio_path))
                from src.tts.kokoro import get_section_durations as _get_wav_durations
                _section_durations = _get_wav_durations(wav_paths)
            else:
                _section_durations = []
        except Exception as exc:
            console.print(f"[yellow]TTS skipped ({exc})[/yellow]")
            _section_durations = []

    breathe(2)
    # ------------------------------------------------------------------ #
    # Step 10 — Render video (FFmpeg, free)
    # ------------------------------------------------------------------ #
    render_label = "Shorts Render (9:16, 60s)" if shorts else "Video Render (FFmpeg)"
    console.print(Rule(f"[bold]Step 10 — {render_label}[/bold]"))
    raw_video_path = out / "raw_video.mp4"
    video_path = out / ("shorts.mp4" if shorts else "final_video.mp4")
    if dry_run:
        console.print("[dim]dry-run: FFmpeg render skipped[/dim]")
        log.info("dry-run render skipped", clips=len(matched_clips))
    elif matched_clips:
        try:
            if shorts:
                from src.pipeline.shorts_renderer import render_shorts
                render_shorts(
                    matched_clips, audio_path, raw_video_path,
                    script_sections=script.sections, hook=script.hook,
                    tmp_dir=out / "_tmp_shorts",
                )
            else:
                from src.pipeline.renderer import render_video
                render_video(matched_clips, audio_path, raw_video_path, music_path=music_path)
        except Exception as exc:
            result.errors.append(f"Render failed: {exc}")
            console.print(f"[red]Render error: {exc}[/red]")
            log.error("render failed", exc=str(exc))
    else:
        console.print("[yellow]No clips matched — render skipped. Add media to media_library/ first.[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 10b — Subtitles + title card overlay (long form only)
    # Shorts already has captions burned in by shorts_renderer
    # ------------------------------------------------------------------ #
    if raw_video_path.exists() and not shorts:
        console.print(Rule("[bold]Step 10b — Subtitles & Text Overlays[/bold]"))
        try:
            from src.pipeline.subtitles import apply_overlays
            apply_overlays(
                video_path=raw_video_path,
                output_path=video_path,
                script_sections=script.sections,
                hook=script.hook,
                call_to_action=script.call_to_action,
                title=script.title,
                tmp_dir=out / "_tmp_overlays",
                section_durations=_section_durations or None,
            )
            result.video_path = video_path
        except Exception as exc:
            console.print(f"[yellow]Overlays skipped ({exc}) — using raw render.[/yellow]")
            import shutil
            shutil.copy2(raw_video_path, video_path)
            result.video_path = video_path
    elif raw_video_path.exists() and shorts:
        result.video_path = raw_video_path

    # ------------------------------------------------------------------ #
    # Step 10c — Thumbnail generation
    # ------------------------------------------------------------------ #
    if result.video_path and result.video_path.exists():
        console.print(Rule("[bold]Step 10c — Thumbnail Generation[/bold]"))
        try:
            from src.thumbnail.generator import generate_thumbnail
            thumb_path = out / "thumbnail.jpg"
            generate_thumbnail(
                video_path=result.video_path,
                title=script.title,
                hook=script.hook,
                output_path=thumb_path,
            )
            result.thumbnail_path = thumb_path
        except Exception as exc:
            console.print(f"[yellow]Thumbnail skipped ({exc})[/yellow]")

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

        # Record hook_style experiment outcome (entropy as proxy until real CTR arrives)
        if hook_exp_variant is not None:
            try:
                from src.experiments.engine import get_engine as _get_exp_engine
                _get_exp_engine().record_outcome(
                    "hook_style", hook_exp_variant, result.entropy_score
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Step 11b — Multi-angle Shorts (4 angles from one long-form script)
    # ------------------------------------------------------------------ #
    if multi_shorts and matched_clips and not shorts:
        console.print(Rule("[bold]Step 11b — Multi-angle Shorts Generation[/bold]"))
        try:
            from src.pipeline.multi_shorts import generate_angles, render_all_angles
            angles = generate_angles(
                topic=topic,
                niche=niche,
                script_sections=script.sections,
                original_hook=script.hook,
                original_cta=script.call_to_action,
                model=ollama_model,
            )
            console.print(f"[green]{len(angles)} Shorts angles generated[/green]")
            angle_results = render_all_angles(
                angles=angles,
                clips=matched_clips,
                audio_dir=out / "audio",
                output_dir=out / "shorts",
                tts_voice=tts_voice,
                ollama_model=ollama_model,
                upload=upload,
                privacy=privacy,
            )
            console.print(f"[bold green]{sum(1 for r in angle_results if r['success'])} / {len(angle_results)} Shorts rendered[/bold green]")
        except Exception as exc:
            console.print(f"[yellow]Multi-Shorts skipped ({exc})[/yellow]")

    # ------------------------------------------------------------------ #
    # Step 12 — Upload to YouTube (free API, optional)
    # ------------------------------------------------------------------ #
    if dry_run:
        console.print("[dim]dry-run: upload skipped[/dim]")
    elif upload and result.video_path and result.video_path.exists():
        console.print(Rule("[bold]Step 12 — YouTube Upload[/bold]"))
        try:
            from src.uploader import upload_video
            thumb_arg = str(result.thumbnail_path) if getattr(result, "thumbnail_path", None) and result.thumbnail_path.exists() else None
            vid_id = upload_video(
                video_path=str(result.video_path),
                title=script.title,
                description=script.description,
                tags=script.tags,
                privacy_status=privacy,
                thumbnail_path=thumb_arg,
            )
            result.youtube_video_id = vid_id
            log.info("video uploaded", video_id=vid_id, title=script.title)
        except Exception as exc:
            result.errors.append(f"Upload failed: {exc}")
            console.print(f"[red]Upload error: {exc}[/red]")
            log.error("upload failed", exc=str(exc))

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    result.success = dry_run or (result.video_path is not None and result.video_path.exists())
    if result.success and not dry_run:
        cp.advance(PipelineStage.COMPLETED)
        cp.clear()
    log.info("pipeline finished", success=result.success, errors=result.errors,
             entropy=result.entropy_score, clips=len(result.asset_ids_used))
    console.print(Rule("[bold green]Pipeline Complete[/bold green]"))
    console.print(f"Output directory: [cyan]{out.resolve()}[/cyan]")
    if result.youtube_video_id:
        console.print(f"YouTube: [cyan]https://www.youtube.com/watch?v={result.youtube_video_id}[/cyan]")
    if result.errors:
        for err in result.errors:
            console.print(f"[red]• {err}[/red]")
    return result
