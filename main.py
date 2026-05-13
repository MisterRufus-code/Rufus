#!/usr/bin/env python3
"""
Rufus — AI Adaptive Media Automation System
100% free, fully local execution.

Requirements:
  pip install -r requirements.txt
  ollama pull mistral
  sudo apt install ffmpeg
  cp .env.example .env  # then add Pexels & Pixabay keys
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Rufus: 100%% free AI-powered YouTube automation engine."""


@cli.command("trends")
@click.option("--keywords", "-k", multiple=True, required=True)
@click.option("--geo", default="US", show_default=True)
@click.option("--timeframe", default="now 7-d", show_default=True)
@click.option("--yt-trending", is_flag=True, default=False)
@click.option("--region", default="US", show_default=True)
def trends_cmd(keywords, geo, timeframe, yt_trending, region):
    """Fetch Google Trends and YouTube trending data."""
    from src.trends import fetch_google_trends, print_trends_table
    from src.trends import fetch_youtube_trending, print_yt_trending_table

    console.print("[bold cyan]Fetching Google Trends...[/bold cyan]")
    topics = fetch_google_trends(list(keywords), geo=geo, timeframe=timeframe)
    print_trends_table(topics)

    if yt_trending:
        console.print("\n[bold cyan]Fetching YouTube Trending...[/bold cyan]")
        try:
            from src.uploader import get_youtube_client
            youtube = get_youtube_client()
            videos = fetch_youtube_trending(youtube, region_code=region)
            print_yt_trending_table(videos)
        except Exception as exc:
            console.print(f"[red]YouTube trending error: {exc}[/red]")


# ---------------------------------------------------------------------------
# ideas  (local Ollama — free)
# ---------------------------------------------------------------------------

@cli.command("ideas")
@click.option("--topic", "-t", required=True)
@click.option("--niche", "-n", default="general", show_default=True)
@click.option("--count", "-c", default=5, show_default=True)
@click.option("--model", "-m", default="mistral", show_default=True,
              help="Ollama model name")
@click.option("--trending-keywords", "-k", multiple=True)
@click.option("--save", "-s", type=click.Path(), default=None)
def ideas_cmd(topic, niche, count, model, trending_keywords, save):
    """Generate viral video ideas using local Ollama LLM (free)."""
    from src.ideas import generate_video_ideas, print_ideas, check_ollama_status

    check_ollama_status()
    console.print(f"[bold cyan]Generating {count} ideas for:[/bold cyan] {topic}")
    ideas = generate_video_ideas(
        topic, niche, count=count, model=model,
        trending_context=", ".join(trending_keywords),
    )
    print_ideas(ideas)

    if save:
        Path(save).write_text(json.dumps(
            [{"title": i.title, "hook": i.hook, "description": i.description,
              "tags": i.tags, "estimated_virality": i.estimated_virality}
             for i in ideas], indent=2
        ))
        console.print(f"[green]Saved to {save}[/green]")


# ---------------------------------------------------------------------------
# script  (local Ollama — free)
# ---------------------------------------------------------------------------

@cli.command("script")
@click.option("--title", "-t", required=True)
@click.option("--hook", "-h", default="")
@click.option("--duration", "-d", default=10, show_default=True)
@click.option("--style", default="engaging and conversational", show_default=True)
@click.option("--model", "-m", default="mistral", show_default=True)
@click.option("--save", "-s", type=click.Path(), default=None)
def script_cmd(title, hook, duration, style, model, save):
    """Generate a full video script using local Ollama (free)."""
    from src.ideas import VideoIdea, generate_video_script, print_script

    idea = VideoIdea(title=title, hook=hook or f"Did you know that {title.lower()}?",
                     description="", tags=[], estimated_virality="High")
    console.print(f"[bold cyan]Generating script for:[/bold cyan] {title}")
    script = generate_video_script(idea, duration_minutes=duration, style=style, model=model)
    print_script(script)

    if save:
        Path(save).write_text(json.dumps({
            "title": script.title, "hook": script.hook, "sections": script.sections,
            "call_to_action": script.call_to_action, "description": script.description,
            "tags": script.tags,
        }, indent=2))
        console.print(f"[green]Script saved to {save}[/green]")


# ---------------------------------------------------------------------------
# fetch  (Pexels + Pixabay — free)
# ---------------------------------------------------------------------------

@cli.command("fetch")
@click.option("--queries", "-q", multiple=True, required=True,
              help="Search queries for footage")
@click.option("--source", "-s", type=click.Choice(["pexels", "ytcc", "both"]),
              default="both", show_default=True)
@click.option("--count", "-c", default=3, show_default=True,
              help="Videos per query")
@click.option("--images", is_flag=True, default=False,
              help="Download images instead of videos")
def fetch_cmd(queries, source, count, images):
    """Download free stock footage from Pexels and/or YouTube CC."""
    queries = list(queries)
    total_paths: list = []

    if source in ("pexels", "both"):
        from src.media_fetch.pexels import download_videos, download_images
        fn = download_images if images else download_videos
        pexels_paths = fn(queries, images_per_query=count) if images else fn(queries, videos_per_query=count)
        total_paths.extend(pexels_paths)
        console.print(f"[green]Pexels: {len(pexels_paths)} files downloaded[/green]")

    if source in ("ytcc", "both"):
        from src.media_fetch.ytcc import download_videos as ytcc_dl
        if not images:
            ytcc_paths = ytcc_dl(queries, videos_per_query=count)
            total_paths.extend(ytcc_paths)
            console.print(f"[green]YouTube CC: {len(ytcc_paths)} files downloaded[/green]")

    console.print(f"[bold green]Total: {len(total_paths)} files[/bold green]")


# ---------------------------------------------------------------------------
# ingest  (CLIP + BLIP + YOLO — free, local)
# ---------------------------------------------------------------------------

@cli.command("ingest")
@click.option("--library", "-l", type=click.Path(), default=None,
              help="Path to media library (default from .env)")
@click.option("--force", is_flag=True, default=False,
              help="Re-index even already-indexed assets")
def ingest_cmd(library, force):
    """Scan media library and index assets into the vector database."""
    from src.ingestion.indexer import index_library
    console.print("[bold cyan]Indexing media library...[/bold cyan]")
    result = index_library(library_path=Path(library) if library else None, force_reindex=force)
    console.print(
        f"[green]Done:[/green] {result['indexed']} indexed, "
        f"{result['skipped']} skipped, {result['errors']} errors"
    )


# ---------------------------------------------------------------------------
# voiceover  (Kokoro TTS — free, local)
# ---------------------------------------------------------------------------

@cli.command("voiceover")
@click.option("--text", "-t", default=None, help="Text to synthesize")
@click.option("--script-file", "-f", type=click.Path(exists=True), default=None,
              help="JSON script file from the script command")
@click.option("--voice", "-v", default="af_heart", show_default=True)
@click.option("--output", "-o", default="voiceover.wav", show_default=True)
def voiceover_cmd(text, script_file, voice, output):
    """Generate free local voiceover using Kokoro TTS."""
    from src.tts.kokoro import synthesize, synthesize_sections, merge_audio_files, list_voices

    console.print(f"[cyan]Available voices:[/cyan] {', '.join(list_voices())}")

    out_path = Path(output)
    if script_file:
        data = json.loads(Path(script_file).read_text())
        sections = [{"script": data.get("hook", "")}] + data.get("sections", []) + \
                   [{"script": data.get("call_to_action", "")}]
        wavs = synthesize_sections(sections, out_path.parent / "audio_sections", voice=voice)
        merge_audio_files(wavs, out_path)
    elif text:
        synthesize(text, out_path, voice=voice)
    else:
        console.print("[red]Provide --text or --script-file[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# entropy  (visual entropy scoring — free)
# ---------------------------------------------------------------------------

@cli.command("entropy")
@click.option("--library", "-l", type=click.Path(), default=None)
def entropy_cmd(library):
    """Score the visual entropy of the current media library."""
    from src.ingestion.scanner import scan_library
    from src.ingestion.extractor import extract_features
    from src.database.models import TimelineClip, MediaAsset
    from src.entropy.engine import score_timeline, print_entropy_report

    assets = []
    for raw in list(scan_library(Path(library) if library else None))[:20]:
        try:
            f = extract_features(raw.path, raw.asset_type)
            assets.append(MediaAsset(
                asset_id=f.asset_id, path=f.path, asset_type=f.asset_type,
                caption=f.caption, detected_objects=f.detected_objects,
                dominant_colors=f.dominant_colors, motion_score=f.motion_score,
                duration_seconds=f.duration_seconds, clip_embedding=f.clip_embedding,
            ))
        except Exception:
            continue

    if not assets:
        console.print("[yellow]No assets found to score.[/yellow]")
        return

    clips = [TimelineClip(asset=a, start_time=i*5, end_time=i*5+5,
                          scene_index=i, out_point=min(a.duration_seconds, 5.0))
             for i, a in enumerate(assets)]
    report = score_timeline(clips)
    print_entropy_report(report)


# ---------------------------------------------------------------------------
# upload  (YouTube Data API — free)
# ---------------------------------------------------------------------------

@cli.command("upload")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--title", "-t", required=True)
@click.option("--description", "-d", default="")
@click.option("--tags", multiple=True)
@click.option("--privacy", default="private", show_default=True,
              type=click.Choice(["public", "unlisted", "private"]))
@click.option("--thumbnail", type=click.Path(), default=None)
@click.option("--from-script", type=click.Path(exists=True), default=None)
def upload_cmd(video_path, title, description, tags, privacy, thumbnail, from_script):
    """Upload video to YouTube (free API)."""
    from src.uploader import upload_video

    if from_script:
        data = json.loads(Path(from_script).read_text())
        title = title or data.get("title", title)
        description = description or data.get("description", "")
        tags = tags or data.get("tags", [])

    vid_id = upload_video(str(video_path), title=title, description=description,
                          tags=list(tags), privacy_status=privacy, thumbnail_path=thumbnail)
    if vid_id:
        console.print(f"[bold green]Uploaded:[/bold green] https://www.youtube.com/watch?v={vid_id}")
    else:
        console.print("[red]Upload failed.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# analytics  (YouTube API — free)
# ---------------------------------------------------------------------------

@cli.command("analytics")
@click.option("--videos", "-v", default=10, show_default=True)
def analytics_cmd(videos):
    """Show channel and video analytics."""
    from src.analytics import (get_channel_stats, get_recent_videos_stats,
                                print_channel_stats, print_video_stats)
    ch = get_channel_stats()
    print_channel_stats(ch)
    vids = get_recent_videos_stats(max_results=videos)
    print_video_stats(vids)


# ---------------------------------------------------------------------------
# sync-analytics  (auto-pull YouTube stats into ML feedback DB)
# ---------------------------------------------------------------------------

@cli.command("sync-analytics")
@click.option("--videos", "-v", default=20, show_default=True,
              help="Max number of recent videos to sync")
@click.option("--niche", "-n", default="general", show_default=True,
              help="Niche label to tag synced records with")
def sync_analytics_cmd(videos, niche):
    """Pull YouTube analytics and auto-save ML feedback for all uploaded videos."""
    from src.analytics import sync_feedback_from_youtube
    sync_feedback_from_youtube(max_videos=videos, niche=niche)


# ---------------------------------------------------------------------------
# feedback  (ML feedback — free, local)
# ---------------------------------------------------------------------------

@cli.command("feedback")
@click.option("--video-id", required=True, help="YouTube video ID")
@click.option("--title", required=True)
@click.option("--niche", required=True)
@click.option("--topic", required=True)
@click.option("--ctr", default=0.0, type=float, help="Click-through rate (0.0–1.0)")
@click.option("--watch-time", default=0.0, type=float, help="Avg watch time in seconds")
@click.option("--retention", default=0.0, type=float, help="Audience retention rate (0.0–1.0)")
@click.option("--views", default=0, type=int)
@click.option("--model", default="mistral", show_default=True)
def feedback_cmd(video_id, title, niche, topic, ctr, watch_time, retention, views, model):
    """Record video performance for ML optimization."""
    from src.ml.feedback import VideoFeedback, record_feedback
    fb = VideoFeedback(
        video_id=video_id, title=title, niche=niche, topic=topic,
        prompt_model=model, ctr=ctr, avg_watch_time_seconds=watch_time,
        retention_rate=retention, views=views,
    )
    record_feedback(fb)


# ---------------------------------------------------------------------------
# optimize  (ML insights — free, local)
# ---------------------------------------------------------------------------

@cli.command("optimize")
def optimize_cmd():
    """Show ML optimization insights from feedback data."""
    from src.ml.optimizer import analyse_feedback, print_insights
    insights = analyse_feedback()
    print_insights(insights)


# ---------------------------------------------------------------------------
# pipeline  (full end-to-end, 100% free)
# ---------------------------------------------------------------------------

@cli.command("pipeline")
@click.option("--topic", "-t", default=None,
              help="Topic to make video about (auto-selected from trending if not provided)")
@click.option("--niche", "-n", default="general", show_default=True)
@click.option("--model", "-m", default="mistral", show_default=True)
@click.option("--voice", default="af_heart", show_default=True)
@click.option("--ideas-count", default=3, show_default=True)
@click.option("--geo", default="US", show_default=True)
@click.option("--no-download", is_flag=True, default=False,
              help="Skip footage download (use existing media_library)")
@click.option("--upload", is_flag=True, default=False)
@click.option("--privacy", default="private", show_default=True,
              type=click.Choice(["public", "unlisted", "private"]))
@click.option("--music", type=click.Path(), default=None,
              help="Optional background music file")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--shorts", is_flag=True, default=False,
              help="Produce a 9:16 YouTube Shorts video (≤60s) instead of long form")
@click.option("--both", "both_formats", is_flag=True, default=False,
              help="Produce both long-form AND Shorts in one run (recommended)")
@click.option("--multi-shorts", "multi_shorts", is_flag=True, default=False,
              help="Generate 4 Shorts angles from the long-form script (4x upload volume)")
@click.option("--low-power", is_flag=True, default=False,
              help="Limit CPU threads and add pauses — quieter PC, slower pipeline")
def pipeline_cmd(topic, niche, model, voice, ideas_count, geo, no_download, upload, privacy, music, output, shorts, both_formats, multi_shorts, low_power):
    """
    Run the FULL free pipeline.
    --topic is optional: auto-selects trending topic if not provided.
    --both produces long-form + Shorts in one run (recommended daily use).
    --low-power reduces fan noise on slower machines.
    """
    from src.trends import get_trending_topic
    from src.pipeline.orchestrator import run_pipeline

    # Auto-select topic from trending if not provided
    if not topic:
        console.print(f"[cyan]No topic provided — finding trending topic for '{niche}'...[/cyan]")
        topic = get_trending_topic(niche=niche, geo=geo, model=model)

    if both_formats:
        # Run long form then Shorts
        success = True
        for label, shorts_flag in [("Long Form", False), ("Shorts", True)]:
            console.print(f"\n[bold magenta]{'='*20} {label} {'='*20}[/bold magenta]")
            result = run_pipeline(
                topic=topic, niche=niche, ollama_model=model, tts_voice=voice,
                ideas_count=ideas_count, geo=geo,
                download_footage=(not no_download) if label == "Long Form" else False,
                output_dir=Path(output) if output else None,
                upload=upload, privacy=privacy,
                music_path=Path(music) if music else None,
                shorts=shorts_flag, low_power=low_power,
            )
            if not result.success:
                success = False
        if not success:
            sys.exit(1)
    else:
        result = run_pipeline(
            topic=topic, niche=niche, ollama_model=model, tts_voice=voice,
            ideas_count=ideas_count, geo=geo, download_footage=not no_download,
            output_dir=Path(output) if output else None,
            upload=upload, privacy=privacy,
            music_path=Path(music) if music else None,
            shorts=shorts, low_power=low_power, multi_shorts=multi_shorts,
        )
        if not result.success:
            sys.exit(1)


# ---------------------------------------------------------------------------
# serve  (FastAPI server for n8n integration)
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve_cmd(host, port):
    """Start the Rufus API server for n8n automation."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed — run: pip install uvicorn fastapi[/red]")
        sys.exit(1)
    console.print(f"[bold green]Rufus API starting on http://{host}:{port}[/bold green]")
    console.print(f"[dim]n8n can reach it at: http://host.docker.internal:{port}[/dim]")
    import uvicorn
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# status  (check all dependencies)
# ---------------------------------------------------------------------------

@cli.command("status")
def status_cmd():
    """Check that all free dependencies are installed and running."""
    from src.ideas import check_ollama_status
    import shutil

    console.print("[bold]Rufus System Status[/bold]\n")

    check_ollama_status()

    # FFmpeg
    if shutil.which("ffmpeg"):
        console.print("[green]FFmpeg installed[/green]")
    else:
        console.print("[red]FFmpeg missing[/red] — run: sudo apt install ffmpeg")

    # Kokoro
    try:
        from kokoro_onnx import Kokoro
        console.print("[green]Kokoro TTS installed[/green]")
    except ImportError:
        console.print("[yellow]Kokoro TTS not installed[/yellow] — run: pip install kokoro-onnx soundfile")

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333).get_collections()
        console.print("[green]Qdrant running[/green]")
    except Exception:
        console.print("[yellow]Qdrant not running[/yellow] — run: docker compose up -d qdrant")

    # API keys
    import os
    from dotenv import load_dotenv
    load_dotenv()
    for key, label in [("PEXELS_API_KEY", "Pexels"), ("PIXABAY_API_KEY", "Pixabay")]:
        if os.getenv(key):
            console.print(f"[green]{label} API key set[/green]")
        else:
            console.print(f"[yellow]{label} API key missing[/yellow] — add to .env")


if __name__ == "__main__":
    cli()
