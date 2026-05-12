#!/usr/bin/env python3
"""Rufus — YouTube Viral Automation CLI."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """Rufus: AI-powered YouTube viral content automation."""


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------

@cli.command("trends")
@click.option("--keywords", "-k", multiple=True, required=True,
              help="Keywords to check (pass multiple: -k ai -k chatgpt)")
@click.option("--geo", default="US", show_default=True, help="Country code")
@click.option("--timeframe", default="now 7-d", show_default=True,
              help="Pytrends timeframe string")
@click.option("--yt-trending", is_flag=True, default=False,
              help="Also show YouTube trending videos")
@click.option("--region", default="US", show_default=True,
              help="Region code for YouTube trending")
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
# ideas
# ---------------------------------------------------------------------------

@cli.command("ideas")
@click.option("--topic", "-t", required=True, help="Main topic or niche")
@click.option("--niche", "-n", default="general", show_default=True,
              help="Channel niche (e.g. 'tech', 'finance', 'fitness')")
@click.option("--count", "-c", default=5, show_default=True,
              help="Number of ideas to generate")
@click.option("--trending-keywords", "-k", multiple=True,
              help="Trending keywords to incorporate")
@click.option("--save", "-s", type=click.Path(), default=None,
              help="Save ideas to a JSON file")
def ideas_cmd(topic, niche, count, trending_keywords, save):
    """Generate AI-powered viral video ideas."""
    from src.ideas import generate_video_ideas, print_ideas

    trending_ctx = ", ".join(trending_keywords) if trending_keywords else ""

    console.print(f"[bold cyan]Generating {count} video ideas for:[/bold cyan] {topic}")
    ideas = generate_video_ideas(topic, niche, count=count, trending_context=trending_ctx)
    print_ideas(ideas)

    if save:
        data = [
            {
                "title": i.title,
                "hook": i.hook,
                "description": i.description,
                "tags": i.tags,
                "estimated_virality": i.estimated_virality,
            }
            for i in ideas
        ]
        Path(save).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Saved to {save}[/green]")


# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------

@cli.command("script")
@click.option("--title", "-t", required=True, help="Video title")
@click.option("--hook", "-h", default="", help="Opening hook line")
@click.option("--duration", "-d", default=10, show_default=True,
              help="Target duration in minutes")
@click.option("--style", default="engaging and conversational", show_default=True,
              help="Video style/tone")
@click.option("--save", "-s", type=click.Path(), default=None,
              help="Save script to a JSON file")
def script_cmd(title, hook, duration, style, save):
    """Generate a full AI video script."""
    from src.ideas import VideoIdea, generate_video_script, print_script

    idea = VideoIdea(
        title=title,
        hook=hook or f"Did you know that {title.lower()}?",
        description="",
        tags=[],
        estimated_virality="High",
    )

    console.print(f"[bold cyan]Generating script for:[/bold cyan] {title}")
    script = generate_video_script(idea, duration_minutes=duration, style=style)
    print_script(script)

    if save:
        data = {
            "title": script.title,
            "hook": script.hook,
            "sections": script.sections,
            "call_to_action": script.call_to_action,
            "description": script.description,
            "tags": script.tags,
        }
        Path(save).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Script saved to {save}[/green]")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------

@cli.command("upload")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--title", "-t", required=True, help="Video title")
@click.option("--description", "-d", default="", help="Video description")
@click.option("--tags", multiple=True, help="Tags (pass multiple: --tags ai --tags tech)")
@click.option("--privacy", default="private", show_default=True,
              type=click.Choice(["public", "unlisted", "private"]))
@click.option("--category", default="22", show_default=True,
              help="YouTube category ID (22=People & Blogs)")
@click.option("--thumbnail", type=click.Path(), default=None,
              help="Path to thumbnail image (JPG)")
@click.option("--from-script", type=click.Path(exists=True), default=None,
              help="Load title/description/tags from a saved script JSON")
def upload_cmd(video_path, title, description, tags, privacy, category, thumbnail, from_script):
    """Upload a video to YouTube."""
    from src.uploader import upload_video

    if from_script:
        data = json.loads(Path(from_script).read_text())
        title = title or data.get("title", title)
        description = description or data.get("description", "")
        tags = tags or data.get("tags", [])

    console.print(f"[bold cyan]Uploading:[/bold cyan] {video_path}")
    video_id = upload_video(
        video_path=video_path,
        title=title,
        description=description,
        tags=list(tags),
        category_id=category,
        privacy_status=privacy,
        thumbnail_path=thumbnail,
    )

    if video_id:
        console.print(f"[bold green]Video ID:[/bold green] {video_id}")
    else:
        console.print("[red]Upload failed.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

@cli.command("analytics")
@click.option("--videos", "-v", default=10, show_default=True,
              help="Number of recent videos to show")
def analytics_cmd(videos):
    """Show channel and video performance analytics."""
    from src.analytics import (
        get_channel_stats,
        get_recent_videos_stats,
        print_channel_stats,
        print_video_stats,
    )

    console.print("[bold cyan]Fetching channel analytics...[/bold cyan]")
    ch = get_channel_stats()
    print_channel_stats(ch)

    console.print("\n[bold cyan]Fetching recent video stats...[/bold cyan]")
    vids = get_recent_videos_stats(max_results=videos)
    print_video_stats(vids)


# ---------------------------------------------------------------------------
# pipeline — run trends → ideas → script in one shot
# ---------------------------------------------------------------------------

@cli.command("pipeline")
@click.option("--topic", "-t", required=True, help="Topic to research and create content for")
@click.option("--niche", "-n", default="general", show_default=True)
@click.option("--ideas-count", default=3, show_default=True)
@click.option("--geo", default="US", show_default=True)
@click.option("--save-dir", type=click.Path(), default="output", show_default=True,
              help="Directory to save output files")
def pipeline_cmd(topic, niche, ideas_count, geo, save_dir):
    """Full pipeline: trends → ideas → script (saves everything to disk)."""
    from src.trends import fetch_google_trends
    from src.ideas import generate_video_ideas, generate_video_script

    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Step 1: Trends
    console.rule("[bold]Step 1: Fetching Trends[/bold]")
    topics = fetch_google_trends([topic], geo=geo)
    trending_ctx = ", ".join(
        [t.keyword] + t.related_queries for t in topics
    )

    # Step 2: Ideas
    console.rule("[bold]Step 2: Generating Ideas[/bold]")
    ideas = generate_video_ideas(topic, niche, count=ideas_count, trending_context=trending_ctx)

    ideas_file = output / "ideas.json"
    ideas_file.write_text(
        json.dumps(
            [{"title": i.title, "hook": i.hook, "description": i.description,
              "tags": i.tags, "estimated_virality": i.estimated_virality}
             for i in ideas],
            indent=2,
        )
    )
    console.print(f"[green]Ideas saved → {ideas_file}[/green]")

    # Step 3: Script for top idea
    console.rule("[bold]Step 3: Generating Script for Top Idea[/bold]")
    best_idea = max(ideas, key=lambda i: {"Low": 1, "Medium": 2, "High": 3, "Viral": 4}.get(i.estimated_virality, 0))
    console.print(f"[cyan]Top idea:[/cyan] {best_idea.title} [{best_idea.estimated_virality}]")

    script = generate_video_script(best_idea)
    script_file = output / "script.json"
    script_file.write_text(
        json.dumps(
            {"title": script.title, "hook": script.hook, "sections": script.sections,
             "call_to_action": script.call_to_action, "description": script.description,
             "tags": script.tags},
            indent=2,
        )
    )
    console.print(f"[green]Script saved → {script_file}[/green]")
    console.rule("[bold green]Pipeline complete![/bold green]")
    console.print(f"Output directory: [cyan]{output.resolve()}[/cyan]")


if __name__ == "__main__":
    cli()
