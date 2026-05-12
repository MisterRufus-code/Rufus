"""
Visual Entropy Engine.

Scores how "fresh" and "varied" a proposed video timeline is.
Prevents the system from producing repetitive, low-retention content.

Measures:
  - Scene similarity     (CLIP cosine distance between consecutive clips)
  - Color repetition     (dominant color palette overlap)
  - Motion repetition    (motion score variance)
  - Emotional monotony   (caption semantic similarity)

Final entropy score: 0.0 (fully repetitive) → 1.0 (maximally varied)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.panel import Panel

from src.database.models import MediaAsset, TimelineClip

console = Console()


@dataclass
class EntropyReport:
    overall_score: float                 # 0–1
    scene_variety: float                 # 0–1
    color_variety: float                 # 0–1
    motion_variety: float                # 0–1
    caption_variety: float               # 0–1
    flagged_clips: list[int] = field(default_factory=list)   # scene indices too similar
    passed: bool = True
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _scene_variety(clips: list[TimelineClip]) -> tuple[float, list[int]]:
    """
    Measure embedding diversity across consecutive clip pairs.
    High similarity (>0.92) between neighbours = repetition detected.
    """
    if len(clips) < 2:
        return 1.0, []

    similarities = []
    flagged = []
    for i in range(1, len(clips)):
        a_emb = clips[i - 1].asset.clip_embedding
        b_emb = clips[i].asset.clip_embedding
        if not a_emb or not b_emb:
            continue
        sim = _cosine_similarity(a_emb, b_emb)
        similarities.append(sim)
        if sim > 0.92:
            flagged.append(i)

    avg_sim = float(np.mean(similarities)) if similarities else 0.5
    return float(np.clip(1.0 - avg_sim, 0.0, 1.0)), flagged


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_variety(clips: list[TimelineClip]) -> float:
    """
    Measure how diverse the dominant color palettes are across clips.
    """
    if not clips:
        return 1.0

    all_colors: list[np.ndarray] = []
    for clip in clips:
        for hex_col in clip.asset.dominant_colors:
            try:
                rgb = _hex_to_rgb(hex_col)
                all_colors.append(np.array(rgb, dtype=np.float32))
            except (ValueError, IndexError):
                continue

    if len(all_colors) < 2:
        return 0.5

    # Standard deviation of each channel — higher = more variety
    arr = np.stack(all_colors)
    std_per_channel = arr.std(axis=0)
    # Normalize: max possible std for 0-255 range is ~127
    variety = float(np.clip(std_per_channel.mean() / 80.0, 0.0, 1.0))
    return variety


def _motion_variety(clips: list[TimelineClip]) -> float:
    """
    Variety in motion scores. Mix of static and dynamic clips is best.
    Low variance = everything is the same pace.
    """
    scores = [c.asset.motion_score for c in clips if c.asset.motion_score > 0]
    if len(scores) < 2:
        return 0.5
    std = float(np.std(scores))
    # std of 0.25 or more is considered fully varied
    return float(np.clip(std / 0.25, 0.0, 1.0))


def _caption_variety(clips: list[TimelineClip]) -> float:
    """
    Measure semantic diversity of BLIP captions using simple TF-IDF overlap.
    """
    captions = [c.asset.caption for c in clips if c.asset.caption]
    if len(captions) < 2:
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        vec = TfidfVectorizer(stop_words="english", max_features=200)
        matrix = vec.fit_transform(captions)
        sim_matrix = sk_cosine(matrix)
        # Average off-diagonal similarity
        n = sim_matrix.shape[0]
        off_diag = (sim_matrix.sum() - n) / max(n * n - n, 1)
        return float(np.clip(1.0 - off_diag, 0.0, 1.0))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_timeline(
    clips: list[TimelineClip],
    min_score: float | None = None,
) -> EntropyReport:
    """
    Score the visual entropy of a proposed timeline.
    Returns an EntropyReport with overall score and per-metric breakdown.
    """
    import config as cfg
    threshold = min_score if min_score is not None else cfg.ENTROPY_MIN_SCORE

    sv, flagged = _scene_variety(clips)
    cv = _color_variety(clips)
    mv = _motion_variety(clips)
    capv = _caption_variety(clips)

    # Weighted average — scene variety matters most
    overall = (sv * 0.40) + (cv * 0.20) + (mv * 0.20) + (capv * 0.20)

    notes = []
    if sv < 0.4:
        notes.append("Many consecutive clips look too similar — swap some assets.")
    if cv < 0.3:
        notes.append("Color palette is monotone — mix warm/cool scenes.")
    if mv < 0.3:
        notes.append("All clips have similar motion — add static/dynamic contrast.")
    if capv < 0.3:
        notes.append("Captions are repetitive — diversify scene subjects.")

    report = EntropyReport(
        overall_score=round(overall, 3),
        scene_variety=round(sv, 3),
        color_variety=round(cv, 3),
        motion_variety=round(mv, 3),
        caption_variety=round(capv, 3),
        flagged_clips=flagged,
        passed=overall >= threshold,
        notes=notes,
    )
    return report


def print_entropy_report(report: EntropyReport) -> None:
    color = "green" if report.passed else "red"
    status = "PASSED" if report.passed else "FAILED"
    body = (
        f"[bold]Overall Score:[/bold] [{color}]{report.overall_score:.2f}[/{color}]  [{color}]{status}[/{color}]\n\n"
        f"  Scene Variety  : {_bar(report.scene_variety)}\n"
        f"  Color Variety  : {_bar(report.color_variety)}\n"
        f"  Motion Variety : {_bar(report.motion_variety)}\n"
        f"  Caption Variety: {_bar(report.caption_variety)}\n"
    )
    if report.flagged_clips:
        body += f"\n[yellow]Repetitive scene pairs at indices: {report.flagged_clips}[/yellow]\n"
    if report.notes:
        body += "\n" + "\n".join(f"[yellow]• {n}[/yellow]" for n in report.notes)

    console.print(Panel(body, title="Visual Entropy Report", border_style=color))


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    color = "green" if score >= 0.6 else ("yellow" if score >= 0.35 else "red")
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {score:.2f}"
