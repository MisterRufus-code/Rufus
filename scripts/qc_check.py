#!/usr/bin/env python3
"""
qc_check.py — Automated output quality control for rendered Shorts.

Runs after every render and answers one question: is this file actually a
publishable YouTube Short? Catches the failure modes that silently produce a
broken upload — missing audio track, wrong resolution, truncated render,
whisper-quiet or clipping mix — before the uploader ever sees the file.

Verdict model:
  CRITICAL  → the file is broken; main.py holds the upload (saved for review)
  WARNING   → publishable but off-target; printed so trends are visible
  ok == True when there are no criticals.

Usage:
    python scripts/qc_check.py media_library/output/short_20260629.mp4
"""

import json
import subprocess
import sys
from pathlib import Path

import video_format as _vf

# What a publishable render looks like, from the active format profile.
#
# THE SHAPE THIS RENDER WAS ASKED FOR. Hard-coded as 1080x1920 until a second
# format existed — and this is the last gate before upload, so a long-form
# render would have been failed here as "wrong resolution" for being exactly
# the resolution it was told to be. The check is still worth having: it catches
# an encode that came out at the wrong size, which is a real failure and looks
# identical in every other respect.
REQ_W, REQ_H     = _vf.dimensions()
# Outside this = broken render. Per-FORMAT: the Shorts cap is 3 minutes, and
# a nine-minute explainer is not a broken render — it is the other format
# doing exactly what it was asked to. One ceiling cannot mean both.
MIN_DUR, MAX_DUR = _vf.get("qc_min_s", 10.0), _vf.get("qc_max_s", 180.0)
# The retention sweet spot, and a WARNING rather than a failure. It is a
# Shorts number: 25-60s is where a vertical video holds, and a nine-minute
# explainer is outside it by design, so long-form takes its own band from the
# profile rather than being told every time that it is too long to retain.
IDEAL_MIN, IDEAL_MAX = ((25.0, 60.0) if not _vf.is_long()
                        else (_vf.get("qc_min_s", 240.0),
                              _vf.get("qc_max_s", 1500.0)))
MIN_BYTES        = 1_000_000       # <1MB at either format means the encode died
FPS_MIN, FPS_MAX = 24.0, 61.0
# volumedetect mean_volume sanity band (dB). Outside = mix likely broken.
MEAN_VOL_MIN, MEAN_VOL_MAX = -33.0, -6.0


def _probe(video_path: Path) -> dict | None:
    """ffprobe → parsed JSON, or None if ffprobe unavailable/failed."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except Exception:
        return None


def _mean_volume_db(video_path: Path) -> float | None:
    """Single-pass ffmpeg volumedetect → mean_volume in dB, or None."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(video_path),
             "-af", "volumedetect", "-vn", "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        for line in r.stderr.splitlines():
            if "mean_volume:" in line:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
    except Exception:
        pass
    return None


def _extract_info(probe: dict) -> dict:
    """Pull the QC-relevant facts out of an ffprobe JSON blob. Pure function."""
    info: dict = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0,
                  "has_video": False, "has_audio": False, "vcodec": "", "acodec": ""}
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"]:
            info["has_video"] = True
            info["width"]  = int(s.get("width", 0) or 0)
            info["height"] = int(s.get("height", 0) or 0)
            info["vcodec"] = s.get("codec_name", "")
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
        elif s.get("codec_type") == "audio" and not info["has_audio"]:
            info["has_audio"] = True
            info["acodec"] = s.get("codec_name", "")
    try:
        info["duration"] = float(probe.get("format", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        info["duration"] = 0.0
    return info


# The longest this format may sit on one unchanging picture before it reads as
# a slideshow. Chosen from the retention failure it describes rather than from
# a round number: a viewer who has seen everything in the frame and is given no
# new information is a viewer deciding whether to swipe. Per-format because it
# is 2.5x the format's own average shot — five seconds is a stall in a Short
# and an ordinary beat in an explainer.
MAX_STATIC_RUN = _vf.get("max_hold_s", 5.0)


def _pacing_warnings(cuts: list[float] | None, duration: float) -> list[str]:
    """Stretches longer than MAX_STATIC_RUN with no cut. Warning, never critical.

    A gate here would be the wrong tool. The picture may be genuinely moving
    inside a long beat (Wan/Hunyuan motion, a strong Ken Burns push), which
    this cannot see, so a hard failure would reject good videos for a
    heuristic's benefit — the rejection-ladder mistake CLAUDE.md warns about.
    What it can do is make a slideshow impossible to ship unnoticed.
    """
    if not cuts or duration <= 0:
        return []

    marks = [0.0] + sorted(c for c in cuts if 0 < c < duration) + [duration]
    gaps = []
    for start, end in zip(marks, marks[1:]):
        run = end - start
        if run > MAX_STATIC_RUN:
            gaps.append(f"{start:.1f}-{end:.1f}s ({run:.1f}s)")

    if not gaps:
        return []
    return [f"{len(gaps)} stretch(es) over {MAX_STATIC_RUN:.0f}s without a cut: "
            + ", ".join(gaps)]


def _evaluate(info: dict, size_bytes: int,
              mean_vol: float | None = None,
              cuts: list[float] | None = None) -> tuple[list[str], list[str]]:
    """Turn extracted facts into (critical, warnings). Pure function."""
    critical: list[str] = []
    warnings: list[str] = []

    if size_bytes < MIN_BYTES:
        critical.append(f"file too small ({size_bytes} bytes) — encode likely failed")
    if not info["has_video"]:
        critical.append("no video stream")
    elif (info["width"], info["height"]) != (REQ_W, REQ_H):
        critical.append(f"resolution {info['width']}x{info['height']} — must be {REQ_W}x{REQ_H}")
    if not info["has_audio"]:
        critical.append("no audio stream — voice/music missing from mux")

    d = info["duration"]
    if d < MIN_DUR or d > MAX_DUR:
        critical.append(f"duration {d:.1f}s outside {MIN_DUR:.0f}-{MAX_DUR:.0f}s — broken render")
    elif not (IDEAL_MIN <= d <= IDEAL_MAX):
        warnings.append(f"duration {d:.1f}s outside the {IDEAL_MIN:.0f}-{IDEAL_MAX:.0f}s retention sweet spot")

    if info["has_video"] and not (FPS_MIN <= info["fps"] <= FPS_MAX):
        warnings.append(f"unusual frame rate {info['fps']:.1f} fps")

    if mean_vol is not None and not (MEAN_VOL_MIN <= mean_vol <= MEAN_VOL_MAX):
        which = "near-silent" if mean_vol < MEAN_VOL_MIN else "hot/clipping"
        warnings.append(f"mean volume {mean_vol:.1f} dB looks {which} (sane band "
                        f"{MEAN_VOL_MIN:.0f}..{MEAN_VOL_MAX:.0f})")

    warnings.extend(_pacing_warnings(cuts, d))

    return critical, warnings


def run_qc(video_path: Path, check_loudness: bool = True,
           cuts: list[float] | None = None) -> dict:
    """Full QC pass on a rendered Short.

    `cuts` are the render's own cut timestamps, when the caller has them — the
    pacing check cannot be recovered from the finished file, so a caller that
    omits them simply gets no pacing warning.

    Returns {"ok": bool, "critical": [...], "warnings": [...], "info": {...}}.
    If ffprobe is unavailable the result is ok=True with a warning — QC must
    never block a pipeline that renders fine but lacks probing tools.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"ok": False, "critical": [f"file not found: {video_path}"],
                "warnings": [], "info": {}}

    size = video_path.stat().st_size
    probe = _probe(video_path)
    if probe is None:
        return {"ok": True, "critical": [],
                "warnings": ["ffprobe unavailable — QC skipped"],
                "info": {"size_bytes": size}}

    info = _extract_info(probe)
    mean_vol = _mean_volume_db(video_path) if (check_loudness and info["has_audio"]) else None
    critical, warnings = _evaluate(info, size, mean_vol, cuts)

    info["size_bytes"] = size
    if mean_vol is not None:
        info["mean_volume_db"] = mean_vol
    return {"ok": not critical, "critical": critical, "warnings": warnings, "info": info}


def print_report(result: dict) -> None:
    """Human-readable one-glance QC summary for the run log."""
    i = result.get("info", {})
    if i.get("has_video"):
        print(f"           QC: {i['width']}x{i['height']}  {i['fps']:.0f}fps  "
              f"{i['duration']:.1f}s  {i['size_bytes'] // 1024}KB"
              + (f"  {i['mean_volume_db']:.1f}dB mean" if "mean_volume_db" in i else ""))
    for c in result["critical"]:
        print(f"           QC ✗ CRITICAL: {c}")
    for w in result["warnings"]:
        print(f"           QC ⚠ {w}")
    if result["ok"] and not result["warnings"]:
        print("           QC ✓ all checks passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qc_check.py <video.mp4>")
        sys.exit(1)
    res = run_qc(Path(sys.argv[1]))
    print_report(res)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)
