"""
Pre-render quality gate — validates script and footage before expensive TTS/render work.

Runs after script generation (Step 6) to catch low-quality outputs early:
  - Word count per section and total
  - Filler phrases that survived stripping
  - Insufficient footage vs estimated audio duration
  - Empty sections

Returns a QualityReport with PASS / WARN / FAIL level.
Orchestrator retries script generation once if level is FAIL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.console import Console

console = Console()

_FILLER_CHECK = re.compile(
    r"\b(?:stay (?:tuned|with me)|don['’]t (?:miss|skip|go anywhere)|"
    r"i['’]ll (?:tell|reveal|get to|come back) (?:you )?(?:this|that|it)|"
    r"keep watching|let that sink in|this changes everything|"
    r"you['’]re (?:probably )?wondering|here['’]s where it gets|"
    r"but first[,\s]|before (?:we|i) (?:continue|get to that)|"
    r"in (?:today|this)['’]?s video)\b",
    re.IGNORECASE,
)

# Thresholds — a 5-min video needs ~650 words; sections must carry real content
WARN_WORDS_PER_SECTION = 120
FAIL_WORDS_PER_SECTION = 80
WARN_TOTAL_WORDS = 650
FAIL_TOTAL_WORDS = 400
WARN_FOOTAGE_RATIO = 0.70   # footage_seconds / audio_seconds
WARN_HOOK_VIRAL = 5.5
MIN_CLIPS = 3


@dataclass
class QualityReport:
    passed: bool = True
    level: str = "PASS"          # "PASS", "WARN", "FAIL"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    word_count_total: int = 0
    section_word_counts: list[int] = field(default_factory=list)
    filler_count: int = 0
    footage_ratio: float = 1.0
    hook_viral_score: float = 0.0

    def _escalate(self, level: str) -> None:
        order = ["PASS", "WARN", "FAIL"]
        if order.index(level) > order.index(self.level):
            self.level = level

    def summary(self) -> str:
        lines = [
            f"Quality Gate: {self.level}  "
            f"words={self.word_count_total}  "
            f"filler={self.filler_count}  "
            f"footage={self.footage_ratio:.0%}"
        ]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        for issue in self.issues:
            lines.append(f"  FAIL: {issue}")
        return "\n".join(lines)


def check_script_quality(script, hook_viral_score: float = 0.0) -> QualityReport:
    """Validate a VideoScript for quality before TTS + render."""
    report = QualityReport(hook_viral_score=hook_viral_score)

    sections = getattr(script, "sections", []) or []
    hook = str(getattr(script, "hook", "") or "")
    cta = str(getattr(script, "call_to_action", "") or "")

    # ── Per-section word counts ───────────────────────────────────────────────
    section_counts = []
    for s in sections:
        text = str(s.get("script") or s.get("text") or "")
        section_counts.append(len(text.split()))
    report.section_word_counts = section_counts

    for i, (count, section) in enumerate(zip(section_counts, sections)):
        heading = section.get("heading", f"section {i + 1}")
        if count < FAIL_WORDS_PER_SECTION:
            report.issues.append(f"'{heading}': only {count} words (minimum {FAIL_WORDS_PER_SECTION})")
            report._escalate("FAIL")
            report.passed = False
        elif count < WARN_WORDS_PER_SECTION:
            report.warnings.append(f"'{heading}': {count} words (recommended {WARN_WORDS_PER_SECTION}+)")
            report._escalate("WARN")

    # ── Total word count ──────────────────────────────────────────────────────
    all_text = " ".join(
        [hook]
        + [str(s.get("script") or s.get("text") or "") for s in sections]
        + [cta]
    )
    total_words = len(all_text.split())
    report.word_count_total = total_words

    if total_words < FAIL_TOTAL_WORDS:
        report.issues.append(f"Script total {total_words} words (minimum {FAIL_TOTAL_WORDS})")
        report._escalate("FAIL")
        report.passed = False
    elif total_words < WARN_TOTAL_WORDS:
        report.warnings.append(f"Script total {total_words} words (recommended {WARN_TOTAL_WORDS}+)")
        report._escalate("WARN")

    # ── Filler phrase check ───────────────────────────────────────────────────
    filler_matches = _FILLER_CHECK.findall(all_text)
    report.filler_count = len(filler_matches)
    if filler_matches:
        report.warnings.append(
            f"{len(filler_matches)} filler phrase(s) still present: "
            + ", ".join(repr(m) for m in filler_matches[:3])
        )
        report._escalate("WARN")

    # ── Empty sections ────────────────────────────────────────────────────────
    empty = [
        s.get("heading", f"section {i+1}")
        for i, s in enumerate(sections)
        if not (s.get("script") or s.get("text"))
    ]
    if empty:
        report.issues.append(f"Empty section(s): {', '.join(str(e) for e in empty)}")
        report._escalate("FAIL")
        report.passed = False

    # ── Hook quality ──────────────────────────────────────────────────────────
    if hook_viral_score > 0 and hook_viral_score < WARN_HOOK_VIRAL:
        report.warnings.append(f"Hook viral score low: {hook_viral_score:.1f}/10")
        report._escalate("WARN")

    return report


def check_footage_quality(clips, estimated_audio_seconds: float = 0.0) -> QualityReport:
    """Validate that footage quantity is adequate for the script length."""
    report = QualityReport()

    if not clips:
        report.issues.append("No footage clips — render impossible without media")
        report._escalate("FAIL")
        report.passed = False
        return report

    if len(clips) < MIN_CLIPS:
        report.warnings.append(f"Only {len(clips)} clip(s) — video will look repetitive")
        report._escalate("WARN")

    if estimated_audio_seconds > 0:
        total_footage = sum(
            getattr(c, "out_point", 0.0) - getattr(c, "in_point", 0.0) for c in clips
        )
        ratio = total_footage / max(estimated_audio_seconds, 1.0)
        report.footage_ratio = ratio

        if ratio < WARN_FOOTAGE_RATIO:
            report.warnings.append(
                f"Footage {total_footage:.0f}s vs audio {estimated_audio_seconds:.0f}s "
                f"({ratio:.0%}) — renderer will loop footage"
            )
            report._escalate("WARN")

    return report


def print_quality_report(report: QualityReport, console_obj=None) -> None:
    c = console_obj or console
    colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(report.level, "white")
    c.print(
        f"[{colour}]■ Quality Gate: {report.level}[/{colour}]  "
        f"words={report.word_count_total}  "
        f"filler={report.filler_count}  "
        f"sections={len(report.section_word_counts)}"
    )
    for w in report.warnings:
        c.print(f"  [yellow]▲[/yellow] {w}")
    for issue in report.issues:
        c.print(f"  [red]✗[/red] {issue}")
