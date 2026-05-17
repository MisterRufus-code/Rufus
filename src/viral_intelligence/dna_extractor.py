"""
ViralDNA — extracts the winning patterns from top-performing YouTube videos.

Most AI systems write scripts based on general copywriting theory.
ViralDNA learns from what is *actually working on YouTube right now*:
  - Downloads captions from the top 20 viral videos in your niche
  - Extracts exact phrases, sentence structures, emotional triggers
  - Builds a "DNA profile" that Ollama uses as a writing template
  - Updates automatically on every pipeline run

This means the system gets smarter over time — not from a static model,
but from live YouTube performance data.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

_DNA_DIR = Path("data/viral_dna")

# YouTube search queries per niche to find top viral videos
_NICHE_QUERIES: dict[str, list[str]] = {
    "finance":  ["personal finance tips", "how to invest money", "passive income secrets"],
    "tech":     ["AI tools 2025", "best technology", "future of AI"],
    "health":   ["healthy habits", "weight loss tips", "longevity secrets"],
    "wellness": ["morning routine", "meditation benefits", "stress relief"],
    "business": ["how to start a business", "entrepreneur secrets", "side hustle"],
    "general":  ["shocking facts", "mind blowing science", "you won't believe"],
}

# Emotional trigger words to track frequency
_TRIGGER_WORDS = [
    "shocking", "secret", "truth", "never", "always", "mistake",
    "destroy", "collapse", "hidden", "reveal", "exposed", "jaw-dropping",
    "insane", "wild", "brutal", "actually", "really", "stop", "warning",
    "dangerous", "powerful", "hack", "trick", "method", "system",
]

# High-value opening structures to detect
_OPENING_PATTERNS = [
    r"^(stop|never|don't|avoid)",
    r"^(what|why|how) (most|nobody|everyone)",
    r"^(the real reason|the truth about|the secret)",
    r"^(most people|9[05]% of people)",
    r"^(i (lost|made|spent|saved)|we (lost|made))",
    r"^\$[\d,]+(k|m| million| thousand)?",
    r"^\d+ (ways|things|secrets|mistakes|habits|rules)",
    r"^(you('re| are) (doing|making|losing|wasting))",
]


@dataclass
class DNAProfile:
    niche:            str
    updated_at:       str
    sample_size:      int                 = 0
    top_phrases:      list[str]           = field(default_factory=list)
    opening_patterns: list[str]           = field(default_factory=list)
    trigger_freq:     dict[str, int]      = field(default_factory=dict)
    avg_hook_words:   int                 = 12
    proven_structures: list[str]          = field(default_factory=list)
    top_numbers:      list[str]           = field(default_factory=list)

    def to_ollama_context(self) -> str:
        """
        Format this DNA profile as a rich context block for Ollama.
        The LLM uses it to write scripts that mirror what's actually viral.
        """
        lines = [
            f"=== VIRAL DNA PROFILE: {self.niche.upper()} ===",
            f"Analysed from {self.sample_size} top-performing YouTube videos.",
            "",
            "TOP VIRAL PHRASES (use these exact words and structures):",
        ]
        for phrase in self.top_phrases[:15]:
            lines.append(f"  • \"{phrase}\"")

        if self.opening_patterns:
            lines.append("")
            lines.append("PROVEN OPENING STRUCTURES (copy these patterns):")
            for p in self.opening_patterns[:8]:
                lines.append(f"  • {p}")

        if self.proven_structures:
            lines.append("")
            lines.append("HIGH-RETENTION SCRIPT STRUCTURES:")
            for s in self.proven_structures[:5]:
                lines.append(f"  • {s}")

        top_triggers = sorted(self.trigger_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_triggers:
            lines.append("")
            lines.append(f"MOST EFFECTIVE EMOTIONAL TRIGGERS: "
                         f"{', '.join(w for w, _ in top_triggers)}")

        if self.top_numbers:
            lines.append("")
            lines.append(f"SPECIFICITY PATTERNS THAT WORK: {', '.join(self.top_numbers[:6])}")

        lines.append("")
        lines.append(f"AVERAGE HOOK LENGTH: {self.avg_hook_words} words")
        lines.append("=== END DNA PROFILE ===")

        return "\n".join(lines)

    def save(self) -> None:
        _DNA_DIR.mkdir(parents=True, exist_ok=True)
        path = _DNA_DIR / f"{self.niche}.json"
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, niche: str) -> Optional["DNAProfile"]:
        path = _DNA_DIR / f"{niche}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except Exception:
            return None


class ViralDNA:
    """
    Extracts viral patterns from top YouTube videos and stores them
    as a DNA profile that improves script quality over time.
    """

    def __init__(self, niche: str):
        self.niche = niche.lower()

    # ------------------------------------------------------------------
    # Video discovery via yt-dlp (free, no API key needed)
    # ------------------------------------------------------------------

    def _search_top_videos(self, query: str, max_results: int = 8) -> list[str]:
        """
        Use yt-dlp to search YouTube and return video URLs.
        Filters for videos 3-15 minutes long (ideal retention format).
        """
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    f"ytsearch{max_results}:{query}",
                    "--get-id",
                    "--match-filter", "duration > 180 & duration < 900",
                    "--no-playlist",
                    "--quiet",
                ],
                capture_output=True, text=True, timeout=30,
            )
            ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in ids]
        except Exception:
            return []

    def _get_captions(self, url: str) -> str:
        """Download auto-generated captions for a YouTube video."""
        try:
            result = subprocess.run(
                [
                    "yt-dlp", url,
                    "--write-auto-sub", "--sub-lang", "en",
                    "--skip-download",
                    "--sub-format", "vtt",
                    "--output", "/tmp/rufus_dna_%(id)s",
                    "--quiet",
                ],
                capture_output=True, text=True, timeout=45,
            )
            # Find the downloaded vtt file
            import glob
            files = glob.glob("/tmp/rufus_dna_*.vtt")
            if not files:
                return ""
            vtt = sorted(files)[-1]
            raw = Path(vtt).read_text(errors="replace")
            Path(vtt).unlink(missing_ok=True)
            # Strip VTT formatting: timestamps, tags
            text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}.*?-->.*?\n", "", raw)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"WEBVTT.*?\n\n", "", text)
            return " ".join(text.split())
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Pattern extraction
    # ------------------------------------------------------------------

    def _extract_ngrams(self, text: str, n: int = 3) -> Counter:
        """Extract n-grams from text."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        return Counter(ngrams)

    def _extract_openings(self, captions: list[str]) -> list[str]:
        """Find opening sentence structures that appear in multiple videos."""
        openings: list[str] = []
        for cap in captions:
            sentences = re.split(r"[.!?]", cap[:500])
            if sentences:
                first = sentences[0].strip().lower()
                for pattern in _OPENING_PATTERNS:
                    if re.match(pattern, first, re.IGNORECASE):
                        # Normalize to template form
                        openings.append(first[:80])
                        break
        return openings

    def _extract_numbers(self, text: str) -> list[str]:
        """Find specific numbers/amounts used in viral content."""
        patterns = [
            r"\$[\d,]+(?:k|m| million| thousand)?",
            r"\d+(?:\.\d+)?%",
            r"\d+ (ways|steps|secrets|mistakes|habits|tips|rules|things)",
            r"in \d+ (days|weeks|months|minutes|seconds)",
            r"\d+x (more|better|faster|cheaper)",
        ]
        found = []
        for p in patterns:
            matches = re.findall(p, text.lower())
            found.extend(matches if isinstance(matches[0], str) else
                         [" ".join(m) for m in matches] if matches else [])
        return found[:20]

    def _extract_structures(self, captions: list[str]) -> list[str]:
        """Identify high-level script structures from transcript flow."""
        structures = []
        for cap in captions:
            parts = cap.split(".")
            has_hook    = any(re.search(r"\b(stop|never|secret|truth|mistake)\b", p, re.I) for p in parts[:5])
            has_number  = bool(re.search(r"\b\d+\s*(ways|things|steps|secrets)\b", cap[:200], re.I))
            has_story   = bool(re.search(r"\b(i (was|had|lost|made)|when i|my)\b", cap[:400], re.I))
            has_cta     = bool(re.search(r"\b(subscribe|comment|like|let me know)\b", cap[-200:], re.I))

            parts_desc = []
            if has_hook:   parts_desc.append("loss-aversion hook")
            if has_number: parts_desc.append("numbered list")
            if has_story:  parts_desc.append("personal story")
            if has_cta:    parts_desc.append("CTA close")

            if len(parts_desc) >= 2:
                structures.append(" → ".join(parts_desc))

        # Return most common structures
        return [s for s, _ in Counter(structures).most_common(5)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, max_videos: int = 15, force: bool = False) -> DNAProfile:
        """
        Build or refresh the DNA profile for this niche.
        Uses cached profile if it's less than 24 hours old.
        """
        existing = DNAProfile.load(self.niche)
        if existing and not force:
            updated = datetime.fromisoformat(existing.updated_at)
            age_h = (datetime.now() - updated).total_seconds() / 3600
            if age_h < 24:
                console.print(f"[dim]ViralDNA: using cached {self.niche} profile "
                               f"({existing.sample_size} videos, {age_h:.0f}h old)[/dim]")
                return existing

        console.print(f"[cyan]ViralDNA[/cyan] extracting patterns from top {self.niche} videos…")

        queries  = _NICHE_QUERIES.get(self.niche, _NICHE_QUERIES["general"])
        all_urls: list[str] = []
        for q in queries[:2]:
            all_urls.extend(self._search_top_videos(q, max_results=8))
        all_urls = list(dict.fromkeys(all_urls))[:max_videos]

        if not all_urls:
            console.print(f"[yellow]ViralDNA: no videos found — using default profile[/yellow]")
            return self._default_profile()

        captions:     list[str]  = []
        all_text = ""

        for i, url in enumerate(all_urls):
            cap = self._get_captions(url)
            if cap and len(cap) > 200:
                captions.append(cap)
                all_text += " " + cap
                console.print(f"[dim]  [{i+1}/{len(all_urls)}] extracted {len(cap)} chars[/dim]")

        if not captions:
            return self._default_profile()

        # Extract patterns
        trigram_counts  = self._extract_ngrams(all_text, n=3)
        bigram_counts   = self._extract_ngrams(all_text, n=2)

        top_phrases = (
            [p for p, _ in bigram_counts.most_common(30)
             if not re.match(r"^(the|a|and|or|of|to|in|is|it|you|that|this)\b", p)][:10] +
            [p for p, _ in trigram_counts.most_common(20)][:8]
        )

        trigger_freq = {}
        text_lower = all_text.lower()
        for word in _TRIGGER_WORDS:
            count = text_lower.count(word)
            if count > 0:
                trigger_freq[word] = count

        openings   = self._extract_openings(captions)
        numbers    = self._extract_numbers(all_text)
        structures = self._extract_structures(captions)

        # Average hook length
        hook_lengths = []
        for cap in captions:
            first_sent = re.split(r"[.!?]", cap[:300])
            if first_sent:
                hook_lengths.append(len(first_sent[0].split()))
        avg_hook = int(sum(hook_lengths) / len(hook_lengths)) if hook_lengths else 12

        profile = DNAProfile(
            niche            = self.niche,
            updated_at       = datetime.now().isoformat(),
            sample_size      = len(captions),
            top_phrases      = top_phrases,
            opening_patterns = list(dict.fromkeys(openings))[:8],
            trigger_freq     = trigger_freq,
            avg_hook_words   = avg_hook,
            proven_structures= structures,
            top_numbers      = list(dict.fromkeys(numbers))[:10],
        )
        profile.save()
        console.print(
            f"[green]✓ ViralDNA[/green] built from {len(captions)} videos — "
            f"{len(top_phrases)} phrases, {len(structures)} structures"
        )
        return profile

    def _default_profile(self) -> DNAProfile:
        """Fallback profile based on universal viral content patterns."""
        return DNAProfile(
            niche            = self.niche,
            updated_at       = datetime.now().isoformat(),
            sample_size      = 0,
            top_phrases      = [
                "most people don't know", "the real reason", "shocking truth",
                "what nobody tells you", "stop doing this", "this changed everything",
                "they don't want you to know", "the hidden secret",
            ],
            opening_patterns = [
                "Stop [doing X] — it's costing you everything",
                "What nobody tells you about [topic]",
                "The real reason most people fail at [topic]",
                "[N] shocking [topic] secrets revealed",
                "You're making this [topic] mistake right now",
            ],
            trigger_freq     = {w: 5 for w in _TRIGGER_WORDS[:8]},
            avg_hook_words   = 12,
            proven_structures= [
                "loss-aversion hook → numbered list → personal story → CTA close",
                "shocking claim → proof → solution → CTA close",
                "question hook → story → numbered tips → CTA close",
            ],
            top_numbers = ["3 ways", "5 secrets", "$1,000", "30 days", "90%", "in 7 days"],
        )
