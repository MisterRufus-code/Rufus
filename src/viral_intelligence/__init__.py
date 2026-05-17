"""
Viral Intelligence — three-layer competitive advantage engine.

  1. TrendSniper    — detects topics 3-7 days before they peak
  2. ViralDNA       — extracts winning patterns from top-performing videos
  3. RetentionArch  — structures every script for maximum watch time

Together they form a closed loop: snipe → learn → build → measure → repeat.
"""

from .trend_sniper import TrendSniper, SniperSignal
from .dna_extractor import ViralDNA, DNAProfile
from .retention_engine import RetentionEngine, RetentionScore

__all__ = [
    "TrendSniper", "SniperSignal",
    "ViralDNA", "DNAProfile",
    "RetentionEngine", "RetentionScore",
]
