"""Autonomous experimentation engine — runs concurrent A/B tests, detects winners
using Welch's t-test, and auto-promotes winning variants.

Built-in experiments (registered on import):
  hook_style    — question / stat / story / controversy / prediction
  tts_voice     — niche-aware voice pool
  upload_time   — morning / afternoon / evening
  video_length  — short / medium / long
  subtitle_density — sparse / medium / dense
"""

from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentVariant:
    name: str
    config: dict
    assignments: int = 0
    wins: int = 0
    metric_values: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.assignments if self.assignments > 0 else 0.0

    @property
    def mean_metric(self) -> float:
        return sum(self.metric_values) / len(self.metric_values) if self.metric_values else 0.0


@dataclass
class Experiment:
    experiment_id: str
    name: str
    metric: str                          # "ctr" | "retention_rate" | "views" | "performance_score"
    variants: list[ExperimentVariant]
    status: str = "active"               # active | concluded | paused
    winner: Optional[str] = None
    min_samples_per_variant: int = 8
    created_at: str = field(default_factory=lambda: _now_iso())
    concluded_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _experiment_to_dict(exp: Experiment) -> dict:
    d = asdict(exp)
    return d


def _experiment_from_dict(d: dict) -> Experiment:
    d = dict(d)
    d["variants"] = [ExperimentVariant(**v) for v in d.get("variants", [])]
    return Experiment(**d)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine: Optional["ExperimentEngine"] = None
_engine_lock = threading.Lock()


class ExperimentEngine:
    """Singleton A/B experimentation engine with epsilon-greedy assignment."""

    def __init__(self) -> None:
        raw_path = os.getenv("RUFUS_EXPERIMENTS_PATH", "data/experiments.json")
        self._db_path = Path(raw_path)
        self._lock = threading.Lock()
        self._experiments: dict[str, Experiment] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            raw = self._db_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for item in data.get("experiments", []):
                try:
                    exp = _experiment_from_dict(item)
                    self._experiments[exp.name] = exp
                except Exception:
                    pass
        except Exception:
            pass

    def _save(self) -> None:
        """Persist experiments to JSON atomically."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiments": [_experiment_to_dict(e) for e in self._experiments.values()]
        }
        tmp = self._db_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, experiment: Experiment) -> None:
        """Add an experiment if one with the same name does not already exist."""
        with self._lock:
            if experiment.name not in self._experiments:
                self._experiments[experiment.name] = experiment
                self._save()

    def assign_variant(self, experiment_name: str) -> Optional[ExperimentVariant]:
        """Epsilon-greedy assignment: 20% explore, 80% exploit best variant.

        Returns None if experiment is not found or paused.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None or exp.status == "paused":
                return None

            if exp.status == "concluded" and exp.winner:
                # Always return the winner once concluded
                for v in exp.variants:
                    if v.name == exp.winner:
                        v.assignments += 1
                        self._save()
                        return v
                return None

            # Epsilon-greedy
            explore = random.random() < 0.20
            if explore or not any(v.metric_values for v in exp.variants):
                chosen = random.choice(exp.variants)
            else:
                chosen = max(exp.variants, key=lambda v: v.mean_metric)

            chosen.assignments += 1
            self._save()
            return chosen

    def record_outcome(
        self,
        experiment_name: str,
        variant_name: str,
        metric_value: float,
    ) -> None:
        """Record a metric observation for a variant.

        A variant wins a round if its metric_value is above the median of all
        current metric values across all variants in the experiment.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                return

            target: Optional[ExperimentVariant] = None
            for v in exp.variants:
                if v.name == variant_name:
                    target = v
                    break
            if target is None:
                return

            target.metric_values.append(metric_value)

            # Determine win: beat the median of all collected values
            all_values: list[float] = []
            for v in exp.variants:
                all_values.extend(v.metric_values)

            if all_values:
                med = median(all_values)
                if metric_value > med:
                    target.wins += 1

            self._save()

    def check_significance(self, experiment_name: str) -> Optional[str]:
        """Run Welch's t-test between the best and second-best variant.

        Returns the winner's name if p < 0.05 AND min_samples are met,
        otherwise returns None.

        Falls back to simple mean comparison if scipy is unavailable.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                return None

        # Work outside lock for potentially slow stats
        exp = self._experiments[experiment_name]

        eligible = [
            v for v in exp.variants
            if len(v.metric_values) >= exp.min_samples_per_variant
        ]
        if len(eligible) < 2:
            return None

        sorted_variants = sorted(eligible, key=lambda v: v.mean_metric, reverse=True)
        best = sorted_variants[0]
        second = sorted_variants[1]

        try:
            from scipy.stats import ttest_ind  # type: ignore
            _, p_value = ttest_ind(
                best.metric_values, second.metric_values, equal_var=False
            )
            if p_value < 0.05:
                return best.name
        except ImportError:
            # Fallback: require best mean > second mean by at least 5%
            if best.mean_metric > second.mean_metric * 1.05:
                return best.name
        except Exception:
            pass

        return None

    def auto_conclude(self) -> list[str]:
        """Check significance for all active experiments and conclude winners.

        Returns a list of experiment names that were concluded in this call.
        """
        concluded: list[str] = []
        active_names = [
            name for name, exp in self._experiments.items()
            if exp.status == "active"
        ]
        for name in active_names:
            winner_name = self.check_significance(name)
            if winner_name:
                with self._lock:
                    exp = self._experiments.get(name)
                    if exp and exp.status == "active":
                        exp.status = "concluded"
                        exp.winner = winner_name
                        exp.concluded_at = _now_iso()
                        self._save()
                        concluded.append(name)
        return concluded

    def get_best_config(self, experiment_name: str) -> Optional[dict]:
        """Return the config dict of the current best (or concluded winner) variant."""
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                return None

            if exp.status == "concluded" and exp.winner:
                for v in exp.variants:
                    if v.name == exp.winner:
                        return dict(v.config)
                return None

            # Return config of variant with highest mean metric
            candidates = [v for v in exp.variants if v.metric_values]
            if not candidates:
                # No data yet — return first variant
                return dict(exp.variants[0].config) if exp.variants else None

            best = max(candidates, key=lambda v: v.mean_metric)
            return dict(best.config)

    def print_report(self) -> None:
        """Print a Rich table showing all experiments and their current status."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title="A/B Experiment Report",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Experiment", style="bold")
            table.add_column("Status")
            table.add_column("Metric")
            table.add_column("Variants")
            table.add_column("Winner")

            with self._lock:
                experiments = list(self._experiments.values())

            for exp in experiments:
                variant_summary = ", ".join(
                    f"{v.name}(n={len(v.metric_values)}, μ={v.mean_metric:.3f})"
                    for v in exp.variants
                )
                status_color = (
                    "green" if exp.status == "concluded"
                    else "yellow" if exp.status == "active"
                    else "dim"
                )
                table.add_row(
                    exp.name,
                    f"[{status_color}]{exp.status}[/{status_color}]",
                    exp.metric,
                    variant_summary,
                    exp.winner or "—",
                )

            console.print(table)

        except ImportError:
            # Rich not available — plain text fallback
            with self._lock:
                experiments = list(self._experiments.values())
            for exp in experiments:
                print(f"[{exp.status}] {exp.name} ({exp.metric}) winner={exp.winner}")
                for v in exp.variants:
                    print(
                        f"  {v.name}: n={len(v.metric_values)} "
                        f"mean={v.mean_metric:.3f} win_rate={v.win_rate:.2f}"
                    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_engine() -> ExperimentEngine:
    """Return the module-level singleton ExperimentEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ExperimentEngine()
    return _engine


# ---------------------------------------------------------------------------
# Built-in experiment registration
# ---------------------------------------------------------------------------

def _make_variant(name: str, config: dict) -> ExperimentVariant:
    return ExperimentVariant(name=name, config=config)


def _register_builtins() -> None:
    engine = get_engine()

    # hook_style
    engine.register(Experiment(
        experiment_id="builtin_hook_style",
        name="hook_style",
        metric="ctr",
        variants=[
            _make_variant("question",     {"style": "question"}),
            _make_variant("stat",         {"style": "stat"}),
            _make_variant("story",        {"style": "story"}),
            _make_variant("controversy",  {"style": "controversy"}),
            _make_variant("prediction",   {"style": "prediction"}),
        ],
    ))

    # upload_time
    engine.register(Experiment(
        experiment_id="builtin_upload_time",
        name="upload_time",
        metric="views",
        variants=[
            _make_variant("morning",   {"hour_range": [6, 9]}),
            _make_variant("afternoon", {"hour_range": [14, 17]}),
            _make_variant("evening",   {"hour_range": [19, 22]}),
        ],
    ))

    # video_length
    engine.register(Experiment(
        experiment_id="builtin_video_length",
        name="video_length",
        metric="retention_rate",
        variants=[
            _make_variant("short",  {"target_duration": 300}),
            _make_variant("medium", {"target_duration": 600}),
            _make_variant("long",   {"target_duration": 900}),
        ],
    ))

    # subtitle_density
    engine.register(Experiment(
        experiment_id="builtin_subtitle_density",
        name="subtitle_density",
        metric="retention_rate",
        variants=[
            _make_variant("sparse",  {"words_per_line": 4}),
            _make_variant("medium",  {"words_per_line": 6}),
            _make_variant("dense",   {"words_per_line": 8}),
        ],
    ))


_register_builtins()
