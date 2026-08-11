#!/usr/bin/env python3
"""
analyze_scripts.py — Offline analysis of script-writer JSONL logs.

Reads logs/scripts/*.jsonl and produces:
  - distribution: score histogram per niche/seed_type/temperature
  - rejections:   most common rejection reasons
  - winners:      top-scoring hooks (also writes config/winning_hooks.json)
  - losers:       lowest-scoring hooks (also writes config/losing_hooks.json)
  - funnel:       avg attempts-to-pass per niche/seed_type
  - cost:         total $ spent on script step (7/30 days)

All output is plain text to stdout. Pure stdlib.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT      = Path(__file__).parent.parent
LOG_DIR   = ROOT / "logs" / "scripts"
CONFIG_DIR = ROOT / "config"


def _iter_records(days: int | None = None):
    """Yield every JSONL record, optionally filtered to last N days."""
    if not LOG_DIR.exists():
        return
    cutoff = None
    if days is not None:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        if cutoff and path.stem < cutoff:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue


def _final_records():
    """Records for finished scripts only (phase=='final')."""
    return [r for r in _iter_records() if r.get("phase") == "final"]


def _print_table(rows: list[tuple], headers: list[str]) -> None:
    if not rows:
        print("  (no data)")
        return
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt    = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def cmd_distribution(_args):
    by_niche:  dict = defaultdict(list)
    by_seed:   dict = defaultdict(list)
    by_temp:   dict = defaultdict(list)
    for r in _final_records():
        s = r.get("total_score")
        if s is None:
            continue
        by_niche[r.get("niche") or "?"].append(s)
        by_seed[r.get("seed_type") or "?"].append(s)
        by_temp[round(r.get("temperature") or 0, 2)].append(s)

    def _stats(d):
        out = []
        for k, vs in sorted(d.items()):
            if not vs:
                continue
            out.append((k, len(vs), f"{statistics.mean(vs):.2f}",
                       f"{min(vs)}/{max(vs)}",
                       sum(1 for v in vs if v >= 8)))
        return out

    print("\n── score distribution by niche ──")
    _print_table(_stats(by_niche), ["niche", "n", "avg", "min/max", "≥8"])
    print("\n── score distribution by seed_type ──")
    _print_table(_stats(by_seed), ["seed_type", "n", "avg", "min/max", "≥8"])
    print("\n── score distribution by temperature ──")
    _print_table(_stats(by_temp), ["temp", "n", "avg", "min/max", "≥8"])


def cmd_rejections(_args):
    c: Counter = Counter()
    by_phase: dict = defaultdict(Counter)
    for r in _iter_records():
        reason = r.get("rejected_reason")
        if not reason:
            continue
        c[reason] += 1
        by_phase[r.get("phase") or "?"][reason] += 1
    print("\n── top rejection reasons (all phases) ──")
    _print_table([(k, v) for k, v in c.most_common(20)], ["reason", "count"])
    for phase, counter in by_phase.items():
        print(f"\n── rejections in phase '{phase}' ──")
        _print_table([(k, v) for k, v in counter.most_common(10)], ["reason", "count"])


def _hook_score_records():
    """Walk hook_score phase records and map hook → its score."""
    # hook_score is logged once per call with a body=raw JSON.
    # Easier path: aggregate body_gen finals — each carries its hook + final score.
    out = []
    for r in _final_records():
        hook  = (r.get("hook") or "").strip()
        score = r.get("total_score")
        if hook and score is not None:
            out.append((hook, score, r.get("niche") or "?",
                        r.get("seed_type") or "?"))
    return out


def cmd_winners(args):
    records = _hook_score_records()
    records.sort(key=lambda x: x[1], reverse=True)
    top = records[: args.top]
    print(f"\n── top {len(top)} winning hooks (final body score) ──")
    _print_table([(h[:60], s, n, st) for h, s, n, st in top],
                 ["hook", "score", "niche", "seed"])

    out_path = CONFIG_DIR / "winning_hooks.json"
    out_path.write_text(json.dumps(
        [{"hook": h, "score": s, "niche": n, "seed_type": st}
         for h, s, n, st in top],
        indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")


def cmd_losers(args):
    records = _hook_score_records()
    records.sort(key=lambda x: x[1])
    bottom = records[: args.top]
    print(f"\n── bottom {len(bottom)} losing hooks ──")
    _print_table([(h[:60], s, n, st) for h, s, n, st in bottom],
                 ["hook", "score", "niche", "seed"])

    out_path = CONFIG_DIR / "losing_hooks.json"
    out_path.write_text(json.dumps(
        [{"hook": h, "score": s, "niche": n, "seed_type": st}
         for h, s, n, st in bottom],
        indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")


def cmd_funnel(_args):
    """For each run_id, count body_gen attempts and look up final score + niche."""
    attempts_per_run: dict = defaultdict(int)
    final_per_run:    dict = {}
    for r in _iter_records():
        rid = r.get("run_id")
        if not rid:
            continue
        if r.get("phase") == "body_gen":
            attempts_per_run[rid] += 1
        if r.get("phase") == "final":
            final_per_run[rid] = r

    by_niche:  dict = defaultdict(list)
    by_seed:   dict = defaultdict(list)
    for rid, attempts in attempts_per_run.items():
        f = final_per_run.get(rid, {})
        by_niche[f.get("niche") or "?"].append(attempts)
        by_seed[f.get("seed_type") or "?"].append(attempts)

    print("\n── avg body attempts per niche ──")
    rows = [(k, len(vs), f"{statistics.mean(vs):.2f}", max(vs))
            for k, vs in sorted(by_niche.items()) if vs]
    _print_table(rows, ["niche", "runs", "avg_attempts", "max"])

    print("\n── avg body attempts per seed_type ──")
    rows = [(k, len(vs), f"{statistics.mean(vs):.2f}", max(vs))
            for k, vs in sorted(by_seed.items()) if vs]
    _print_table(rows, ["seed_type", "runs", "avg_attempts", "max"])


def cmd_cost(_args):
    for days in (7, 30):
        total = 0.0
        n_runs = 0
        seen_runs: set = set()
        for r in _iter_records(days=days):
            c = r.get("cost_usd") or 0
            total += float(c)
            if r.get("phase") == "final" and r.get("run_id"):
                seen_runs.add(r["run_id"])
                n_runs += 1
        avg = total / n_runs if n_runs else 0.0
        print(f"\n── last {days} days ──")
        print(f"  total spend: ${total:.4f}")
        print(f"  runs:        {n_runs}")
        print(f"  avg / run:   ${avg:.4f}")


def main():
    p   = argparse.ArgumentParser(description="Analyze script-writer JSONL logs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("distribution")
    sub.add_parser("rejections")
    w = sub.add_parser("winners");  w.add_argument("--top", type=int, default=20)
    l = sub.add_parser("losers");   l.add_argument("--top", type=int, default=20)
    sub.add_parser("funnel")
    sub.add_parser("cost")
    args = p.parse_args()

    dispatch = {
        "distribution": cmd_distribution,
        "rejections":   cmd_rejections,
        "winners":      cmd_winners,
        "losers":       cmd_losers,
        "funnel":       cmd_funnel,
        "cost":         cmd_cost,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
