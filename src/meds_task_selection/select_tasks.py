"""Select a fine-tuning/evaluation task registry from measured grid statistics.

A task is one ``(code, duration_days)`` pair. Eligibility is defined on measured
label statistics (enough uncensored rows, prevalence inside a band, censoring
below a cap); among eligible tasks, selection greedily prefers prevalence close
to a target while enforcing diversity: per-category and per-code caps plus
minimum counts of short-horizon (ICU-scale) and long-horizon (EHR-scale) tasks.
Fully deterministic; ties break on (code, duration).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml


@dataclass(frozen=True)
class SelectionConfig:
    n_tasks: int = 10
    min_uncensored: int = 500
    min_prevalence: float = 0.05
    max_prevalence: float = 0.50
    max_censor_rate: float = 0.80
    target_prevalence: float = 0.15
    max_per_category: int = 3
    max_per_code: int = 1
    min_short: int = 4
    min_long: int = 4
    short_max_days: float = 14.0


def task_id(code: str, duration_days: float) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in code).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{slug}_{duration_days:g}d"


def constraint_report(stats: pl.DataFrame, config: SelectionConfig) -> list[tuple[str, int, int]]:
    """Per-constraint attrition: (name, n_passing_alone, n_passing_cumulatively).

    A zero-task outcome is almost always one binding threshold — usually
    ``min_uncensored`` set above the measurement grid's rows-per-task. Reporting
    both columns separates "this filter is strict" from "this filter is the one
    that emptied the set".
    """
    constraints = [
        (f"n_uncensored >= {config.min_uncensored}", pl.col("n_uncensored") >= config.min_uncensored),
        (f"prevalence >= {config.min_prevalence}", pl.col("prevalence") >= config.min_prevalence),
        (f"prevalence <= {config.max_prevalence}", pl.col("prevalence") <= config.max_prevalence),
        (f"censor_rate <= {config.max_censor_rate}", pl.col("censor_rate") <= config.max_censor_rate),
    ]
    report = []
    cumulative = pl.lit(True)
    for name, predicate in constraints:
        cumulative = cumulative & predicate
        report.append((name, stats.filter(predicate).height, stats.filter(cumulative).height))
    return report


def _eligible(stats: pl.DataFrame, config: SelectionConfig) -> list[dict]:
    rows = (
        stats.filter(
            (pl.col("n_uncensored") >= config.min_uncensored)
            & (pl.col("prevalence") >= config.min_prevalence)
            & (pl.col("prevalence") <= config.max_prevalence)
            & (pl.col("censor_rate") <= config.max_censor_rate)
        )
        .with_columns(
            pl.col("query").str.split("//").list.first().alias("category"),
            (pl.col("prevalence") / config.target_prevalence)
            .log()
            .abs()
            .alias("score"),
        )
        .sort("score", "query", "duration_days")
        .to_dicts()
    )
    for row in rows:
        row["short"] = row["duration_days"] <= config.short_max_days
    return rows


def select_tasks(stats: pl.DataFrame, config: SelectionConfig) -> list[dict]:
    """Return up to ``config.n_tasks`` selected task rows, best-scored first within each phase."""
    eligible = _eligible(stats, config)
    selected: list[dict] = []
    per_category: dict[str, int] = {}
    per_code: dict[str, int] = {}

    def try_pick(row: dict) -> bool:
        if len(selected) >= config.n_tasks:
            return False
        if per_category.get(row["category"], 0) >= config.max_per_category:
            return False
        if per_code.get(row["query"], 0) >= config.max_per_code:
            return False
        selected.append(row)
        per_category[row["category"]] = per_category.get(row["category"], 0) + 1
        per_code[row["query"]] = per_code.get(row["query"], 0) + 1
        return True

    def picked(row: dict) -> bool:
        return any(row is s for s in selected)

    def fill(candidates: list[dict], quota: int) -> None:
        count = 0
        for row in candidates:
            if count >= quota:
                return
            if not picked(row) and try_pick(row):
                count += 1

    fill([r for r in eligible if r["short"]], config.min_short)
    fill([r for r in eligible if not r["short"]], config.min_long)
    fill(eligible, math.inf)

    for row in selected:
        row["task_id"] = task_id(row["query"], row["duration_days"])
        row.pop("score", None)
        row.pop("short", None)
    return selected


def write_registry(selected: list[dict], out_path: Path) -> None:
    """Write tasks.yaml plus grid-axis siblings: selected_codes.yaml and selected_durations.txt.

    The code list is consumable directly by ``grid.query_codes``; the duration
    file holds a comma-joined integer list for ``grid.durations=[$(cat ...)]``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(selected, sort_keys=False))
    codes = sorted({row["query"] for row in selected})
    durations = sorted({int(row["duration_days"]) for row in selected})
    (out_path.parent / "selected_codes.yaml").write_text(yaml.safe_dump(codes))
    (out_path.parent / "selected_durations.txt").write_text(",".join(str(d) for d in durations) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True, help="task_stats.parquet")
    parser.add_argument("--out", type=Path, required=True, help="output tasks.yaml registry")
    defaults = SelectionConfig()
    parser.add_argument("--n-tasks", type=int, default=defaults.n_tasks)
    parser.add_argument("--min-uncensored", type=int, default=defaults.min_uncensored)
    parser.add_argument("--min-prevalence", type=float, default=defaults.min_prevalence)
    parser.add_argument("--max-prevalence", type=float, default=defaults.max_prevalence)
    parser.add_argument("--max-censor-rate", type=float, default=defaults.max_censor_rate)
    parser.add_argument("--target-prevalence", type=float, default=defaults.target_prevalence)
    parser.add_argument("--max-per-category", type=int, default=defaults.max_per_category)
    parser.add_argument("--max-per-code", type=int, default=defaults.max_per_code)
    parser.add_argument("--min-short", type=int, default=defaults.min_short)
    parser.add_argument("--min-long", type=int, default=defaults.min_long)
    parser.add_argument("--short-max-days", type=float, default=defaults.short_max_days)
    args = parser.parse_args()

    config = SelectionConfig(
        n_tasks=args.n_tasks,
        min_uncensored=args.min_uncensored,
        min_prevalence=args.min_prevalence,
        max_prevalence=args.max_prevalence,
        max_censor_rate=args.max_censor_rate,
        target_prevalence=args.target_prevalence,
        max_per_category=args.max_per_category,
        max_per_code=args.max_per_code,
        min_short=args.min_short,
        min_long=args.min_long,
        short_max_days=args.short_max_days,
    )
    stats = pl.read_parquet(args.stats)
    selected = select_tasks(stats, config)
    if len(selected) < config.n_tasks:
        print(
            f"WARNING: only {len(selected)}/{config.n_tasks} tasks satisfied the "
            f"constraints ({stats.height} measured tasks; max rows/task "
            f"{stats['n_rows'].max()}). Attrition per constraint:"
        )
        for name, alone, cumulative in constraint_report(stats, config):
            print(f"  {name:<32} {alone:>6} alone {cumulative:>6} cumulative")
        print(
            "  then diversity caps (max-per-category/code, min-short/long) apply.\n"
            "  Relax the binding threshold, or enlarge the measurement grid "
            "(more prediction_times_per_subject / no subsampling)."
        )

    write_registry(selected, args.out)
    for row in selected:
        print(
            f"{row['task_id']}: prevalence={row['prevalence']:.3f} "
            f"n_uncensored={row['n_uncensored']} censor_rate={row['censor_rate']:.2%}"
        )
    print(f"{len(selected)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
