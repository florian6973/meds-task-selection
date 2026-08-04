"""Measure per-(code, duration) label statistics from a labeled dense task grid.

Reads ``{grid_dir}/{split}/*.parquet`` produced by ``meds-generate-task-grid``
(schema: subject_id, prediction_time, query, duration_days, boolean_value with
null meaning censored) and writes one row of statistics per task:
n_rows, n_censored, n_pos, n_neg, prevalence (over uncensored rows), censor_rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def measure_prevalence(grid: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate grid rows into per-(query, duration_days) task statistics."""
    return (
        grid.group_by("query", "duration_days")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("boolean_value").null_count().alias("n_censored"),
            pl.col("boolean_value").sum().alias("n_pos"),
            pl.col("subject_id").n_unique().alias("n_subjects"),
        )
        .with_columns(
            (pl.col("n_rows") - pl.col("n_censored")).alias("n_uncensored"),
        )
        .with_columns(
            (pl.col("n_uncensored") - pl.col("n_pos")).alias("n_neg"),
            (pl.col("n_pos") / pl.col("n_uncensored")).alias("prevalence"),
            (pl.col("n_censored") / pl.col("n_rows")).alias("censor_rate"),
        )
        .sort("query", "duration_days")
        .collect(engine="streaming")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True, help="task-grid output root")
    parser.add_argument("--split", default="tuning", help="split subdirectory (default: tuning)")
    parser.add_argument("--out", type=Path, required=True, help="output task_stats.parquet")
    args = parser.parse_args()

    stats = measure_prevalence(pl.scan_parquet(args.grid_dir / args.split / "*.parquet"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.write_parquet(args.out)

    labeled = stats.filter(pl.col("n_uncensored") > 0)
    print(f"{stats.height} tasks -> {args.out}")
    print(
        f"prevalence over uncensored rows: median {labeled['prevalence'].median():.4f}, "
        f"censor rate: median {stats['censor_rate'].median():.2%}"
    )


if __name__ == "__main__":
    main()
