"""Slice per-task MEDS label files out of labeled dense task grids.

For each task in the registry and each requested split, filters the grid to that
task's ``(query, duration_days)`` pair and writes
``{out_dir}/{task_id}/{split}.parquet`` with columns
``subject_id, prediction_time, boolean_value`` — the shape a supervised or
fine-tuned template model consumes. Censored (null) rows are dropped unless
``--keep-censored`` is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

LABEL_COLUMNS = ("subject_id", "prediction_time", "boolean_value")


def extract_task_labels(
    grid: pl.LazyFrame, code: str, duration_days: float, keep_censored: bool = False
) -> pl.DataFrame:
    """Return the label frame for one task from a labeled grid."""
    rows = grid.filter(
        (pl.col("query") == code) & (pl.col("duration_days") == duration_days)
    )
    if not keep_censored:
        rows = rows.filter(pl.col("boolean_value").is_not_null())
    return rows.select(LABEL_COLUMNS).sort("subject_id", "prediction_time").collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="tasks.yaml registry")
    parser.add_argument("--grid-dir", type=Path, required=True, help="task-grid output root")
    parser.add_argument("--splits", nargs="+", default=["train", "tuning", "held_out"])
    parser.add_argument("--out-dir", type=Path, required=True, help="label-file output root")
    parser.add_argument("--keep-censored", action="store_true")
    args = parser.parse_args()

    tasks = yaml.safe_load(args.tasks.read_text())
    for split in args.splits:
        grid = pl.scan_parquet(args.grid_dir / split / "*.parquet")
        for task in tasks:
            labels = extract_task_labels(
                grid, task["query"], task["duration_days"], keep_censored=args.keep_censored
            )
            out_path = args.out_dir / task["task_id"] / f"{split}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            labels.write_parquet(out_path)
            n_pos = labels["boolean_value"].sum()
            print(f"{task['task_id']}/{split}: {labels.height} rows ({n_pos} positive)")


if __name__ == "__main__":
    main()
