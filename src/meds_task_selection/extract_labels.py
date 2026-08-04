"""Slice per-task MEDS label files out of labeled dense task grids.

For each task in the registry, filters the grid to that task's
``(query, duration_days)`` pair and writes label files with columns
``subject_id, prediction_time, boolean_value``. Censored (null) rows are dropped
unless ``--keep-censored`` is passed.

Two layouts:

``sharded`` (default)
    ``{out_dir}/{task_id}/{split}/{shard}.parquet``, one file per *data* shard,
    mirroring ``{data_dir}/data/{split}/{shard}.parquet``. This is what MEDS-DEV
    (via ACES) produces and what MEDS-Tab requires: it resolves a shard's labels
    as ``input_label_dir / shard_fp.relative_to(shard_fp.parents[1])`` and
    ``scan_parquet``s that path for *every* data shard, so a shard with no rows
    for this task still needs an empty file with the right schema.

``flat``
    ``{out_dir}/{task_id}/{split}.parquet``, one file per split — simpler for
    consumers that just read a whole split at once.
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
    rows = grid.filter((pl.col("query") == code) & (pl.col("duration_days") == duration_days))
    if not keep_censored:
        rows = rows.filter(pl.col("boolean_value").is_not_null())
    return rows.select(LABEL_COLUMNS).sort("subject_id", "prediction_time").collect()


def grid_shards(grid_dir: Path, split: str) -> list[Path]:
    """Every grid shard parquet for ``split``, in sorted order."""
    return sorted((grid_dir / split).glob("*.parquet"))


def write_sharded_labels(
    grid_dir: Path,
    out_dir: Path,
    task: dict,
    split: str,
    keep_censored: bool = False,
) -> tuple[int, int, int]:
    """Write one label parquet per data shard; return (n_shards, n_rows, n_positive).

    Empty shards still get a file: MEDS-Tab scans the label path for every data
    shard and fails on a missing one.
    """
    split_out = out_dir / task["task_id"] / split
    split_out.mkdir(parents=True, exist_ok=True)
    n_rows = n_pos = 0
    shards = grid_shards(grid_dir, split)
    for shard in shards:
        labels = extract_task_labels(
            pl.scan_parquet(shard), task["query"], task["duration_days"], keep_censored
        )
        labels.write_parquet(split_out / shard.name)
        n_rows += labels.height
        n_pos += int(labels["boolean_value"].sum() or 0)
    return len(shards), n_rows, n_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="tasks.yaml registry")
    parser.add_argument("--grid-dir", type=Path, required=True, help="task-grid output root")
    parser.add_argument("--splits", nargs="+", default=["train", "tuning", "held_out"])
    parser.add_argument("--out-dir", type=Path, required=True, help="label-file output root")
    parser.add_argument("--keep-censored", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("sharded", "flat"),
        default="sharded",
        help="sharded (default, MEDS-DEV/MEDS-Tab compatible) or flat (one file per split)",
    )
    args = parser.parse_args()

    tasks = yaml.safe_load(args.tasks.read_text())
    for split in args.splits:
        for task in tasks:
            if args.layout == "sharded":
                n_shards, n_rows, n_pos = write_sharded_labels(
                    args.grid_dir, args.out_dir, task, split, keep_censored=args.keep_censored
                )
                print(
                    f"{task['task_id']}/{split}: {n_rows} rows ({n_pos} positive) "
                    f"across {n_shards} shards"
                )
            else:
                labels = extract_task_labels(
                    pl.scan_parquet(args.grid_dir / split / "*.parquet"),
                    task["query"],
                    task["duration_days"],
                    keep_censored=args.keep_censored,
                )
                out_path = args.out_dir / task["task_id"] / f"{split}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                labels.write_parquet(out_path)
                n_pos = int(labels["boolean_value"].sum() or 0)
                print(f"{task['task_id']}/{split}: {labels.height} rows ({n_pos} positive)")


if __name__ == "__main__":
    main()
