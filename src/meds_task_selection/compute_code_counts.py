"""Enrich a MEDS codes.parquet with per-code counts from one split's event shards.

Counts are computed on the split you pass (use ``train`` so later selection never
sees held-out statistics). Codes present in the data but absent from
``metadata/codes.parquet`` (e.g. ``MEDS_DEATH``, admission/discharge events) are
kept via a full join, so they remain selectable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

OCCURRENCES = "code/n_occurrences"
SUBJECTS = "code/n_subjects"
PREVALENCE = "code/subject_prevalence"


def compute_code_counts(data_dir: Path, split: str) -> tuple[pl.DataFrame, int]:
    """Return per-code counts over ``{data_dir}/data/{split}/*.parquet`` and the split's subject count."""
    events = pl.scan_parquet(data_dir / "data" / split / "*.parquet")
    # Streaming engine: the train split holds hundreds of millions of events, and the
    # default in-memory engine would materialize the projected columns all at once.
    counts = (
        events.group_by("code")
        .agg(
            pl.len().alias(OCCURRENCES),
            pl.col("subject_id").n_unique().alias(SUBJECTS),
        )
        .collect(engine="streaming")
    )
    n_subjects = events.select(pl.col("subject_id").n_unique()).collect(engine="streaming").item()
    return counts, n_subjects


def enrich_codes(codes: pl.DataFrame, counts: pl.DataFrame, n_subjects: int) -> pl.DataFrame:
    """Full-join metadata with counts and add subject prevalence; unseen codes get zero counts."""
    return (
        codes.join(counts, on="code", how="full", coalesce=True)
        .with_columns(
            pl.col(OCCURRENCES).fill_null(0),
            pl.col(SUBJECTS).fill_null(0),
        )
        .with_columns((pl.col(SUBJECTS) / n_subjects).alias(PREVALENCE))
        .sort(SUBJECTS, "code", descending=[True, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="MEDS dataset root")
    parser.add_argument("--split", default="train", help="split to count over (default: train)")
    parser.add_argument("--out", type=Path, required=True, help="output codes_with_counts.parquet")
    args = parser.parse_args()

    codes = pl.read_parquet(args.data_dir / "metadata" / "codes.parquet")
    counts, n_subjects = compute_code_counts(args.data_dir, args.split)
    enriched = enrich_codes(codes, counts, n_subjects)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(args.out)
    print(f"{enriched.height} codes ({counts.height} seen in {args.split}, {n_subjects} subjects) -> {args.out}")


if __name__ == "__main__":
    main()
