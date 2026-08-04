"""Build a candidate query-code panel from an enriched codes_with_counts.parquet.

Stratified per category (first ``//`` segment): the head stratum takes the most
frequent codes outright; the torso and tail strata take evenly spaced ranks
inside a subject-prevalence band, so mid-frequency and rare codes are covered
without any RNG. Anchor and extra codes are always included verbatim.

Frequency here is only a prefilter to keep the measurement grid tractable —
final task selection happens on measured grid prevalence (see select_tasks).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

from .compute_code_counts import PREVALENCE, SUBJECTS

DEFAULT_CATEGORIES = (
    "LAB",
    "DIAGNOSIS",
    "PROCEDURE",
    "INFUSION_START",
    "SUBJECT_FLUID_OUTPUT",
    "HCPCS",
)
DEFAULT_EXTRA_CODES = ("MEDS_DEATH",)


def evenly_spaced(rows: pl.DataFrame, k: int) -> pl.DataFrame:
    """Take up to ``k`` evenly spaced rows (by current order), deterministically."""
    if rows.height <= k:
        return rows
    idx = [round(i * (rows.height - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return rows[sorted(set(idx))]


def select_candidates(
    codes: pl.DataFrame,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    n_head: int = 20,
    n_torso: int = 20,
    n_tail: int = 10,
    torso_band: tuple[float, float] = (0.01, 0.10),
    tail_band: tuple[float, float] = (0.001, 0.01),
    anchors: tuple[str, ...] = (),
    extra_codes: tuple[str, ...] = DEFAULT_EXTRA_CODES,
) -> list[str]:
    """Return a deterministic candidate code list; order groups by category and stratum."""
    ranked = (
        codes.with_columns(pl.col("code").str.split("//").list.first().alias("category"))
        .filter(pl.col(SUBJECTS) > 0)
        .sort(SUBJECTS, "code", descending=[True, False])
    )
    selected: list[str] = []
    for category in categories:
        cat = ranked.filter(pl.col("category") == category)
        head = cat.head(n_head)
        torso = evenly_spaced(
            cat.filter(pl.col(PREVALENCE).is_between(*torso_band, closed="left")), n_torso
        )
        tail = evenly_spaced(
            cat.filter(pl.col(PREVALENCE).is_between(*tail_band, closed="left")), n_tail
        )
        for stratum in (head, torso, tail):
            selected.extend(stratum["code"].to_list())

    for code in (*anchors, *extra_codes):
        selected.append(code)

    return list(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, required=True, help="codes_with_counts.parquet")
    parser.add_argument("--out", type=Path, required=True, help="output candidates.yaml")
    parser.add_argument("--categories", nargs="*", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--n-head", type=int, default=20)
    parser.add_argument("--n-torso", type=int, default=20)
    parser.add_argument("--n-tail", type=int, default=10)
    parser.add_argument("--anchors", type=Path, help="YAML list of always-included codes")
    parser.add_argument(
        "--extra-codes", nargs="*", default=list(DEFAULT_EXTRA_CODES),
        help="codes appended verbatim even if absent from metadata (default: MEDS_DEATH)",
    )
    args = parser.parse_args()

    codes = pl.read_parquet(args.codes)
    anchors: tuple[str, ...] = ()
    if args.anchors is not None:
        anchors = tuple(yaml.safe_load(args.anchors.read_text()))

    candidates = select_candidates(
        codes,
        categories=tuple(args.categories),
        n_head=args.n_head,
        n_torso=args.n_torso,
        n_tail=args.n_tail,
        anchors=anchors,
        extra_codes=tuple(args.extra_codes),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(candidates, sort_keys=False))
    print(f"{len(candidates)} candidate codes -> {args.out}")


if __name__ == "__main__":
    main()
