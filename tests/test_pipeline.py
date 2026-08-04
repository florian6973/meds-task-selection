"""End-to-end pipeline tests on synthetic MEDS-shaped data. No real dataset is read."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from meds_task_selection.compute_code_counts import (
    OCCURRENCES,
    PREVALENCE,
    SUBJECTS,
    compute_code_counts,
    enrich_codes,
)
from meds_task_selection.extract_labels import extract_task_labels
from meds_task_selection.measure_prevalence import measure_prevalence
from meds_task_selection.select_candidates import select_candidates
from meds_task_selection.select_tasks import (
    SelectionConfig,
    constraint_report,
    select_tasks,
    task_id,
    write_registry,
)


@pytest.fixture
def meds_dir(tmp_path: Path) -> Path:
    """A tiny two-shard MEDS train split plus a codes.parquet missing MEDS_DEATH."""
    t = datetime(2020, 1, 1)
    shard0 = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [t] * 5,
            "code": ["LAB//A", "LAB//A", "DIAGNOSIS//X", "LAB//A", "MEDS_DEATH"],
        }
    )
    shard1 = pl.DataFrame(
        {
            "subject_id": [3, 3],
            "time": [t] * 2,
            "code": ["DIAGNOSIS//X", "LAB//B"],
        }
    )
    data = tmp_path / "data" / "train"
    data.mkdir(parents=True)
    shard0.write_parquet(data / "0.parquet")
    shard1.write_parquet(data / "1.parquet")

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    pl.DataFrame(
        {
            "code": ["LAB//A", "LAB//B", "DIAGNOSIS//X", "DIAGNOSIS//UNSEEN"],
            "description": ["a", "b", "x", "unseen"],
        }
    ).write_parquet(metadata / "codes.parquet")
    return tmp_path


def test_code_counts(meds_dir: Path):
    counts, n_subjects = compute_code_counts(meds_dir, "train")
    assert n_subjects == 3
    codes = pl.read_parquet(meds_dir / "metadata" / "codes.parquet")
    enriched = enrich_codes(codes, counts, n_subjects)

    by_code = {row["code"]: row for row in enriched.to_dicts()}
    assert by_code["LAB//A"][OCCURRENCES] == 3
    assert by_code["LAB//A"][SUBJECTS] == 2
    assert by_code["LAB//A"][PREVALENCE] == pytest.approx(2 / 3)
    assert by_code["DIAGNOSIS//UNSEEN"][SUBJECTS] == 0
    # data-only code survives the full join even though metadata lacks it
    assert by_code["MEDS_DEATH"][SUBJECTS] == 1
    assert by_code["MEDS_DEATH"]["description"] is None


def test_candidate_selection_stratified():
    n = 200
    codes = pl.DataFrame(
        {
            "code": [f"LAB//{i:03d}" for i in range(n)],
            SUBJECTS: list(range(n, 0, -1)),
            PREVALENCE: [(n - i) / n for i in range(n)],
        }
    )
    candidates = select_candidates(
        codes, categories=("LAB",), n_head=5, n_torso=3, n_tail=2, extra_codes=("MEDS_DEATH",)
    )
    assert candidates[:5] == [f"LAB//{i:03d}" for i in range(5)]  # head = top by count
    assert candidates[-1] == "MEDS_DEATH"
    assert len(candidates) == len(set(candidates))
    torso = codes.filter(pl.col(PREVALENCE).is_between(0.01, 0.10, closed="left"))["code"]
    assert sum(c in set(torso) for c in candidates) == 3
    # deterministic
    assert candidates == select_candidates(
        codes, categories=("LAB",), n_head=5, n_torso=3, n_tail=2, extra_codes=("MEDS_DEATH",)
    )


def make_grid(tasks: dict[tuple[str, float], tuple[int, int, int]]) -> pl.DataFrame:
    """Build grid rows per (code, duration) from (n_pos, n_neg, n_censored) triples."""
    t = datetime(2020, 6, 1)
    rows = []
    for (code, duration), (n_pos, n_neg, n_cens) in tasks.items():
        labels = [True] * n_pos + [False] * n_neg + [None] * n_cens
        for i, label in enumerate(labels):
            rows.append(
                {
                    "subject_id": i,
                    "prediction_time": t,
                    "query": code,
                    "duration_days": duration,
                    "boolean_value": label,
                }
            )
    return pl.DataFrame(rows)


def test_measure_prevalence():
    grid = make_grid({("LAB//A", 7.0): (10, 30, 10), ("DIAGNOSIS//X", 365.0): (2, 2, 46)})
    stats = measure_prevalence(grid.lazy())
    by_task = {(r["query"], r["duration_days"]): r for r in stats.to_dicts()}

    lab = by_task[("LAB//A", 7.0)]
    assert lab["n_rows"] == 50
    assert lab["n_uncensored"] == 40
    assert lab["prevalence"] == pytest.approx(0.25)
    assert lab["censor_rate"] == pytest.approx(0.2)
    assert by_task[("DIAGNOSIS//X", 365.0)]["censor_rate"] == pytest.approx(0.92)


def test_select_tasks_respects_bands_and_coverage():
    stats = measure_prevalence(
        make_grid(
            {
                ("LAB//A", 1.0): (20, 80, 0),  # short, prevalence .20 -> eligible
                ("LAB//B", 7.0): (10, 90, 0),  # short, prevalence .10 -> eligible
                ("LAB//C", 7.0): (1, 99, 0),  # prevalence .01 -> below band
                ("DIAGNOSIS//X", 365.0): (15, 85, 0),  # long, eligible
                ("DIAGNOSIS//Y", 90.0): (30, 70, 0),  # long, eligible
                ("PROCEDURE//P", 365.0): (10, 10, 80),  # censor rate .8... eligible at cap
                ("PROCEDURE//Q", 30.0): (50, 5, 45),  # prevalence .91 -> above band
            }
        ).lazy()
    )
    config = SelectionConfig(
        n_tasks=4, min_uncensored=20, min_short=2, min_long=2, max_per_category=2
    )
    selected = select_tasks(stats, config)
    assert len(selected) == 4
    ids = {row["task_id"] for row in selected}
    assert "LAB_C_7d" not in ids and "PROCEDURE_Q_30d" not in ids
    n_short = sum(row["duration_days"] <= config.short_max_days for row in selected)
    assert n_short >= 2 and len(selected) - n_short >= 2
    for category in ("LAB", "DIAGNOSIS", "PROCEDURE"):
        assert sum(r["query"].startswith(category) for r in selected) <= 2
    assert select_tasks(stats, config) == selected  # deterministic


def test_select_tasks_max_per_code():
    stats = measure_prevalence(
        make_grid({("LAB//A", 1.0): (15, 85, 0), ("LAB//A", 7.0): (16, 84, 0)}).lazy()
    )
    config = SelectionConfig(n_tasks=2, min_uncensored=20, min_short=0, min_long=0)
    assert len(select_tasks(stats, config)) == 1  # same code picked once


def test_extract_labels_drops_censored():
    grid = make_grid({("LAB//A", 7.0): (3, 5, 2), ("LAB//B", 7.0): (1, 1, 0)})
    labels = extract_task_labels(grid.lazy(), "LAB//A", 7.0)
    assert labels.columns == ["subject_id", "prediction_time", "boolean_value"]
    assert labels.height == 8
    assert labels["boolean_value"].null_count() == 0
    assert labels["boolean_value"].sum() == 3

    kept = extract_task_labels(grid.lazy(), "LAB//A", 7.0, keep_censored=True)
    assert kept.height == 10


def test_constraint_report_identifies_binding_threshold():
    # Every task has healthy prevalence but only 100 rows: min_uncensored is the sole blocker.
    stats = measure_prevalence(
        make_grid({("LAB//A", 7.0): (20, 80, 0), ("LAB//B", 7.0): (15, 85, 0)}).lazy()
    )
    config = SelectionConfig(min_uncensored=500)
    assert select_tasks(stats, config) == []

    report = dict((name, cumulative) for name, _, cumulative in constraint_report(stats, config))
    assert report["n_uncensored >= 500"] == 0
    assert all(alone == 2 for name, alone, _ in constraint_report(stats, config) if "prevalence" in name)


def test_write_registry_axis_files(tmp_path: Path):
    import yaml

    selected = [
        {"task_id": "LAB_A_7d", "query": "LAB//A", "duration_days": 7.0},
        {"task_id": "DIAGNOSIS_X_365d", "query": "DIAGNOSIS//X", "duration_days": 365.0},
        {"task_id": "LAB_A_365d", "query": "LAB//A", "duration_days": 365.0},
    ]
    write_registry(selected, tmp_path / "tasks.yaml")

    assert yaml.safe_load((tmp_path / "tasks.yaml").read_text()) == selected
    assert yaml.safe_load((tmp_path / "selected_codes.yaml").read_text()) == [
        "DIAGNOSIS//X",
        "LAB//A",
    ]
    assert (tmp_path / "selected_durations.txt").read_text() == "7,365\n"


def test_task_id_slug():
    assert task_id("DIAGNOSIS//ICD//10//I509", 365.0) == "DIAGNOSIS_ICD_10_I509_365d"
    assert task_id("LAB//50813//mmol/L", 1.0) == "LAB_50813_mmol_L_1d"
    assert task_id("MEDS_DEATH", 30.0) == "MEDS_DEATH_30d"
