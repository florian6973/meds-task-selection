#!/usr/bin/env bash
# Task-selection runbook: MIMIC-IV MEDS -> 10 fine-tuning tasks + label files.
#
# Usage:  ./run_pipeline.sh <step>
#   counts | candidates | smoke | measure | prevalence | select | final | labels
#   all              runs every step in order (smoke included)
#
# Resource policy: <= 4 cores, well under 20 GB.  The sampler pins its own polars pool to 1
# thread per process, so 4 concurrent shard jobs ~= 4 cores; this package's polars pool is
# capped by POLARS_MAX_THREADS below.
set -euo pipefail

MEDS="${MEDS:-$HOME/Datasets/MIMIC-IV/MEDS_cohort}"
# Default output root: `work/` inside this repo, which .gitignore excludes.  Every generated
# artifact (grids, labels, stats, Hydra logs) lands here and nowhere else.
WORK="${WORK:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/work}"
JOBS="${JOBS:-4}"
MEASURE_SPLIT="${MEASURE_SPLIT:-tuning}"
FINAL_SPLITS="${FINAL_SPLITS:-train tuning held_out}"

export POLARS_MAX_THREADS="$JOBS"
unset VIRTUAL_ENV # a foreign active venv would shadow the per-project ones below

SAMPLER=/home/fpollet/Git/meds-random-task-sampler
SELECTION=/home/fpollet/Git/meds-task-selection
grid() { uv run --project "$SAMPLER" meds-generate-task-grid "$@"; }
ts() { local cmd="$1"; shift; uv run --project "$SELECTION" "$cmd" "$@"; }

mkdir -p "$WORK"

# Build one grid per shard, JOBS at a time.  Each job is a separate process reading one ~10 MB
# shard; per-shard Hydra log dirs keep concurrent jobs from clobbering each other's config snapshot.
grid_split() {
    local out_dir="$1" split="$2" codes="$3" durations="$4" unique="$5"
    ls "$MEDS/data/$split" | sed 's/\.parquet$//' | sort |
        xargs -P "$JOBS" -I{} -- \
            env uv run --project "$SAMPLER" meds-generate-task-grid \
            "data_dir=$MEDS" "out_dir=$out_dir" "split=$split" "input_shard={}" \
            "grid.query_codes=$codes" "grid.durations=[$durations]" \
            "grid.write_unique_prediction_times=$unique" \
            "log_dir=$WORK/logs/$split/{}"
}

step="${1:-all}"
run_step() { [[ "$step" == "all" || "$step" == "$1" ]]; }

# 1. Per-code counts on the TRAIN split only (selecting on held-out statistics would leak).
if run_step counts; then
    echo "== counts"
    ts meds-ts-count-codes --data-dir "$MEDS" --split train --out "$WORK/codes_with_counts.parquet"
fi

# 2. Candidate panel: stratified head/torso/tail per category, plus MEDS_DEATH.
if run_step candidates; then
    echo "== candidates"
    ts meds-ts-candidates --codes "$WORK/codes_with_counts.parquet" --out "$WORK/candidates.yaml"
fi

# 3. Smoke test: one shard under the *same* settings as the real measurement run, so its peak
# RSS and wall time extrapolate honestly.  /usr/bin/time needs a real binary, not the `grid`
# shell function, hence the explicit `env uv run` here.
if run_step smoke; then
    echo "== smoke (1 shard, measurement settings)"
    /usr/bin/time -v env uv run --project "$SAMPLER" meds-generate-task-grid \
        "data_dir=$MEDS" "out_dir=$WORK/smoke_grid" "split=$MEASURE_SPLIT" input_shard=0 \
        "grid.query_codes=$WORK/candidates.yaml" 'grid.durations=[1,2,3,7,14,30,90,180,365,731]' \
        grid.write_unique_prediction_times=false "log_dir=$WORK/logs/smoke" \
        2>&1 | grep -E "Maximum resident|Elapsed \(wall|grid rows to"
    echo "-- peak RSS above is per shard; x$JOBS concurrent jobs is the run's total"
fi

# 4. Measurement grid: candidate codes x both horizon scales, on the tuning split.
if run_step measure; then
    echo "== measurement grid ($MEASURE_SPLIT)"
    grid_split "$WORK/measure_grid" "$MEASURE_SPLIT" "$WORK/candidates.yaml" \
        "1,2,3,7,14,30,90,180,365,731" false
fi

# 5. Measured per-(code, duration) label statistics.
if run_step prevalence; then
    echo "== prevalence"
    ts meds-ts-prevalence --grid-dir "$WORK/measure_grid" --split "$MEASURE_SPLIT" \
        --out "$WORK/task_stats.parquet"
fi

# 6. Pick 10 tasks; also writes selected_codes.yaml + selected_durations.txt for the final grids.
if run_step select; then
    echo "== select"
    ts meds-ts-select --stats "$WORK/task_stats.parquet" --out "$WORK/tasks.yaml"
fi

# 7. Final grids over the selected axes only, for every split.
if run_step final; then
    echo "== final grids"
    for split in $FINAL_SPLITS; do
        echo "-- $split"
        grid_split "$WORK/final_grid" "$split" "$WORK/selected_codes.yaml" \
            "$(cat "$WORK/selected_durations.txt")" true
    done
fi

# 8. Per-task MEDS label files for the template's supervised stages.
if run_step labels; then
    echo "== labels"
    # shellcheck disable=SC2086  # word splitting is the intent: --splits takes a list
    ts meds-ts-labels --tasks "$WORK/tasks.yaml" --grid-dir "$WORK/final_grid" \
        --splits $FINAL_SPLITS --out-dir "$WORK/labels"
fi

echo "done: $step"
