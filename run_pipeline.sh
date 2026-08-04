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
WORK_FROM_ENV="${WORK:+1}"
WORK="${WORK:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/work}"
JOBS="${JOBS:-4}"
MEASURE_SPLIT="${MEASURE_SPLIT:-tuning}"
FINAL_SPLITS="${FINAL_SPLITS:-train tuning held_out}"

export POLARS_MAX_THREADS="$JOBS"

# An activated venv shadows the per-project environments used below (and MEDS-DEV's tool env in
# the baseline step), so drop it from PATH entirely rather than just unsetting the marker variable.
_ACTIVE_VENV_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    _ACTIVE_VENV_BIN="$VIRTUAL_ENV/bin"
    PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$_ACTIVE_VENV_BIN" | paste -sd:)"
    export PATH
fi
unset VIRTUAL_ENV

SAMPLER=/home/fpollet/Git/meds-random-task-sampler
SELECTION=/home/fpollet/Git/meds-task-selection
grid() { uv run --project "$SAMPLER" meds-generate-task-grid "$@"; }
ts() { local cmd="$1"; shift; uv run --project "$SELECTION" "$cmd" "$@"; }

mkdir -p "$WORK"
echo "WORK=$WORK"

# Fail fast with an actionable message rather than deep inside a long-running model command.
require_input() {
    local path="$1" what="$2"
    if [[ ! -e "$path" ]]; then
        echo "missing $what: $path" >&2
        if [[ -n "${WORK_FROM_ENV:-}" ]]; then
            echo "  WORK came from your environment ($WORK). If that is stale, run: unset WORK" >&2
        fi
        exit 1
    fi
}

# `/usr/bin/time -v` appends ~23 lines of resource stats, which would bury the real error in a
# plain `tail`.  Surface the exception lines instead, falling back to the tail if there are none.
report_failure() {
    local log="$1"
    echo "FAILED — $log" >&2
    if ! grep -m3 -E "^[A-Za-z_.]*(Error|Exception):" "$log" >&2; then
        grep -vE '^[[:space:]]+(Command being timed|User time|System time|Percent of CPU|Elapsed \(wall|Average|Maximum resident|Major |Minor |Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Exit status)' "$log" |
            tail -20 >&2
    fi
}

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

# 9. MEDS-DEV baseline: meds_tab/tiny, one task at a time.  Tabularization is task-specific
# (MEDS-Tab materializes features only at each task's label prediction times), so every task pays
# a full tabularization pass — run one first and time it before looping over all ten.
# Set TASKS_TO_RUN to a subset, e.g. TASKS_TO_RUN="LAB_51146_3d".
if run_step baseline; then
    echo "== meds_tab/tiny baseline"
    : "${DATASET_NAME:=MIMIC-IV}"
    # MEDS-DEV's CLIs shell out to sibling tools -- meds-dev-evaluation invokes
    # meds-evaluation-cli -- and expect to find them alongside themselves.  With a `uv tool`
    # install only MEDS-DEV's *own* entry points are symlinked into ~/.local/bin; its
    # dependencies' scripts (meds-evaluation-cli among them) exist solely inside the tool
    # environment.  So dropping a stray venv from PATH is necessary but not sufficient: without
    # the tool env on PATH, the inner command is either missing or served by whichever unrelated
    # venv comes first, which fails with a misleading config-override traceback.  Do both.
    if ! command -v meds-dev-model >/dev/null; then
        echo "meds-dev-model not found; install MEDS-DEV (uv tool install meds-dev)" >&2
        exit 1
    fi
    MEDS_DEV_BIN="$(dirname "$(sed -n '1s|^#!||p' "$(command -v meds-dev-model)")")"
    if [[ -n "${_ACTIVE_VENV_BIN:-}" ]]; then # stripped from PATH at the top of this script
        echo "-- dropped $_ACTIVE_VENV_BIN from PATH"
    fi
    export PATH="$MEDS_DEV_BIN:$PATH"
    require_input "$WORK/tasks.yaml" "task registry"
    : "${TASKS_TO_RUN:=$(uv run --project "$SELECTION" python -c "
import yaml, sys
print(' '.join(t['task_id'] for t in yaml.safe_load(open('$WORK/tasks.yaml'))))")}"
    mkdir -p "$WORK/logs/baseline"
    for task in $TASKS_TO_RUN; do
        echo "-- $task"
        # MEDS-Tab needs a label dir per data shard; check before paying for tabularization.
        require_input "$WORK/labels/$task" "labels for task '$task'"
        require_input "$WORK/labels/$task/held_out" "held_out label shards for '$task'"
        log="$WORK/logs/baseline/$task.log"
        # OMP_NUM_THREADS caps XGBoost's thread pool; POLARS_MAX_THREADS (exported above) caps
        # MEDS-Tab's polars.  The recipe itself pins joblib to one worker and XGBoost tuning to
        # n_jobs=1, so JOBS here bounds threads-within-one-process, not concurrent processes.
        if ! /usr/bin/time -v env OMP_NUM_THREADS="$JOBS" meds-dev-model \
            model=meds_tab/tiny \
            "dataset_dir=$MEDS" \
            "labels_dir=$WORK/labels/$task" \
            "output_dir=$WORK/baseline" \
            "dataset_name=$DATASET_NAME" \
            "task_name=$task" \
            dataset_type=supervised \
            mode=full >"$log" 2>&1; then
            report_failure "$log"
            exit 1
        fi
        grep -E "Maximum resident set size|Elapsed \(wall" "$log" |
            sed 's/^\s*/   /' # peak RSS + wall time for this task
        du -sh "$WORK/baseline/$DATASET_NAME/$task" | sed 's/^/   disk: /'

        meds-dev-evaluation \
            "predictions_dir=$WORK/baseline/$DATASET_NAME/$task/predict" \
            "output_dir=$WORK/baseline_eval/$task" >>"$log" 2>&1
        uv run --project "$SELECTION" python -c "
import json, sys
m = json.load(open('$WORK/baseline_eval/$task/results.json'))['samples_equally_weighted']
print('   AUROC {:.4f}  AP {:.4f}  calib_err {:.4f}'.format(
    m['roc_auc_score'], m['average_precision_score'], m['calibration_error']))"
    done
fi

echo "done: $step"
