#!/usr/bin/env bash
# One SLURM array element = the meds_tab/tiny baseline for one task.
#
# Resources come from `slurm/submit.sh` on the sbatch command line rather than #SBATCH lines here, so
# a one-off can override them without editing this file:
#   sbatch --array=1-1 --time=12:00:00 slurm/job.sbatch tasks.txt
#SBATCH --job-name=meds-tab-baseline
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --error=slurm-logs/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=config.sh
source slurm/config.sh

TASKS_FILE=${1:?job.sbatch needs a tasks file (one task_id per line)}
task=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKS_FILE")
[ -n "$task" ] || { echo "no entry ${SLURM_ARRAY_TASK_ID:-1} in $TASKS_FILE" >&2; exit 1; }

if [ -n "$SETUP" ]; then eval "$SETUP"; fi

# An activated venv shadows MEDS-DEV's environment; drop it and put MEDS-DEV's own first.  Its CLIs
# shell out to sibling tools (meds-evaluation-cli) that a `uv tool` install keeps *inside* the tool
# env rather than symlinking onto PATH, so prepending that env is required, not merely tidy.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd:)"
  export PATH
  unset VIRTUAL_ENV
fi
command -v meds-dev-model >/dev/null || { echo "meds-dev-model not on PATH after SETUP='$SETUP'" >&2; exit 1; }
MEDS_DEV_BIN="$(dirname "$(sed -n '1s|^#!||p' "$(command -v meds-dev-model)")")"
export PATH="$MEDS_DEV_BIN:$PATH"

# Keep the job inside its cgroup allocation: the recipe pins joblib to one worker, so these bound
# threads-within-a-process (XGBoost, polars) rather than concurrent processes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}"

labels="$LABELS_ROOT/$task"
out="$RUN_DIR/$task"
[ -d "$labels/held_out" ] || { echo "missing label shards: $labels/held_out" >&2; exit 1; }

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} task=$task"

tab_dir="$out/$DATASET_NAME/$task/train/meds_tab"
pred_dir="$out/$DATASET_NAME/$task/predict"

# Fast path: issue MEDS-Tab's own CLIs with WORKERS joblib workers.  These are the exact commands
# `meds_tab/tiny` runs -- only `worker="range(0,N)"` differs, which the recipe pins to 1 and cannot be
# overridden through meds-dev-model.  Workers coordinate via MEDS-Transforms' rwlock and skip shards
# another worker finished, so raising the count only adds throughput.
if [ "$MODEL" = "meds_tab/tiny" ] && [ "${WORKERS:-1}" -gt 1 ]; then
  [ -n "${VENV_DIR:-}" ] && compgen -G "$VENV_DIR/.installed.*.txt" >/dev/null ||
    { echo "WORKERS>1 needs a prebuilt VENV_DIR; run slurm/prebuild-venv.sh" >&2; exit 1; }
  VB="$VENV_DIR/bin"
  TAB_ARGS=(
    "input_dir=$MEDS_ROOT/data" "output_dir=$tab_dir" do_overwrite=False
    "input_label_dir=$labels"
    "tabularization.aggs=[code/count,value/sum]"
    "tabularization.window_sizes=[7d,30d]"
    "tabularization.max_included_codes=100"
  )
  echo "[$(date -Is)] tabularizing with $WORKERS workers"
  srun --unbuffered "$VB/meds-tab-describe" "input_dir=$MEDS_ROOT/data" "output_dir=$tab_dir"
  srun --unbuffered "$VB/meds-tab-tabularize-static" "${TAB_ARGS[@]}"
  srun --unbuffered "$VB/meds-tab-tabularize-time-series" --multirun \
    "worker=range(0,$WORKERS)" hydra/launcher=joblib "${TAB_ARGS[@]}"
  echo "[$(date -Is)] training xgboost"
  srun --unbuffered "$VB/meds-tab-xgboost" --multirun \
    "input_dir=$MEDS_ROOT/data" "output_dir=$tab_dir" \
    "output_model_dir=$out/$DATASET_NAME/$task/train/results" \
    task_name=null do_overwrite=False \
    "hydra.sweeper.n_trials=20" "hydra.sweeper.n_jobs=1" \
    "input_tabularized_cache_dir=$tab_dir/tabularize/" \
    "input_label_cache_dir=$labels" \
    "tabularization.aggs=[code/count,value/sum]" \
    "tabularization.window_sizes=[7d,30d]" \
    "tabularization.max_included_codes=100" tabularization.min_code_inclusion_count=null
  latest=$(ls -td "$out/$DATASET_NAME/$task/train/results"/*/ | head -n 1)
  mkdir -p "$pred_dir"
  cp "$latest/best_trial/held_out_predictions.parquet" "$pred_dir/predictions.parquet"
else

MODEL_ARGS=(
  "model=$MODEL"
  "dataset_dir=$MEDS_ROOT"
  "labels_dir=$labels"
  "output_dir=$out"
  "dataset_name=$DATASET_NAME"
  "task_name=$task"
  dataset_type=supervised
  mode=full
)
# A prebuilt venv is reused only if its install marker is already there; without one MEDS-DEV would
# rebuild it with `uv venv --clear` and concurrent elements would clobber each other.  Only offer it
# to meds_tab: another model's requirements hash would not match, and MEDS-DEV's response to a
# mismatch is to delete the venv -- which would destroy the shared meds-tab environment.
if [ -n "${VENV_DIR:-}" ] && [ "$MODEL" = "meds_tab/tiny" ]; then
  compgen -G "$VENV_DIR/.installed.*.txt" >/dev/null ||
    { echo "VENV_DIR=$VENV_DIR is not a built venv; run slurm/prebuild-venv.sh first" >&2; exit 1; }
  MODEL_ARGS+=("venv_dir=$VENV_DIR")
fi

# GNU time is a convenience for peak-RSS reporting, not a dependency: plenty of compute nodes have a
# bare /usr/bin without it, and `time` alone is a shell keyword that cannot wrap an srun task. Fall
# back to running unwrapped -- `sacct -j <jobid> --format=MaxRSS` reports the same number afterwards.
TIMER=()
for candidate in /usr/bin/time /bin/time "$(command -v gtime || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then TIMER=("$candidate" -v); break; fi
done
[ "${#TIMER[@]}" -gt 0 ] || echo "note: GNU time not found; use 'sacct -j \$SLURM_JOB_ID --format=MaxRSS' for peak memory"

# `${TIMER[@]+...}` rather than a bare `"${TIMER[@]}"`: expanding an empty array under `set -u` is an
# unbound-variable error on bash < 4.4, which is still what RHEL 7 era compute nodes ship.
srun --unbuffered ${TIMER[@]+"${TIMER[@]}"} meds-dev-model "${MODEL_ARGS[@]}"
fi

srun --unbuffered meds-dev-evaluation \
  "predictions_dir=$out/$DATASET_NAME/$task/predict" \
  "output_dir=$RUN_DIR/eval/$task"

echo "[$(date -Is)] done task=$task"
# Use MEDS-DEV's own interpreter: a bare `python` may not exist on a compute node, and this one is
# guaranteed present because its CLIs just ran.
"$MEDS_DEV_BIN/python" - "$RUN_DIR/eval/$task/results.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["samples_equally_weighted"]
print("AUROC {:.4f}  AP {:.4f}  calib_err {:.4f}".format(
    m["roc_auc_score"], m["average_precision_score"], m["calibration_error"]))
PY
