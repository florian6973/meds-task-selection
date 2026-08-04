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

MODEL_ARGS=(
  model=meds_tab/tiny
  "dataset_dir=$MEDS_ROOT"
  "labels_dir=$labels"
  "output_dir=$out"
  "dataset_name=$DATASET_NAME"
  "task_name=$task"
  dataset_type=supervised
  mode=full
)
# A prebuilt venv is reused only if its install marker is already there; without one MEDS-DEV would
# rebuild it with `uv venv --clear` and concurrent elements would clobber each other.
if [ -n "${VENV_DIR:-}" ]; then
  compgen -G "$VENV_DIR/.installed.*.txt" >/dev/null ||
    { echo "VENV_DIR=$VENV_DIR is not a built venv; run slurm/prebuild-venv.sh first" >&2; exit 1; }
  MODEL_ARGS+=("venv_dir=$VENV_DIR")
fi

srun --unbuffered /usr/bin/time -v meds-dev-model "${MODEL_ARGS[@]}"

srun --unbuffered meds-dev-evaluation \
  "predictions_dir=$out/$DATASET_NAME/$task/predict" \
  "output_dir=$RUN_DIR/eval/$task"

echo "[$(date -Is)] done task=$task"
python - "$RUN_DIR/eval/$task/results.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["samples_equally_weighted"]
print("AUROC {:.4f}  AP {:.4f}  calib_err {:.4f}".format(
    m["roc_auc_score"], m["average_precision_score"], m["calibration_error"]))
PY
