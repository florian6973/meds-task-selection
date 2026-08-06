#!/usr/bin/env bash
# Progress of the meds_tab/tiny baseline array: one line per task.
#
#   ./slurm/status.sh            # snapshot
#   watch -n 60 ./slurm/status.sh
#
# Tabularization writes one npz per (shard x window_size x agg) and takes the bulk of the runtime, so
# its file count is the only fine-grained progress signal the pipeline emits — SLURM just shows
# "RUNNING" for hours. Stages after it are reported by the artifacts they leave behind.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=config.sh
source slurm/config.sh

# A bare `python` is not guaranteed on a login node either; prefer MEDS-DEV's own interpreter.
PY_BIN=python
if command -v meds-dev-model >/dev/null; then
    PY_BIN="$(dirname "$(sed -n '1s|^#!||p' "$(command -v meds-dev-model)")")/python"
elif command -v python3 >/dev/null; then
    PY_BIN=python3
fi

# Expected npz = shards x |window_sizes| x |aggs|; the tiny recipe uses 2 windows and 2 aggs.
shards=0
if [ -d "$MEDS_ROOT/data" ]; then
    shards=$(find "$MEDS_ROOT/data" -mindepth 2 -maxdepth 2 -name '*.parquet' | wc -l)
fi
expected=$((shards * 4))

# tasks.yaml records the tuning-subsample statistics used for task selection. Parse only its simple
# top-level list of mappings here so status.sh does not acquire a PyYAML dependency on login nodes.
task_metadata() {
    awk -v wanted="$1" '
        function emit() {
            if (task == wanted) {
                printf "%s\t%.1f%%\t%.1f%%\n", category, 100 * prevalence, 100 * censor
                found=1
            }
        }
        /^- / {
            if (seen) emit()
            seen=1; task=""; category=prevalence=censor="-"
        }
        /^[[:space:]-]*task_id:/ {
            task=$0; sub(/^.*task_id:[[:space:]]*/, "", task); gsub(/["'\'']/, "", task)
        }
        /^[[:space:]]+category:/ {
            category=$0; sub(/^.*category:[[:space:]]*/, "", category); gsub(/["'\'']/, "", category)
        }
        /^[[:space:]]+prevalence:/ {
            prevalence=$0; sub(/^.*prevalence:[[:space:]]*/, "", prevalence)
        }
        /^[[:space:]]+censor_rate:/ {
            censor=$0; sub(/^.*censor_rate:[[:space:]]*/, "", censor)
        }
        END { if (seen && !found) emit() }
    ' "$TASKS_YAML"
}

printf '%-42s %-14s %8s %8s %-18s %s\n' TASK CATEGORY PREV CENSOR TABULARIZE RESULT
for dir in "$RUN_DIR"/*/; do
    task=$(basename "$dir")
    [ "$task" = "eval" ] && continue
    metadata=$(task_metadata "$task")
    if [ -n "$metadata" ]; then
        IFS=$'\t' read -r category prevalence censor_rate <<< "$metadata"
    else
        category=- prevalence=- censor_rate=-
    fi

    tab="$dir/$DATASET_NAME/$task/train/meds_tab/tabularize"
    n=0
    [ -d "$tab" ] && n=$(find "$tab" -name '*.npz' | wc -l)
    if [ "$expected" -gt 0 ]; then
        progress=$(printf '%d/%d (%d%%)' "$n" "$expected" $((n * 100 / expected)))
    else
        progress="$n npz"
    fi

    results="$RUN_DIR/eval/$task/results.json"
    if [ -f "$results" ]; then
        line=$("$PY_BIN" - "$results" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["samples_equally_weighted"]
print("AUROC {:.4f}  AP {:.4f}".format(m["roc_auc_score"], m["average_precision_score"]))
PY
        )
    elif [ -f "$dir/$DATASET_NAME/$task/predict/predictions.parquet" ]; then
        line="predicted, awaiting eval"
    elif [ "$n" -ge "$expected" ] && [ "$expected" -gt 0 ]; then
        line="training xgboost"
    else
        line="-"
    fi
    printf '%-42s %-14s %8s %8s %-18s %s\n' \
        "$task" "$category" "$prevalence" "$censor_rate" "$progress" "$line"
done

if command -v squeue >/dev/null; then
    echo
    squeue -u "$USER" -o '%.10i %.9P %.20j %.2t %.10M %R' 2>/dev/null | head -15
fi
