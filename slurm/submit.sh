#!/usr/bin/env bash
# Submit the meds_tab/tiny baseline to SLURM as a job array, one element per task.
#
#   ./slurm/submit.sh                       every task in $TASKS_YAML
#   ./slurm/submit.sh --tasks a b           only these task_ids
#   ./slurm/submit.sh --dry-run             print the sbatch command and submit nothing
#
# Settings live in slurm/config.sh; see slurm/README.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=config.sh
source slurm/config.sh

die() { echo "slurm/submit.sh: $*" >&2; exit 1; }

DRY=0
SELECTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tasks) shift; while [ $# -gt 0 ] && [ "${1:0:2}" != "--" ]; do SELECTED+=("$1"); shift; done ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

[ -f "$TASKS_YAML" ] || die "no task registry at $TASKS_YAML (set TASKS_YAML)"
[ -d "$LABELS_ROOT" ] || die "no labels at $LABELS_ROOT (set LABELS_ROOT; see slurm/README.md)"

mkdir -p slurm-logs "$RUN_DIR"
TASKS_FILE="$RUN_DIR/tasks.txt"

if [ "${#SELECTED[@]}" -gt 0 ]; then
  printf '%s\n' "${SELECTED[@]}" > "$TASKS_FILE"
else
  # Read task_ids without importing this project: the registry is a plain YAML list of mappings.
  grep -E '^\s*-?\s*task_id:' "$TASKS_YAML" | sed -E 's/.*task_id:\s*//; s/^["'\'']//; s/["'\'']$//' > "$TASKS_FILE"
fi

N=$(wc -l < "$TASKS_FILE")
[ "$N" -gt 0 ] || die "no tasks resolved from $TASKS_YAML"

# Every task must have labels before we queue anything: a missing dir would otherwise surface hours
# later as a failed array element.
while read -r task; do
  [ -d "$LABELS_ROOT/$task/held_out" ] || die "missing label shards for '$task' at $LABELS_ROOT/$task/held_out"
done < "$TASKS_FILE"

ARRAY="1-$N"
[ -n "$MAX_CONCURRENT" ] && ARRAY="$ARRAY%$MAX_CONCURRENT"

CMD=(sbatch "--array=$ARRAY" "--cpus-per-task=$CPUS" "--mem=$MEM" "--time=$TIME")
[ -n "$ACCOUNT" ] && CMD+=("--account=$ACCOUNT")
[ -n "$QOS" ] && CMD+=("--qos=$QOS")
[ -n "$PARTITION" ] && CMD+=("--partition=$PARTITION")
[ -n "$GRES" ] && CMD+=("--gres=$GRES")
# shellcheck disable=SC2206  # deliberate word splitting: a list of extra sbatch flags
[ -n "$SBATCH_EXTRA" ] && CMD+=($SBATCH_EXTRA)
CMD+=(slurm/job.sbatch "$TASKS_FILE")

echo "$N task(s) -> $TASKS_FILE"
printf '  %s\n' "$(cat "$TASKS_FILE" | tr '\n' ' ')"
echo "${CMD[*]}"
[ "$DRY" -eq 1 ] && exit 0
"${CMD[@]}"
