# MEDS Task Selection

Selects fine-tuning and evaluation tasks for MEDS models from **measured dense-grid prevalence**,
using [`meds-random-task-sampler`](https://github.com/florian6973/meds-random-task-sampler)'s
`meds-generate-task-grid` as the measurement instrument.

Overall code frequency is only a prefilter to keep the measurement grid tractable. What a task is
selected on is the realized label prevalence of `(code, duration_days)` at eligible prediction
times — the same code can have ~0% prevalence at 1 day and 30% at 731 days, and only a labeled
grid measures that.

## Quick start

[`run_pipeline.sh`](run_pipeline.sh) runs every step below with 4-core / low-memory settings:

Every generated artifact lands in `work/` inside this repo, which `.gitignore` excludes. Override
with `WORK=...` if you want it elsewhere.

```bash
export MEDS=~/Datasets/MIMIC-IV/MEDS_cohort
export JOBS=4

./run_pipeline.sh counts       # ~5 min, streams the train split
./run_pipeline.sh candidates   # instant
./run_pipeline.sh smoke        # one shard, prints peak RSS and wall time -> extrapolate
./run_pipeline.sh measure      # the measurement grid, JOBS shards at a time
./run_pipeline.sh prevalence
./run_pipeline.sh select       # inspect $WORK/tasks.yaml before continuing
./run_pipeline.sh final
./run_pipeline.sh labels
```

`./run_pipeline.sh all` runs them in sequence. Stop after `select` and read `tasks.yaml`: if fewer
than 10 tasks come out, the command prints per-constraint attrition naming the binding threshold
(most often `--min-uncensored` set above the measurement grid's rows-per-task).

## Pipeline

The equivalent explicit commands. All selection decisions use `train`/`tuning` statistics only —
`held_out` is never used to choose tasks.

```bash
MEDS=~/Datasets/MIMIC-IV/MEDS_cohort
WORK=./work                     # gitignored; anywhere outside the dataset works

# 1. Per-code counts on the train split (writes a new file; never modifies the dataset)
meds-ts-count-codes --data-dir $MEDS --split train --out $WORK/codes_with_counts.parquet

# 2. Candidate panel: stratified head/torso/tail per category + MEDS_DEATH (~300 codes)
meds-ts-candidates --codes $WORK/codes_with_counts.parquet --out $WORK/candidates.yaml

# 3. Measurement grid on the tuning split, union of ICU- and EHR-scale horizons
meds-generate-task-grid data_dir=$MEDS out_dir=$WORK/measure_grid split=tuning \
    "grid.query_codes=$WORK/candidates.yaml" \
    'grid.durations=[1,2,3,7,14,30,90,180,365,731]' \
    grid.subject_subsample_fraction=0.25

# 4. Per-task label statistics
meds-ts-prevalence --grid-dir $WORK/measure_grid --split tuning --out $WORK/task_stats.parquet

# 5. Select the task registry (default: 10 tasks, >=4 short-horizon, >=4 long-horizon).
#    Also writes selected_codes.yaml and selected_durations.txt next to tasks.yaml.
meds-ts-select --stats $WORK/task_stats.parquet --out $WORK/tasks.yaml

# 6. Final grids restricted to the selected axes, then per-task label files
for split in train tuning held_out; do
    meds-generate-task-grid data_dir=$MEDS out_dir=$WORK/final_grid split=$split \
        "grid.query_codes=$WORK/selected_codes.yaml" \
        "grid.durations=[$(cat $WORK/selected_durations.txt)]"
done

meds-ts-labels --tasks $WORK/tasks.yaml --grid-dir $WORK/final_grid --out-dir $WORK/labels
```

Output: `labels/{task_id}/{split}.parquet` with `subject_id, prediction_time, boolean_value` —
one MEDS-style label file per task per split, ready for the template `preprocess_task` /
`supervised_train` stages. Censored rows are dropped by default (`--keep-censored` retains them).

The final grid crosses all selected codes with all selected durations, so it also doubles as a
small dense evaluation grid for zero-shot models; `meds-ts-labels` slices out just the registry's
`(code, duration)` pairs for supervised fine-tuning.

## Selection criteria (`meds-ts-select`)

| Constraint | Default | Rationale |
| --- | --- | --- |
| `--min-uncensored` | 500 | metric stability on the measurement subsample |
| `--min/max-prevalence` | 0.05 / 0.50 | enough positives to fine-tune; excludes near-constant tasks |
| `--max-censor-rate` | 0.80 | task must be observable at that horizon |
| `--target-prevalence` | 0.15 | picks prevalence closest to this, in log space |
| `--max-per-category`, `--max-per-code` | 3 / 1 | diversity across LAB/DIAGNOSIS/PROCEDURE/... |
| `--min-short` / `--min-long` | 4 / 4 | ICU-scale (<=14d) and EHR-scale (>=30d) coverage |

MIMIC caveats: ICD codes are timestamped at discharge, so short-horizon ICD tasks mostly measure
imminent discharge — the prevalence band usually removes them, but review the registry. The head
stratum may contain ICD-9/ICD-10 duplicates of one concept; `--max-per-code` does not catch those,
so check `tasks.yaml` descriptions before freezing it.

## MEDS-DEV baseline (meds_tab/tiny)

```bash
./run_pipeline.sh baseline                        # every task in tasks.yaml
TASKS_TO_RUN="LAB_51146_3d" ./run_pipeline.sh baseline   # or just one
```

which is, per task:

```bash
meds-dev-model \
    model=meds_tab/tiny \
    dataset_dir=$MEDS \
    labels_dir=$WORK/labels/$TASK \
    output_dir=$WORK/baseline \
    dataset_name=MIMIC-IV task_name=$TASK \
    dataset_type=supervised mode=full

meds-dev-evaluation \
    predictions_dir=$WORK/baseline/MIMIC-IV/$TASK/predict \
    output_dir=$WORK/baseline_eval/$TASK
```

`mode=full` runs supervised train then predict, sets the predict split to `held_out` itself (passing
`split=` alongside `mode=full` is an error), and writes to
`$WORK/baseline/{dataset_name}/{task_name}/{train,predict}/`. MEDS-DEV builds its own venv holding
`meds-tab==0.2.0`, so nothing is installed into this project.

**Labels must be sharded** — `meds-ts-labels` defaults to the `sharded` layout for exactly this
reason. MEDS-Tab resolves a shard's labels as
`input_label_dir / shard_fp.relative_to(shard_fp.parents[1])` and `scan_parquet`s that path for
*every* data shard, so `{labels_dir}/{split}/{shard}.parquet` must mirror
`{data_dir}/data/{split}/{shard}.parquet` name for name, including empty files for shards with no
rows. The `flat` layout will not work here. Don't leave both layouts in one directory: MEDS-Tab's
task caching globs `**/*.parquet` and would read the same labels twice.

**PATH matters, and dropping the stray venv is not enough.** `meds-dev-evaluation` shells out to
`meds-evaluation-cli`. With a `uv tool` install, only MEDS-DEV's *own* entry points are symlinked
into `~/.local/bin`; its dependencies' scripts live solely inside the tool environment. So removing
an activated venv from `PATH` leaves the inner command missing, and leaving it there lets an
unrelated copy serve it — which fails with an opaque omegaconf traceback that reads like a
config-override bug. `run_pipeline.sh` does both: strips `$VIRTUAL_ENV/bin` from `PATH` and prepends
MEDS-DEV's tool environment (derived from the `meds-dev-model` shebang).

**Cost warning.** Tabularization is *task-specific* — MEDS-Tab materializes features only at each
task's label prediction times (`label_df` feeds `get_flat_static_rep`), so all 10 tasks each pay a
full tabularize-static + tabularize-time-series pass over every shard, and `meds_tab/tiny` pins
`worker="range(0,1)"` (a single worker). Run one task first and time it before looping.

### Storage and footprint

Tabularized features land in
`$WORK/baseline/{dataset}/{task}/train/meds_tab/tabularize/{split}/{shard}/{window}/{agg}.npz` —
one file per `shard x window_size x agg`, so 366 shards x 2 x 2 = **1,464 npz per task**, 14,640
across ten. `tabularize-static` writes nothing here: `static_aggs` keeps only static aggregations,
and this recipe's `aggs=[code/count,value/sum]` contains none.

**Measured on MIMIC-IV** (`LAB_51146_3d`, 196k label rows): mean npz ~0.16 MB, so **~0.23 GB per
task, ~2.3 GB for all ten** — the dense end of the range, since real ICU 7-day windows touch most
of the top-100 codes. Add ~1.2 GB once for the MEDS-DEV model venv, which lives at
`$WORK/baseline/.venv` and is shared by every task. Storage is not the constraint.

**Time is.** Tabularization ran at ~0.14 npz/s single-worker, i.e. **~3 hours per task** for 1,464
files, so all ten would be ~30 hours serially. The cheapest fix is a cluster: see
[slurm/README.md](slurm/README.md), which runs one array element per task and turns ~30 h serial
into ~3 h wall. Otherwise: run fewer tasks; raise `worker="range(0,N)"` (the recipe pins one joblib
worker — this needs a local model.yaml variant, since the `uv tool` install executes, not the
checkout); or tabularize once and share across tasks, which is sound because all ten share the
*same* sampled prediction times and differ only in the label column and which rows censoring drops.

### Parallelism and memory

`JOBS` bounds *threads inside one process*, not concurrent processes: `POLARS_MAX_THREADS` for
MEDS-Tab's polars and `OMP_NUM_THREADS` for XGBoost. The `meds_tab/tiny` recipe itself pins
`worker="range(0,1)"` (one joblib worker) and `hydra.sweeper.n_jobs=1` (serial hyperparameter
trials), and this script runs tasks one at a time — so a baseline run is essentially single-process.
Peak RSS is measured and printed per task rather than assumed. To trade memory for wall time, run
several tasks concurrently with a smaller `JOBS` each.

## Resource limits

To stay within ~4 cores / 20 GB on a larger machine:

```bash
export POLARS_MAX_THREADS=4   # caps this package's polars thread pool (default: all cores)
```

- `meds-generate-task-grid` is already conservative: it processes shards **sequentially in one
  process**, and the sampler package pins its own polars pool to 1 thread. Peak memory is one
  shard's events plus one shard's grid (well under 1 GB per shard here). To use your 4 cores,
  fan out single-shard invocations instead, e.g.
  `ls $MEDS/data/tuning | sed 's/.parquet//' | xargs -P 4 -I{} meds-generate-task-grid ... input_shard={}`.
- `meds-sample-random-tasks` (training rows, later) defaults `sampling.max_workers` to **all
  cores** — pass `sampling.max_workers=4` explicitly.
- `meds-ts-count-codes` streams the train split (polars streaming engine), so memory stays
  bounded regardless of split size.

## Development

```bash
uv sync --group dev
uv run pytest -v
```

Tests run entirely on synthetic data; no real dataset is required or read. The full runbook was
validated end to end against a synthetic 1,200-subject MEDS dataset (all three splits, 13 codes),
confirming that the final grid reproduces the measured prevalence exactly.
