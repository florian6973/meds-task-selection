# Running the baseline on a SLURM cluster

One array element per task. Tabularization measured ~3 h per task single-worker, so ten tasks run in
~3 h wall instead of ~30 h serially.

## 0. Install MEDS-DEV on the cluster

You do **not** need to clone MEDS-DEV. It ships console scripts, and `uv tool install` puts them on
PATH in their own isolated environment — clone only if you intend to edit a recipe (e.g. raising
`worker="range(0,N)"` in `meds_tab/tiny/model.yaml`), since the installed copy is what executes.

Install from **git**, not PyPI: the released version lags the repository (0.0.14 versus the
0.0.15.dev main-branch build these scripts were validated against).

```bash
# Home quotas on HPC are small and uv's cache is not; keep both in group space.
export UV_CACHE_DIR=/groups/mm6677_gp/$USER/.cache/uv
export UV_TOOL_DIR=/groups/mm6677_gp/$USER/.local/uv-tools
export UV_TOOL_BIN_DIR=/groups/mm6677_gp/$USER/.local/bin

python -m pip install --user uv          # skip if uv is already available
uv tool install git+https://github.com/Medical-Event-Data-Standard/MEDS-DEV.git

export PATH="$UV_TOOL_BIN_DIR:$PATH"
meds-dev-model --help | head -3          # verify
```

The tool environment is ~540 MB. Pin the exact build with
`uv tool install "meds-dev @ git+https://github.com/Medical-Event-Data-Standard/MEDS-DEV.git@<sha>"`
if you want the array reproducible later.

Then set `SETUP` in `slurm/config.sh` so compute nodes see it:

```bash
: "${SETUP=export PATH=/groups/mm6677_gp/$USER/.local/bin:$PATH}"
```

`uv` itself is only needed on the login node: with a prebuilt `VENV_DIR` (step 3) the jobs skip
installation entirely, so they need nothing but the MEDS-DEV scripts.

## 1. Copy the labels and the registry

Only the labels and `tasks.yaml` need to move — the MEDS data is already on the cluster. The labels
are ~24 MB for ten tasks (one small parquet per data shard per split).

```bash
rsync -avz --info=progress2 work/labels/ CLUSTER:/scratch/$USER/meds-task-selection/labels/
rsync -avz work/tasks.yaml            CLUSTER:/scratch/$USER/meds-task-selection/
rsync -avz slurm/                     CLUSTER:/scratch/$USER/meds-task-selection/slurm/
```

The label shard names must match the cluster's MEDS shards exactly (`{split}/{shard}.parquet`), so
copy the whole tree rather than individual files. If the cluster's copy of MIMIC-IV was sharded
differently from the workstation's, regenerate labels there instead — `meds-ts-labels` needs only
`final_grid`, not the raw data.

## 1b. Get the code onto the cluster

The labels are data and travel by rsync; everything else is in git. If the destination directory
already holds only the rsynced `work/`, attach it to the remote rather than cloning over it —
`work/` is gitignored, so checkout leaves it alone:

```bash
cd /groups/mm6677_gp/ffp2106/Git/meds-task-selection
git init -q
git remote add origin https://github.com/florian6973/meds-task-selection.git
git fetch -q origin && git checkout -B main origin/main
```

## 2. Point `slurm/config.sh` at the cluster

Edit `MEDS_ROOT`, `LABELS_ROOT`, `TASKS_YAML`, `RUN_DIR`, and `SETUP` (whatever puts `meds-dev-model`
and `uv` on PATH — a module load, a conda activate, or nothing if `meds-dev` is a `uv tool` install
already on PATH). Scheduler knobs — `ACCOUNT`, `QOS`, `PARTITION`, `CPUS`, `MEM`, `TIME`,
`MAX_CONCURRENT` — are all overridable from the environment for one-offs.

## 3. Build the model venv once (recommended)

MEDS-DEV builds a model's venv at job time with `uv`, which needs a package index. Compute nodes
frequently have no outbound network, so build it on the login node first — this also lets all ten
elements share one 1.2 GB environment instead of building their own:

```bash
export VENV_DIR=$PWD/work/runs/meds-tab-venv
./slurm/prebuild-venv.sh
```

Then keep `VENV_DIR` exported (or set it in `config.sh`) when submitting. Jobs log
`Requirements already installed in …` and skip the build. Skip this step only if you know compute
nodes can reach PyPI.

## 4. Submit

```bash
./slurm/submit.sh --dry-run                 # prints the sbatch line, submits nothing
./slurm/submit.sh --tasks LAB_51146_3d      # just one, to validate before the full array
./slurm/submit.sh                           # every task in tasks.yaml
squeue -u $USER                             # watch; logs land in slurm-logs/
```

Start with one task. It validates the whole chain — venv build, tabularization, XGBoost, evaluation
— for the cost of a single element, and tells you the real per-task time and memory on your
hardware.

For the group-space layout used on the cluster, the complete 16-core submission is:

```bash
cd /groups/mm6677_gp/ffp2106/Git/meds-task-selection

# Preview the ten-element array without submitting it.
VENV_DIR="$PWD/work/runs/meds-tab-venv" CPUS=16 WORKERS=16 \
    ./slurm/submit.sh --dry-run

# Submit all ten tasks from work/tasks.yaml (assuming TASKS_YAML is configured accordingly).
VENV_DIR="$PWD/work/runs/meds-tab-venv" CPUS=16 WORKERS=16 \
    ./slurm/submit.sh
```

This requests 16 CPUs for **each** array element, so ten elements can consume up to 160 CPUs at
once. Add `MAX_CONCURRENT=2`, for example, to run at most two elements concurrently. `WORKERS=16`
requires the already-built `VENV_DIR`; without `WORKERS=16`, merely allocating 16 CPUs provides
little benefit because the stock recipe uses one tabularization worker.

The command uses `PARTITION` from `slurm/config.sh`, or the site's default partition when it is
empty. `PARTITION=gpu GRES=gpu:1` explicitly requests a GPU partition, but this baseline is
CPU-only and the GPU will sit idle; use that only when site policy requires the GPU partition.

Both `submit.sh` and `job.sbatch` check that a task's label shards exist before doing any work, so a
bad path fails in seconds on the login node rather than hours into an array element.

## Making it faster

The `meds_tab/tiny` recipe hardcodes `worker="range(0,1)"` — a single tabularization worker — which
is what makes a task take hours. `WORKERS` lifts that:

```bash
CPUS=8 WORKERS=8 ./slurm/submit.sh
```

Above 1, the job issues MEDS-Tab's CLIs directly with that many joblib workers instead of going
through `meds-dev-model`. They are the same commands the recipe runs, with only the worker count
changed; workers coordinate through MEDS-Transforms' file lock and skip shards another worker
finished. Verified to produce identical metrics to the `meds-dev-model` path on the same input.
Requires a prebuilt `VENV_DIR`, and the job refuses to start without one.

Tabularization is I/O- and CPU-bound per shard, so scaling is close to linear until the filesystem
saturates: 8 workers should turn ~3 h into well under an hour. Raise `CPUS` alongside `WORKERS`,
since joblib runs them as processes inside the element's allocation.

### A quicker model

```bash
MODEL=random_predictor ./slurm/submit.sh
```

`random_predictor` has no training step and finishes in **seconds per task**. It is not a real
baseline — it draws uniform random probabilities — but it validates the whole
labels -> predict -> evaluate loop across all ten tasks immediately, and its average precision is
each task's empirical chance floor, which is the number you want to compare meds-tab against.
Run this first; it costs nothing and tells you whether anything downstream is misconfigured before
you spend hours.

`VENV_DIR` is deliberately ignored for models other than `meds_tab/tiny`: another model's
requirements hash would not match, and MEDS-DEV's response to a mismatch is to delete the venv,
which would destroy the shared MEDS-Tab environment.

## Tracking progress

```bash
./slurm/status.sh              # one line per task, plus the current Slurm queue
watch -n 60 ./slurm/status.sh  # refresh every minute
squeue -u "$USER"              # scheduler state only
```

```
TASK                                       CATEGORY           PREV   CENSOR TABULARIZE         RESULT
LAB_51146_3d                               LAB               12.4%     3.1% 114/1464 (7%)      -
INFUSION_START_220949_180d                 INFUSION          15.1%    10.8% 1464/1464 (100%)   training xgboost
DIAGNOSIS_ICD_10_E785_365d                 DIAGNOSIS         13.8%    22.0% 1464/1464 (100%)   AUROC 0.7312  AP 0.2841
```

SLURM alone reports only `RUNNING` for hours. Tabularization writes one npz per
`shard x window x agg`, so that file count is the pipeline's one fine-grained progress signal;
later stages are inferred from the artifacts they leave. Category, prevalence, and censoring come
from `tasks.yaml`; the rates are the tuning-subsample measurements used for task selection, not
live statistics from model training. `status.sh` appends `squeue` output when it is available.

Other useful views:

```bash
tail -f slurm-logs/meds-tab-baseline-*_1.out       # live log for array element 1
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem  # after the fact: real peak memory
sacct -j <jobid> --format=JobID,State,ExitCode | grep -v COMPLETED   # what failed
```

`MaxRSS` from `sacct` is the number to trust when tuning `MEM` for later runs — each job also runs
under `/usr/bin/time -v`, so its peak RSS is at the end of the element's log.

## Cancelling and restarting

Resubmitting **resumes**, it does not start over — but do one cleanup step first:

```bash
./slurm/repair.sh              # report half-written files
./slurm/repair.sh --delete     # remove them
./slurm/submit.sh --skip-done  # requeue only tasks without a results.json
```

What actually happens per stage:

| Stage | On restart |
| --- | --- |
| `meds-tab-describe`, tabularize-static | skipped if output exists |
| tabularize-time-series | **resumes** — each existing npz is skipped (`… exists; returning`) |
| XGBoost | re-runs from scratch (minutes; writes a new timestamped results dir) |
| via `meds-dev-model` (`WORKERS=1`) | a stage that fully succeeded is skipped via its `.done` sentinel |

Two details behind `repair.sh`:

*Stale `.lock` files are harmless.* MEDS-Transforms locks with `filelock`, which takes an OS-level
lock the kernel drops when the process dies. Verified: after `kill -9` on a lock holder, the lock
file remains on disk but is immediately re-acquirable. They are deleted only to reduce confusion.

*Truncated `.npz` files are not harmless.* MEDS-Transforms decides a shard is done with
`Path.is_file()` for `.npz` — only `.parquet` gets a real completeness check. A worker killed
mid-write leaves a partial file that every later run skips, so the corruption is silent and
permanent. `repair.sh` opens each npz as the zip archive it is and verifies every member, then
deletes the unreadable ones so a resumed run regenerates them. A `scancel` can damage at most one
file per active worker, so this is quick to fix but worth always doing after an interrupted run.

## Layout and sizing

```
$RUN_DIR/<task>/.venv                              MEDS-DEV's model venv, ~1.2 GB
$RUN_DIR/<task>/<dataset>/<task>/train/meds_tab/   tabularized features, ~0.23 GB
$RUN_DIR/<task>/<dataset>/<task>/predict/          predictions.parquet
$RUN_DIR/eval/<task>/results.json                  AUROC, average precision, calibration error
```

Budget roughly **1.5 GB per task**, so ~15 GB for ten.

Each task gets its own `output_dir`, and therefore its own venv, on purpose: MEDS-DEV builds model
venvs with `uv venv --clear`, which wipes the target unconditionally. Concurrent array elements
sharing one venv would delete it out from under each other. The cost is ~1.2 GB and a couple of
minutes of install per element (uv's download cache is shared, so only the first pays full price).
To trade disk for that time, pre-build one venv, let a single warm-up element populate its
`.installed.<hash>.txt` marker, then point every element at it with `venv_dir=` — `temp_env` skips
installation when that marker matches, but only once it exists, so the warm-up must complete first.

`CPUS` bounds threads *within* an element; the recipe pins joblib to one worker and MEDS-Tab pins
XGBoost to `nthread: 1`, so beyond polars this buys little past ~4. Prefer more concurrent array
elements over more cores each.

### GPU partitions

**This baseline does not use a GPU.** MEDS-Tab's XGBoost defaults to `device: cpu` with
`nthread: 1`, and `meds_tab/tiny` never overrides either; the run's cost is dominated by
tabularization, which is CPU-bound polars work. A GPU would sit idle.

Submit to a GPU partition only if that is where your allocation or the short queue is:

```bash
PARTITION=gpu GRES=gpu:1 ./slurm/submit.sh
```

`SBATCH_EXTRA` passes anything else through verbatim (`--constraint=…`, `--exclude=…`). Actually
using a GPU would mean overriding `model_launcher.model.device=cuda` in the `meds-tab-xgboost`
call, which lives in MEDS-DEV's `meds_tab/tiny/model.yaml` — so it needs an editable install of
MEDS-DEV, and would still only touch the minutes spent training, not the hours spent tabularizing.
