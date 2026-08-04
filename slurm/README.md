# Running the baseline on a SLURM cluster

One array element per task. Tabularization measured ~3 h per task single-worker, so ten tasks run in
~3 h wall instead of ~30 h serially.

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

Both `submit.sh` and `job.sbatch` check that a task's label shards exist before doing any work, so a
bad path fails in seconds on the login node rather than hours into an array element.

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

`CPUS` bounds threads *within* an element (XGBoost, polars); the recipe pins joblib to one worker, so
raising it past ~4 buys little. Prefer more concurrent array elements over more cores each.
