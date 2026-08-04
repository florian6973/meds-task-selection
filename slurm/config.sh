# Site and run configuration for the meds_tab/tiny baseline. **This is the file you edit.**
#
# Every value is also overridable from the environment, so a one-off does not need an edit:
#   TIME=12:00:00 CPUS=8 ./slurm/submit.sh
#
# Sourced by both `slurm/submit.sh` (login node) and `slurm/job.sbatch` (compute node), so keep it
# free of anything that only works in one of those places.

# ---------------------------------------------------------------------------------------------------
# Where the data is (all paths are cluster-side)
# ---------------------------------------------------------------------------------------------------

#: MEDS dataset sharded by split (data/train/…, data/tuning/…, data/held_out/…).
: "${MEDS_ROOT:=/path/to/MIMIC-IV/MEDS_cohort}"

#: Labels root copied from the workstation: one dir per task, each holding {split}/{shard}.parquet
#: mirroring MEDS_ROOT's shards. See `slurm/README.md` for the rsync command.
: "${LABELS_ROOT:=$PWD/labels}"

#: The task registry that names which tasks to run (task_id per entry).
: "${TASKS_YAML:=$PWD/tasks.yaml}"

#: Root for everything the jobs produce. Each task gets `$RUN_DIR/<task>/`, which keeps their model
#: venvs separate — MEDS-Tab's installer runs `uv venv --clear`, so a venv shared by concurrent array
#: elements would be wiped out from under whichever element started first.
: "${RUN_DIR:=$PWD/runs/baseline}"

#: Only used to name output subdirectories, so it need not match a MEDS-DEV dataset registry entry.
: "${DATASET_NAME:=MIMIC-IV}"

#: Optional shared model venv, built by `slurm/prebuild-venv.sh`. Set this when compute nodes have no
#: network access (MEDS-DEV otherwise builds the venv with `uv` at job time) or to avoid ten copies of
#: the same 1.2 GB environment. Leave empty to let each element build its own under its output_dir.
#: Only ever point this at a venv that is already fully built: MEDS-DEV creates venvs with
#: `uv venv --clear`, so concurrent elements racing to populate one would wipe it out from under
#: each other. Once the install marker exists, they only read it.
: "${VENV_DIR:=}"

# ---------------------------------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------------------------------
# Empty values are omitted from the sbatch command line rather than passed as empty flags, so leaving
# ACCOUNT/QOS/PARTITION unset just means "whatever the cluster defaults to".

: "${ACCOUNT:=}"       # --account
: "${QOS:=}"           # --qos
: "${PARTITION:=}"     # --partition
: "${CPUS:=4}"         # --cpus-per-task; also caps XGBoost and polars threads inside the job
: "${MEM:=32G}"        # --mem
: "${TIME:=08:00:00}"  # --time; one task measured ~3 h on a workstation, so this has headroom
: "${MAX_CONCURRENT:=}" # array throttle, e.g. 4 -> `--array=1-10%4`. Empty means no limit.

#: --gres, e.g. `gpu:1`. Needed only to satisfy a GPU partition's requirements: this baseline does
#: not use a GPU. MEDS-Tab pins XGBoost to `device: cpu` and `nthread: 1`, the `meds_tab/tiny` recipe
#: never overrides either, and the run's real cost is tabularization, which is CPU-bound polars work.
#: Requesting a GPU will not make it faster.
: "${GRES:=}"

#: Any further sbatch flags, word-split verbatim, e.g. `--constraint=avx512 --exclude=node[01-04]`.
: "${SBATCH_EXTRA:=}"

# ---------------------------------------------------------------------------------------------------
# What to run, and how hard
# ---------------------------------------------------------------------------------------------------

#: Which MEDS-DEV model. `random_predictor` has no training step and finishes in seconds — use it to
#: validate the whole labels -> predict -> evaluate loop and to get each task's empirical chance
#: baseline before spending hours on a real model.
: "${MODEL:=meds_tab/tiny}"

#: Tabularization workers. The `meds_tab/tiny` recipe hardcodes ONE (`worker="range(0,1)"`), which is
#: what makes a task take hours. Setting this above 1 runs MEDS-Tab's CLIs directly — the same
#: commands the recipe issues, only with more joblib workers — instead of going through
#: `meds-dev-model`. Workers coordinate through MEDS-Transforms' file lock and skip finished shards,
#: so this is safe. Requires VENV_DIR. Set it to $CPUS.
: "${WORKERS:=1}"

# ---------------------------------------------------------------------------------------------------
# The job environment
# ---------------------------------------------------------------------------------------------------

#: Runs inside the job, before anything else. Whatever puts `meds-dev-model` and `uv` on PATH —
#: a module load, a conda activate, `source ~/.venv/bin/activate`, …
#: Assigned only when *unset* (no colon), so `SETUP=` genuinely means "nothing to do".
: "${SETUP=}"
