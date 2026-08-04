#!/usr/bin/env bash
# Build the meds_tab/tiny model venv once, on a node that has network access.
#
# Why this exists: MEDS-DEV builds a model's venv at job time with `uv`, which needs to reach a
# package index.  Many clusters give login nodes network access and compute nodes none, so the venv
# has to be created up front.  Running this also means the ten array elements share one 1.2 GB venv
# instead of building their own.
#
#   ./slurm/prebuild-venv.sh              # builds at $VENV_DIR from slurm/config.sh
#   VENV_DIR=/scratch/me/venv ./slurm/prebuild-venv.sh
#
# Afterwards set VENV_DIR in slurm/config.sh (or export it) and submit as usual.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=config.sh
source slurm/config.sh

die() { echo "slurm/prebuild-venv.sh: $*" >&2; exit 1; }

[ -n "${VENV_DIR:-}" ] || die "VENV_DIR is unset; set it in slurm/config.sh or the environment"
if [ -n "$SETUP" ]; then eval "$SETUP"; fi
command -v uv >/dev/null || die "uv is not on PATH (MEDS-DEV uses it to build model venvs)"
command -v meds-dev-model >/dev/null || die "meds-dev-model is not on PATH after SETUP='$SETUP'"

# Ask the interpreter that actually runs MEDS-DEV where its packaged requirements file lives, rather
# than assuming a checkout is present.
MEDS_DEV_BIN="$(dirname "$(sed -n '1s|^#!||p' "$(command -v meds-dev-model)")")"
REQ="$("$MEDS_DEV_BIN/python" -c "
import importlib.resources as r
print(r.files('MEDS_DEV') / 'models' / 'meds_tab' / 'tiny' / 'requirements.txt')")"
[ -f "$REQ" ] || die "could not locate meds_tab/tiny requirements.txt (got '$REQ')"

# MEDS-DEV skips installation when a marker named for the sha256 of the requirements file is present
# in the venv (see MEDS_DEV.utils.temp_env / file_hash).  Reproduce that name so the jobs reuse this.
HASH="$(sha256sum "$REQ" | cut -d' ' -f1)"

echo "requirements: $REQ"
echo "venv:         $VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
uv venv --python "$("$MEDS_DEV_BIN/python" -c 'import sys; print(sys.executable)')" "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -r "$REQ"
touch "$VENV_DIR/.installed.$HASH.txt"

echo "built. Jobs pointed at VENV_DIR=$VENV_DIR will reuse it instead of reinstalling."
