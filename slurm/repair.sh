#!/usr/bin/env bash
# Delete tabularized files that a cancelled job left half-written, so a resumed run regenerates them.
#
#   ./slurm/repair.sh            # report only
#   ./slurm/repair.sh --delete   # actually remove the bad ones
#
# Why this is needed: MEDS-Transforms decides a shard is done by `Path.is_file()` for .npz outputs
# (only .parquet gets a real completeness check).  A worker SIGKILLed by `scancel` mid-write leaves a
# truncated .npz that every later run will happily skip, so the corruption is permanent and silent.
# Stale `.lock` files, by contrast, are harmless -- filelock uses an OS-level lock that the kernel
# releases when the process dies -- but they are removed here too, since they only add confusion.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=config.sh
source slurm/config.sh

DELETE=0
[ "${1:-}" = "--delete" ] && DELETE=1

PY_BIN=python3
if command -v meds-dev-model >/dev/null; then
    PY_BIN="$(dirname "$(sed -n '1s|^#!||p' "$(command -v meds-dev-model)")")/python"
fi

locks=$(find "$RUN_DIR" -name '*.lock' 2>/dev/null | wc -l)
echo "stale lock files: $locks (harmless; the kernel released the underlying lock)"
if [ "$DELETE" -eq 1 ] && [ "$locks" -gt 0 ]; then
    find "$RUN_DIR" -name '*.lock' -delete
fi

"$PY_BIN" - "$RUN_DIR" "$DELETE" <<'PY'
import sys, zipfile
from pathlib import Path

run_dir, delete = Path(sys.argv[1]), sys.argv[2] == "1"
bad = []
files = sorted(run_dir.rglob("*.npz"))
for fp in files:
    try:
        # An .npz is a zip archive; testzip() reads every member and catches truncation, which a
        # bare np.load() would not surface until the array is actually touched.
        with zipfile.ZipFile(fp) as zf:
            if zf.testzip() is not None:
                raise zipfile.BadZipFile("member CRC mismatch")
    except Exception as e:
        bad.append((fp, type(e).__name__))

print(f"checked {len(files)} npz, {len(bad)} unreadable")
for fp, err in bad:
    print(f"  {'deleting' if delete else 'BAD'} {fp} ({err})")
    if delete:
        fp.unlink()
if bad and not delete:
    print("\nre-run with --delete, then resubmit; the resumed job regenerates them")
PY
