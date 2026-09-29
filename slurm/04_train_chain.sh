#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:?set CONFIG=paper or CONFIG=code}"
NEG="${NEG:?set NEG=random or NEG=file}"
JOBS="${JOBS:?set JOBS to the number of chained 12 h jobs}"
prev=""
for i in $(seq "$JOBS"); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --export=ALL,CONFIG="$CONFIG",NEG="$NEG",GPU_ID="${GPU_ID:-0}" slurm/04_train.sbatch)
    prev="${prev%%;*}"
    echo "Job $i/$JOBS: $prev"
done
