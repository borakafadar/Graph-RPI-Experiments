#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:?set CONFIG=paper or CONFIG=code}"
NEG="${NEG:?set NEG=random or NEG=file}"
JOBS="${JOBS:?set JOBS to the number of chained 12 h jobs}"
prev=""
for i in $(seq "$JOBS"); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --output="GraphRPI_train_${CONFIG}_${NEG}.txt" --export=ALL,CONFIG="$CONFIG",NEG="$NEG" slurm/04_train.sbatch)
    prev="${prev%%;*}"
    echo "Job $i/$JOBS: $prev"
done
