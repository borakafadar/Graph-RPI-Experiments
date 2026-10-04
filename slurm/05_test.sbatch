#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=GraphRPI_test.txt
#SBATCH --open-mode=append

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "===== job $SLURM_JOB_ID started $(date '+%F %T') on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
set -ux
RUN="${RUN:?set RUN to a run name, e.g. RUN=paper_random}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
nvidia-smi
python Graph-RPI-Model/ciceklab_test.py --run-dir "ciceklab_processed/runs/$RUN"
