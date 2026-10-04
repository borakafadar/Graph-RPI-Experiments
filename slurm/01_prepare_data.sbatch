#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=GraphRPI_prepare.txt
#SBATCH --open-mode=append

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "===== job $SLURM_JOB_ID started $(date '+%F %T') on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
set -ux
python Graph-RPI-Model/ciceklab_prepare.py --ciceklab ciceklab --out ciceklab_processed --shards 8
