#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=GraphRPI_features.txt
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "===== job $SLURM_JOB_ID started $(date '+%F %T') on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
set -ux
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
python Graph-RPI-Model/ciceklab_features.py --processed ciceklab_processed --shards 8 --workers 8
