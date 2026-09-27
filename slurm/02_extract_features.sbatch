#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=8

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
set -ux
hostname
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
python Graph-RPI-Model/ciceklab_features.py --processed ciceklab_processed --shards 8 --workers 8
