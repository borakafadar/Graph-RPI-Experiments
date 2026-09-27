#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
set -ux
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
hostname
nvidia-smi
python Graph-RPI-Model/main_ciceklab.py --config paper --neg random --epochs 2 --run-dir ciceklab_processed/runs/smoke
