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
hostname
python Graph-RPI-Model/ciceklab_prepare.py --ciceklab ciceklab --out ciceklab_processed --shards 8
