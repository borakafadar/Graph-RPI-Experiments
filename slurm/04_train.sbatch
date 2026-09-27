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
CONFIG="${CONFIG:?set CONFIG=paper or CONFIG=code}"
NEG="${NEG:?set NEG=random or NEG=file}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
hostname
nvidia-smi
python Graph-RPI-Model/main_ciceklab.py --config "$CONFIG" --neg "$NEG"
