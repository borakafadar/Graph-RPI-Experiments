#!/bin/bash
#SBATCH --job-name=GraphRPI
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --time=12:00:00
#SBATCH --output=GraphRPI_train.txt
#SBATCH --open-mode=append

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "===== job $SLURM_JOB_ID started $(date '+%F %T') on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphrpi
CONFIG="${CONFIG:?set CONFIG=paper or CONFIG=code}"
NEG="${NEG:?set NEG=random or NEG=file}"
MIN_FREE_MB="${MIN_FREE_MB:-22528}"
CLAIMS="/tmp/graphrpi_gpu_claims_$USER"
mkdir -p "$CLAIMS"
exec 9>"$CLAIMS/lock"
while true; do
    flock 9
    GPU=""
    while IFS=", " read -r idx free; do
        [[ -n "${GPU_ID:-}" && "$idx" != "$GPU_ID" ]] && continue
        owner=$(cat "$CLAIMS/$idx" 2>/dev/null || true)
        [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null && continue
        (( free >= MIN_FREE_MB )) && { GPU="$idx"; break; }
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    [[ -n "$GPU" ]] && echo $$ > "$CLAIMS/$GPU"
    flock -u 9
    [[ -n "$GPU" ]] && break
    echo "$(date '+%F %T') no GPU${GPU_ID:+ $GPU_ID} with ${MIN_FREE_MB} MiB free, retrying in 60 s"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    sleep 60
done
echo "Using GPU $GPU"
set -ux
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
nvidia-smi
python Graph-RPI-Model/main_ciceklab.py --config "$CONFIG" --neg "$NEG" --resume
