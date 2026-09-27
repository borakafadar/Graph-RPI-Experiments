#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n graphrpi python=3.10
conda activate graphrpi
set -ux
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt
huggingface-cli download facebook/esm2_t6_8M_UR50D --local-dir Graph-RPI-Model/ESM_Pre_model
python -c "import torch, torch_sparse, torch_geometric, iFeatureOmegaCLI; print(torch.__version__, torch.version.cuda, torch_geometric.__version__)"
