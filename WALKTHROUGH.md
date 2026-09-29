# Graph-RPI on the ciceklab dataset — walkthrough

This guide reproduces Graph-RPI (`GraphRPI.md`) on the ciceklab RNA–protein data using the lab cluster.
The model, loss, masking and metrics come from `Graph-RPI-Model/Model.py`, which is unchanged. Only the data layer is new.

## What is run

| Item | Setting |
|---|---|
| Training data | `rna-protein` rows of `ciceklab/training_chunks/` only (rna-rna rows are skipped). Positive pairs are deduplicated and used as graph edges. |
| Validation / test | `ciceklab/data_with_negatives/rna_protein/final_{valid,test}_unseen_{pair,protein,rna}.jsonl`, negatives used as given. `rna_rna/` and `novel_rna_clusters/` contain no rna-protein interactions. |
| Missing sequences | Rows whose RNA or protein ID is not in `rna.fa` / `protein.fa` are dropped (as the original `build_edge_index` does). The dropped counts are in `ciceklab_processed/data/prepare_stats.json` and in `test_results.csv`. |
| Graph | Inductive. The training graph contains only nodes that appear in the training rows. At evaluation, val/test nodes are added as isolated nodes and message passing uses training edges only. To fit in the 24 GB of an RTX 4090, only the training nodes' features stay on the GPU. Val/test node features stay in CPU memory and are encoded in chunks as isolated nodes (`encode_all` in `ciceklab_common.py`). This gives the same embeddings as encoding the full graph at once, up to float32 rounding. The training jobs also set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to reduce GPU memory fragmentation. |
| Features | Original `generate_features_rna`, `generate_features_protein` and ESM-2 (`esm2_t6_8M_UR50D`) mean embeddings, zero-padded to 909 as in `main.py`. |
| Non-standard letters | iFeatureOmega cannot handle some letters. A single RNA with `N` makes DPCP/PseDNC/PCPseDNC/PseKNC fail for the whole file. A protein file that contains both `U` and `X`/`Z`/`B` gets its type detected as "Unknown", and PAAC/QSOrder/GTPC/GDPC/DistancePair then fail. `ciceklab_features.py` therefore masks these letters as gaps (`-`) before the library's own type check runs: `N` in RNA, everything outside the 20 standard amino acids in proteins. The library ignores gaps. For proteins this is what the library already does when it detects the type correctly. The library still converts U→T internally, as before. On files without these letters the features are identical to the unmodified library. ESM-2 and DDE still see the raw protein sequences. |
| Model selection | As in `main.py` (best ACC over epochs), but on validation instead of test. One checkpoint per validation split; each test split is scored with the checkpoint chosen on its matching validation split. Threshold unchanged (decoder output ≥ 0.5). |
| Hyperparameter sets | `code`: 3000 epochs, α = 0.4, no LR schedule (values in `main.py`). `paper`: 500 epochs, α = 0.5, LR × 0.95 every 5 epochs (values in the paper). Both use lr 1e-3, weight decay 5e-5, mask p = 0.4. |
| Negatives in training | `random`: original random negatives (main result). `file`: our training negatives sampled per epoch instead (extra run, via the `RPIFileNegatives` subclass). |
| Seed | 2024 (the original uses 2023 + fold; this is fold 1's seed). Change with `--seed`. |
| Device | GPU (the original runs on CPU; results are not bit-identical across devices). |

## 0. Environment (once, on the login node `leylak`)

```bash
cd Graph-RPI-Experiments
bash slurm/setup_env.sh
```

This creates the conda env `graphrpi` (Python 3.10) in `~/miniconda3` and installs torch 2.0.0 (CUDA 11.8), torch-scatter and torch-sparse, then `requirements.txt`. It also downloads ESM-2 into `Graph-RPI-Model/ESM_Pre_model`.

The `ciceklab/` dataset folder must be at the repo root.

## 1. Submit the jobs

Submit every job **from the repo root**. Logs are written there as `GraphRPI_<jobid>.txt`. Run the steps in order and check each log before moving on.

```bash
sbatch slurm/01_prepare_data.sbatch
sbatch slurm/02_extract_features.sbatch
sbatch slurm/03_train_smoke.sbatch
CONFIG=paper NEG=random JOBS=2 bash slurm/04_train_chain.sh
CONFIG=code NEG=random JOBS=10 bash slurm/04_train_chain.sh
CONFIG=paper NEG=file JOBS=2 bash slurm/04_train_chain.sh
CONFIG=code NEG=file JOBS=10 bash slurm/04_train_chain.sh
sbatch --export=ALL,RUN=paper_random,GPU_ID=0 slurm/05_test.sbatch
```

What each step does:

1. **`01_prepare_data`** (CPU) reads the JSONL files and FASTAs. It writes the graph indices, ID lists, 8 FASTA shards per molecule type and `prepare_stats.json`.
2. **`02_extract_features`** (CPU, 8 processes)
   - First runs a probe on the first sequences, all sequences with non-standard letters, and the shortest and longest sequences. It stops with an error if a descriptor fails or yields NaN.
   - Then extracts the 16 shards in parallel and merges them.
   - Finished shards are kept, so a resubmitted job continues where it stopped.
3. **`03_train_smoke`** runs 2 epochs of the `paper` / `random` setting. The log shows the peak GPU memory, the time per epoch and the estimated hours for both hyperparameter sets. Check that the run fits within the 12 h limit before submitting step 4.
4. **`04_train`** runs one training per `CONFIG` × `NEG` combination.
   - A full run does not fit in one 12 h job (about 124 s per epoch on job 13289: `paper` ≈ 17 h, `code` ≈ 103 h). The run is split into chained jobs:
     - `04_train_chain.sh` submits `JOBS` copies of `04_train.sbatch`, each with `--dependency=afterany` on the previous one.
     - After every epoch, `main_ciceklab.py` saves `last_state.pt`: model, optimizer, LR scheduler, best checkpoints so far, and the Python/NumPy/Torch/CUDA RNG states.
     - `04_train.sbatch` passes `--resume`, so each job continues after the last saved epoch. A job stopped by the time limit loses at most the epoch in progress. `history.csv` is cut back to the saved epoch before training continues.
     - A resumed run gives the same history, checkpoints and `valid_results.csv` as an uninterrupted one (checked on a small graph, CPU and GPU, `random` and `file`).
     - Jobs left over after the run has finished exit immediately.
   - `--resume` refuses to start in a run directory that has epochs in `history.csv` but no `last_state.pt`, e.g. a run from before resume support. Move that directory away first. A `history.csv` with only the header (a job that crashed during its first epoch) does not block; the run starts from epoch 1.
   - GPU choice:
     - Slurm on this cluster does not manage GPUs (no GRES), so jobs of other users share the same cards. A run needs about 19 GiB of the 24 GiB card (peak 18.79 GiB on job 13289), so any other process holding more than about 4.5 GiB on that card causes CUDA OOM.
     - `04_train.sbatch` therefore picks, at start, a GPU with at least `MIN_FREE_MB` (default 20480) MiB free. If none has, it prints the GPU usage and retries every 60 s. The wait counts towards the 12 h limit.
     - Two of your own jobs on the same node never pick the same GPU: the chosen GPU is claimed in `/tmp/graphrpi_gpu_claims_$USER/` on the node for as long as the job runs.
     - Set `GPU_ID=0` or `GPU_ID=1` to only consider that GPU.
     - Another process can still take memory after the job has started. The job then fails with CUDA OOM and the next job in the chain resumes after the last finished epoch.
5. **`05_test`** scores the test splits once per run (`RUN` = `<config>_<neg>`).
   - It refuses to run again if `test_results.csv` already exists.
   - Run it only after model selection is final.

## Outputs

```
ciceklab_processed/
  data/graph.pt, rna_ids.txt, protein_ids.txt, prepare_stats.json
  fasta/{rna,protein}_shard_{0..7}.fa
  features/rna_features.npy, protein_features.npy, shards/, probe/
  runs/<config>_<neg>/
    config.json                         arguments and hyperparameters
    history.csv                         per-epoch time, lr, validation AUC/ACC
    best_model_valid_unseen_{pair,protein,rna}.pth
    last_state.pt                       state after the last finished epoch, used by --resume
    valid_results.csv                   train and validation metrics of each selected checkpoint
    test_results.csv                    written by 05_test
  runs/smoke/                           output of 03_train_smoke
```

## Files added or changed relative to the original code

Added:
- `Graph-RPI-Model/ciceklab_prepare.py`: JSONL/FASTA → graph indices and FASTA shards.
- `Graph-RPI-Model/ciceklab_features.py`: parallel feature extraction with the original feature functions.
- `Graph-RPI-Model/ciceklab_common.py`: data loader, model builder, and the `RPIFileNegatives` subclass for the `file` runs.
- `Graph-RPI-Model/main_ciceklab.py`: training driver (the counterpart of `main.py`).
- `Graph-RPI-Model/ciceklab_test.py`: one-time test evaluation.
- `slurm/setup_env.sh`, `slurm/01`–`05` job scripts and `slurm/04_train_chain.sh`.
- `WALKTHROUGH.md`.

Changed: none. `main.py`, `Model.py`, `RNA_feature.py` and `protein_feature.py` are untouched.
