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
sbatch --export=ALL,CONFIG=paper,NEG=random,GPU_ID=0 slurm/04_train.sbatch
sbatch --export=ALL,CONFIG=code,NEG=random,GPU_ID=1 slurm/04_train.sbatch
sbatch --export=ALL,CONFIG=paper,NEG=file,GPU_ID=0 slurm/04_train.sbatch
sbatch --export=ALL,CONFIG=code,NEG=file,GPU_ID=1 slurm/04_train.sbatch
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
   - The `GPU_ID` parameter:
     - `--gres` does not work on this cluster, so choose the GPU with `GPU_ID` (0 or 1).
     - Two jobs that land on the same node must use different `GPU_ID`s.
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
- `slurm/setup_env.sh` and `slurm/01`–`05` job scripts.
- `WALKTHROUGH.md`.

Changed: none. `main.py`, `Model.py`, `RNA_feature.py` and `protein_feature.py` are untouched.
