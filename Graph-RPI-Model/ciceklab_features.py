import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import iFeatureOmegaCLI
import numpy as np

ESM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ESM_Pre_model")
STANDARD = {"rna": set("ACGU"), "protein": set("ACDEFGHIKLMNPQRSTVWY")}


def rna_sequence_type(self):
    for record in self.fasta_list:
        record[1] = record[1].replace("N", "-")
    return iFeatureOmegaCLI.Sequence.check_sequence_type(self)


def protein_sequence_type(self):
    for record in self.fasta_list:
        record[1] = re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "-", record[1])
    return iFeatureOmegaCLI.Sequence.check_sequence_type(self)


iFeatureOmegaCLI.iRNA.check_sequence_type = rna_sequence_type
iFeatureOmegaCLI.iProtein.check_sequence_type = protein_sequence_type


def read_records(path):
    with open(path) as fh:
        lines = fh.read().split("\n")
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def count_records(path):
    with open(path) as fh:
        return sum(1 for line in fh if line.startswith(">"))


def check(feats, fasta):
    n = count_records(fasta)
    if feats.shape[0] != n:
        raise RuntimeError(f"{fasta}: {feats.shape[0]} feature rows for {n} sequences")
    bad = np.where(~np.isfinite(feats).all(axis=1))[0]
    if len(bad):
        raise RuntimeError(f"{fasta}: non-finite features in {len(bad)} records, first: {bad[:20].tolist()}")


def compute_rna(fasta):
    from RNA_feature import generate_features_rna
    feats = generate_features_rna(fasta).astype(np.float32)
    check(feats, fasta)
    return feats


def compute_protein(fasta):
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer, AutoModel
    from Model import read_protein_sequences_from_fasta, generate_features_protein_bert
    from protein_feature import generate_features_protein
    protein_sequences = read_protein_sequences_from_fasta(fasta)
    tokenizer = AutoTokenizer.from_pretrained(ESM_DIR, trust_remote_code=True)
    transformer_model = AutoModel.from_pretrained(ESM_DIR, trust_remote_code=True)
    protein_bert_features = generate_features_protein_bert(protein_sequences, tokenizer, transformer_model)
    protein_new_features = generate_features_protein(fasta)
    feats = np.hstack((protein_bert_features, protein_new_features)).astype(np.float32)
    check(feats, fasta)
    return feats


COMPUTE = {"rna": compute_rna, "protein": compute_protein}


def run_shard(kind, fasta, out):
    if os.path.exists(out):
        return out, None
    start = time.time()
    feats = COMPUTE[kind](fasta)
    tmp = out[:-4] + ".tmp.npy"
    np.save(tmp, feats)
    os.replace(tmp, out)
    return out, time.time() - start


def write_probe(kind, shards, path):
    records = [rec for shard in shards for rec in read_records(shard)]
    by_length = sorted(records, key=lambda rec: len(rec[1]))
    unusual = [rec for rec in records if set(rec[1]) - STANDARD[kind]]
    picked = {}
    for rec in records[:50] + unusual + by_length[:20] + by_length[-3:]:
        picked[rec[0]] = rec[1]
    with open(path, "w") as fh:
        for header, seq in picked.items():
            fh.write(f"{header}\n{seq}\n")
    print(f"Probe {kind}: {len(picked)} sequences, {len(unusual)} with non-standard letters, lengths {len(by_length[0][1])}..{len(by_length[-1][1])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="ciceklab_processed")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    fasta_dir = os.path.join(args.processed, "fasta")
    feat_dir = os.path.join(args.processed, "features")
    shard_dir = os.path.join(feat_dir, "shards")
    probe_dir = os.path.join(feat_dir, "probe")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)

    shards = {kind: [os.path.join(fasta_dir, f"{kind}_shard_{i}.fa") for i in range(args.shards)] for kind in ("rna", "protein")}

    for kind in ("rna", "protein"):
        probe_fasta = os.path.join(probe_dir, f"{kind}_probe.fa")
        write_probe(kind, shards[kind], probe_fasta)
        start = time.time()
        feats = COMPUTE[kind](probe_fasta)
        print(f"Probe {kind} OK: shape {feats.shape}, {time.time() - start:.1f}s")

    tasks = [(kind, fasta, os.path.join(shard_dir, f"{kind}_shard_{i}.npy")) for kind in ("rna", "protein") for i, fasta in enumerate(shards[kind])]
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, *task): task for task in tasks}
        for future in as_completed(futures):
            kind, fasta, out = futures[future]
            try:
                _, seconds = future.result()
                print(f"Done {out}" + ("" if seconds is None else f" in {seconds / 3600:.2f}h"), flush=True)
            except Exception as e:
                errors.append(fasta)
                print(f"FAILED {fasta}: {e!r}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} shards failed: {errors}")

    for kind, ids_file in (("rna", "rna_ids.txt"), ("protein", "protein_ids.txt")):
        feats = np.concatenate([np.load(os.path.join(shard_dir, f"{kind}_shard_{i}.npy")) for i in range(args.shards)])
        with open(os.path.join(args.processed, "data", ids_file)) as fh:
            n_ids = sum(1 for line in fh if line.strip())
        if feats.shape[0] != n_ids:
            raise SystemExit(f"{kind}: {feats.shape[0]} feature rows for {n_ids} ids")
        np.save(os.path.join(feat_dir, f"{kind}_features.npy"), feats)
        print(f"Saved {kind}_features.npy with shape {feats.shape}")


if __name__ == "__main__":
    main()
