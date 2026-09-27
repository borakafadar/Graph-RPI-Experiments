import argparse
import glob
import json
import math
import os
from collections import Counter

import torch

EVAL_SPLITS = [f"{part}_unseen_{kind}" for part in ("valid", "test") for kind in ("pair", "protein", "rna")]


def read_rows(path):
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d["interaction_type"] == "rna-protein":
                yield d["RNA_id"], d["target_id"], int(d["interaction_label"])


def read_fasta(path, wanted):
    seqs = {}
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None and name not in seqs:
                    seqs[name] = "".join(buf)
                header = line[1:].strip()
                name = header if header in wanted else None
                buf = []
            elif name is not None:
                buf.append(line.strip())
    if name is not None and name not in seqs:
        seqs[name] = "".join(buf)
    return seqs


def write_shards(order, seqs, fasta_dir, prefix, tag, shards):
    size = math.ceil(len(order) / shards)
    for i in range(shards):
        with open(os.path.join(fasta_dir, f"{prefix}_shard_{i}.fa"), "w") as fh:
            for k in range(i * size, min((i + 1) * size, len(order))):
                fh.write(f">{tag}{k}\n{seqs[order[k]]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciceklab", default="ciceklab")
    parser.add_argument("--out", default="ciceklab_processed")
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()

    data_dir = os.path.join(args.out, "data")
    fasta_dir = os.path.join(args.out, "fasta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fasta_dir, exist_ok=True)

    train_counts = Counter()
    chunk_files = sorted(glob.glob(os.path.join(args.ciceklab, "training_chunks", "chunk_*.jsonl")))
    for f in chunk_files:
        train_counts.update(read_rows(f))
    print(f"Read {len(chunk_files)} training chunks: {sum(train_counts.values())} rna-protein rows")

    eval_rows = {}
    for split in EVAL_SPLITS:
        eval_rows[split] = list(read_rows(os.path.join(args.ciceklab, "data_with_negatives", "rna_protein", f"final_{split}.jsonl")))
        print(f"Read {split}: {len(eval_rows[split])} rows")

    wanted_rna = {r for r, _, _ in train_counts} | {r for rows in eval_rows.values() for r, _, _ in rows}
    wanted_protein = {p for _, p, _ in train_counts} | {p for rows in eval_rows.values() for _, p, _ in rows}
    rna_seqs = read_fasta(os.path.join(args.ciceklab, "rna.fa"), wanted_rna)
    protein_seqs = read_fasta(os.path.join(args.ciceklab, "protein.fa"), wanted_protein)
    print(f"Sequences found: RNA {len(rna_seqs)}/{len(wanted_rna)}, protein {len(protein_seqs)}/{len(wanted_protein)}")

    def available(r, p):
        return r in rna_seqs and p in protein_seqs

    kept = {k: c for k, c in train_counts.items() if available(k[0], k[1])}
    pos = sorted({(r, p) for r, p, y in kept if y == 1})
    neg = sorted({(r, p) for r, p, y in kept if y == 0})
    stats = {
        "train": {
            "rows": sum(train_counts.values()),
            "dropped_rows_missing_sequence": sum(train_counts.values()) - sum(kept.values()),
            "unique_positive_pairs": len(pos),
            "unique_negative_pairs": len(neg),
            "pairs_labelled_both": len(set(pos) & set(neg)),
        }
    }

    eval_kept = {}
    for split, rows in eval_rows.items():
        eval_kept[split] = [row for row in rows if available(row[0], row[1])]
        stats[split] = {
            "rows": len(rows),
            "dropped_rows_missing_sequence": len(rows) - len(eval_kept[split]),
            "dropped_positives": sum(y for r, p, y in rows if not available(r, p)),
            "kept_positives": sum(y for _, _, y in eval_kept[split]),
            "kept_negatives": sum(1 - y for _, _, y in eval_kept[split]),
        }

    rna_train = sorted({r for r, _, _ in kept})
    protein_train = sorted({p for _, p, _ in kept})
    rna_extra = sorted({r for rows in eval_kept.values() for r, _, _ in rows} - set(rna_train))
    protein_extra = sorted({p for rows in eval_kept.values() for _, p, _ in rows} - set(protein_train))
    rna_order = rna_train + rna_extra
    protein_order = protein_train + protein_extra
    ntr, ntp, ner = len(rna_train), len(protein_train), len(rna_extra)
    rna_node = {r: i if i < ntr else ntp + i for i, r in enumerate(rna_order)}
    protein_node = {p: ntr + i if i < ntp else ntr + ner + i for i, p in enumerate(protein_order)}
    stats["nodes"] = {
        "train_rna": ntr,
        "train_protein": ntp,
        "extra_rna": ner,
        "extra_protein": len(protein_extra),
    }

    def to_index(pairs):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([[rna_node[r] for r, _ in pairs], [protein_node[p] for _, p in pairs]], dtype=torch.long)

    graph = {
        "n_train_rna": ntr,
        "n_train_protein": ntp,
        "n_extra_rna": ner,
        "n_extra_protein": len(protein_extra),
        "train_pos": to_index(pos),
        "train_neg": to_index(neg),
        "eval": {
            split: {
                "pairs": to_index([(r, p) for r, p, _ in rows]),
                "labels": torch.tensor([y for _, _, y in rows], dtype=torch.long),
            }
            for split, rows in eval_kept.items()
        },
    }
    torch.save(graph, os.path.join(data_dir, "graph.pt"))

    with open(os.path.join(data_dir, "rna_ids.txt"), "w") as fh:
        fh.write("\n".join(rna_order) + "\n")
    with open(os.path.join(data_dir, "protein_ids.txt"), "w") as fh:
        fh.write("\n".join(protein_order) + "\n")
    write_shards(rna_order, rna_seqs, fasta_dir, "rna", "R", args.shards)
    write_shards(protein_order, protein_seqs, fasta_dir, "protein", "P", args.shards)

    with open(os.path.join(data_dir, "prepare_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
