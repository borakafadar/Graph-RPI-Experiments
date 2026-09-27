import argparse
import csv
import json
import os

import torch

from ciceklab_common import SPLITS, METRICS, build_model, load_ciceklab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processed", default="ciceklab_processed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = os.path.join(args.run_dir, "test_results.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: the test set has already been evaluated for this run")

    device = torch.device(args.device)
    feature, train_data, eval_data = load_ciceklab(args.processed, device)
    with open(os.path.join(args.processed, "data", "prepare_stats.json")) as fh:
        stats = json.load(fh)

    model = build_model().to(device)
    rows = []
    for s in SPLITS:
        model.load_state_dict(torch.load(os.path.join(args.run_dir, f"best_model_valid_{s}.pth"), map_location=device))
        model.eval()
        with torch.no_grad():
            z = model.encoder(feature, train_data.edge_index)
            metrics = model.test(z, *eval_data[f"test_{s}"])
        pos, neg = eval_data[f"test_{s}"]
        dropped = stats[f"test_{s}"]["dropped_rows_missing_sequence"]
        auc, ap, acc, sen, pre, spe, f1, mcc = metrics
        print(f"test_{s} ({pos.size(1)} positive, {neg.size(1)} negative, {dropped} rows dropped for missing sequences):")
        print(f"  Test metrics:  ACC={acc:.4f}, SEN={sen:.4f}, SPE={spe:.4f}, MCC={mcc:.4f}, F1={f1:.4f}, Precision={pre:.4f}, AUC={auc:.4f}")
        rows.append([f"test_{s}", f"valid_{s}", pos.size(1), neg.size(1), dropped, *metrics])

    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Test_split", "Selected_on", "Positives", "Negatives", "Dropped_missing_sequence"] + METRICS)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
