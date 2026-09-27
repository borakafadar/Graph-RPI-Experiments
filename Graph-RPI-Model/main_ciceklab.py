import argparse
import csv
import json
import sys
import time

from Model import *
from ciceklab_common import SPLITS, METRICS, build_model, load_ciceklab

CONFIGS = {
    "code": {"num_epochs": 3000, "alpha": 0.4, "lr_step": False},
    "paper": {"num_epochs": 500, "alpha": 0.5, "lr_step": True},
}


def train_and_validate(model, train_data, feature, valid, optimizer, scheduler, num_epochs, alpha, run_dir):
    best_acc = {s: 0 for s in SPLITS}
    best_state = {s: None for s in SPLITS}
    best_epoch = {s: None for s in SPLITS}
    epoch_seconds = []
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "seconds", "lr"] + [f"valid_{s}_{m}" for s in SPLITS for m in ("AUC", "ACC")])
        for epoch in range(num_epochs):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            model.train()
            model.train_epoch(train_data, optimizer, alpha)
            if scheduler is not None:
                scheduler.step()
            model.eval()
            row = []
            with torch.no_grad():
                z = model.encoder(feature, train_data.edge_index)
                for s in SPLITS:
                    metrics = model.test(z, *valid[f"valid_{s}"])
                    if metrics[2] > best_acc[s]:
                        best_acc[s] = metrics[2]
                        best_state[s] = copy.deepcopy(model.state_dict())
                        best_epoch[s] = epoch + 1
                        torch.save(best_state[s], os.path.join(run_dir, f"best_model_valid_{s}.pth"))
                    row += [metrics[0], metrics[2]]
            epoch_seconds.append(time.time() - start)
            writer.writerow([epoch + 1, f"{epoch_seconds[-1]:.1f}", lr] + [f"{v:.4f}" for v in row])
            fh.flush()
            summary = ", ".join(f"{s} ACC={row[2 * i + 1]:.4f}" for i, s in enumerate(SPLITS))
            print(f"Epoch {epoch + 1:04d} ({epoch_seconds[-1]:.1f}s, lr={lr:.2e}): {summary}", flush=True)
            if epoch == 0 and torch.cuda.is_available():
                print(f"Peak GPU memory after first epoch: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
    mean_seconds = sum(epoch_seconds) / len(epoch_seconds)
    print(f"Mean epoch time: {mean_seconds:.1f}s -> estimated {mean_seconds * CONFIGS['paper']['num_epochs'] / 3600:.1f}h for 'paper', {mean_seconds * CONFIGS['code']['num_epochs'] / 3600:.1f}h for 'code'")
    return best_state, best_epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--neg", choices=["random", "file"], required=True)
    parser.add_argument("--processed", default="ciceklab_processed")
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = dict(CONFIGS[args.config])
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    run_dir = args.run_dir or os.path.join(args.processed, "runs", f"{args.config}_{args.neg}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "args": vars(args), "hyperparameters": config, "lr": 1e-3, "weight_decay": 5e-5, "mask_p": 0.4}, fh, indent=2)
    print(f"Run dir: {run_dir}\nConfig: {config}")

    device = torch.device(args.device)
    set_seed(args.seed)
    feature, train_data, eval_data = load_ciceklab(args.processed, device)

    num_pos_train = train_data.pos_edge_label_index.size(1)
    train_neg = train_data.neg_edge_label_index
    if train_neg.size(1) < num_pos_train:
        indices = torch.randint(0, train_neg.size(1), (num_pos_train,), device=train_neg.device)
        train_neg = train_neg[:, indices]
    elif train_neg.size(1) > num_pos_train:
        indices = torch.randperm(train_neg.size(1))[:num_pos_train]
        train_neg = train_neg[:, indices]

    neg_pool = train_data.neg_edge_label_index if args.neg == "file" else None
    model = build_model(neg_pool).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95) if config["lr_step"] else None

    best_state, best_epoch = train_and_validate(model, train_data, feature, eval_data, optimizer, scheduler, config["num_epochs"], config["alpha"], run_dir)

    results = []
    for s in SPLITS:
        model.load_state_dict(best_state[s])
        model.eval()
        with torch.no_grad():
            z_train = model.encoder(train_data.x, train_data.edge_index)
            train_metrics = model.test(z_train, train_data.pos_edge_label_index, train_neg)
            z = model.encoder(feature, train_data.edge_index)
            valid_metrics = model.test(z, *eval_data[f"valid_{s}"])

        train_auc, train_ap, train_acc, train_sen, train_pre, train_spe, train_f1, train_mcc = train_metrics
        valid_auc, valid_ap, valid_acc, valid_sen, valid_pre, valid_spe, valid_f1, valid_mcc = valid_metrics
        valid_pos_count = eval_data[f"valid_{s}"][0].size(1)
        valid_neg_count = eval_data[f"valid_{s}"][1].size(1)

        print(f"Checkpoint selected on valid_{s} (epoch {best_epoch[s]}):")
        print(f"  Train set: {num_pos_train} positive, {train_neg.size(1)} negative samples")
        print(f"  Valid set: {valid_pos_count} positive, {valid_neg_count} negative samples")
        print(
            f"  Train metrics: ACC={train_acc:.4f}, SEN={train_sen:.4f}, SPE={train_spe:.4f}, MCC={train_mcc:.4f}, F1={train_f1:.4f}, Precision={train_pre:.4f}, AUC={train_auc:.4f}")
        print(
            f"  Valid metrics: ACC={valid_acc:.4f}, SEN={valid_sen:.4f}, SPE={valid_spe:.4f}, MCC={valid_mcc:.4f}, F1={valid_f1:.4f}, Precision={valid_pre:.4f}, AUC={valid_auc:.4f}")

        results.append([s, best_epoch[s], "train", *train_metrics])
        results.append([s, best_epoch[s], "valid", *valid_metrics])

    with open(os.path.join(run_dir, "valid_results.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Selected_on", "Epoch", "Set"] + METRICS)
        writer.writerows(results)
