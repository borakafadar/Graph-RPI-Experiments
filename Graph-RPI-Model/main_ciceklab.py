import argparse
import csv
import json
import sys
import time

from Model import *
from ciceklab_common import SPLITS, METRICS, build_model, encode_all, load_ciceklab

CONFIGS = {
    "code": {"num_epochs": 3000, "alpha": 0.4, "lr_step": False},
    "paper": {"num_epochs": 500, "alpha": 0.5, "lr_step": True},
}


def save_state(path, epoch, model, optimizer, scheduler, best_acc, best_state, best_epoch, epoch_seconds):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "best_acc": best_acc,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "epoch_seconds": epoch_seconds,
        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_state(state, model, optimizer, scheduler):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    py_rng, np_rng, torch_rng, cuda_rng = state["rng"]
    random.setstate(py_rng)
    np.random.set_state(np_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


def train_and_validate(model, train_data, extra_x, valid, optimizer, scheduler, num_epochs, alpha, run_dir, state):
    best_acc = {s: 0 for s in SPLITS}
    best_state = {s: None for s in SPLITS}
    best_epoch = {s: None for s in SPLITS}
    epoch_seconds = []
    start_epoch = 0
    if state is not None:
        load_state(state, model, optimizer, scheduler)
        start_epoch, best_acc, best_state, best_epoch, epoch_seconds = state["epoch"], state["best_acc"], state["best_state"], state["best_epoch"], state["epoch_seconds"]
        for s in SPLITS:
            if best_state[s] is not None:
                torch.save(best_state[s], os.path.join(run_dir, f"best_model_valid_{s}.pth"))
    history = os.path.join(run_dir, "history.csv")
    rows = [["epoch", "seconds", "lr"] + [f"valid_{s}_{m}" for s in SPLITS for m in ("AUC", "ACC")]]
    if start_epoch > 0:
        with open(history, newline="") as fh:
            rows += [row for row in list(csv.reader(fh))[1:] if int(row[0]) <= start_epoch]
    state_path = os.path.join(run_dir, "last_state.pt")
    with open(history, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            model.train()
            model.train_epoch(train_data, optimizer, alpha)
            if scheduler is not None:
                scheduler.step()
            model.eval()
            row = []
            with torch.no_grad():
                z = encode_all(model, train_data, extra_x)
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
            save_state(state_path, epoch + 1, model, optimizer, scheduler, best_acc, best_state, best_epoch, epoch_seconds)
            summary = ", ".join(f"{s} ACC={row[2 * i + 1]:.4f}" for i, s in enumerate(SPLITS))
            print(f"Epoch {epoch + 1:04d} ({epoch_seconds[-1]:.1f}s, lr={lr:.2e}): {summary}", flush=True)
            if epoch == start_epoch and torch.cuda.is_available():
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = dict(CONFIGS[args.config])
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    run_dir = args.run_dir or os.path.join(args.processed, "runs", f"{args.config}_{args.neg}")
    os.makedirs(run_dir, exist_ok=True)
    state = None
    if args.resume and os.path.exists(os.path.join(run_dir, "last_state.pt")):
        state = torch.load(os.path.join(run_dir, "last_state.pt"), map_location="cpu")
        if state["epoch"] >= config["num_epochs"] and os.path.exists(os.path.join(run_dir, "valid_results.csv")):
            print(f"{run_dir} already finished {state['epoch']} epochs, nothing to do")
            sys.exit(0)
        print(f"Resuming {run_dir} after epoch {state['epoch']}")
    elif args.resume and os.path.exists(os.path.join(run_dir, "history.csv")):
        with open(os.path.join(run_dir, "history.csv"), newline="") as fh:
            if len(list(csv.reader(fh))) > 1:
                sys.exit(f"{run_dir} has epochs in history.csv but no last_state.pt; move it away or choose another --run-dir")
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "args": vars(args), "hyperparameters": config, "lr": 1e-3, "weight_decay": 5e-5, "mask_p": 0.4}, fh, indent=2)
    print(f"Run dir: {run_dir}\nConfig: {config}")

    device = torch.device(args.device)
    set_seed(args.seed)
    extra_x, train_data, eval_data = load_ciceklab(args.processed, device)

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

    best_state, best_epoch = train_and_validate(model, train_data, extra_x, eval_data, optimizer, scheduler, config["num_epochs"], config["alpha"], run_dir, state)

    results = []
    for s in SPLITS:
        model.load_state_dict(best_state[s])
        model.eval()
        with torch.no_grad():
            z_train = model.encoder(train_data.x, train_data.edge_index)
            train_metrics = model.test(z_train, train_data.pos_edge_label_index, train_neg)
            z = encode_all(model, train_data, extra_x)
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
