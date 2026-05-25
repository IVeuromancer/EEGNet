"""
MLflow-tracked training.

Usage:
    python -m src.train                          # random 70/15/15 split, CH3+CH4
    python -m src.train --split loso             # leave-one-session-out CV
    python -m src.train --channels 2,3           # CH3+CH4 only (default)
    python -m src.train --channels 0,1,2,3,4,5,6,7  # all 8 channels
    python -m src.train --arch unet1d --epochs 100 --lr 5e-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report

from src.model import build_model
from src.dataset import make_splits, make_loso_splits

# ── config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        n += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


def _make_model(args, n_channels):
    return build_model(
        arch=args.arch,
        n_channels=n_channels,
        n_samples=1000,
        n_classes=2,
        dropout=args.dropout,
    ).to(DEVICE)


def _fit(model, train_loader, eval_loader, args, log_prefix=""):
    """Train with early stopping. Returns (best_eval_acc, best_state_dict)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    pfx = f"{log_prefix}/" if log_prefix else ""

    best_acc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        eval_loss, eval_acc, _, _ = eval_epoch(model, eval_loader, criterion)
        scheduler.step()

        mlflow.log_metrics({
            f"{pfx}train_loss": train_loss, f"{pfx}train_acc": train_acc,
            f"{pfx}eval_loss": eval_loss,   f"{pfx}eval_acc": eval_acc,
        }, step=epoch)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"| eval_loss {eval_loss:.4f} eval_acc {eval_acc:.3f}")

    return best_acc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="eegnet", choices=["eegnet", "unet1d"])
    parser.add_argument("--split", default="random", choices=["random", "loso"],
                        help="random: stratified 70/15/15 split; loso: leave-one-session-out CV")
    parser.add_argument("--channels", default="2,3",
                        help="Comma-separated 0-indexed channel indices (default: 2,3 = CH3+CH4)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data", default="data/processed/dataset.npz")
    args = parser.parse_args()

    channels = [int(c) for c in args.channels.split(",")]

    data = np.load(args.data)
    X, y = data["X"], data["y"]
    print(f"Dataset: {X.shape} | channels {channels} | split: {args.split} | Device: {DEVICE}")

    mlflow.set_experiment("EEGMotorImagery")

    with mlflow.start_run(run_name=f"{args.arch}_{args.split}"):
        mlflow.log_params(vars(args))

        manifest_path = Path(args.data).parent / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            mlflow.set_tags({
                "dataset.preprocessed_at": manifest["preprocessed_at"],
                "dataset.sha256": manifest["dataset_sha256"],
                "dataset.n_trials": manifest["total_trials"],
                "dataset.n_sessions": len(manifest["sessions"]),
                "dataset.sessions": ", ".join(s["filename"] for s in manifest["sessions"]),
            })
            mlflow.log_artifact(str(manifest_path), artifact_path="dataset")

        best_model = None

        # ── random split ──────────────────────────────────────────────────────
        if args.split == "random":
            train_ds, dev_ds, test_ds, _ = make_splits(X, y, channels=channels)
            print(f"Split: {len(train_ds)} train / {len(dev_ds)} dev / {len(test_ds)} test")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

            model = _make_model(args, len(channels))
            best_dev_acc, best_state = _fit(model, train_loader, dev_loader, args)
            model.load_state_dict(best_state)

            criterion = nn.CrossEntropyLoss()
            _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
            print(f"\nBest dev acc: {best_dev_acc:.3f}  |  Test acc: {test_acc:.3f}")
            print(classification_report(test_labels, test_preds, target_names=["Left", "Right"]))
            mlflow.log_metrics({"best_dev_acc": best_dev_acc, "test_acc": test_acc})
            best_model = model

        # ── LOSO ─────────────────────────────────────────────────────────────
        elif args.split == "loso":
            sessions = data["sessions"]
            unique_sessions = np.unique(sessions)
            n_sessions = len(unique_sessions)
            print(f"LOSO: {n_sessions} folds")

            X_ch = X[:, channels, :]
            fold_accs = []
            best_overall_acc = 0.0
            best_overall_state = None
            criterion = nn.CrossEntropyLoss()

            for fold, test_sess in enumerate(unique_sessions):
                n_test = (sessions == test_sess).sum()
                print(f"\nFold {fold+1}/{n_sessions} — test session {test_sess} ({n_test} trials)")
                train_ds, test_ds, _ = make_loso_splits(X_ch, y, sessions, test_session=test_sess)

                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
                test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

                model = _make_model(args, len(channels))
                best_fold_acc, best_state = _fit(model, train_loader, test_loader, args,
                                                  log_prefix=f"fold{fold+1}")
                model.load_state_dict(best_state)

                _, _, preds, labels = eval_epoch(model, test_loader, criterion)
                print(f"  Fold {fold+1} best acc: {best_fold_acc:.3f}")
                print(classification_report(labels, preds, target_names=["Left", "Right"]))
                fold_accs.append(best_fold_acc)

                if best_fold_acc > best_overall_acc:
                    best_overall_acc = best_fold_acc
                    best_overall_state = best_state

            mean_acc = float(np.mean(fold_accs))
            std_acc  = float(np.std(fold_accs))
            print(f"\nLOSO accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            mlflow.log_metrics({"loso_mean_acc": mean_acc, "loso_std_acc": std_acc})

            best_model = _make_model(args, len(channels))
            best_model.load_state_dict(best_overall_state)

        # ── save + register ───────────────────────────────────────────────────
        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)
        model_path = out_dir / f"best_{args.arch}.pt"
        torch.save(best_model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")

        mlflow.pytorch.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=f"EEGMotorImagery_{args.arch}",
        )
        print(f"Model registered as EEGMotorImagery_{args.arch}")


if __name__ == "__main__":
    main()
