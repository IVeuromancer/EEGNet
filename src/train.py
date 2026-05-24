"""
MLflow-tracked training with stratified train/dev/test split.

Usage:
    python -m src.train
    python -m src.train --arch unet1d --epochs 100 --lr 5e-4
    python -m src.train --channels 2,3   # CH3+CH4 only (default)
    python -m src.train --channels 0,1,2,3,4,5,6,7  # all 8 channels
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
from sklearn.metrics import confusion_matrix, classification_report

from src.model import build_model
from src.dataset import make_splits

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="eegnet", choices=["eegnet", "unet1d"])
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
    print(f"Dataset: {X.shape} | channels {channels} | Device: {DEVICE}")

    mlflow.set_experiment("EEGMotorImagery")

    with mlflow.start_run(run_name=f"{args.arch}"):
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

        train_ds, dev_ds, test_ds, _ = make_splits(X, y, channels=channels)
        print(f"Split: {len(train_ds)} train / {len(dev_ds)} dev / {len(test_ds)} test")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_model(
            arch=args.arch,
            n_channels=len(channels),
            n_samples=1000,
            n_classes=2,
            dropout=args.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_dev_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader, criterion)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }, step=epoch)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss {train_loss:.4f} acc {train_acc:.3f} "
                      f"| dev_loss {dev_loss:.4f} dev_acc {dev_acc:.3f}")

        model.load_state_dict(best_state)
        _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
        print(f"\nBest dev acc: {best_dev_acc:.3f}  |  Test acc: {test_acc:.3f}")
        print(classification_report(test_labels, test_preds, target_names=["Left", "Right"]))

        mlflow.log_metrics({"best_dev_acc": best_dev_acc, "test_acc": test_acc})

        # save local .pt for game.py / realtime.py
        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)
        model_path = out_dir / f"best_{args.arch}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")

        # register versioned model in MLflow Model Registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"EEGMotorImagery_{args.arch}",
        )
        print(f"Model registered as EEGMotorImagery_{args.arch}")


if __name__ == "__main__":
    main()
