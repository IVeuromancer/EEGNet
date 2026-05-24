"""
MLflow-tracked training with leave-one-session-out cross-validation.

Usage:
    python src/train.py
    python src/train.py --arch unet1d --epochs 100 --lr 5e-4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report

from src.model import build_model
from src.dataset import make_loso_splits

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


def train_fold(fold: int, train_ds, test_ds, args) -> dict:
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        arch=args.arch,
        n_channels=8,
        n_samples=1000,
        n_classes=2,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, labels = eval_epoch(model, test_loader, criterion)
        scheduler.step()

        mlflow.log_metrics({
            f"fold{fold}/train_loss": train_loss,
            f"fold{fold}/train_acc": train_acc,
            f"fold{fold}/val_loss": val_loss,
            f"fold{fold}/val_acc": val_acc,
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"    Early stop at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"| val_loss {val_loss:.4f} val_acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    _, final_acc, final_preds, final_labels = eval_epoch(model, test_loader, criterion)
    cm = confusion_matrix(final_labels, final_preds)
    print(f"\n  Fold {fold} best val acc: {best_val_acc:.3f}")
    print(classification_report(final_labels, final_preds, target_names=["Left", "Right"]))

    return {"acc": best_val_acc, "cm": cm, "model": model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="eegnet", choices=["eegnet", "unet1d"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data", default="data/processed/dataset.npz")
    args = parser.parse_args()

    data = np.load(args.data)
    X, y, sessions = data["X"], data["y"], data["sessions"]
    n_sessions = len(np.unique(sessions))
    print(f"Dataset: {X.shape}, {n_sessions} sessions | Device: {DEVICE}")

    mlflow.set_experiment("EEGMotorImagery")

    with mlflow.start_run(run_name=f"{args.arch}_loso"):
        mlflow.log_params(vars(args))

        fold_accs = []
        best_overall_acc = 0.0
        best_model = None

        for fold, test_sess in enumerate(range(n_sessions)):
            print(f"\nFold {fold+1}/{n_sessions} — test session {test_sess}")
            train_ds, test_ds, _ = make_loso_splits(X, y, sessions, test_session=test_sess)
            result = train_fold(fold + 1, train_ds, test_ds, args)
            fold_accs.append(result["acc"])

            if result["acc"] > best_overall_acc:
                best_overall_acc = result["acc"]
                best_model = result["model"]

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"\nLOSO accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        mlflow.log_metrics({"loso_mean_acc": mean_acc, "loso_std_acc": std_acc})

        # save best model
        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)
        model_path = out_dir / f"best_{args.arch}.pt"
        torch.save(best_model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))
        print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
