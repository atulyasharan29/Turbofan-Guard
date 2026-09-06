"""Training script for Step 1 Baseline Denoising Autoencoder on nominal flight data (DS02).

Trains TurbofanBaselineAE on nominal cruise flights to learn clean thermodynamic relationships,
evaluates on validation split, saves the best checkpoint, and calibrates baseline residual standard deviations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES, TurbofanDataset, create_dataloader
from src.models.baseline_ae import TurbofanBaselineAE
from src.utils.config import BackboneConfig


def train_baseline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("TurbofanGuard: Training Step 1 Baseline Autoencoder")
    print("=" * 70)
    print(f"  Target suite      : {args.suite} (Nominal Cruise Flights)")
    print(f"  Hardware device   : {device}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Learning rate     : {args.lr}")
    print(f"  Sequence window W : {args.window_length}")
    print(f"  Channel mask prob : {args.mask_prob}")
    print(f"  Output directory  : {output_dir.resolve()}")

    # 2. Dataset and DataLoader loading
    print("\n[1/4] Loading and scaling dataset...")
    train_dataset = TurbofanDataset(
        suite_name=args.suite,
        split="train",
        window_length=args.window_length,
        is_train=True,
        mask_prob=args.mask_prob,
    )

    scaler = train_dataset.get_scaler()
    scaler_path = output_dir / f"scaler_{args.suite}.json"
    scaler.save(scaler_path)
    print(f"  Saved fitted scaler parameters to: {scaler_path.name}")

    val_dataset = TurbofanDataset(
        suite_name=args.suite,
        split="val",
        window_length=args.window_length,
        scaler=scaler,
        is_train=False,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"  Train windows: {len(train_dataset):,} ({len(train_loader)} batches)")
    print(f"  Val windows  : {len(val_dataset):,} ({len(val_loader)} batches)")

    # 3. Model, Criterion, Optimizer, Scheduler
    print("\n[2/4] Instantiating model and optimizer...")
    config = BackboneConfig(
        input_dim=18,
        window_length=args.window_length,
        latent_dim=64,
        dropout=args.dropout,
        activation="gelu",
    )
    model = TurbofanBaselineAE(config=config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training loop with early stopping
    print("\n[3/4] Starting training loop...")
    best_val_loss = float("inf")
    best_model_path = output_dir / "baseline_ae_best.pt"
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)  # (B, W, 18)
            y = batch["y"].to(device)  # (B, 14)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(1, val_batches)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "val_loss": best_val_loss,
                    "history": history,
                },
                best_model_path,
            )
            marker = " [BEST SAVED]"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Val MSE: {avg_val_loss:.6f} | "
            f"LR: {current_lr:.1e}{marker}"
        )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s. Best Val MSE: {best_val_loss:.6f}")

    # 5. Nominal Residual Calibration on Validation Split
    print("\n[4/4] Calibrating nominal residual statistics on healthy validation data...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)  # (B, W, 18)
            x_obs = x[:, -1, 4:18]     # endpoint observed sensors (B, 14)
            y_hat = model(x)           # predicted clean sensors (B, 14)

            error = (x_obs - y_hat).cpu().numpy()  # (B, 14)
            all_errors.append(error)

    all_errors = np.vstack(all_errors)  # (N_val, 14)

    residual_mean = np.mean(all_errors, axis=0).tolist()
    residual_std = np.std(all_errors, axis=0).tolist()

    stats = {
        "suite": args.suite,
        "sample_count": len(all_errors),
        "sensor_names": SENSOR_NAMES,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "recommended_threshold": 3.5,
    }

    stats_path = output_dir / "baseline_residual_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"  Calibrated nominal statistics across {len(all_errors):,} validation flight cycles.")
    print(f"  Saved calibration parameters to: {stats_path.name}")
    print(f"  Average residual std across 14 channels: {np.mean(residual_std):.4f}")

    print("\n" + "=" * 70)
    print("STEP 1 BASELINE AUTOENCODER TRAINING & CALIBRATION COMPLETE!")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TurbofanGuard Baseline Autoencoder on DS02")
    parser.add_argument("--suite", type=str, default="DS02", help="Dataset suite name (default: DS02)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--window-length", type=int, default=16, help="Sequence window length W (default: 16)")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Channel masking probability (default: 0.15)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('mps', 'cuda', 'cpu')")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())
