"""Training script for Step 2 Dual-Head Multi-Task Network (Physics + Diagnostic AI).

Trains TurbofanDualHeadFDI jointly on healthy flights (DS02) and single-fault flights (DS03),
combining clean signal reconstruction with weighted multi-label fault classification.
Calibrates nominal residual statistics and saves the best model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES, TurbofanDataset, create_dataloader
from src.data.scaler import TurbofanScaler
from src.models.dual_head_fdi import TurbofanDualHeadFDI
from src.training.losses import MultiTaskFDILoss
from src.utils.config import BackboneConfig, DualHeadConfig


def train_dual_head(args: argparse.Namespace) -> None:
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
    print("TurbofanGuard: Training Step 2 Dual-Head Multi-Task Network")
    print("=" * 70)
    print(f"  Nominal suite (Healthy) : {args.nominal_suite}")
    print(f"  Fault suite (Failures)  : {args.fault_suite}")
    print(f"  Hardware device         : {device}")
    print(f"  Epochs                  : {args.epochs}")
    print(f"  Batch size              : {args.batch_size}")
    print(f"  Learning rate           : {args.lr}")
    print(f"  Loss weight (lambda)    : {args.lambda_fdi}")
    print(f"  Pos class weight        : {args.pos_weight}")
    print(f"  Output directory        : {output_dir.resolve()}")

    # 2. Scaler Loading or Fitting
    print("\n[1/4] Preparing dataset and scalers...")
    if args.scaler and Path(args.scaler).exists():
        scaler = TurbofanScaler.load(args.scaler)
        print(f"  Loaded pre-fitted scaler from: {args.scaler}")
    else:
        # Fit on nominal suite training split
        ds_fit = TurbofanDataset(
            suite_name=args.nominal_suite,
            split="train",
            window_length=args.window_length,
            is_train=True,
        )
        scaler = ds_fit.get_scaler()
        scaler_save_path = output_dir / f"scaler_{args.nominal_suite}.json"
        scaler.save(scaler_save_path)
        print(f"  Fitted and saved scaler to: {scaler_save_path.name}")

    # 3. Create Multi-Suite Datasets
    # Training splits
    ds_nominal_train = TurbofanDataset(
        suite_name=args.nominal_suite,
        split="train",
        window_length=args.window_length,
        scaler=scaler,
        is_train=True,
        mask_prob=args.mask_prob,
    )
    ds_fault_train = TurbofanDataset(
        suite_name=args.fault_suite,
        split="train",
        window_length=args.window_length,
        scaler=scaler,
        is_train=True,
        mask_prob=args.mask_prob,
    )
    train_dataset = ConcatDataset([ds_nominal_train, ds_fault_train])

    # Validation splits
    ds_nominal_val = TurbofanDataset(
        suite_name=args.nominal_suite,
        split="val",
        window_length=args.window_length,
        scaler=scaler,
        is_train=False,
    )
    ds_fault_val = TurbofanDataset(
        suite_name=args.fault_suite,
        split="val",
        window_length=args.window_length,
        scaler=scaler,
        is_train=False,
    )
    val_dataset = ConcatDataset([ds_nominal_val, ds_fault_val])

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

    print(f"  Train windows : {len(train_dataset):,} ({len(ds_nominal_train):,} healthy + {len(ds_fault_train):,} fault)")
    print(f"  Val windows   : {len(val_dataset):,} ({len(ds_nominal_val):,} healthy + {len(ds_fault_val):,} fault)")

    # 4. Model Instantiation & Optional Transfer Learning
    print("\n[2/4] Initializing model and multi-task loss...")
    if args.config and Path(args.config).exists():
        config = DualHeadConfig.from_json(args.config)
        config = config.copy_with(
            lambda_fdi=args.lambda_fdi,
            pos_weight=args.pos_weight,
            tau_high=args.tau_high,
            tau_low=args.tau_low,
        )
    else:
        config = DualHeadConfig(
            backbone=BackboneConfig(
                input_dim=18,
                window_length=args.window_length,
                latent_dim=64,
                dropout=args.dropout,
            ),
            lambda_fdi=args.lambda_fdi,
            pos_weight=args.pos_weight,
            tau_high=args.tau_high,
            tau_low=args.tau_low,
        )

    model = TurbofanDualHeadFDI(config=config).to(device)

    # Transfer learning from Step 1 baseline if available
    if args.pretrained and Path(args.pretrained).exists():
        print(f"  Warm-starting from pretrained checkpoint: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        baseline_state = ckpt.get("model_state", ckpt)

        # Map state dict keys
        model_state = model.state_dict()
        transferred_keys = 0
        for k, v in baseline_state.items():
            # Transfer backbone directly
            if k.startswith("backbone.") and k in model_state:
                model_state[k] = v
                transferred_keys += 1
            # Transfer decoder weights to reconstruction head (decoder -> recon_head)
            elif k.startswith("decoder."):
                recon_k = k.replace("decoder.", "recon_head.")
                if recon_k in model_state:
                    model_state[recon_k] = v
                    transferred_keys += 1

        model.load_state_dict(model_state)
        print(f"  Transferred {transferred_keys} weight tensors to Backbone and Head 1.")

    criterion = MultiTaskFDILoss(
        lambda_fdi=config.lambda_fdi,
        pos_weight=config.pos_weight,
        num_channels=config.output_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 5. Training Loop
    print("\n[3/4] Training Dual-Head Multi-Task Network...")
    best_val_loss = float("inf")
    best_model_path = output_dir / "dual_head_best.pt"
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_fdi_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["m"].to(device)

            optimizer.zero_grad()
            out = model(x)
            losses = criterion(
                y_hat=out["y_hat"],
                logits=out["logits"],
                y_true=y,
                m_true=m,
            )

            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_total_loss += losses["loss"].item()
            train_recon_loss += losses["recon_loss"].item()
            train_fdi_loss += losses["fdi_loss"].item()

        scheduler.step()

        avg_train_total = train_total_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_fdi = train_fdi_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_fdi_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                m = batch["m"].to(device)

                out = model(x)
                losses = criterion(
                    y_hat=out["y_hat"],
                    logits=out["logits"],
                    y_true=y,
                    m_true=m,
                )

                val_total_loss += losses["loss"].item()
                val_recon_loss += losses["recon_loss"].item()
                val_fdi_loss += losses["fdi_loss"].item()

                p_fault = out["p_fault"].cpu().numpy()
                m_true = m.cpu().numpy()
                all_preds.append((p_fault > 0.5).astype(np.float32))
                all_targets.append(m_true)

        avg_val_total = val_total_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_fdi = val_fdi_loss / len(val_loader)

        # FDI Classification Metrics (Micro/Macro F1)
        all_preds_arr = np.vstack(all_preds)
        all_targets_arr = np.vstack(all_targets)

        tp = np.sum((all_preds_arr == 1) & (all_targets_arr == 1))
        fp = np.sum((all_preds_arr == 1) & (all_targets_arr == 0))
        fn = np.sum((all_preds_arr == 0) & (all_targets_arr == 1))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        # Checkpoint Saving
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            patience_counter = 0
            marker = " -> BEST"

            checkpoint = {
                "epoch": epoch,
                "val_loss": avg_val_total,
                "val_recon_loss": avg_val_recon,
                "val_fdi_loss": avg_val_fdi,
                "val_f1": float(f1),
                "val_recall": float(recall),
                "val_precision": float(precision),
                "config": model.config.to_dict(),
                "model_state": model.state_dict(),
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train: {avg_train_total:.4f} (Recon: {avg_train_recon:.4f}, FDI: {avg_train_fdi:.4f}) | "
            f"Val: {avg_val_total:.4f} (Recon: {avg_val_recon:.4f}, FDI: {avg_val_fdi:.4f}) | "
            f"F1: {f1:.3f} (Rec: {recall:.3f}){marker}"
        )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training finished in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # 6. Calibrate Nominal Residual Statistics on Healthy Flights
    print("\n[4/4] Calibrating nominal residual statistics on healthy flights (DS02 val)...")
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    nominal_val_loader = create_dataloader(
        ds_nominal_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_errors = []
    with torch.no_grad():
        for batch in nominal_val_loader:
            x = batch["x"].to(device)
            x_obs = x[:, -1, 4:18]
            y_hat = model(x)["y_hat"]

            err = (x_obs - y_hat).cpu().numpy()
            all_errors.append(err)

    all_errors = np.vstack(all_errors)
    res_mean = np.mean(all_errors, axis=0).tolist()
    res_std = np.std(all_errors, axis=0).tolist()

    stats = {
        "nominal_suite": args.nominal_suite,
        "sample_count": len(all_errors),
        "sensor_names": SENSOR_NAMES,
        "residual_mean": res_mean,
        "residual_std": res_std,
        "tau_high": config.tau_high,
        "tau_low": config.tau_low,
    }

    stats_path = output_dir / "dual_head_residual_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"  Calibrated nominal statistics over {len(all_errors):,} validation flight cycles.")
    print(f"  Saved calibration parameters to: {stats_path.name}")
    print(f"  Average residual std: {np.mean(res_std):.4f}")

    print("\n" + "=" * 70)
    print("STEP 2 DUAL-HEAD MULTI-TASK TRAINING & CALIBRATION COMPLETE!")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step 2 TurbofanGuard Dual-Head Network")
    parser.add_argument("--nominal-suite", type=str, default="DS02", help="Healthy suite (default: DS02)")
    parser.add_argument("--fault-suite", type=str, default="DS03", help="Faulted suite (default: DS03)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs (default: 12)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--window-length", type=int, default=16, help="Sequence window length (default: 16)")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Mask probability (default: 0.15)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (default: 0.1)")
    parser.add_argument("--lambda-fdi", type=float, default=0.5, help="Loss weight for FDI (default: 0.5)")
    parser.add_argument("--pos-weight", type=float, default=6.0, help="Positive class BCE weight (default: 6.0)")
    parser.add_argument("--tau-high", type=float, default=4.5, help="High threshold (default: 4.5)")
    parser.add_argument("--tau-low", type=float, default=2.0, help="Low threshold (default: 2.0)")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler_DS02.json", help="Path to scaler")
    parser.add_argument("--pretrained", type=str, default="checkpoints/baseline_ae_best.pt", help="Pretrained baseline")
    parser.add_argument("--config", type=str, default="configs/dual_head_config.json", help="Path to dual head config")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    train_dual_head(parse_args())
