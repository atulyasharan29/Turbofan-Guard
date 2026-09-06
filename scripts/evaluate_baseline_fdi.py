"""Evaluation and benchmarking script for Step 1 Baseline Denoising Autoencoder on DS03.

Evaluates:
1. Signal Reconstruction (Virtual Sensor Denoising): RMSE and MAPE across all 14 sensors in physical units.
2. Fault Detection: False Alarm Rate (FAR on healthy cycles) and True Positive Rate (TPR / Recall on active faults).
3. Fault Isolation: Accuracy of isolated failing sensor (argmax r_i == affected_sensor).
4. Detection Latency: Average flight cycles between fault inception and first confirmed alarm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES, TurbofanDataset, create_dataloader
from src.data.scaler import TurbofanScaler
from src.models.baseline_ae import TurbofanBaselineAE
from src.utils.config import BackboneConfig


def evaluate_baseline_fdi(args: argparse.Namespace) -> None:
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
    print(f"TurbofanGuard: Benchmarking Step 1 Baseline FDI on {args.suite} ({args.split} split)")
    print("=" * 70)
    print(f"  Checkpoint path     : {args.checkpoint}")
    print(f"  Residual stats path : {args.stats}")
    print(f"  Detection threshold : {args.threshold} sigma")
    print(f"  Hardware device     : {device}")

    checkpoint_path = Path(args.checkpoint)
    stats_path = Path(args.stats)
    scaler_path = Path(args.scaler)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path.resolve()}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Residual stats not found at: {stats_path.resolve()}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler parameters not found at: {scaler_path.resolve()}")

    # 2. Load model, scaler, and calibration parameters
    print("\n[1/4] Loading model and calibration parameters...")
    scaler = TurbofanScaler.load(scaler_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    sigma_nominal = torch.tensor(stats["residual_std"], dtype=torch.float32).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = BackboneConfig.from_dict(checkpoint["config"])
    model = TurbofanBaselineAE(config=config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Model loaded from Epoch {checkpoint.get('epoch', 'N/A')} (Val MSE: {checkpoint.get('val_loss', 0.0):.6f})")

    # 3. Load evaluation dataset
    print(f"\n[2/4] Loading {args.suite} ({args.split} split) with sequence windowing...")
    dataset = TurbofanDataset(
        suite_name=args.suite,
        split=args.split,
        window_length=config.window_length,
        scaler=scaler,
        is_train=False,
    )

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(f"  Loaded {len(dataset):,} sliding windows across {dataset.engine_ids.max() - dataset.engine_ids.min() + 1} engines.")

    # 4. Evaluation Loop
    print(f"\n[3/4] Running inference and residual thresholding...")
    all_y_pred_scaled = []
    all_y_true_scaled = []

    # Detection & isolation counters
    total_nominal_cycles = 0
    false_alarm_cycles = 0

    total_active_fault_cycles = 0
    detected_fault_cycles = 0

    correct_isolations = 0
    alarmed_fault_cycles = 0

    # Per-engine tracking for detection latency
    engine_fault_start = {}
    engine_first_alarm = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)       # (B, W, 18)
            y = batch["y"].to(device)       # (B, 14)
            m = batch["m"].cpu().numpy()     # (B, 14) multi-hot ground truth
            cycles = batch["cycle"].numpy()  # (B,)
            engine_ids = batch["engine_id"].numpy()  # (B,)

            # Inference
            res_dict = model.infer_residuals(x, sigma_nominal=sigma_nominal, threshold=args.threshold)
            y_pred = res_dict["y_hat"]
            is_alarm = res_dict["is_alarm"].cpu().numpy()
            isolated_sensor = res_dict["isolated_sensor"].cpu().numpy()

            all_y_pred_scaled.append(y_pred.cpu().numpy())
            all_y_true_scaled.append(y.cpu().numpy())

            # Evaluate each cycle in batch
            for b in range(len(cycles)):
                eng_id = int(engine_ids[b])
                cycle = int(cycles[b])
                has_fault = m[b].sum() > 0
                alarm_active = bool(is_alarm[b])

                if not has_fault:
                    # Nominal healthy cycle
                    total_nominal_cycles += 1
                    if alarm_active:
                        false_alarm_cycles += 1
                else:
                    # Active sensor fault cycle
                    total_active_fault_cycles += 1
                    actual_faulty_channel = int(np.argmax(m[b]))

                    # Record fault start cycle per engine
                    if eng_id not in engine_fault_start:
                        engine_fault_start[eng_id] = cycle

                    if alarm_active:
                        detected_fault_cycles += 1
                        alarmed_fault_cycles += 1

                        # Record first alarm cycle
                        if eng_id not in engine_first_alarm:
                            engine_first_alarm[eng_id] = cycle

                        # Isolation accuracy
                        if isolated_sensor[b] == actual_faulty_channel:
                            correct_isolations += 1

    # 5. Compute Quantitative Metrics
    print(f"\n[4/4] Computing Quantitative Metrics...")
    all_y_pred_scaled = np.vstack(all_y_pred_scaled)
    all_y_true_scaled = np.vstack(all_y_true_scaled)

    # Invert to real physical engineering units
    y_pred_phys = scaler.inverse_transform_y(all_y_pred_scaled)
    y_true_phys = scaler.inverse_transform_y(all_y_true_scaled)

    # Denoising Metrics
    rmse_per_sensor = np.sqrt(np.mean((y_pred_phys - y_true_phys) ** 2, axis=0))
    mape_per_sensor = np.mean(np.abs((y_pred_phys - y_true_phys) / np.maximum(np.abs(y_true_phys), 1e-5)), axis=0) * 100.0

    mean_rmse_scaled = np.sqrt(np.mean((all_y_pred_scaled - all_y_true_scaled) ** 2))
    mean_mape_phys = np.mean(mape_per_sensor)

    # FDI Metrics
    far = (false_alarm_cycles / max(1, total_nominal_cycles)) * 100.0
    tpr = (detected_fault_cycles / max(1, total_active_fault_cycles)) * 100.0
    isolation_acc = (correct_isolations / max(1, alarmed_fault_cycles)) * 100.0

    # Latency
    latencies = []
    for eng_id, start_cyc in engine_fault_start.items():
        if eng_id in engine_first_alarm:
            lat = max(0, engine_first_alarm[eng_id] - start_cyc)
            latencies.append(lat)

    mean_latency = np.mean(latencies) if latencies else float("nan")

    # Display Report
    print("\n" + "=" * 70)
    print("STEP 1 BASELINE BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print(f"Dataset Suite: {args.suite} | Split: {args.split} | Threshold: {args.threshold} sigma")
    print("-" * 70)
    print("1. SIGNAL RECONSTRUCTION & VIRTUAL SENSING (DENOISING):")
    print(f"   Overall Scaled RMSE      : {mean_rmse_scaled:.4f}")
    print(f"   Overall Physical MAPE    : {mean_mape_phys:.2f}%")
    print(f"   Top 3 Accurate Sensors   :")
    sorted_sensors = np.argsort(mape_per_sensor)
    for rank, idx in enumerate(sorted_sensors[:3], 1):
        print(f"     {rank}. {SENSOR_NAMES[idx]:<6}: MAPE = {mape_per_sensor[idx]:.3f}% | RMSE = {rmse_per_sensor[idx]:.4f}")

    print("\n2. FAULT DETECTION & ISOLATION (FDI) PERFORMANCE:")
    print(f"   Total Nominal Cycles     : {total_nominal_cycles:,}")
    print(f"   False Alarms             : {false_alarm_cycles:,}")
    print(f"   False Alarm Rate (FAR)   : {far:.2f}%  (Target: < 1.0%)")
    print(f"   Total Active Fault Cycles: {total_active_fault_cycles:,}")
    print(f"   Detected Fault Cycles    : {detected_fault_cycles:,}")
    print(f"   True Positive Rate (TPR) : {tpr:.2f}%  (Target: > 90.0%)")
    print(f"   Single-Fault Isolation   : {isolation_acc:.2f}%  (Correct: {correct_isolations:,} / {alarmed_fault_cycles:,})")
    print(f"   Average Detection Latency: {mean_latency:.1f} flight cycles")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Step 1 Baseline FDI on DS03")
    parser.add_argument("--suite", type=str, default="DS03", help="Evaluation suite (default: DS03)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--threshold", type=float, default=3.5, help="Static residual threshold tau (default: 3.5)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_ae_best.pt", help="Model checkpoint")
    parser.add_argument("--stats", type=str, default="checkpoints/baseline_residual_stats.json", help="Residual stats")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler_DS02.json", help="Fitted scaler")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_baseline_fdi(parse_args())
