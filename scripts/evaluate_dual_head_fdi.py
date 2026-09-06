"""Evaluation and benchmarking script for Step 2 Dual-Head Multi-Task Network.

Evaluates:
1. Signal Reconstruction (Virtual Sensing): Scaled RMSE and Physical MAPE across 14 channels.
2. Two-Factor Authentication FDI:
   - False Alarm Rate (FAR on healthy flight cycles).
   - True Positive Rate (TPR / Recall on active fault cycles).
   - Single-fault isolation accuracy.
   - Average detection latency (flight cycles elapsed from inception to confirmed alarm).
3. Multi-Fault Accuracy (Hamming Loss and Exact Match Ratio for DS04).
4. Direct comparison report against Step 1 Baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES, TurbofanDataset, create_dataloader
from src.data.scaler import TurbofanScaler
from src.models.dual_head_fdi import TurbofanDualHeadFDI
from src.utils.config import DualHeadConfig


def evaluate_dual_head_fdi(args: argparse.Namespace) -> Dict[str, float]:
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
    print(f"TurbofanGuard: Benchmarking Step 2 Dual-Head FDI on {args.suite} ({args.split} split)")
    print("=" * 70)
    print(f"  Checkpoint path     : {args.checkpoint}")
    print(f"  Residual stats path : {args.stats}")
    print(f"  Dynamic thresholds  : tau_high={args.tau_high}, tau_low={args.tau_low}")
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
    print("\n[1/4] Loading model, scaler, and calibration parameters...")
    scaler = TurbofanScaler.load(scaler_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    sigma_nominal = torch.tensor(stats["residual_std"], dtype=torch.float32).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DualHeadConfig.from_dict(checkpoint["config"])
    model = TurbofanDualHeadFDI(config=config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Model loaded from Epoch {checkpoint.get('epoch', 'N/A')} (Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")

    # 3. Load evaluation dataset
    print(f"\n[2/4] Loading {args.suite} ({args.split} split) with sequence windowing...")
    dataset = TurbofanDataset(
        suite_name=args.suite,
        split=args.split,
        window_length=config.backbone.window_length,
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
    print(f"\n[3/4] Running Two-Factor Authentication inference and adaptive thresholding...")
    all_y_pred_scaled = []
    all_y_true_scaled = []

    # Detection & isolation counters
    total_nominal_cycles = 0
    false_alarm_cycles = 0

    total_active_fault_cycles = 0
    detected_fault_cycles = 0

    correct_isolations = 0
    alarmed_fault_cycles = 0

    # Multi-label fault tracking
    total_sensor_evaluations = 0
    sensor_misclassifications = 0
    exact_matches = 0
    total_cycles_evaluated = 0

    # Latency tracking
    engine_fault_start = {}
    engine_first_alarm = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["m"].cpu().numpy()  # (B, 14)
            cycles = batch["cycle"].numpy()
            engine_ids = batch["engine_id"].numpy()

            # Two-Factor Authentication Inference
            fdi_res = model.infer_adaptive_fdi(
                x,
                sigma_nominal=sigma_nominal,
                tau_high=args.tau_high,
                tau_low=args.tau_low,
            )

            y_pred = fdi_res["y_hat"].cpu().numpy()
            is_alarm = fdi_res["is_alarm"].cpu().numpy()
            isolated_mask = fdi_res["isolated_sensors"].cpu().numpy()
            primary_fault = fdi_res["primary_fault"].cpu().numpy()

            all_y_pred_scaled.append(y_pred)
            all_y_true_scaled.append(y.cpu().numpy())

            for b in range(len(cycles)):
                eng_id = int(engine_ids[b])
                cycle = int(cycles[b])
                has_fault = m[b].sum() > 0
                alarm_active = bool(is_alarm[b])

                total_cycles_evaluated += 1
                total_sensor_evaluations += 14

                # Multi-label Hamming loss & exact match
                pred_binary = (isolated_mask[b] > 0).astype(int)
                true_binary = (m[b] > 0).astype(int)
                mismatches = np.sum(pred_binary != true_binary)
                sensor_misclassifications += mismatches
                if mismatches == 0:
                    exact_matches += 1

                if not has_fault:
                    total_nominal_cycles += 1
                    if alarm_active:
                        false_alarm_cycles += 1
                else:
                    total_active_fault_cycles += 1
                    if eng_id not in engine_fault_start:
                        engine_fault_start[eng_id] = cycle

                    if alarm_active:
                        detected_fault_cycles += 1
                        alarmed_fault_cycles += 1

                        if eng_id not in engine_first_alarm:
                            engine_first_alarm[eng_id] = cycle

                        # Single-fault isolation check
                        actual_faulty_channel = int(np.argmax(m[b]))
                        if isolated_mask[b, actual_faulty_channel] == 1.0 or primary_fault[b] == actual_faulty_channel:
                            correct_isolations += 1

    # 5. Compute Quantitative Metrics
    print(f"\n[4/4] Computing Quantitative Metrics...")
    all_y_pred_scaled = np.vstack(all_y_pred_scaled)
    all_y_true_scaled = np.vstack(all_y_true_scaled)

    # Invert to physical units
    y_pred_phys = scaler.inverse_transform_y(all_y_pred_scaled)
    y_true_phys = scaler.inverse_transform_y(all_y_true_scaled)

    rmse_per_sensor = np.sqrt(np.mean((y_pred_phys - y_true_phys) ** 2, axis=0))
    mape_per_sensor = np.mean(np.abs((y_pred_phys - y_true_phys) / np.maximum(np.abs(y_true_phys), 1e-5)), axis=0) * 100.0

    mean_rmse_scaled = np.sqrt(np.mean((all_y_pred_scaled - all_y_true_scaled) ** 2))
    mean_mape_phys = np.mean(mape_per_sensor)

    far = (false_alarm_cycles / max(1, total_nominal_cycles)) * 100.0
    tpr = (detected_fault_cycles / max(1, total_active_fault_cycles)) * 100.0
    isolation_acc = (correct_isolations / max(1, alarmed_fault_cycles)) * 100.0
    hamming_loss = sensor_misclassifications / max(1, total_sensor_evaluations)
    exact_match_ratio = (exact_matches / max(1, total_cycles_evaluated)) * 100.0

    latencies = []
    for eng_id, start_cyc in engine_fault_start.items():
        if eng_id in engine_first_alarm:
            lat = max(0, engine_first_alarm[eng_id] - start_cyc)
            latencies.append(lat)

    mean_latency = np.mean(latencies) if latencies else float("nan")

    # Display Comprehensive Benchmark Report
    print("\n" + "=" * 70)
    print("STEP 2 DUAL-HEAD MULTI-TASK BENCHMARK REPORT")
    print("=" * 70)
    print(f"Dataset Suite: {args.suite} | Split: {args.split} | Adaptive Tau: [{args.tau_low}, {args.tau_high}]")
    print("-" * 70)
    print("1. SIGNAL RECONSTRUCTION & VIRTUAL SENSING (DENOISING):")
    print(f"   Overall Scaled RMSE        : {mean_rmse_scaled:.4f}")
    print(f"   Overall Physical MAPE      : {mean_mape_phys:.2f}%")
    print(f"   Top 3 Accurate Sensors     :")
    sorted_sensors = np.argsort(mape_per_sensor)
    for rank, idx in enumerate(sorted_sensors[:3], 1):
        print(f"     {rank}. {SENSOR_NAMES[idx]:<6}: MAPE = {mape_per_sensor[idx]:.3f}% | RMSE = {rmse_per_sensor[idx]:.4f}")

    print("\n2. FAULT DETECTION & ISOLATION (TWO-FACTOR AUTHENTICATION):")
    print(f"   Total Nominal Cycles       : {total_nominal_cycles:,}")
    print(f"   False Alarms Triggered     : {false_alarm_cycles:,}")
    print(f"   False Alarm Rate (FAR)     : {far:.2f}%  (Target: < 1.0%)")
    print(f"   Total Active Fault Cycles  : {total_active_fault_cycles:,}")
    print(f"   Detected Fault Cycles      : {detected_fault_cycles:,}")
    print(f"   True Positive Rate (TPR)   : {tpr:.2f}%  (Target: > 90.0%)")
    print(f"   Fault Isolation Accuracy   : {isolation_acc:.2f}%  (Correct: {correct_isolations:,} / {alarmed_fault_cycles:,})")
    print(f"   Average Detection Latency  : {mean_latency:.1f} flight cycles")
    print(f"   Multi-Label Hamming Loss   : {hamming_loss:.4f}")
    print(f"   Exact Match Ratio (Subset) : {exact_match_ratio:.2f}%")

    print("\n3. COMPARATIVE BENCHMARK: STEP 1 (STATIC) vs. STEP 2 (DUAL-HEAD ADAPTIVE)")
    print("-" * 70)
    print(f"  {'Metric':<28} | {'Step 1 (Baseline)':<18} | {'Step 2 (Dual-Head)':<18}")
    print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*18}")
    print(f"  {'False Alarm Rate (FAR)':<28} | {'5.44%':<18} | {f'{far:.2f}%':<18}")
    print(f"  {'True Positive Rate (TPR)':<28} | {'28.64%':<18} | {f'{tpr:.2f}%':<18}")
    print(f"  {'Detection Latency (cycles)':<28} | {'16.0':<18} | {f'{mean_latency:.1f}':<18}")
    print(f"  {'Physical MAPE Error':<28} | {'0.21%':<18} | {f'{mean_mape_phys:.2f}%':<18}")
    print("=" * 70)

    return {
        "far": far,
        "tpr": tpr,
        "latency": mean_latency,
        "mape": mean_mape_phys,
        "isolation_acc": isolation_acc,
        "hamming_loss": hamming_loss,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Step 2 Dual-Head FDI on DS03/DS04")
    parser.add_argument("--suite", type=str, default="DS03", help="Evaluation suite (default: DS03)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--tau-high", type=float, default=4.5, help="High threshold (default: 4.5)")
    parser.add_argument("--tau-low", type=float, default=2.0, help="Low threshold (default: 2.0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dual_head_best.pt", help="Checkpoint path")
    parser.add_argument("--stats", type=str, default="checkpoints/dual_head_residual_stats.json", help="Residual stats")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler_DS02.json", help="Scaler path")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_dual_head_fdi(parse_args())
