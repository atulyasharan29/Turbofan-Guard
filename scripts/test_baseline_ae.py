"""Unit test suite for Step 1 Baseline Denoising Autoencoder (TurbofanBaselineAE).

Verifies:
1. Model instantiation and lightweight parameter footprint (< 100k parameters).
2. Forward pass with 3D temporal sequence windows (B, W, 18) -> (B, 14).
3. Forward pass with 2D single snapshots (B, 18) -> (B, 14).
4. Single-engine streaming inference (B=1).
5. Latent space representation extraction z_t (B, 64).
6. Residual calculation, thresholding, and isolated sensor detection.
7. Full end-to-end gradient flow across backbone and decoder.
8. Apple Silicon GPU (MPS) and CPU device execution.
"""

import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES
from src.models.baseline_ae import TurbofanBaselineAE


def test_baseline_ae():
    print("=" * 70)
    print("TurbofanGuard: Testing Step 1 Baseline Autoencoder Architecture")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Parameter Footprint Verification
    # -------------------------------------------------------------
    print("\n[TEST 1] Instantiating TurbofanBaselineAE...")
    model = TurbofanBaselineAE()

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"  Backbone parameters: {backbone_params:,}")
    print(f"  Decoder parameters : {decoder_params:,}")
    print(f"  Total parameters   : {total_params:,}")

    assert total_params < 100_000, f"Expected < 100k parameters, got {total_params:,}"
    print("  [PASS] Lightweight footprint verified (< 100k parameters).")

    # -------------------------------------------------------------
    # 2. Forward Pass Geometry: 3D Temporal Windows & 2D Snapshots
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Forward Pass Geometry...")
    batch_size = 16
    w_len = 16

    # 3D Sequence Windows: (B, W, 18)
    x_3d = torch.randn(batch_size, w_len, 18)
    y_3d = model(x_3d)
    print(f"  3D Window Input  {tuple(x_3d.shape)} -> Output {tuple(y_3d.shape)}")
    assert y_3d.shape == (batch_size, 14), f"Expected ({batch_size}, 14), got {y_3d.shape}"

    # 2D Single Snapshot Mode: (B, 18)
    x_2d = torch.randn(batch_size, 18)
    y_2d = model(x_2d)
    print(f"  2D Snapshot Input {tuple(x_2d.shape)} -> Output {tuple(y_2d.shape)}")
    assert y_2d.shape == (batch_size, 14), f"Expected ({batch_size}, 14), got {y_2d.shape}"

    # Single-engine streaming: (1, W, 18)
    x_single = torch.randn(1, w_len, 18)
    y_single = model(x_single)
    assert y_single.shape == (1, 14), f"Expected (1, 14), got {y_single.shape}"

    # Latent space extraction
    z = model.get_latent(x_3d)
    assert z.shape == (batch_size, 64), f"Expected ({batch_size}, 64), got {z.shape}"
    print("  [PASS] Forward pass and latent extraction shapes verified.")

    # -------------------------------------------------------------
    # 3. Residual Computation & Fault Isolation Logic
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing Residual Computation & Fault Isolation Logic...")
    # Create clean synthetic baseline where prediction matches observation
    x_clean = torch.zeros(4, w_len, 18)
    # Set simulated healthy sensors in observed slice (indices 4..18)
    # Give it model output so residual is near 0
    with torch.no_grad():
        y_clean_pred = model(x_clean)
    x_clean[:, -1, 4:18] = y_clean_pred.clone()

    # Nominal healthy test (no fault)
    sigma_nominal = torch.ones(14)
    clean_res = model.infer_residuals(x_clean, sigma_nominal=sigma_nominal, threshold=3.5)
    print(f"  Clean input max residual: {clean_res['residuals'].max().item():.4f}")
    assert not clean_res["is_alarm"].any(), "Healthy telemetry triggered false alarm!"

    # Injected Fault Test: Inject massive +10 sigma spike on sensor 12 (T030)
    # Sensor 12 corresponds to column 4 + 12 = 16 in x
    faulty_x = x_clean.clone()
    faulty_sensor_idx = 12
    faulty_x[:, -1, 4 + faulty_sensor_idx] += 10.0

    fault_res = model.infer_residuals(faulty_x, sigma_nominal=sigma_nominal, threshold=3.5)
    print(f"  Faulted input alarm triggered: {fault_res['is_alarm'].tolist()}")
    print(f"  Isolated sensor channel     : {fault_res['isolated_sensor'].tolist()} (Sensor: '{SENSOR_NAMES[faulty_sensor_idx]}')")

    assert fault_res["is_alarm"].all(), "Faulted telemetry failed to trigger alarm!"
    assert (fault_res["isolated_sensor"] == faulty_sensor_idx).all(), f"Failed to isolate sensor {faulty_sensor_idx}!"
    print("  [PASS] Fault detection alarm and isolation verified.")

    # -------------------------------------------------------------
    # 4. Backward Pass & Gradient Flow
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing End-to-End Gradient Backpropagation...")
    # 1. Test 3D Sequence Window Pathway
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x_train = torch.randn(8, w_len, 18)
    target = torch.randn(8, 14)
    pred = model(x_train)

    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    # In window mode, all parameters except snapshot_proj should receive gradients
    missing_win = [name for name, p in model.named_parameters() if "snapshot_proj" not in name and (p.grad is None or (p.grad == 0).all())]
    assert not missing_win, f"Window mode missing gradients in: {missing_win}"
    print(f"  [PASS] 3D Window pathway: all active parameters received non-zero gradients.")

    # 2. Test 2D Snapshot Pathway
    optimizer.zero_grad()
    x_snap = torch.randn(8, 18)
    pred_snap = model(x_snap)
    loss_snap = torch.nn.functional.mse_loss(pred_snap, target)
    loss_snap.backward()

    missing_snap = [name for name, p in model.named_parameters() if "snapshot_proj" in name and (p.grad is None or (p.grad == 0).all())]
    assert not missing_snap, f"Snapshot mode missing gradients in: {missing_snap}"
    print(f"  [PASS] 2D Snapshot pathway: snapshot projection parameters received non-zero gradients.")

    # -------------------------------------------------------------
    # 5. Device Execution: MPS / CUDA / CPU
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Device Execution...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Target acceleration device: {device}")
    model = model.to(device)
    x_dev = x_3d.to(device)
    y_dev = model(x_dev)
    assert y_dev.device.type == device.type, f"Expected {device.type}, got {y_dev.device.type}"
    print(f"  [PASS] Forward execution verified on {device.type}.")

    print("\n" + "=" * 70)
    print("ALL BASELINE AUTOENCODER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_baseline_ae()
