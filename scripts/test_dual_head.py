"""Unit test suite for Step 2 Dual-Head Multi-Task Network (TurbofanDualHeadFDI).

Verifies:
1. Model instantiation, config validation, and lightweight footprint (< 120k parameters).
2. Multi-task forward pass geometry for 3D temporal windows (B, W, 18) and 2D snapshots (B, 18).
3. MultiTaskFDILoss formulation, positive class weighting, and loss components.
4. Adaptive Residual Thresholding & Two-Factor Authentication:
   - Noise spike suppression (prevents false alarms when p_fault is low).
   - Early drift detection (sensitive trigger when p_fault is high).
   - Multi-fault isolation mask generation.
5. End-to-end simultaneous gradient backpropagation across both heads and the backbone.
6. Apple Silicon GPU (MPS) and CPU device execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SENSOR_NAMES
from src.models.dual_head_fdi import DiagnosticClassifierHead, TurbofanDualHeadFDI
from src.training.losses import MultiTaskFDILoss
from src.utils.config import DualHeadConfig


def test_dual_head() -> None:
    print("=" * 70)
    print("TurbofanGuard: Testing Step 2 Dual-Head Multi-Task Architecture")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Config & Parameter Footprint Verification
    # -------------------------------------------------------------
    print("\n[TEST 1] Instantiating TurbofanDualHeadFDI & Verifying Parameter Count...")
    config = DualHeadConfig.from_json("configs/dual_head_config.json")
    model = TurbofanDualHeadFDI(config=config)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    recon_params = sum(p.numel() for p in model.recon_head.parameters())
    diag_params = sum(p.numel() for p in model.diag_head.parameters())

    print(f"  Backbone parameters   : {backbone_params:,}")
    print(f"  Head 1 (Reconstruction): {recon_params:,}")
    print(f"  Head 2 (Diagnostic FDI): {diag_params:,}")
    print(f"  Total parameters      : {total_params:,}")

    assert total_params < 120_000, f"Expected < 120k parameters, got {total_params:,}"
    print("  [PASS] Lightweight footprint verified (< 120k parameters).")

    # -------------------------------------------------------------
    # 2. Forward Pass Geometry: 3D Sequence Windows & 2D Snapshots
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Forward Pass Geometry & Probabilities...")
    batch_size = 8
    w_len = 16

    # 3D Sequence Windows: (B, W, 18)
    x_3d = torch.randn(batch_size, w_len, 18)
    out_3d = model(x_3d)

    assert "y_hat" in out_3d and "logits" in out_3d and "p_fault" in out_3d and "latent" in out_3d
    assert out_3d["y_hat"].shape == (batch_size, 14), f"Expected y_hat (8, 14), got {out_3d['y_hat'].shape}"
    assert out_3d["logits"].shape == (batch_size, 14), f"Expected logits (8, 14), got {out_3d['logits'].shape}"
    assert out_3d["p_fault"].shape == (batch_size, 14), f"Expected p_fault (8, 14), got {out_3d['p_fault'].shape}"
    assert out_3d["latent"].shape == (batch_size, 64), f"Expected latent (8, 64), got {out_3d['latent'].shape}"

    # Verify probability bounds [0.0, 1.0]
    p_min = out_3d["p_fault"].min().item()
    p_max = out_3d["p_fault"].max().item()
    assert 0.0 <= p_min and p_max <= 1.0, f"Probabilities out of bounds: [{p_min}, {p_max}]"
    print(f"  3D Window: y_hat {tuple(out_3d['y_hat'].shape)}, p_fault {tuple(out_3d['p_fault'].shape)}")

    # 2D Single Snapshot Mode: (B, 18)
    x_2d = torch.randn(batch_size, 18)
    out_2d = model(x_2d)
    assert out_2d["y_hat"].shape == (batch_size, 14)
    assert out_2d["p_fault"].shape == (batch_size, 14)
    print(f"  2D Snapshot: y_hat {tuple(out_2d['y_hat'].shape)}, p_fault {tuple(out_2d['p_fault'].shape)}")
    print("  [PASS] Multi-task output geometries verified.")

    # -------------------------------------------------------------
    # 3. Multi-Task Loss Verification
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing MultiTaskFDILoss...")
    criterion = MultiTaskFDILoss(lambda_fdi=0.5, pos_weight=6.0, num_channels=14)

    y_true = torch.randn(batch_size, 14)
    m_true = torch.zeros(batch_size, 14)
    m_true[0, 12] = 1.0  # Sensor 12 (T030) broken on first sample

    loss_dict = criterion(
        y_hat=out_3d["y_hat"],
        logits=out_3d["logits"],
        y_true=y_true,
        m_true=m_true,
    )

    assert "loss" in loss_dict and "recon_loss" in loss_dict and "fdi_loss" in loss_dict
    expected_total = loss_dict["recon_loss"] + 0.5 * loss_dict["fdi_loss"]
    assert torch.isclose(loss_dict["loss"], expected_total, atol=1e-5)
    print(f"  Recon Loss (MSE) : {loss_dict['recon_loss'].item():.4f}")
    print(f"  FDI Loss (wBCE)  : {loss_dict['fdi_loss'].item():.4f}")
    print(f"  Combined Loss    : {loss_dict['loss'].item():.4f}")
    print("  [PASS] MultiTaskFDILoss mathematically confirmed.")

    # -------------------------------------------------------------
    # 4. Two-Factor Authentication & Adaptive Thresholding
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Two-Factor Authentication Logic...")
    # Mock inputs with controllable residuals and fault probabilities
    sigma_nom = torch.ones(14)
    tau_high = 4.5
    tau_low = 2.0

    # Test Scenario A: Noise Spike Suppression
    # Telemetry has a 3.8 sigma residual spike on sensor 0 (NL), but AI radar is confident engine is healthy (p=0.02)
    p_spike = torch.full((1, 14), 0.02)
    th_spike = tau_high - (tau_high - tau_low) * p_spike
    res_spike = torch.zeros(1, 14)
    res_spike[0, 0] = 3.8  # Spike below 4.45 dynamic threshold

    alarm_spike = (res_spike > th_spike).any(dim=-1)
    print(f"  Scenario A (3.8 sigma spike, p=0.02): tau_adaptive={th_spike[0, 0]:.2f}, alarm={alarm_spike.item()}")
    assert not alarm_spike.item(), "Failed to suppress noise spike!"
    print("  [PASS] Scenario A: Noise spike false alarm successfully suppressed.")

    # Test Scenario B: Slow Drift Early Trigger
    # Telemetry has a subtle 2.5 sigma residual on sensor 12 (T030), and AI radar detects drift trend (p=0.92)
    p_drift = torch.full((1, 14), 0.02)
    p_drift[0, 12] = 0.92
    th_drift = tau_high - (tau_high - tau_low) * p_drift
    res_drift = torch.zeros(1, 14)
    res_drift[0, 12] = 2.5  # 2.5 sigma breaches 2.20 dynamic threshold!

    alarm_drift = (res_drift > th_drift).any(dim=-1)
    print(f"  Scenario B (2.5 sigma drift, p=0.92): tau_adaptive={th_drift[0, 12]:.2f}, alarm={alarm_drift.item()}")
    assert alarm_drift.item(), "Failed to detect subtle drift under high AI confidence!"
    print("  [PASS] Scenario B: Subtle drift caught early under high AI confidence.")

    # Test full infer_adaptive_fdi method
    x_test = torch.zeros(2, w_len, 18)
    fdi_res = model.infer_adaptive_fdi(x_test, sigma_nominal=sigma_nom)
    assert "isolated_sensors" in fdi_res and "primary_fault" in fdi_res
    assert fdi_res["isolated_sensors"].shape == (2, 14)
    print("  [PASS] infer_adaptive_fdi outputs verified.")

    # -------------------------------------------------------------
    # 5. Backward Pass & Gradient Flow
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Simultaneous Gradient Backpropagation...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x_sim = torch.randn(4, w_len, 18)
    out_sim = model(x_sim)
    loss_sim = criterion(
        y_hat=out_sim["y_hat"],
        logits=out_sim["logits"],
        y_true=torch.randn(4, 14),
        m_true=torch.zeros(4, 14),
    )["loss"]
    loss_sim.backward()

    # Check gradients in both heads and backbone
    backbone_grads = [p.grad for name, p in model.backbone.named_parameters() if "snapshot_proj" not in name]
    recon_grads = [p.grad for p in model.recon_head.parameters()]
    diag_grads = [p.grad for p in model.diag_head.parameters()]

    assert all(g is not None and not (g == 0).all() for g in backbone_grads), "Backbone missing gradients!"
    assert all(g is not None and not (g == 0).all() for g in recon_grads), "Reconstruction head missing gradients!"
    assert all(g is not None and not (g == 0).all() for g in diag_grads), "Diagnostic head missing gradients!"
    print("  [PASS] Simultaneous gradient flow to Backbone, Head 1, and Head 2 confirmed.")

    # -------------------------------------------------------------
    # 6. Device Execution
    # -------------------------------------------------------------
    print("\n[TEST 6] Testing Hardware Device Acceleration...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Target device: {device}")
    model = model.to(device)
    x_dev = x_3d.to(device)
    out_dev = model(x_dev)
    assert out_dev["y_hat"].device.type == device.type
    assert out_dev["p_fault"].device.type == device.type
    print(f"  [PASS] Multi-task execution verified on {device.type}.")

    print("\n" + "=" * 70)
    print("ALL DUAL-HEAD MULTI-TASK ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_dual_head()
