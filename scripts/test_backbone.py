"""Test suite for the TurbofanGuard Shared Neural Backbone and Configuration Subsystem."""

import sys
from pathlib import Path
import tempfile
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import BackboneConfig
from src.models.backbone import TurbofanBackbone


def test_config_system():
    print(f"\n{'='*65}")
    print("1. TESTING CONFIGURATION SUBSYSTEM")
    print(f"{'='*65}")

    # 1. Load default config
    config_path = PROJECT_ROOT / "configs" / "backbone_config.json"
    print(f"Loading config from {config_path.name}...")
    cfg = BackboneConfig.from_json(config_path)
    assert cfg.input_dim == 18, f"Expected input_dim 18, got {cfg.input_dim}"
    assert cfg.window_length == 16, f"Expected window_length 16, got {cfg.window_length}"
    assert cfg.latent_dim == 64, f"Expected latent_dim 64, got {cfg.latent_dim}"
    print(f"   [PASS] Loaded config: input_dim={cfg.input_dim}, W={cfg.window_length}, z_dim={cfg.latent_dim}")

    # 2. Test serialization round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        modified_cfg = cfg.copy_with(window_length=30, latent_dim=128)
        modified_cfg.to_json(tmp_path)
        reloaded = BackboneConfig.from_json(tmp_path)
        assert reloaded.window_length == 30, "Modified window_length not preserved"
        assert reloaded.latent_dim == 128, "Modified latent_dim not preserved"
        print("   [PASS] Serialization round-trip preserved modified hyperparameters.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # 3. Test validation errors
    try:
        BackboneConfig(input_dim=-1)
        assert False, "Should have raised ValueError for negative input_dim"
    except ValueError as e:
        print(f"   [PASS] Caught expected validation error: {e}")

    try:
        BackboneConfig(activation="invalid_act")
        assert False, "Should have raised ValueError for invalid activation"
    except ValueError as e:
        print(f"   [PASS] Caught expected validation error: {e}")


def test_backbone_architecture():
    print(f"\n{'='*65}")
    print("2. TESTING TURBOFAN BACKBONE ARCHITECTURE")
    print(f"{'='*65}")

    # Instantiate model
    model = TurbofanBackbone()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated successfully.")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    assert total_params < 500_000, f"Model too heavy! {total_params:,} parameters"
    print("   [PASS] Lightweight footprint verified (< 500k parameters).")

    # Test variable window lengths W
    test_windows = [10, 16, 24, 30]
    batch_size = 8
    print("\n3. Testing forward pass with variable window lengths:")
    with torch.no_grad():
        for W in test_windows:
            x = torch.randn(batch_size, W, 18)
            z = model(x)
            assert z.shape == (batch_size, 64), f"Expected (8, 64) for W={W}, got {z.shape}"
            assert not torch.isnan(z).any(), f"NaN detected in output for W={W}"
            print(f"   W={W:2d}: Input shape {tuple(x.shape)} -> Latent z_t shape {tuple(z.shape)} [PASS]")

    # Test streaming single sample (B=1)
    print("\n4. Testing single-engine streaming inference (Batch size B=1):")
    with torch.no_grad():
        x_single = torch.randn(1, 16, 18)
        z_single = model(x_single)
        assert z_single.shape == (1, 64), f"Expected (1, 64), got {z_single.shape}"
        print(f"   Streaming window: {tuple(x_single.shape)} -> {tuple(z_single.shape)} [PASS]")

    # Test snapshot mode (2D tensor: B, 18)
    print("\n5. Testing snapshot mode (2D input):")
    with torch.no_grad():
        x_snap = torch.randn(batch_size, 18)
        z_snap = model(x_snap)
        assert z_snap.shape == (batch_size, 64), f"Expected (8, 64), got {z_snap.shape}"
        assert not torch.isnan(z_snap).any(), "NaN detected in snapshot output"
        print(f"   Snapshot batch:  {tuple(x_snap.shape)} -> {tuple(z_snap.shape)} [PASS]")

        x_snap_single = torch.randn(1, 18)
        z_snap_single = model(x_snap_single)
        assert z_snap_single.shape == (1, 64), f"Expected (1, 64), got {z_snap_single.shape}"
        print(f"   Snapshot single: {tuple(x_snap_single.shape)} -> {tuple(z_snap_single.shape)} [PASS]")


def test_gradient_flow():
    print(f"\n{'='*65}")
    print("6. TESTING BACKPROPAGATION & GRADIENT FLOW")
    print(f"{'='*65}")

    # 1. Test Window Pathway Gradient Flow
    model = TurbofanBackbone()
    model.train()

    x_win = torch.randn(4, 16, 18, requires_grad=True)
    z_win = model(x_win)
    loss_win = (z_win ** 2).sum()
    loss_win.backward()

    # In window mode, all layers EXCEPT snapshot_proj should receive gradients
    missing_win_grads = []
    for name, param in model.named_parameters():
        if "snapshot_proj" not in name:
            if param.grad is None:
                missing_win_grads.append(name)
            elif torch.all(param.grad == 0):
                missing_win_grads.append(f"{name} (zero grad)")

    assert not missing_win_grads, f"Window pathway missing gradients: {missing_win_grads}"
    print("   [PASS] 3D Temporal Window pathway: all conv & dense parameters received non-zero gradients.")

    # 2. Test Snapshot Pathway Gradient Flow
    model.zero_grad()
    x_snap = torch.randn(4, 18, requires_grad=True)
    z_snap = model(x_snap)
    loss_snap = (z_snap ** 2).sum()
    loss_snap.backward()

    # In snapshot mode, snapshot_proj, dense layers, and latent_proj should receive gradients
    snapshot_active_layers = ["snapshot_proj", "dense_1", "dense_2", "latent_proj"]
    missing_snap_grads = []
    for name, param in model.named_parameters():
        if any(layer in name for layer in snapshot_active_layers):
            if param.grad is None:
                missing_snap_grads.append(name)
            elif torch.all(param.grad == 0):
                missing_snap_grads.append(f"{name} (zero grad)")

    assert not missing_snap_grads, f"Snapshot pathway missing gradients: {missing_snap_grads}"
    print("   [PASS] 2D Snapshot pathway: all linear projection & dense parameters received non-zero gradients.")


def test_device_compatibility():
    print(f"\n{'='*65}")
    print("7. TESTING HARDWARE ACCELERATION / DEVICE COMPATIBILITY")
    print(f"{'='*65}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device.type.upper()}")

    model = TurbofanBackbone().to(device)
    model.eval()

    x = torch.randn(8, 16, 18, device=device)
    with torch.no_grad():
        z = model(x)
        assert z.device.type == device.type, f"Output not on target device {device}"
        assert z.shape == (8, 64)
    print(f"   [PASS] Successfully executed forward pass on {device.type.upper()}.")


if __name__ == "__main__":
    test_config_system()
    test_backbone_architecture()
    test_gradient_flow()
    test_device_compatibility()
    print(f"\n{'='*65}")
    print("ALL TURBOFAN BACKBONE TESTS PASSED SUCCESSFULLY!")
    print(f"{'='*65}\n")
