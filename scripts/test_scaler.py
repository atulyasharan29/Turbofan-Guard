"""Test script for simplified TurbofanScaler using scikit-learn StandardScaler."""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_suite_split, TurbofanScaler, INPUT_FEATURES, TARGET_FEATURES


def test_scaler_suite(suite_name: str):
    print(f"\n{'='*65}")
    print(f"TESTING TURBOFAN SCALER: {suite_name}")
    print(f"{'='*65}")

    # 1. Load splits
    print("1. Loading Parquet splits...")
    df_train = load_suite_split(suite_name, split="train")
    df_val = load_suite_split(suite_name, split="val")
    df_test = load_suite_split(suite_name, split="test")
    print(f"   Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")

    # 2. Check explicit feature whitelists
    print(f"2. Explicit whitelists:")
    print(f"   Input features  (X, {len(INPUT_FEATURES)}): {INPUT_FEATURES}")
    print(f"   Target features (Y, {len(TARGET_FEATURES)}): {TARGET_FEATURES}")
    for col in INPUT_FEATURES + TARGET_FEATURES:
        assert not col.startswith("DETA") and not col.startswith("CW"), f"Leakage column found: {col}"
    print("   [PASS] Verified zero leakage columns in feature lists.")

    # 3. Fit scaler strictly on train
    scaler = TurbofanScaler()
    scaler.fit(df_train)
    print(f"3. Scaler fitted on {len(df_train)} training rows.")

    # 4. Transform splits
    X_train, Y_train = scaler.transform(df_train)
    X_val, Y_val = scaler.transform(df_val)
    X_test, Y_test = scaler.transform(df_test)

    assert Y_train is not None and Y_val is not None and Y_test is not None

    # 5. Dimension checks
    assert X_train.shape == (len(df_train), 18), f"Expected shape ({len(df_train)}, 18)"
    assert Y_train.shape == (len(df_train), 14), f"Expected shape ({len(df_train)}, 14)"
    assert X_val.shape == (len(df_val), 18), f"Expected shape ({len(df_val)}, 18)"
    assert Y_val.shape == (len(df_val), 14), f"Expected shape ({len(df_val)}, 14)"
    assert X_test.shape == (len(df_test), 18), f"Expected shape ({len(df_test)}, 18)"
    assert Y_test.shape == (len(df_test), 14), f"Expected shape ({len(df_test)}, 14)"
    print(f"4. Transformed array dimensions verified:")
    print(f"   X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"   X_val  : {X_val.shape},   Y_val  : {Y_val.shape}")
    print(f"   X_test : {X_test.shape},  Y_test : {Y_test.shape}")

    # 6. Distribution checks on training split
    mean_x = np.mean(X_train, axis=0)
    std_x = np.std(X_train, axis=0)
    raw_std_x = np.std(df_train[INPUT_FEATURES].to_numpy(), axis=0)
    non_constant_mask = raw_std_x > 1e-4

    print(f"5. Checking standard distribution on train:")
    print(f"   Max abs mean deviation on X: {np.max(np.abs(mean_x)):.6f}")
    if np.any(non_constant_mask):
        max_std_dev = np.max(np.abs(std_x[non_constant_mask] - 1.0))
        print(f"   Max std deviation from 1 on non-constant features: {max_std_dev:.6f}")
        assert max_std_dev < 1e-3, "Standardized train std deviates from 1.0"
    assert np.all(np.abs(mean_x) < 1e-4), "Standardized train mean deviates from 0"

    # 7. Inverse transformation check
    print("6. Testing inverse transformation round-trip...")
    Y_recovered = scaler.inverse_transform_y(Y_val)

    Y_orig = df_val[TARGET_FEATURES].to_numpy()
    max_rel_diff_y = np.max(np.abs(Y_recovered - Y_orig) / (np.abs(Y_orig) + 1e-8))
    print(f"   Max relative reconstruction error on Y_val: {max_rel_diff_y:.2e}")
    assert np.allclose(Y_recovered, Y_orig, rtol=1e-4, atol=1e-2), "Inverse transform error too large on Y"
    print("   [PASS] Target inverse transformation is exact within float precision.")

    X_recovered = scaler.inverse_transform_x(X_val)
    X_orig = df_val[INPUT_FEATURES].to_numpy()
    max_rel_diff_x = np.max(np.abs(X_recovered - X_orig) / (np.abs(X_orig) + 1e-8))
    print(f"   Max relative reconstruction error on X_val: {max_rel_diff_x:.2e}")
    assert np.allclose(X_recovered, X_orig, rtol=1e-4, atol=1e-2), "Inverse transform error too large on X"
    print("   [PASS] Input inverse transformation is exact within float precision.")

    # 8. Persistence check
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    save_path = checkpoint_dir / f"test_scaler_{suite_name}.json"
    print(f"7. Testing scaler serialization to {save_path.name}...")
    scaler.save(save_path)
    assert save_path.exists(), "Scaler file was not created"

    loaded_scaler = TurbofanScaler.load(save_path)
    X_test_loaded, Y_test_loaded = loaded_scaler.transform(df_test)
    assert np.allclose(X_test, X_test_loaded, atol=1e-6), "Loaded scaler transformed X differs!"
    # pyrefly: ignore [bad-argument-type]
    assert np.allclose(Y_test, Y_test_loaded, atol=1e-6), "Loaded scaler transformed Y differs!"
    print("   [PASS] Scaler successfully saved, reloaded, and verified.")

    save_path.unlink()
    print(f"\n[PASS] {suite_name} scaler verification successful!")


def main():
    print("="*65)
    print("TurbofanGuard: Testing Simplified StandardScaler Pipeline")
    print("="*65)

    test_scaler_suite("DS01")
    test_scaler_suite("DS03")

    print("\n" + "="*65)
    print("ALL SCALER TESTS COMPLETED SUCCESSFULLY!")
    print("="*65)


if __name__ == "__main__":
    main()
