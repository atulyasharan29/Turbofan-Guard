"""PyTorch Dataset and DataLoader module for TurbofanGuard.

Provides sliding temporal sequence windowing, dynamic multi-hot fault label generation,
and training-time random channel masking augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.reader import load_manifests, load_suite_split
from src.data.scaler import INPUT_FEATURES, TARGET_FEATURES, TurbofanScaler

# 14 Clean base sensor names (ordered identically to TARGET_FEATURES)
SENSOR_NAMES: List[str] = [
    "NH", "NL", "WFE", "PS0", "P2", "P023", "P030",
    "P044", "P050", "P134", "T2", "T023", "T030", "T050",
]

SENSOR_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SENSOR_NAMES)}


def build_fault_matrix(
    df: pd.DataFrame,
    manifest_df: Optional[pd.DataFrame],
    suite_name: str,
) -> np.ndarray:
    """Construct multi-hot binary fault matrix M of shape (N, 14).

    Args:
        df: DataFrame containing the split telemetry.
        manifest_df: Engine manifest DataFrame containing fault injection metadata.
        suite_name: Name of the suite ('DS01', 'DS02', 'DS03', 'DS04').

    Returns:
        np.ndarray of shape (len(df), 14) with float32 values in {0.0, 1.0}.
    """
    n_rows = len(df)
    fault_matrix = np.zeros((n_rows, len(SENSOR_NAMES)), dtype=np.float32)

    # DS01 and DS02 are nominal datasets (all healthy, zero faults)
    if suite_name in ["DS01", "DS02"] or manifest_df is None:
        return fault_matrix

    # Working copy with row index preserved
    work_df = df[["engine_id"]].copy()
    work_df["_row_idx"] = np.arange(n_rows)

    if suite_name == "DS03":
        # DS03 has a single structured sensor fault per trajectory
        if "fault_on" in df.columns and "affected_sensor" in manifest_df.columns:
            work_df["fault_on"] = df["fault_on"].to_numpy()
            merged = work_df.merge(
                manifest_df[["engine_id", "affected_sensor"]],
                on="engine_id",
                how="left",
            )
            # Filter rows where fault is actively on
            active_faults = merged[merged["fault_on"] == 1]
            for _, row in active_faults.iterrows():
                sensor = str(row["affected_sensor"]).strip()
                if sensor in SENSOR_TO_INDEX:
                    row_idx = int(row["_row_idx"])
                    channel_idx = SENSOR_TO_INDEX[sensor]
                    fault_matrix[row_idx, channel_idx] = 1.0

    elif suite_name == "DS04":
        # DS04 has concurrent multi-sensor fault families (fault_a, fault_b, fault_c)
        fault_cols_df = [c for c in ["fault_a_on", "fault_b_on", "fault_c_on"] if c in df.columns]
        for col in fault_cols_df:
            work_df[col] = df[col].to_numpy()

        manifest_cols = ["engine_id"]
        for prefix in ["fault_a", "fault_b", "fault_c"]:
            meas_col = f"{prefix}_measurement"
            if meas_col in manifest_df.columns:
                manifest_cols.append(meas_col)

        merged = work_df.merge(manifest_df[manifest_cols], on="engine_id", how="left")

        for prefix in ["fault_a", "fault_b", "fault_c"]:
            on_col = f"{prefix}_on"
            meas_col = f"{prefix}_measurement"
            if on_col in merged.columns and meas_col in merged.columns:
                active = merged[merged[on_col] == 1]
                for _, row in active.iterrows():
                    sensor = str(row[meas_col]).strip()
                    if sensor in SENSOR_TO_INDEX:
                        row_idx = int(row["_row_idx"])
                        channel_idx = SENSOR_TO_INDEX[sensor]
                        fault_matrix[row_idx, channel_idx] = 1.0

    return fault_matrix


class TurbofanDataset(Dataset):
    """PyTorch Dataset providing sliding temporal windows of turbofan telemetry.

    Yields:
        Dictionary containing:
            - 'x': Tensor of shape (window_length, 18) - scaled input window.
            - 'y': Tensor of shape (14,) - scaled clean target at window endpoint.
            - 'm': Tensor of shape (14,) - multi-hot fault indicator at window endpoint.
            - 'cycle': int - flight cycle at window endpoint.
            - 'engine_id': int - engine serial number.
    """

    def __init__(
        self,
        suite_name: str,
        split: str = "train",
        scaler: Optional[TurbofanScaler] = None,
        window_length: int = 16,
        stride: int = 1,
        is_train: bool = True,
        mask_prob: float = 0.15,
        max_masked_channels: int = 2,
        data_dir: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        manifest_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the TurbofanDataset.

        Args:
            suite_name: Dataset suite directory name (e.g. 'DS01', 'DS02', 'DS03', 'DS04').
            split: Split name ('train', 'val', or 'test').
            scaler: Fitted TurbofanScaler. If None and split == 'train', fits a new scaler.
            window_length: Number of consecutive flight cycles in each window W (default: 16).
            stride: Stride between successive sliding windows (default: 1).
            is_train: Whether this dataset is used for training (enables channel masking).
            mask_prob: Probability of applying channel masking augmentation to a window.
            max_masked_channels: Maximum number of sensor channels to mask simultaneously (1 or 2).
            data_dir: Base directory containing dataset suites.
            df: Optional pre-loaded DataFrame (if None, loaded via reader).
            manifest_df: Optional pre-loaded manifest DataFrame.
        """
        super().__init__()
        self.suite_name = suite_name
        self.split = split
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.is_train = is_train
        self.mask_prob = float(mask_prob) if is_train else 0.0
        self.max_masked_channels = int(max_masked_channels)

        if self.window_length <= 0:
            raise ValueError(f"window_length must be positive, got {self.window_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        # 1. Load telemetry and manifest data if not provided
        if df is None:
            df = load_suite_split(suite_name, split=split, data_dir=data_dir)
        if manifest_df is None:
            manifest_df, _ = load_manifests(suite_name, data_dir=data_dir)

        # 2. Sort DataFrame strictly by engine_id and flight_cycle
        df = df.sort_values(by=["engine_id", "flight_cycle"]).reset_index(drop=True)

        # 3. Fit or apply TurbofanScaler
        if scaler is None:
            if split == "train":
                self.scaler = TurbofanScaler().fit(df)
            else:
                raise ValueError(
                    f"A pre-fitted TurbofanScaler must be provided for non-train split '{split}'."
                )
        else:
            self.scaler = scaler

        x_scaled, y_scaled = self.scaler.transform(df)
        self.x_data = x_scaled  # (N, 18) float32
        self.y_data = y_scaled if y_scaled is not None else np.zeros((len(df), 14), dtype=np.float32)

        # 4. Build multi-hot binary fault ground truth
        self.m_data = build_fault_matrix(df, manifest_df, suite_name=suite_name)  # (N, 14) float32

        # 5. Extract metadata arrays for indexing
        self.engine_ids = df["engine_id"].to_numpy()
        self.flight_cycles = df["flight_cycle"].to_numpy()

        # 6. Pre-calculate all valid window slice indices (no boundary crossing between engines)
        self.windows: List[Tuple[int, int, int, int, int]] = []
        # Group indices by engine_id
        unique_engines, engine_start_indices = np.unique(self.engine_ids, return_index=True)
        # Add total length as end boundary
        engine_boundaries = list(engine_start_indices) + [len(df)]

        for i in range(len(unique_engines)):
            start_row = engine_boundaries[i]
            end_row = engine_boundaries[i + 1]
            engine_len = end_row - start_row

            if engine_len >= self.window_length:
                for w_start in range(start_row, end_row - self.window_length + 1, self.stride):
                    w_end = w_start + self.window_length
                    endpoint_idx = w_end - 1
                    eng_id = int(self.engine_ids[endpoint_idx])
                    cycle = int(self.flight_cycles[endpoint_idx])
                    self.windows.append((w_start, w_end, endpoint_idx, eng_id, cycle))

    def __len__(self) -> int:
        """Return total number of valid temporal windows in this split."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch and return a single temporal sequence sample."""
        w_start, w_end, endpoint_idx, eng_id, cycle = self.windows[idx]

        # Extract window array (W, 18)
        x_win = self.x_data[w_start:w_end].copy()
        y_endpoint = self.y_data[endpoint_idx].copy()
        m_endpoint = self.m_data[endpoint_idx].copy()

        # Channel masking augmentation (applied strictly during training)
        # Note: Features 0..3 are operating conditions (XM, ALT, DTISA, EPR).
        # We only mask sensor telemetry channels (indices 4..17).
        if self.is_train and self.mask_prob > 0.0:
            if np.random.rand() < self.mask_prob:
                num_to_mask = np.random.randint(1, self.max_masked_channels + 1)
                # Sensor channels lie in indices 4 through 17
                masked_indices = np.random.choice(range(4, 18), size=num_to_mask, replace=False)
                # Setting to 0.0 replaces with the standardized training mean
                x_win[:, masked_indices] = 0.0

        return {
            "x": torch.from_numpy(x_win),
            "y": torch.from_numpy(y_endpoint),
            "m": torch.from_numpy(m_endpoint),
            "cycle": cycle,
            "engine_id": eng_id,
        }

    def get_scaler(self) -> TurbofanScaler:
        """Return the fitted TurbofanScaler associated with this dataset."""
        return self.scaler


def create_dataloader(
    dataset: TurbofanDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Factory creating a standard PyTorch DataLoader for TurbofanDataset.

    Args:
        dataset: Instantiated TurbofanDataset.
        batch_size: Number of window samples per batch.
        shuffle: Whether to shuffle windows at each epoch.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies Tensors into CUDA/MPS pinned memory.

    Returns:
        torch.utils.data.DataLoader yielding batched dictionary tensors.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
