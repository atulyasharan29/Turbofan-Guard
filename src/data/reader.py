"""Raw data reader and column grouping utilities for TurbofanGuard.

Provides functions to load raw split Parquet files and suite metadata
without applying any transformations or sequence windowing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Default project root: TurbofanGuard/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_suite_split(
    suite_name: str,
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load a raw split Parquet file for a dataset suite.

    Args:
        suite_name: Dataset suite directory name (e.g. 'DS01', 'DS02', 'DS03', 'DS04').
        split: Split name ('train', 'val', or 'test').
        data_dir: Base directory containing the suite folders. If None, defaults to the project root.

    Returns:
        pd.DataFrame containing the raw split data.
    """
    base_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT
    parquet_path = base_dir / suite_name / "parquet" / f"{split}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found at: {parquet_path.resolve()}\n"
            f"Please ensure '{suite_name}/parquet/{split}.parquet' exists in {base_dir.resolve()}."
        )

    return pd.read_parquet(parquet_path)


def load_manifests(
    suite_name: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Load the engine manifest CSV and suite metadata JSON for a suite.

    Args:
        suite_name: Dataset suite directory name (e.g. 'DS01', 'DS03').
        data_dir: Base directory containing the suite folders. If None, defaults to the project root.

    Returns:
        Tuple of (engine_manifest_df, suite_metadata_dict).
    """
    base_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT
    metadata_dir = base_dir / suite_name / "metadata"
    manifest_csv = metadata_dir / "engine_manifest.csv"
    metadata_json = metadata_dir / "suite_metadata.json"

    manifest_df = pd.read_csv(manifest_csv) if manifest_csv.exists() else None

    metadata_dict = None
    if metadata_json.exists():
        with open(metadata_json, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

    return manifest_df, metadata_dict


def get_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Partition DataFrame columns into standard domain categories.

    Categories:
        - conditions: 4 flight operating condition variables (XM, ALT, DTISA, EPR).
        - sensors_obs: 14 observed sensor channels (*_obs).
        - sensors_truth: 14 clean ground-truth target channels (*_truth).
        - health_indices: 10 internal degradation states (*DETA*, *CW*) - isolated as leakage.
        - input_features: 18 features (conditions + sensors_obs) used for model input.
        - metadata: Identifiers and cycle counters (suite_id, engine_id, fault_id, fault_on, flight_cycle).

    Args:
        df: DataFrame loaded from a split parquet.

    Returns:
        Dictionary mapping group names to lists of column names.
    """
    cols = list(df.columns)

    conditions = [c for c in cols if c in ["XM", "ALT", "DTISA", "EPR"]]
    sensors_obs = [c for c in cols if c.endswith("_obs")]
    sensors_truth = [c for c in cols if c.endswith("_truth")]
    health_indices = [c for c in cols if c.startswith("DETA") or c.startswith("CW")]
    meta_cols = [c for c in cols if c in ["suite_id", "engine_id", "fault_id", "fault_on", "flight_cycle"]]
    input_features = conditions + sensors_obs

    return {
        "conditions": conditions,
        "sensors_obs": sensors_obs,
        "sensors_truth": sensors_truth,
        "health_indices": health_indices,
        "input_features": input_features,
        "metadata": meta_cols,
    }


def get_split_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a high-level summary of a loaded split DataFrame."""
    groups = get_column_groups(df)
    return {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "num_engines": df["engine_id"].nunique() if "engine_id" in df.columns else None,
        "cycle_range": (int(df["flight_cycle"].min()), int(df["flight_cycle"].max())) if "flight_cycle" in df.columns else None,
        "num_conditions": len(groups["conditions"]),
        "num_observed_sensors": len(groups["sensors_obs"]),
        "num_truth_sensors": len(groups["sensors_truth"]),
        "num_health_indices_isolated": len(groups["health_indices"]),
    }
