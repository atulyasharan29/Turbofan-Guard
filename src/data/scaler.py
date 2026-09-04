"""Feature normalization pipeline using scikit-learn StandardScaler.

Provides explicit feature whitelists and TurbofanScaler to scale the 18 input features (X)
and 14 clean target features (Y) strictly using training split statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Explicit whitelists of required features (no manual slicing or blacklisting)
INPUT_FEATURES: List[str] = [
    # 4 Flight Operating Conditions
    "XM", "ALT", "DTISA", "EPR",
    # 14 Observed Sensor Telemetry Channels
    "NH_obs", "NL_obs", "WFE_obs", "PS0_obs", "P2_obs", "P023_obs", "P030_obs",
    "P044_obs", "P050_obs", "P134_obs", "T2_obs", "T023_obs", "T030_obs", "T050_obs",
]

TARGET_FEATURES: List[str] = [
    # 14 Clean Ground-Truth Sensors
    "NH_truth", "NL_truth", "WFE_truth", "PS0_truth", "P2_truth", "P023_truth", "P030_truth",
    "P044_truth", "P050_truth", "P134_truth", "T2_truth", "T023_truth", "T030_truth", "T050_truth",
]


class TurbofanScaler:
    """Compact scaler wrapping scikit-learn StandardScaler for TurbofanGuard.

    Manages:
        - scaler_x: StandardScaler fitted on the 18 INPUT_FEATURES
        - scaler_y: StandardScaler fitted on the 14 TARGET_FEATURES
    """

    def __init__(self) -> None:
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "TurbofanScaler":
        """Fit scalers strictly on the training split."""
        self.scaler_x.fit(df_train[INPUT_FEATURES].to_numpy(dtype=np.float64))
        self.scaler_y.fit(df_train[TARGET_FEATURES].to_numpy(dtype=np.float64))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform input features X and target features Y into float32 arrays."""
        self._check_fitted()
        x_raw = df[INPUT_FEATURES].to_numpy(dtype=np.float64)
        x_scaled = self.scaler_x.transform(x_raw).astype(np.float32)

        has_targets = set(TARGET_FEATURES).issubset(df.columns)
        if has_targets:
            y_raw = df[TARGET_FEATURES].to_numpy(dtype=np.float64)
            y_scaled = self.scaler_y.transform(y_raw).astype(np.float32)
        else:
            y_scaled = None

        return x_scaled, y_scaled

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on df_train and return scaled (X, Y)."""
        return self.fit(df_train).transform(df_train)  # type: ignore

    def inverse_transform_x(self, x_scaled: np.ndarray) -> np.ndarray:
        """Invert scaled X values back to real physical engineering units."""
        self._check_fitted()
        return self.scaler_x.inverse_transform(x_scaled)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert scaled Y values back to real physical engineering units."""
        self._check_fitted()
        return self.scaler_y.inverse_transform(y_scaled)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save mean and scale parameters to a JSON file."""
        self._check_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "scaler_x": {
                "mean": self.scaler_x.mean_.tolist(),
                "scale": self.scaler_x.scale_.tolist(),
            },
            "scaler_y": {
                "mean": self.scaler_y.mean_.tolist(),
                "scale": self.scaler_y.scale_.tolist(),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TurbofanScaler":
        """Load scaler parameters from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        instance = cls()
        instance.scaler_x.mean_ = np.array(state["scaler_x"]["mean"], dtype=np.float64)
        instance.scaler_x.scale_ = np.array(state["scaler_x"]["scale"], dtype=np.float64)
        instance.scaler_x.var_ = instance.scaler_x.scale_ ** 2

        instance.scaler_y.mean_ = np.array(state["scaler_y"]["mean"], dtype=np.float64)
        instance.scaler_y.scale_ = np.array(state["scaler_y"]["scale"], dtype=np.float64)
        instance.scaler_y.var_ = instance.scaler_y.scale_ ** 2

        instance.is_fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("TurbofanScaler is not fitted yet. Call 'fit' before transforming.")
