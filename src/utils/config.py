"""Configuration management for TurbofanGuard models and pipelines.

Provides typed dataclasses with JSON serialization, deserialization, and parameter validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


VALID_ACTIVATIONS = {"gelu", "relu", "silu", "leaky_relu", "tanh"}


@dataclass
class BackboneConfig:
    """Hyperparameter configuration for the TurbofanGuard Shared Neural Backbone.

    Attributes:
        input_dim: Number of input telemetry features (default: 18 = 4 conditions + 14 sensors).
        window_length: Temporal sequence window length W (default: 16 flight cycles).
        conv_channels: Output channels for 1D temporal convolution layers.
        kernel_sizes: Receptive field kernel sizes for multi-scale temporal convolutions.
        dense_hidden_dims: Intermediate dimensions for thermodynamic cross-feature dense layers.
        latent_dim: Dimensionality of the shared latent state z_t (default: 64).
        dropout: Dropout rate applied across dense representation layers.
        activation: Non-linear activation function name ('gelu', 'relu', 'silu', etc.).
    """

    input_dim: int = 18
    window_length: int = 16
    conv_channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])
    dense_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    latent_dim: int = 64
    dropout: float = 0.1
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self.validate()

    def validate(self) -> None:
        """Assert validity of all hyperparameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.window_length <= 0:
            raise ValueError(f"window_length must be positive, got {self.window_length}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.activation.lower() not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. Choose from {sorted(VALID_ACTIVATIONS)}"
            )
        if not self.conv_channels:
            raise ValueError("conv_channels cannot be empty")
        if not self.kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty")
        if not self.dense_hidden_dims:
            raise ValueError("dense_hidden_dims cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackboneConfig:
        """Instantiate BackboneConfig from a dictionary."""
        return cls(
            input_dim=data.get("input_dim", 18),
            window_length=data.get("window_length", 16),
            conv_channels=list(data.get("conv_channels", [32, 64])),
            kernel_sizes=list(data.get("kernel_sizes", [3, 5])),
            dense_hidden_dims=list(data.get("dense_hidden_dims", [128, 64])),
            latent_dim=data.get("latent_dim", 64),
            dropout=float(data.get("dropout", 0.1)),
            activation=str(data.get("activation", "gelu")).lower(),
        )

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Serialize configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> BackboneConfig:
        """Load and instantiate BackboneConfig from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy_with(self, **kwargs: Any) -> BackboneConfig:
        """Return a copy of the configuration with specified overrides."""
        current = self.to_dict()
        current.update(kwargs)
        return BackboneConfig.from_dict(current)


@dataclass
class DualHeadConfig:
    """Hyperparameter configuration for Step 2 Dual-Head Multi-Task Network.

    Attributes:
        backbone: BackboneConfig for the shared neural encoder.
        recon_hidden_dims: Dense layer dimensions for the Reconstruction Decoder (Head 1).
        diag_hidden_dims: Dense layer dimensions for the Diagnostic Classifier (Head 2).
        output_dim: Number of sensor channels (default: 14).
        lambda_fdi: Balancing hyperparameter between reconstruction and classification loss.
        pos_weight: Multiplier for positive fault class loss in weighted BCE.
        tau_high: Conservative high alarm threshold for unconfident states (default: 4.5 sigma).
        tau_low: Sensitive low alarm threshold for confident fault states (default: 2.0 sigma).
    """

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    recon_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    diag_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 14
    lambda_fdi: float = 0.5
    pos_weight: float = 6.0
    tau_high: float = 4.5
    tau_low: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self.validate()

    def validate(self) -> None:
        """Assert validity of all hyperparameters."""
        self.backbone.validate()
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.lambda_fdi < 0.0:
            raise ValueError(f"lambda_fdi must be non-negative, got {self.lambda_fdi}")
        if self.pos_weight <= 0.0:
            raise ValueError(f"pos_weight must be positive, got {self.pos_weight}")
        if self.tau_low >= self.tau_high:
            raise ValueError(f"tau_low ({self.tau_low}) must be strictly less than tau_high ({self.tau_high})")
        if not self.recon_hidden_dims:
            raise ValueError("recon_hidden_dims cannot be empty")
        if not self.diag_hidden_dims:
            raise ValueError("diag_hidden_dims cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a nested dictionary."""
        return {
            "backbone": self.backbone.to_dict(),
            "recon_hidden_dims": list(self.recon_hidden_dims),
            "diag_hidden_dims": list(self.diag_hidden_dims),
            "output_dim": self.output_dim,
            "lambda_fdi": self.lambda_fdi,
            "pos_weight": self.pos_weight,
            "tau_high": self.tau_high,
            "tau_low": self.tau_low,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DualHeadConfig:
        """Instantiate DualHeadConfig from a dictionary."""
        backbone_data = data.get("backbone", {})
        if isinstance(backbone_data, BackboneConfig):
            backbone_cfg = backbone_data
        else:
            backbone_cfg = BackboneConfig.from_dict(backbone_data) if backbone_data else BackboneConfig()

        return cls(
            backbone=backbone_cfg,
            recon_hidden_dims=list(data.get("recon_hidden_dims", [128, 64])),
            diag_hidden_dims=list(data.get("diag_hidden_dims", [128, 64])),
            output_dim=int(data.get("output_dim", 14)),
            lambda_fdi=float(data.get("lambda_fdi", 0.5)),
            pos_weight=float(data.get("pos_weight", 6.0)),
            tau_high=float(data.get("tau_high", 4.5)),
            tau_low=float(data.get("tau_low", 2.0)),
        )

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Serialize configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> DualHeadConfig:
        """Load and instantiate DualHeadConfig from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy_with(self, **kwargs: Any) -> DualHeadConfig:
        """Return a copy of the configuration with specified overrides."""
        current = self.to_dict()
        current.update(kwargs)
        return DualHeadConfig.from_dict(current)

