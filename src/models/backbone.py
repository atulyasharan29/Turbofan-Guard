"""Shared Neural Backbone module for TurbofanGuard (The Engine Brain).

Processes 18-dimensional telemetry inputs (snapshot or temporal sequence windows),
extracts coupled thermodynamic and temporal dynamics, and projects to the shared
latent state z_t that drives both virtual sensor reconstruction and diagnostic FDI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from src.utils.config import BackboneConfig

# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "backbone_config.json"


def _get_activation(name: str) -> nn.Module:
    """Return activation layer instance from string name."""
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: '{name}'")


class TurbofanBackbone(nn.Module):
    """Shared neural backbone extracting thermodynamic and temporal representations.

    Input Support:
        - 2D Snapshot Tensors: (batch_size, input_dim)
        - 3D Temporal Window Tensors: (batch_size, window_length, input_dim)

    Output:
        - Shared Latent State z_t: (batch_size, latent_dim)
    """

    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        if config is None:
            if DEFAULT_CONFIG_PATH.exists():
                self.config = BackboneConfig.from_json(DEFAULT_CONFIG_PATH)
            else:
                self.config = BackboneConfig()
        else:
            self.config = config

        self.input_dim = self.config.input_dim
        self.latent_dim = self.config.latent_dim
        c_out_1 = self.config.conv_channels[0]
        c_out_2 = self.config.conv_channels[1] if len(self.config.conv_channels) > 1 else c_out_1

        # -------------------------------------------------------------
        # 1. Temporal Feature Extractor (for 3D inputs: B, W, input_dim)
        # -------------------------------------------------------------
        k1, k2 = self.config.kernel_sizes[0], self.config.kernel_sizes[1]
        # Multi-scale 1D convolutions: branch 1 captures fast changes (k1), branch 2 captures trends (k2)
        self.conv_branch_1 = nn.Conv1d(self.input_dim, c_out_1, kernel_size=k1, padding="same")
        self.conv_branch_2 = nn.Conv1d(self.input_dim, c_out_1, kernel_size=k2, padding="same")
        self.conv_norm_1 = nn.GroupNorm(num_groups=1, num_channels=c_out_1 * 2)
        self.conv_act_1 = _get_activation(self.config.activation)

        # Second temporal stage
        self.conv_stage_2 = nn.Conv1d(c_out_1 * 2, c_out_2, kernel_size=3, padding="same")
        self.conv_norm_2 = nn.GroupNorm(num_groups=1, num_channels=c_out_2)
        self.conv_act_2 = _get_activation(self.config.activation)

        # Feature dimension after temporal pooling (concatenating current endpoint + temporal mean)
        temporal_feature_dim = c_out_2 * 2

        # -------------------------------------------------------------
        # 2. Snapshot Fallback Pathway (for 2D inputs: B, input_dim)
        # -------------------------------------------------------------
        self.snapshot_proj = nn.Sequential(
            nn.Linear(self.input_dim, temporal_feature_dim),
            nn.LayerNorm(temporal_feature_dim),
            _get_activation(self.config.activation),
        )

        # -------------------------------------------------------------
        # 3. Thermodynamic Cross-Feature Block (Dense Residual Layers)
        # -------------------------------------------------------------
        h1 = self.config.dense_hidden_dims[0]
        h2 = self.config.dense_hidden_dims[1] if len(self.config.dense_hidden_dims) > 1 else h1

        self.dense_1 = nn.Linear(temporal_feature_dim, h1)
        self.norm_1 = nn.LayerNorm(h1)
        self.act_1 = _get_activation(self.config.activation)
        self.drop_1 = nn.Dropout(self.config.dropout)

        # Optional linear projection for residual skip if dimensions differ
        self.skip_1 = (
            nn.Linear(temporal_feature_dim, h1)
            if temporal_feature_dim != h1
            else nn.Identity()
        )

        self.dense_2 = nn.Linear(h1, h2)
        self.norm_2 = nn.LayerNorm(h2)
        self.act_2 = _get_activation(self.config.activation)
        self.drop_2 = nn.Dropout(self.config.dropout)

        self.skip_2 = nn.Linear(h1, h2) if h1 != h2 else nn.Identity()

        # -------------------------------------------------------------
        # 4. Latent Space Projection (Output: z_t)
        # -------------------------------------------------------------
        self.latent_proj = nn.Linear(h2, self.latent_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shared neural backbone.

        Args:
            x: Input tensor. Either:
               - 2D Tensor of shape (batch_size, input_dim)
               - 3D Tensor of shape (batch_size, window_length, input_dim)

        Returns:
            z: Shared latent state tensor of shape (batch_size, latent_dim).
        """
        if x.dim() == 2:
            # Snapshot mode: (B, input_dim) -> (B, temporal_feature_dim)
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected input feature dim {self.input_dim}, got {x.shape[-1]}"
                )
            feat = self.snapshot_proj(x)

        elif x.dim() == 3:
            # Temporal window mode: (B, W, input_dim)
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected input feature dim {self.input_dim}, got {x.shape[-1]}"
                )
            # Transpose to (B, input_dim, W) for PyTorch Conv1d
            x_t = x.transpose(1, 2)

            # Multi-scale convolutions
            b1 = self.conv_branch_1(x_t)
            b2 = self.conv_branch_2(x_t)
            c1 = torch.cat([b1, b2], dim=1)  # (B, c_out_1 * 2, W)
            c1 = self.conv_act_1(self.conv_norm_1(c1))

            c2 = self.conv_stage_2(c1)  # (B, c_out_2, W)
            c2 = self.conv_act_2(self.conv_norm_2(c2))

            # Temporal Aggregation:
            # 1. Endpoint state: last flight cycle in window (instantaneous state)
            endpoint = c2[:, :, -1]  # (B, c_out_2)
            # 2. Mean temporal summary: average across all cycles in window
            mean_pool = c2.mean(dim=-1)  # (B, c_out_2)

            feat = torch.cat([endpoint, mean_pool], dim=-1)  # (B, c_out_2 * 2)

        else:
            raise ValueError(
                f"Expected 2D or 3D input tensor, got shape {tuple(x.shape)} with {x.dim()} dims"
            )

        # Thermodynamic Cross-Feature Processing with residual skips
        h1 = self.drop_1(self.act_1(self.norm_1(self.dense_1(feat))))
        h1 = h1 + self.skip_1(feat)

        h2 = self.drop_2(self.act_2(self.norm_2(self.dense_2(h1))))
        h2 = h2 + self.skip_2(h1)

        # Final Latent Projection
        z = self.latent_norm(self.latent_proj(h2))
        return z

    def get_latent_dim(self) -> int:
        """Return the dimension of the shared latent space."""
        return self.latent_dim

    def get_config(self) -> BackboneConfig:
        """Return a copy of the backbone configuration."""
        return self.config.copy_with()
