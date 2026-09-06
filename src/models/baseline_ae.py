"""Step 1: Baseline Denoising Autoencoder / Virtual Sensor Model for TurbofanGuard.

Combines TurbofanBackbone with a Reconstruction Decoder to estimate clean physical
target signals ŷ_t ∈ ℝ¹⁴ and compute normalized physical residuals for Fault Detection & Isolation (FDI).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from src.models.backbone import TurbofanBackbone, _get_activation
from src.utils.config import BackboneConfig


class ReconstructionDecoder(nn.Module):
    """Reconstruction decoder head mapping shared latent state z_t to clean sensors ŷ_t."""

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 14,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                _get_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final linear projection to the 14 clean sensor channels
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through reconstruction decoder.

        Args:
            z: Latent state tensor of shape (batch_size, latent_dim).

        Returns:
            y_hat: Reconstructed clean sensors of shape (batch_size, output_dim).
        """
        return self.net(z)


class TurbofanBaselineAE(nn.Module):
    """Complete Step 1 Baseline Denoising Autoencoder & Virtual Sensor.

    Architecture:
        Input (B, W, 18) or (B, 18) ──> TurbofanBackbone ──> Latent z_t (B, 64)
                                                                  │
                                                                  ▼
                                                      ReconstructionDecoder
                                                                  │
                                                                  ▼
                                                       Clean ŷ_t (B, 14)
    """

    def __init__(
        self,
        config: Optional[BackboneConfig] = None,
        decoder_hidden_dims: Optional[list[int]] = None,
        output_dim: int = 14,
    ) -> None:
        super().__init__()
        self.backbone = TurbofanBackbone(config=config)
        self.config = self.backbone.get_config()
        self.output_dim = output_dim

        self.decoder = ReconstructionDecoder(
            latent_dim=self.config.latent_dim,
            output_dim=self.output_dim,
            hidden_dims=decoder_hidden_dims if decoder_hidden_dims else [128, 64],
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate clean thermodynamic sensors ŷ_t from telemetry inputs.

        Args:
            x: Input telemetry tensor. Either:
               - 3D Temporal window: (batch_size, window_length, 18)
               - 2D Single snapshot: (batch_size, 18)

        Returns:
            y_hat: Clean physical sensor predictions of shape (batch_size, 14).
        """
        z = self.backbone(x)
        return self.decoder(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the shared latent representation z_t."""
        return self.backbone(x)

    def infer_residuals(
        self,
        x: torch.Tensor,
        sigma_nominal: Optional[torch.Tensor] = None,
        threshold: float = 3.5,
    ) -> Dict[str, torch.Tensor]:
        """Perform real-time residual inference and threshold-based FDI.

        Args:
            x: Input telemetry tensor of shape (B, W, 18) or (B, 18).
            sigma_nominal: Optional baseline nominal std of shape (14,) or (1, 14).
            threshold: Standard deviation threshold tau for alarm trigger (default: 3.5).

        Returns:
            Dictionary containing:
                - 'y_hat': Model clean predictions (B, 14).
                - 'residuals': Normalized residuals r_i (B, 14).
                - 'is_alarm': Boolean alarm flags (B,).
                - 'isolated_sensor': Predicted failing sensor index (B,).
        """
        self.eval()
        with torch.no_grad():
            y_hat = self.forward(x)

            # Extract the observed sensor slice (indices 4..17 in input features)
            # If 3D window (B, W, 18), extract endpoint cycle (last cycle in window)
            if x.dim() == 3:
                x_obs = x[:, -1, 4:18]
            elif x.dim() == 2:
                x_obs = x[:, 4:18]
            else:
                raise ValueError(f"Expected 2D or 3D input tensor, got shape {tuple(x.shape)}")

            # Raw absolute error
            abs_err = torch.abs(x_obs - y_hat)

            # Normalize by nominal healthy standard deviation if available
            if sigma_nominal is not None:
                # Ensure sigma is on the same device and broadcastable
                sigma = sigma_nominal.to(x.device).view(1, -1)
                # Clamp minimum sigma to prevent division by zero
                sigma = torch.clamp(sigma, min=1e-5)
                residuals = abs_err / sigma
            else:
                residuals = abs_err

            # Threshold check
            is_alarm = (residuals > threshold).any(dim=-1)
            isolated_sensor = torch.argmax(residuals, dim=-1)

            return {
                "y_hat": y_hat,
                "residuals": residuals,
                "is_alarm": is_alarm,
                "isolated_sensor": isolated_sensor,
            }
