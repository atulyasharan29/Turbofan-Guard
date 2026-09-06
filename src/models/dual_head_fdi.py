"""Step 2: Dual-Head Multi-Task Network for TurbofanGuard (Physics + Diagnostic AI).

Combines TurbofanBackbone with:
- Head 1 (ReconstructionDecoder): Predicts continuous physical clean signals ŷ_t ∈ ℝ¹⁴
- Head 2 (DiagnosticClassifierHead): Predicts multi-label fault probabilities p_t ∈ [0, 1]¹⁴
- infer_adaptive_fdi: Fuses physical residuals with AI probabilities for Two-Factor Authentication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.models.backbone import TurbofanBackbone, _get_activation
from src.models.baseline_ae import ReconstructionDecoder
from src.utils.config import BackboneConfig, DualHeadConfig


class DiagnosticClassifierHead(nn.Module):
    """Multi-label fault diagnostic classifier head mapping latent state z_t to fault logits."""

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 14,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: List[nn.Module] = []
        in_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                _get_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final linear projection to unnormalized logits for all 14 sensor channels
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized fault logits from latent state z.

        Args:
            z: Latent state tensor of shape (batch_size, latent_dim).

        Returns:
            logits: Fault logits tensor of shape (batch_size, output_dim).
        """
        return self.net(z)

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Compute fault probabilities using Sigmoid activation."""
        return torch.sigmoid(self.forward(z))


class TurbofanDualHeadFDI(nn.Module):
    """Step 2 Complete Dual-Head Multi-Task Network.

    Architecture:
        Input Sequence (B, W, 18) ──> TurbofanBackbone ──> Latent z_t (B, 64)
                                                                 │
                                ┌────────────────────────────────┴──────────────────────────────┐
                                ▼                                                               ▼
                     Head 1: ReconstructionDecoder                                  Head 2: DiagnosticClassifierHead
                                │                                                               │
                                ▼                                                               ▼
                     Clean Sensors ŷ_t (B, 14)                                      Fault Probabilities p_t (B, 14)
    """

    def __init__(
        self,
        config: Optional[DualHeadConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = DualHeadConfig()
        self.config = config

        # Shared Backbone Encoder
        self.backbone = TurbofanBackbone(config=self.config.backbone)

        # Head 1: Physics Reconstruction Decoder
        self.recon_head = ReconstructionDecoder(
            latent_dim=self.config.backbone.latent_dim,
            output_dim=self.config.output_dim,
            hidden_dims=self.config.recon_hidden_dims,
            activation=self.config.backbone.activation,
            dropout=self.config.backbone.dropout,
        )

        # Head 2: Diagnostic FDI Classifier Head
        self.diag_head = DiagnosticClassifierHead(
            latent_dim=self.config.backbone.latent_dim,
            output_dim=self.config.output_dim,
            hidden_dims=self.config.diag_hidden_dims,
            activation=self.config.backbone.activation,
            dropout=self.config.backbone.dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-task forward pass.

        Args:
            x: Input telemetry tensor of shape (B, W, 18) or (B, 18).

        Returns:
            Dictionary containing:
                - 'y_hat': Clean sensor predictions (B, 14).
                - 'logits': Unnormalized fault logits (B, 14).
                - 'p_fault': Sigmoid fault probabilities (B, 14).
                - 'latent': Shared latent representation z_t (B, 64).
        """
        z = self.backbone(x)
        y_hat = self.recon_head(z)
        logits = self.diag_head(z)
        p_fault = torch.sigmoid(logits)

        return {
            "y_hat": y_hat,
            "logits": logits,
            "p_fault": p_fault,
            "latent": z,
        }

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared latent representation z_t."""
        return self.backbone(x)

    def infer_adaptive_fdi(
        self,
        x: torch.Tensor,
        sigma_nominal: Optional[torch.Tensor] = None,
        tau_high: Optional[float] = None,
        tau_low: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform real-time Two-Factor Authentication FDI using adaptive thresholding.

        Decision Rule:
            tau_adaptive(p_i) = tau_high - (tau_high - tau_low) * p_i
            Trigger Alarm on Sensor i <=> residual_i > tau_adaptive(p_i)

        Args:
            x: Input telemetry tensor of shape (B, W, 18) or (B, 18).
            sigma_nominal: Baseline nominal std of shape (14,) or (1, 14).
            tau_high: Conservative high threshold when AI confidence is 0% (default: from config).
            tau_low: Sensitive low threshold when AI confidence is 100% (default: from config).

        Returns:
            Dictionary containing:
                - 'y_hat': Model clean physical predictions (B, 14).
                - 'p_fault': Diagnostic fault probabilities (B, 14).
                - 'residuals': Normalized physical residuals r_i (B, 14).
                - 'tau_adaptive': Dynamic adaptive thresholds (B, 14).
                - 'is_alarm': Boolean overall engine alarm flag (B,).
                - 'is_sensor_alarm': Boolean per-sensor alarm mask (B, 14).
                - 'isolated_sensors': Multi-hot isolated fault mask (B, 14).
                - 'primary_fault': Sensor index with highest residual excess (B,).
        """
        self.eval()
        th_high = tau_high if tau_high is not None else self.config.tau_high
        th_low = tau_low if tau_low is not None else self.config.tau_low

        with torch.no_grad():
            out = self.forward(x)
            y_hat = out["y_hat"]
            p_fault = out["p_fault"]

            # Extract observed sensor readings (indices 4..17)
            if x.dim() == 3:
                x_obs = x[:, -1, 4:18]
            elif x.dim() == 2:
                x_obs = x[:, 4:18]
            else:
                raise ValueError(f"Expected 2D or 3D input tensor, got shape {tuple(x.shape)}")

            abs_err = torch.abs(x_obs - y_hat)

            # Normalize by nominal noise standard deviation
            if sigma_nominal is not None:
                sigma = sigma_nominal.to(x.device).view(1, -1)
                sigma = torch.clamp(sigma, min=1e-5)
                residuals = abs_err / sigma
            else:
                residuals = abs_err

            # Adaptive dynamic threshold computation:
            # tau_adaptive = tau_high - (tau_high - tau_low) * p_fault
            tau_adaptive = th_high - (th_high - th_low) * p_fault

            # Two-Factor Trigger: residual exceeds adaptive threshold
            is_sensor_alarm = residuals > tau_adaptive
            is_alarm = is_sensor_alarm.any(dim=-1)
            isolated_sensors = is_sensor_alarm.to(torch.float32)

            # Primary fault index: largest margin over adaptive threshold
            margin = residuals - tau_adaptive
            primary_fault = torch.argmax(margin, dim=-1)

            return {
                "y_hat": y_hat,
                "p_fault": p_fault,
                "residuals": residuals,
                "tau_adaptive": tau_adaptive,
                "is_alarm": is_alarm,
                "is_sensor_alarm": is_sensor_alarm,
                "isolated_sensors": isolated_sensors,
                "primary_fault": primary_fault,
            }
