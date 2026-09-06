"""Multi-Task Loss functions for TurbofanGuard Step 2.

Combines continuous physics signal reconstruction loss (MSE) with
weighted multi-label fault classification loss (BCE with logits and positive class weighting).
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import torch
import torch.nn as nn


class MultiTaskFDILoss(nn.Module):
    """Joint Multi-Task Loss combining Physics Reconstruction and Diagnostic FDI.

    Formulation:
        L_total = L_recon + lambda_fdi * L_fdi

    Where:
        L_recon = MSE(y_hat, y_true)
        L_fdi   = BCEWithLogits(logits, m_true, pos_weight=w_pos)
    """

    def __init__(
        self,
        lambda_fdi: float = 0.5,
        pos_weight: Union[float, torch.Tensor] = 6.0,
        num_channels: int = 14,
    ) -> None:
        """Initialize MultiTaskFDILoss.

        Args:
            lambda_fdi: Balancing multiplier for FDI classification loss (default: 0.5).
            pos_weight: Positive weight factor for BCE to combat class imbalance.
            num_channels: Number of sensor channels (default: 14).
        """
        super().__init__()
        self.lambda_fdi = float(lambda_fdi)

        if isinstance(pos_weight, (int, float)):
            pos_weight_tensor = torch.full((num_channels,), float(pos_weight), dtype=torch.float32)
        else:
            pos_weight_tensor = pos_weight.to(torch.float32)

        self.register_buffer("pos_weight", pos_weight_tensor)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(
        self,
        y_hat: torch.Tensor,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        m_true: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss components.

        Args:
            y_hat: Predicted clean physical sensors of shape (B, 14).
            logits: Predicted unnormalized fault logits of shape (B, 14).
            y_true: Ground-truth clean physical targets of shape (B, 14).
            m_true: Ground-truth multi-hot binary fault matrix of shape (B, 14).

        Returns:
            Dictionary containing:
                - 'loss': Total combined scalar loss for backward pass.
                - 'recon_loss': Physical reconstruction MSE loss scalar.
                - 'fdi_loss': Weighted classification BCE loss scalar.
        """
        recon_loss = self.mse_loss(y_hat, y_true)
        fdi_loss = self.bce_loss(logits, m_true)
        total_loss = recon_loss + self.lambda_fdi * fdi_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "fdi_loss": fdi_loss,
        }
