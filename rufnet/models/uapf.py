"""Uncertainty-Aware Posterior Fusion (UAPF)."""

from __future__ import annotations

import torch
from torch import nn


class UncertaintyAwarePosteriorFusion(nn.Module):
    """Predicts pixel-wise variance and fuses meta predictions with priors."""

    def __init__(self, channels: int, hidden_channels: int | None = None, alpha: float = 1.0):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 2, 32)
        self.alpha = alpha
        self.feature_head = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.logvar_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.posterior_refine = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, fused_features: torch.Tensor, prior: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_head(fused_features)
        mean_logits = self.mean_head(features)
        log_variance = self.logvar_head(features).clamp(min=-8.0, max=8.0)
        variance = torch.exp(log_variance)
        meta_prob = torch.sigmoid(mean_logits)

        fusion_weight = torch.exp(-self.alpha * variance).clamp(0.0, 1.0)
        posterior_prob = fusion_weight * meta_prob + (1.0 - fusion_weight) * prior.clamp(0.0, 1.0)
        final_logits = self.posterior_refine(posterior_prob)
        final_prob = torch.sigmoid(final_logits)

        return {
            "meta_logits": mean_logits,
            "meta_prob": meta_prob,
            "log_variance": log_variance,
            "variance": variance,
            "fusion_weight": fusion_weight,
            "posterior_prob": posterior_prob,
            "final_logits": final_logits,
            "final_prob": final_prob,
        }

