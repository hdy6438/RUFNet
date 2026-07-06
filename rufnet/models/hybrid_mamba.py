"""Hybrid Mamba support-query interaction blocks."""

from __future__ import annotations

import torch
from torch import nn


def _load_mamba_class():
    try:
        from mamba_ssm import Mamba
    except Exception as exc:  # pragma: no cover - depends on optional CUDA package.
        raise ImportError(
            "RUFNet requires the official mamba-ssm package for Hybrid Mamba blocks. "
            "Install it with `pip install mamba-ssm` in a compatible PyTorch/CUDA "
            "environment."
        ) from exc
    return Mamba


def masked_average_pool(
    features: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes shot-wise masked prototypes from feature maps.

    Parameters
    ----------
    features:
        Tensor of shape ``(B, K, C, H, W)``.
    masks:
        Tensor of shape ``(B, K, 1, H, W)``.
    """

    weights = masks.float().clamp(0.0, 1.0)
    numerator = (features * weights).sum(dim=(-1, -2))
    denominator = weights.sum(dim=(-1, -2)).clamp_min(eps)
    return numerator / denominator


class SupportResetMambaBlock(nn.Module):
    """Mamba block with periodic support-token resets and query gating."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        support_reset_interval: int = 64,
    ):
        super().__init__()
        Mamba = _load_mamba_class()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.support_reset_interval = max(int(support_reset_interval), 1)
        self.query_gate = nn.Linear(dim, dim)
        self.support_gate = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.LayerNorm(dim)

    def _run_with_support_resets(
        self,
        query_tokens: torch.Tensor,
        support_token: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        chunk_lengths = []
        for start in range(0, query_tokens.shape[1], self.support_reset_interval):
            chunk = query_tokens[:, start : start + self.support_reset_interval]
            chunks.append(support_token)
            chunks.append(chunk)
            chunk_lengths.append(chunk.shape[1])

        sequence = torch.cat(chunks, dim=1)
        mixed = self.mamba(self.norm(sequence))

        outputs = []
        cursor = 0
        for chunk_len in chunk_lengths:
            cursor += 1
            outputs.append(mixed[:, cursor : cursor + chunk_len])
            cursor += chunk_len
        return torch.cat(outputs, dim=1)

    def forward(self, query_tokens: torch.Tensor, support_token: torch.Tensor) -> torch.Tensor:
        mixed = self._run_with_support_resets(query_tokens, support_token)
        support_context = support_token.expand(-1, query_tokens.shape[1], -1)
        gate = torch.sigmoid(self.query_gate(query_tokens) + self.support_gate(support_context))
        isolated = gate * mixed + (1.0 - gate) * query_tokens
        return self.out_norm(isolated + query_tokens)


class HybridMambaInteraction(nn.Module):
    """Stacks support-reset Mamba blocks over query feature tokens."""

    def __init__(
        self,
        channels: int,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        support_reset_interval: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SupportResetMambaBlock(
                    dim=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    support_reset_interval=support_reset_interval,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        bsz, channels, height, width = query_features.shape
        prototypes = masked_average_pool(support_features, support_masks)
        support_token = prototypes.mean(dim=1, keepdim=True)

        query_tokens = query_features.flatten(2).transpose(1, 2)
        for block in self.blocks:
            query_tokens = block(query_tokens, support_token)
        return query_tokens.transpose(1, 2).reshape(bsz, channels, height, width)

