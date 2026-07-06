"""Attention-Guided Mask Refinement (AGMR)."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentionGuidedMaskRefinement(nn.Module):
    """Refines support masks using support-to-query cross-attention.

    Inputs are support features ``(B, K, C, H, W)``, query features
    ``(B, C, H, W)``, and support masks ``(B, K, 1, H, W)``. The output is a
    soft refined support mask at feature resolution.
    """

    def __init__(self, channels: int, attention_dim: int | None = None, mask_mix: float = 0.25):
        super().__init__()
        attention_dim = attention_dim or max(channels // 2, 32)
        self.query_proj = nn.Conv2d(channels, attention_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(channels, attention_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(channels, attention_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(attention_dim, channels, kernel_size=1, bias=False)
        self.mask_embed = nn.Conv2d(1, channels, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.mask_mix = mask_mix

    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        bsz, shots, channels, height, width = support_features.shape
        masks = support_masks.float()
        if masks.shape[-2:] != (height, width):
            masks = F.interpolate(
                masks.flatten(0, 1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(bsz, shots, 1, height, width)

        support_flat = support_features.flatten(0, 1)
        query_flat = (
            query_features[:, None]
            .expand(bsz, shots, channels, height, width)
            .contiguous()
            .flatten(0, 1)
        )

        q = self.query_proj(support_flat).flatten(2).transpose(1, 2)
        k = self.key_proj(query_flat).flatten(2).transpose(1, 2)
        v = self.value_proj(query_flat).flatten(2).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        else:  # pragma: no cover - PyTorch < 2.0 fallback.
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
            attended = torch.bmm(attn, v)

        attended = attended.transpose(1, 2).reshape(bsz * shots, -1, height, width)
        enhanced = support_flat + self.gamma * self.out_proj(attended)
        mask_embed = self.mask_embed(masks.flatten(0, 1))
        refined_logits = self.fuse(torch.cat([enhanced, support_flat, mask_embed], dim=1))
        refined = torch.sigmoid(refined_logits)

        if self.mask_mix > 0:
            refined = (1.0 - self.mask_mix) * refined + self.mask_mix * masks.flatten(0, 1)
        return refined.view(bsz, shots, 1, height, width).clamp(0.0, 1.0)

