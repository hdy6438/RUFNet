"""RUFNet end-to-end model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .agmr import AttentionGuidedMaskRefinement
from .encoder import ResNetPSPEncoder
from .hybrid_mamba import HybridMambaInteraction, masked_average_pool
from .uapf import UncertaintyAwarePosteriorFusion


class RUFNet(nn.Module):
    """Query-guided support mask refinement and uncertainty fusion network.

    The model expects a binary 1-way few-shot segmentation episode:

    * ``support_images``: ``(B, K, C, H, W)`` or ``(B, C, H, W)``
    * ``support_masks``: ``(B, K, 1, H, W)`` or ``(B, 1, H, W)``
    * ``query_images``: ``(B, C, H, W)``
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_channels: int = 256,
        encoder_pretrained: bool = False,
        mamba_depth: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        support_reset_interval: int = 64,
        prior_scale: float = 20.0,
        uncertainty_alpha: float = 1.0,
        use_agmr: bool = True,
        use_uapf: bool = True,
    ):
        super().__init__()
        self.encoder = ResNetPSPEncoder(
            in_channels=in_channels,
            out_channels=feature_channels,
            pretrained=encoder_pretrained,
        )
        self.agmr = AttentionGuidedMaskRefinement(feature_channels)
        self.interaction = HybridMambaInteraction(
            channels=feature_channels,
            depth=mamba_depth,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            support_reset_interval=support_reset_interval,
        )
        self.uapf = UncertaintyAwarePosteriorFusion(
            channels=feature_channels,
            alpha=uncertainty_alpha,
        )
        self.prior_scale = prior_scale
        self.use_agmr = use_agmr
        self.use_uapf = use_uapf

    @staticmethod
    def _ensure_episode_dims(
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_images.ndim == 4:
            support_images = support_images[:, None]
        if support_masks.ndim == 4:
            support_masks = support_masks[:, None]
        if support_images.ndim != 5 or support_masks.ndim != 5:
            raise ValueError("support images and masks must have shape (B,K,C,H,W)/(B,K,1,H,W).")
        return support_images, support_masks

    def _query_prior(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = masked_average_pool(support_features, support_masks).mean(dim=1)
        query_norm = F.normalize(query_features, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        similarity = torch.einsum("bchw,bc->bhw", query_norm, proto_norm).unsqueeze(1)
        return torch.sigmoid(self.prior_scale * similarity)

    @staticmethod
    def _resize_output(value: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        return F.interpolate(value, size=size, mode=mode, align_corners=False)

    def forward(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        support_images, support_masks = self._ensure_episode_dims(support_images, support_masks)
        bsz, shots, channels, image_height, image_width = support_images.shape
        target_size = query_images.shape[-2:]

        support_features = self.encoder(support_images.reshape(bsz * shots, channels, image_height, image_width))
        _, feature_channels, feat_height, feat_width = support_features.shape
        support_features = support_features.view(bsz, shots, feature_channels, feat_height, feat_width)
        query_features = self.encoder(query_images)

        support_masks_low = F.interpolate(
            support_masks.flatten(0, 1).float(),
            size=(feat_height, feat_width),
            mode="bilinear",
            align_corners=False,
        ).view(bsz, shots, 1, feat_height, feat_width)

        if self.use_agmr:
            refined_support_masks = self.agmr(support_features, query_features, support_masks_low)
        else:
            refined_support_masks = support_masks_low

        fused_query = self.interaction(query_features, support_features, refined_support_masks)
        prior = self._query_prior(fused_query, support_features, refined_support_masks)
        outputs = self.uapf(fused_query, prior)

        if not self.use_uapf:
            outputs["final_logits"] = outputs["meta_logits"]
            outputs["final_prob"] = outputs["meta_prob"]
            outputs["posterior_prob"] = outputs["meta_prob"]
            outputs["fusion_weight"] = torch.ones_like(outputs["meta_prob"])

        resized: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            resized[key] = self._resize_output(value, target_size)
        resized["prior_prob"] = self._resize_output(prior, target_size)
        resized["refined_support_masks"] = F.interpolate(
            refined_support_masks.flatten(0, 1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).view(bsz, shots, 1, *target_size)
        return resized

