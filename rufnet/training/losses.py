"""Loss functions for RUFNet."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class RUFNetLoss(nn.Module):
    """Combined BCE, Dice, auxiliary meta, and variance regularization loss."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        aux_meta_weight: float = 0.3,
        variance_weight: float = 1e-4,
        use_monai: bool = True,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.aux_meta_weight = aux_meta_weight
        self.variance_weight = variance_weight
        self.monai_loss = None
        if use_monai:
            try:
                from monai.losses import DiceCELoss

                self.monai_loss = DiceCELoss(
                    sigmoid=True,
                    lambda_dice=dice_weight,
                    lambda_ce=bce_weight,
                )
            except Exception:
                self.monai_loss = None

    def _segmentation_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.monai_loss is not None:
            return self.monai_loss(logits, target.float())
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = soft_dice_loss(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        final_loss = self._segmentation_loss(outputs["final_logits"], target)
        meta_loss = self._segmentation_loss(outputs["meta_logits"], target)
        variance_loss = outputs["variance"].mean()
        total = final_loss + self.aux_meta_weight * meta_loss + self.variance_weight * variance_loss
        return {
            "loss": total,
            "final_loss": final_loss.detach(),
            "meta_loss": meta_loss.detach(),
            "variance_loss": variance_loss.detach(),
        }
