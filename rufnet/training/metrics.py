"""Segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch


def dice_coefficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = (prediction >= threshold).float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    if not mask.any():
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def hausdorff_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Computes symmetric 2D Hausdorff distance for a single mask pair."""

    from scipy import ndimage

    pred = prediction.detach().cpu().numpy() >= threshold
    gt = target.detach().cpu().numpy() > 0.5
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")

    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    gt_distance = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    return float(max(pred_distance[gt_surface].max(), gt_distance[pred_surface].max()))


def batch_metrics(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, float]:
    probs = outputs["final_prob"].detach()
    dice = dice_coefficient(probs, target).mean().item()
    hd_values = [hausdorff_distance(prob, mask) for prob, mask in zip(probs, target)]
    finite_hd = [value for value in hd_values if np.isfinite(value)]
    hd = float(np.mean(finite_hd)) if finite_hd else float("inf")
    return {"dice": dice, "hausdorff": hd}

