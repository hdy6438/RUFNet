"""Visualization utilities for predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def save_prediction_panel(
    query_image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    uncertainty: torch.Tensor | None,
    output_path: str | Path,
    modality_index: int = 0,
) -> None:
    """Saves a four-panel query/ground-truth/prediction/uncertainty figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = query_image.detach().cpu().numpy()[modality_index]
    gt = ground_truth.detach().cpu().numpy().squeeze()
    pred = prediction.detach().cpu().numpy().squeeze()
    uncertainty_np = None if uncertainty is None else uncertainty.detach().cpu().numpy().squeeze()

    columns = 4 if uncertainty_np is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].imshow(normalize_for_display(image), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(normalize_for_display(image), cmap="gray")
    axes[1].imshow(gt, alpha=0.45, cmap="magma")
    axes[1].set_title("Ground truth")
    axes[2].imshow(normalize_for_display(image), cmap="gray")
    axes[2].imshow(pred, alpha=0.45, cmap="viridis")
    axes[2].set_title("RUFNet")
    if uncertainty_np is not None:
        axes[3].imshow(uncertainty_np, cmap="inferno")
        axes[3].set_title("Variance")
    for axis in axes:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

