"""Optional adapters for external meta-learning libraries."""

from __future__ import annotations

import torch


def wrap_maml(model: torch.nn.Module, inner_lr: float = 1e-3, first_order: bool = True):
    """Wraps a model with learn2learn's MAML implementation.

    RUFNet's default training is episodic prototype-style segmentation because
    it must preserve patient-level support/query exclusion. This helper is
    provided for ablation studies that need a library-backed MAML wrapper.
    """

    try:
        import learn2learn as l2l
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "learn2learn is required for MAML experiments. Install it with "
            "`pip install learn2learn`."
        ) from exc
    return l2l.algorithms.MAML(model, lr=inner_lr, first_order=first_order)

