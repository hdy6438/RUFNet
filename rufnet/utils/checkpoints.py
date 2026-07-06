"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint}

