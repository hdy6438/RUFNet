"""Training loop for RUFNet."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import RUFNetLoss
from .metrics import batch_metrics


class PolynomialLR:
    """Polynomial learning-rate decay used by the paper."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_steps: int, power: float = 0.9):
        self.optimizer = optimizer
        self.max_steps = max(max_steps, 1)
        self.power = power
        self.step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        factor = (1.0 - min(self.step_count, self.max_steps) / self.max_steps) ** self.power
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor


class RUFNetTrainer:
    """Minimal trainer with separate main and Mamba optimizers."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        main_optimizer: torch.optim.Optimizer,
        mamba_optimizer: torch.optim.Optimizer | None = None,
        criterion: RUFNetLoss | None = None,
        schedulers: Iterable[PolynomialLR] | None = None,
        output_dir: str | Path = "checkpoints",
        amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.main_optimizer = main_optimizer
        self.mamba_optimizer = mamba_optimizer
        self.optimizers = [main_optimizer] + ([mamba_optimizer] if mamba_optimizer is not None else [])
        self.criterion = criterion or RUFNetLoss()
        self.schedulers = list(schedulers or [])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    def _move_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch["support_images"].to(self.device, non_blocking=True),
            batch["support_masks"].to(self.device, non_blocking=True),
            batch["query_images"].to(self.device, non_blocking=True),
            batch["query_masks"].to(self.device, non_blocking=True),
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        stats: dict[str, list[float]] = {"loss": [], "final_loss": [], "meta_loss": [], "variance_loss": []}
        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for batch in progress:
            support_images, support_masks, query_images, query_masks = self._move_batch(batch)
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(support_images, support_masks, query_images)
                losses = self.criterion(outputs, query_masks)

            self.scaler.scale(losses["loss"]).backward()
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
            for scheduler in self.schedulers:
                scheduler.step()

            for key in stats:
                stats[key].append(float(losses[key].item()))
            progress.set_postfix(loss=stats["loss"][-1])
        return {key: sum(values) / max(len(values), 1) for key, values in stats.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        dice_values: list[float] = []
        hd_values: list[float] = []
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            support_images, support_masks, query_images, query_masks = self._move_batch(batch)
            outputs = self.model(support_images, support_masks, query_images)
            metrics = batch_metrics(outputs, query_masks)
            dice_values.append(metrics["dice"])
            hd_values.append(metrics["hausdorff"])
        return {
            "dice": sum(dice_values) / max(len(dice_values), 1),
            "hausdorff": sum(hd_values) / max(len(hd_values), 1),
        }

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "main_optimizer": self.main_optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.mamba_optimizer is not None:
            payload["mamba_optimizer"] = self.mamba_optimizer.state_dict()
        torch.save(payload, path)

    def fit(self, epochs: int) -> dict[str, float]:
        best_metrics: dict[str, float] = {"dice": -1.0}
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate()
            metrics = {**{f"train_{k}": v for k, v in train_stats.items()}, **val_stats}
            self.save_checkpoint(self.output_dir / "last.pt", epoch, metrics)
            if val_stats and val_stats.get("dice", -1.0) > best_metrics.get("dice", -1.0):
                best_metrics = val_stats
                self.save_checkpoint(self.output_dir / "best.pt", epoch, metrics)
            print({"epoch": epoch, **metrics})
        return best_metrics

