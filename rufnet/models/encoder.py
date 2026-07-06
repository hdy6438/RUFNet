"""Backbone encoder used by RUFNet.

The paper initializes the encoder from a PSPNet ResNet-50 model. This module
implements the same family of encoder: a ResNet-50 feature extractor followed
by a PSP-style context pooling head.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    """Pyramid pooling head used to add global context to CNN features."""

    def __init__(self, in_channels: int, out_channels: int, bins: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        branch_channels = max(out_channels // len(bins), 1)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(bin_size, bin_size)),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for bin_size in bins
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + branch_channels * len(bins), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]
        return self.bottleneck(torch.cat([x, *pooled], dim=1))


def _replace_first_conv(module: nn.Module, in_channels: int) -> None:
    old_conv = module.conv1
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        elif in_channels > old_conv.in_channels:
            new_conv.weight[:, : old_conv.in_channels].copy_(old_conv.weight)
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for channel in range(old_conv.in_channels, in_channels):
                new_conv.weight[:, channel : channel + 1].copy_(mean_weight)
        else:
            new_conv.weight.copy_(old_conv.weight[:, :in_channels])
    module.conv1 = new_conv


class ResNetPSPEncoder(nn.Module):
    """ResNet-50 + PSP context encoder.

    Parameters
    ----------
    in_channels:
        Number of image modalities. BraTS uses four modalities by default.
    out_channels:
        Feature width consumed by AGMR, Mamba interaction, and UAPF.
    pretrained:
        If true, loads ImageNet weights from torchvision and adapts the first
        convolution to the requested number of channels.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except Exception as exc:  # pragma: no cover - import error message is environment-specific.
            raise ImportError("torchvision is required for ResNetPSPEncoder.") from exc

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        _replace_first_conv(backbone, in_channels)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.psp = PSPModule(in_channels=1024, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.psp(x)

