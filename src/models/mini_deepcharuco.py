from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

try:
    from losses.heatmap_loss import HeatmapFocalLoss
    from losses.offset_loss import MaskedL1Loss
except ImportError:
    from ..losses.heatmap_loss import HeatmapFocalLoss
    from ..losses.offset_loss import MaskedL1Loss


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _FallbackBackbone(nn.Module):
    """Simple fallback backbone with output stride=4."""

    def __init__(self, in_channels: int = 1, out_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBnRelu(in_channels, 32, stride=2),
            _ConvBnRelu(32, 64, stride=2),
            _ConvBnRelu(64, 128, stride=1),
            _ConvBnRelu(128, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class _ResNet18Backbone(nn.Module):
    """ResNet18 feature extractor adjusted to grayscale input, output stride=4."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except Exception:
            self.model = _FallbackBackbone(in_channels=in_channels, out_channels=128)
            self.out_channels = 128
            self._fallback = True
            return

        backbone = resnet18(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Keep up to layer1 to maintain output stride=4.
        self.model = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.out_channels = 64
        self._fallback = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MiniDeepCharuco(nn.Module):
    def __init__(self, num_corners: int, in_channels: int = 1, backbone: str = "resnet18"):
        super().__init__()
        self.num_corners = num_corners

        if backbone.lower() == "resnet18":
            self.backbone = _ResNet18Backbone(in_channels=in_channels)
            feat_ch = self.backbone.out_channels
        else:
            self.backbone = _FallbackBackbone(in_channels=in_channels, out_channels=128)
            feat_ch = 128

        self.neck = nn.Sequential(
            _ConvBnRelu(feat_ch, 128, stride=1),
            _ConvBnRelu(128, 128, stride=1),
        )
        self.heatmap_head = nn.Conv2d(128, num_corners, kernel_size=1)
        self.offset_head = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.neck(feat)
        heatmap = self.heatmap_head(feat)
        offset = self.offset_head(feat)
        return {"heatmap": heatmap, "offset": offset}

    @torch.no_grad()
    def infer_image(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if img.ndim == 3:
            img = img.unsqueeze(0)
        out = self.forward(img)
        return out["heatmap"], out["offset"]


@dataclass
class MiniLossConfig:
    lambda_offset: float = 1.0
    heatmap_alpha: float = 2.0
    heatmap_beta: float = 4.0


class lMiniModel(pl.LightningModule):
    def __init__(self, model: MiniDeepCharuco, loss_config: MiniLossConfig, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss_config = loss_config
        self.lr = lr

        self.heatmap_loss = HeatmapFocalLoss(alpha=loss_config.heatmap_alpha, beta=loss_config.heatmap_beta)
        self.offset_loss = MaskedL1Loss()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def infer_image(self, img: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(img, np.ndarray):
            img = torch.as_tensor(img, dtype=torch.float32, device=self.device)
        img = img.to(self.device)
        return self.model.infer_image(img)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        image = batch["image"]
        heatmap_gt = batch["heatmap"]
        offset_gt = batch["offset"]
        offset_mask = batch["offset_mask"]

        pred = self.model(image)
        heatmap_pred = pred["heatmap"]
        offset_pred = pred["offset"]

        loss_hm = self.heatmap_loss(heatmap_pred, heatmap_gt)
        loss_off = self.offset_loss(offset_pred, offset_gt, offset_mask)
        loss = loss_hm + self.loss_config.lambda_offset * loss_off

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_loss_heatmap", loss_hm)
        self.log(f"{stage}_loss_offset", loss_off)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
