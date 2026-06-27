import torch
from torch import nn


class MaskedL1Loss(nn.Module):
    """L1 loss computed only where mask==1."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask shape: [B, 1, H, W]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        mask = mask.float()
        loss = (pred - target).abs() * mask
        denom = mask.sum() * pred.shape[1] + self.eps
        return loss.sum() / denom
