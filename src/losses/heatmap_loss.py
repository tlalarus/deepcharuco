import torch
from torch import nn


class HeatmapFocalLoss(nn.Module):
    """CenterNet-style focal loss for heatmap regression."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits).clamp(self.eps, 1.0 - self.eps)

        pos_mask = (target >= 1.0 - self.eps).float()
        neg_mask = (target < 1.0 - self.eps).float()
        neg_weights = (1.0 - target).pow(self.beta)

        pos_loss = -torch.log(pred) * (1.0 - pred).pow(self.alpha) * pos_mask
        neg_loss = -torch.log(1.0 - pred) * pred.pow(self.alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        total_loss = pos_loss.sum() + neg_loss.sum()

        if num_pos > 0:
            return total_loss / num_pos
        return neg_loss.sum()
