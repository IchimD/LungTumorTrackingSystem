import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = torch.sum(probs * targets)
    cardinality = torch.sum(probs) + torch.sum(targets)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
