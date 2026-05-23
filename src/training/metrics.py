import torch


def _binary_stats(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    preds = preds.float()
    targets = targets.float()
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    return intersection, union, torch.sum(preds), torch.sum(targets)


def threshold_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = threshold_predictions(logits, threshold)
    intersection, union, _, _ = _binary_stats(preds, targets, eps)
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice)


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = threshold_predictions(logits, threshold)
    intersection, union, sum_preds, sum_targets = _binary_stats(preds, targets, eps)
    union_area = sum_preds + sum_targets - intersection
    return float((intersection + eps) / (union_area + eps))


def sensitivity_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = threshold_predictions(logits, threshold)
    _, _, sum_preds, sum_targets = _binary_stats(preds, targets, eps)
    true_positive = torch.sum(preds * targets)
    return float((true_positive + eps) / (sum_targets + eps))


def precision_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = threshold_predictions(logits, threshold)
    true_positive, _, sum_preds, _ = _binary_stats(preds, targets, eps)
    return float((true_positive + eps) / (sum_preds + eps))
