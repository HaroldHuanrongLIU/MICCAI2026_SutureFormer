"""Average Displacement Error (ADE) metric."""

import torch


def compute_ade(
    predicted: torch.Tensor, ground_truth: torch.Tensor, pred_mask: torch.Tensor
) -> float:
    """Masked average L2 distance across all valid prediction steps.

    Args:
        predicted: [B, max_pred, 2] predicted positions (pixel space).
        ground_truth: [B, max_pred, 2] ground truth positions (pixel space).
        pred_mask: [B, max_pred] boolean mask (True = valid, False = pad).

    Returns:
        Mean ADE across batch.
    """
    errors = torch.norm(predicted - ground_truth, dim=-1)  # [B, max_pred]
    errors = errors * pred_mask.float()
    per_sample = errors.sum(dim=1) / pred_mask.sum(dim=1).float().clamp(min=1)
    return per_sample.mean().item()
