"""Final Displacement Error (FDE) metric."""

import torch


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor, n_pred: torch.Tensor) -> float:
    """L2 distance at the per-sample final prediction step.

    Args:
        predicted: [B, max_pred, 2] predicted positions (pixel space).
        ground_truth: [B, max_pred, 2] ground truth positions (pixel space).
        n_pred: [B] number of valid prediction steps per sample.

    Returns:
        Mean FDE across batch.
    """
    B = predicted.shape[0]
    last_idx = (n_pred - 1).clamp(min=0)  # [B]
    final_pred = predicted[torch.arange(B, device=predicted.device), last_idx, :]
    final_gt = ground_truth[torch.arange(B, device=ground_truth.device), last_idx, :]
    return torch.norm(final_pred - final_gt, dim=-1).mean().item()
