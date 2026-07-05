"""Confidence-weighted reward function for trajectory prediction."""

import torch


def compute_reward(
    predicted_pos: torch.FloatTensor,
    gt_pos: torch.FloatTensor,
    confidence: torch.FloatTensor,
    is_keyframe: torch.FloatTensor,
    step_ratio: torch.FloatTensor,
    is_terminal: torch.BoolTensor,
    r_time: float = -0.01,
    d_thresh: float = 0.02,
    d_max: float = 0.1,
    r_prox_max: float = 0.5,
    r_prox_neg: float = -0.25,
    r_terminal_max: float = 1.0,
    sigma: float = 0.03,
) -> torch.FloatTensor:
    """Compute the reward for one prediction step.

    Reward = r_time + w(k) * r_proximity(d_k) + r_terminal * is_terminal

    Args:
        predicted_pos: [B, 2] predicted position (normalized).
        gt_pos: [B, 2] ground truth position (normalized).
        confidence: [B] confidence score (1.0 for keyframe, 0.45-0.9 for interpolated).
        is_keyframe: [B] boolean flags (as float).
        step_ratio: [B] normalized step ratio (k / n_pred_i per sample).
        is_terminal: [B] boolean tensor, True if this is the last step for sample i.
        r_time: Per-step time penalty.
        d_thresh: Threshold for "close enough" (normalized).
        d_max: Max distance for penalty scaling (normalized).
        r_prox_max: Maximum proximity reward.
        r_prox_neg: Negative proximity penalty.
        r_terminal_max: Maximum terminal reward.
        sigma: Terminal reward decay rate.

    Returns:
        reward: [B] reward for each sample in batch.
    """
    # Distance to GT
    d_k = torch.norm(predicted_pos - gt_pos, dim=-1)  # [B]

    # Proximity reward
    close_mask = (d_k < d_thresh).float()
    r_proximity = close_mask * r_prox_max * (1.0 - d_k / d_thresh) + (
        1.0 - close_mask
    ) * r_prox_neg * (d_k / d_max).clamp(max=1.0)

    # Confidence weighting
    w = torch.where(
        is_keyframe.bool(),
        torch.ones_like(confidence),
        0.5 + 0.5 * confidence,
    )

    # Terminal reward (per-sample masking)
    r_terminal = torch.zeros_like(d_k)
    if is_terminal.any():
        r_terminal[is_terminal] = r_terminal_max * torch.exp(-d_k[is_terminal] / sigma)

    # Total reward
    reward = r_time + w * r_proximity + r_terminal

    return reward
