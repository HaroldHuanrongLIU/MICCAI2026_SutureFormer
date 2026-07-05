"""Guidance augmentation for training robustness."""

import torch


def augment_guidance(
    gt_guidance: torch.FloatTensor,
    noise_std: float = 0.005,
    p: float = 0.3,
) -> torch.FloatTensor:
    """Randomly perturb GT guidance during training.

    Makes the agent robust to imperfect guidance at test time.

    Args:
        gt_guidance: [B, F, 2] ground truth guidance coordinates.
        noise_std: Standard deviation of Gaussian noise.
        p: Probability of applying augmentation per sample.

    Returns:
        augmented_guidance: [B, F, 2] with optional noise added.
    """
    mask = (torch.rand(gt_guidance.shape[0], 1, 1, device=gt_guidance.device) < p).float()
    noise = torch.randn_like(gt_guidance) * noise_std
    return (gt_guidance + mask * noise).clamp(0.0, 1.0)
