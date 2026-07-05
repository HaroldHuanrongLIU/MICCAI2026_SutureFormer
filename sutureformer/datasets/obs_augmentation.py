"""Observation augmentation for training: coordinate jitter, color jitter, temporal dropout."""

import torch


def augment_observations(
    obs_images: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_mask: torch.BoolTensor,
    n_obs: torch.LongTensor,
    coord_jitter_std: float = 0.002,
    color_jitter_brightness: float = 0.1,
    temporal_dropout_prob: float = 0.2,
) -> tuple:
    """Apply observation augmentation in-place on batch tensors.

    Args:
        obs_images: [B, max_obs, 4, H, W] float32 (RGB+guidance).
        obs_coords: [B, max_obs, 2] float32 normalized coords.
        obs_mask: [B, max_obs] bool mask.
        n_obs: [B] per-sample observation lengths.
        coord_jitter_std: Std of Gaussian noise on coordinates.
        color_jitter_brightness: Max brightness shift (±) on RGB channels.
        temporal_dropout_prob: Probability of dropping a middle observation frame per sample.

    Returns:
        Tuple of (obs_images, obs_coords, obs_mask, n_obs) — augmented copies.
    """
    B, max_obs = obs_images.shape[:2]
    device = obs_images.device

    # 1. Coordinate jitter: add Gaussian noise to valid observation coords
    if coord_jitter_std > 0:
        noise = torch.randn_like(obs_coords) * coord_jitter_std
        obs_coords = (obs_coords + noise * obs_mask.unsqueeze(-1).float()).clamp(0.0, 1.0)

    # 2. Color jitter: random brightness shift on RGB channels only (not guidance ch 3)
    if color_jitter_brightness > 0:
        # Per-sample brightness shift
        shift = (torch.rand(B, 1, 1, 1, 1, device=device) * 2 - 1) * color_jitter_brightness
        obs_images = obs_images.clone()
        obs_images[:, :, :3] = (obs_images[:, :, :3] + shift).clamp(0.0, 1.0)

    # 3. Temporal dropout: randomly drop 1 middle observation frame per sample
    #    Only drop if n_obs >= 4 (keep at least 3 frames: first, one middle, last)
    if temporal_dropout_prob > 0:
        drop_mask = torch.rand(B, device=device) < temporal_dropout_prob
        eligible = n_obs >= 4  # need at least 4 to safely drop one
        drop_mask = drop_mask & eligible

        if drop_mask.any():
            obs_images = obs_images.clone()
            obs_coords = obs_coords.clone()
            obs_mask = obs_mask.clone()
            n_obs = n_obs.clone()

            for i in range(B):
                if not drop_mask[i]:
                    continue
                ni = n_obs[i].item()
                # Pick a random middle frame (not first, not last)
                drop_idx = torch.randint(1, ni - 1, (1,)).item()
                # Shift subsequent frames left by 1
                obs_images[i, drop_idx : ni - 1] = obs_images[i, drop_idx + 1 : ni].clone()
                obs_coords[i, drop_idx : ni - 1] = obs_coords[i, drop_idx + 1 : ni].clone()
                # Zero the last (now-unused) slot
                obs_images[i, ni - 1] = 0
                obs_coords[i, ni - 1] = 0
                obs_mask[i, ni - 1] = False
                n_obs[i] = ni - 1

    return obs_images, obs_coords, obs_mask, n_obs
