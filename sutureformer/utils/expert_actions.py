"""Expert action extraction for offline CQL training."""

import torch
import torch.nn.functional as F

from sutureformer.datasets.augmentation import augment_guidance
from sutureformer.utils.direction_map import DIRECTION_VECTORS, compute_displacement
from sutureformer.utils.reward import compute_reward


def discretize_direction(displacement: torch.Tensor) -> torch.LongTensor:
    """Map a continuous displacement vector to the nearest discrete direction index.

    Args:
        displacement: [B, 2] displacement vectors.

    Returns:
        direction: [B] LongTensor in {0..8}.
    """
    norms = displacement.norm(dim=-1)  # [B]
    idle_mask = norms < 1e-6

    # Normalize to unit vectors (safe division)
    unit = displacement / norms.unsqueeze(-1).clamp(min=1e-8)  # [B, 2]

    # Compare against 8 directional unit vectors (exclude idle)
    dirs = DIRECTION_VECTORS[:8].to(displacement.device)  # [8, 2]
    cos_sim = F.cosine_similarity(
        unit.unsqueeze(1), dirs.unsqueeze(0), dim=-1
    )  # [B, 8]

    direction = cos_sim.argmax(dim=-1)  # [B]
    direction[idle_mask] = 8
    return direction


def extract_expert_transitions(model, z_ctx, batch, config):
    """Extract expert transitions from GT trajectories for offline CQL training.

    Walks GT positions step-by-step, computing states via the model's state encoder
    (with z_ctx gradients flowing), discretizing expert actions, and computing rewards.

    Args:
        model: SutureFormer model instance.
        z_ctx: [B, 256] observation context (with gradients).
        batch: Dict from collate_variable_length.
        config: Trainer config with reward params, guidance noise params.

    Returns:
        Dict with keys: states, actions, rewards, next_states, dones, mag_gt.
        All tensors concatenated across active (sample, step) pairs.
    """
    device = z_ctx.device
    pred_coords = batch["pred_coords"].to(device)  # [B, max_pred, 2]
    pred_conf = batch["pred_conf"].to(device)  # [B, max_pred]
    pred_is_kf = batch["pred_is_kf"].to(device)  # [B, max_pred]
    obs_coords = batch["obs_coords"].to(device)  # [B, max_obs, 2]
    n_pred = batch["n_pred"].to(device)  # [B]
    n_obs = batch["n_obs"].to(device)  # [B]

    B = z_ctx.shape[0]
    max_pred = pred_coords.shape[1]

    # Augment guidance for training
    guidance = augment_guidance(
        pred_coords,
        noise_std=config.guidance_noise_std,
        p=config.guidance_noise_prob,
    )  # [B, max_pred, 2]

    # Start from last valid observed position
    last_obs_idx = (n_obs - 1).clamp(min=0)
    start_pos = obs_coords[torch.arange(B, device=device), last_obs_idx, :]  # [B, 2]

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_mag_gt = []
    all_sample_ids = []
    all_step_ids = []

    for k in range(max_pred):
        active_mask = k < n_pred  # [B] bool
        if not active_mask.any():
            break

        # Current position: start_pos for k=0, pred_coords[:, k-1] for k>0
        if k == 0:
            current_pos = start_pos
        else:
            current_pos = pred_coords[:, k - 1, :]

        target_pos = pred_coords[:, k, :]  # [B, 2]

        # Step ratio
        step_ratio = torch.zeros(B, 1, device=device)
        step_ratio[active_mask, 0] = k / n_pred[active_mask].float()

        guidance_pos = guidance[:, k, :]  # [B, 2]

        # Compute state (z_ctx NOT detached - gradients flow)
        state = model.state_encoder(z_ctx, current_pos, guidance_pos, step_ratio)  # [B, 480]

        # Expert action from GT displacement
        gt_displacement = target_pos - current_pos  # [B, 2]
        action = discretize_direction(gt_displacement)  # [B]

        # Magnitude GT
        mag_gt = gt_displacement.norm(dim=-1, keepdim=True)  # [B, 1]

        # Expert predicted position (with quantization error from discretization)
        predicted_pos = (current_pos + compute_displacement(action, mag_gt)).clamp(0.0, 1.0)

        # Terminal flag
        is_terminal = k == n_pred - 1  # [B] bool

        # Reward
        reward = compute_reward(
            predicted_pos,
            target_pos,
            pred_conf[:, k],
            pred_is_kf[:, k],
            step_ratio.squeeze(1),
            is_terminal,
            r_time=config.r_time,
            d_thresh=config.d_thresh,
            d_max=config.d_max,
            r_prox_max=config.r_prox_max,
            r_prox_neg=config.r_prox_neg,
            r_terminal_max=config.r_terminal_max,
            sigma=config.sigma_terminal,
        )

        # Next state for non-terminal steps
        if k < max_pred - 1:
            next_step_ratio = torch.zeros(B, 1, device=device)
            next_active = (k + 1) < n_pred
            if next_active.any():
                next_step_ratio[next_active, 0] = (k + 1) / n_pred[next_active].float()
            next_guidance_pos = guidance[:, min(k + 1, max_pred - 1), :]
            next_state = model.state_encoder(
                z_ctx, target_pos, next_guidance_pos, next_step_ratio
            )  # [B, 480]
        else:
            next_state = state  # terminal

        # Collect only active transitions
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        all_states.append(state[active_idx])
        all_actions.append(action[active_idx])
        all_rewards.append(reward[active_idx])
        all_next_states.append(next_state[active_idx])
        all_dones.append(is_terminal[active_idx].float())
        all_mag_gt.append(mag_gt[active_idx])
        all_sample_ids.append(active_idx)
        all_step_ids.append(torch.full_like(active_idx, k))

    return {
        "states": torch.cat(all_states, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "rewards": torch.cat(all_rewards, dim=0),
        "next_states": torch.cat(all_next_states, dim=0),
        "dones": torch.cat(all_dones, dim=0),
        "mag_gt": torch.cat(all_mag_gt, dim=0),
        "sample_ids": torch.cat(all_sample_ids, dim=0),
        "step_ids": torch.cat(all_step_ids, dim=0),
    }
