"""Test-time guidance generation via polynomial extrapolation."""

import torch


def generate_test_guidance(
    obs_coords: torch.FloatTensor,
    obs_mask: torch.BoolTensor,
    n_pred: torch.LongTensor,
    lookback: int = 10,
) -> torch.FloatTensor:
    """Generate pseudo-guidance for prediction phase during testing.

    Uses quadratic polynomial extrapolation (least-squares fit) when >=5
    frames are available, falling back to linear extrapolation otherwise.
    Handles variable-length observations via obs_mask.

    Args:
        obs_coords: [B, max_obs, 2] observed coordinates (normalized), padded.
        obs_mask: [B, max_obs] boolean mask (True = valid).
        n_pred: [B] number of prediction steps per sample.
        lookback: Number of recent frames to use for fitting.

    Returns:
        pseudo_guidance: [B, max_pred, 2] extrapolated guidance positions.
    """
    B = obs_coords.shape[0]
    device = obs_coords.device
    max_pred = n_pred.max().item()

    n_obs = obs_mask.sum(dim=1).long()  # [B]

    guidance = torch.zeros(B, max_pred, 2, device=device)

    for b in range(B):
        n = n_obs[b].item()
        if n < 2:
            guidance[b] = obs_coords[b, 0, :].unsqueeze(0).expand(max_pred, -1)
            continue

        lb = min(lookback, n)
        start = n - lb
        recent = obs_coords[b, start:n, :]  # [lb, 2]
        last_pos = recent[-1]

        if lb >= 5:
            # Quadratic polynomial fit: pos = a*t^2 + b*t + c
            t_obs = torch.arange(lb, dtype=torch.float32, device=device)
            V = torch.stack([t_obs ** 2, t_obs, torch.ones_like(t_obs)], dim=1)  # [lb, 3]
            coeffs = torch.linalg.lstsq(V, recent).solution  # [3, 2]

            t_pred = torch.arange(lb, lb + max_pred, dtype=torch.float32, device=device)
            V_pred = torch.stack([t_pred ** 2, t_pred, torch.ones_like(t_pred)], dim=1)  # [max_pred, 3]
            guidance[b] = V_pred @ coeffs  # [max_pred, 2]
        else:
            # Linear extrapolation fallback (vectorized over prediction steps)
            velocity = (recent[-1] - recent[0]) / max(lb - 1, 1)
            steps = torch.arange(1, max_pred + 1, dtype=torch.float32, device=device).unsqueeze(1)  # [max_pred, 1]
            guidance[b] = last_pos.unsqueeze(0) + velocity.unsqueeze(0) * steps  # [max_pred, 2]

    return guidance.clamp(0.0, 1.0)
