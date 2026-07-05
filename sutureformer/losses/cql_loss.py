"""CQL-Discrete Q-loss: Bellman backup + conservative regularization penalty."""

import torch
import torch.nn.functional as F


def compute_nstep_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_states: torch.Tensor,
    sample_ids: torch.Tensor,
    step_ids: torch.Tensor,
    gamma: float,
    n_step: int,
) -> tuple:
    """Compute N-step returns by accumulating rewards along trajectories.

    For each transition (sample_id, step_id), accumulates rewards for up to
    n_step steps forward within the same sample, and returns the bootstrapping
    next_state at the n-th step.

    Args:
        rewards: [N] per-transition rewards.
        dones: [N] terminal flags (1.0 = terminal).
        next_states: [N, state_dim] next states.
        sample_ids: [N] which sample each transition belongs to.
        step_ids: [N] which step within the sample.
        gamma: Discount factor.
        n_step: Number of steps to accumulate.

    Returns:
        (nstep_rewards, nstep_next_states, nstep_dones, nstep_gamma):
            nstep_rewards: [N] accumulated discounted rewards.
            nstep_next_states: [N, state_dim] bootstrapping next states.
            nstep_dones: [N] whether the n-step trajectory hit a terminal.
            nstep_gamma: effective gamma^n for each transition.
    """
    N = rewards.shape[0]
    device = rewards.device

    # Build lookup: (sample_id, step_id) -> index in flat tensor
    # Use a tensor-based approach: encode (sample_id, step_id) as unique key
    max_step = step_ids.max().item() + 1
    flat_key = sample_ids * max_step + step_ids  # [N] unique key per transition
    key_to_idx = torch.full((flat_key.max().item() + 1,), -1, dtype=torch.long, device=device)
    key_to_idx[flat_key] = torch.arange(N, device=device)

    nstep_rewards = torch.zeros(N, device=device)
    nstep_next_states = next_states.clone()
    nstep_dones = dones.clone()
    nstep_gamma = torch.full((N,), gamma ** n_step, device=device)

    # Vectorized accumulation: iterate over steps (n_step is small, typically 3)
    discount = torch.ones(N, device=device)
    actual_steps = torch.zeros(N, dtype=torch.long, device=device)
    hit_done = torch.zeros(N, dtype=torch.bool, device=device)
    last_valid_idx = torch.arange(N, device=device)

    for step in range(n_step):
        future_key = sample_ids * max_step + (step_ids + step)
        in_range = future_key < key_to_idx.shape[0]
        future_idx = torch.where(in_range, key_to_idx[future_key.clamp(max=key_to_idx.shape[0] - 1)], torch.tensor(-1, device=device))
        valid = (future_idx >= 0) & ~hit_done

        if not valid.any():
            break

        nstep_rewards[valid] += discount[valid] * rewards[future_idx[valid]]
        actual_steps[valid] = step + 1
        last_valid_idx[valid] = future_idx[valid]

        newly_done = valid & (dones[future_idx.clamp(min=0)] > 0.5)
        hit_done = hit_done | newly_done
        discount[valid & ~newly_done] *= gamma

    nstep_dones[hit_done] = 1.0
    nstep_gamma = gamma ** actual_steps.float()
    nstep_next_states = next_states[last_valid_idx]

    return nstep_rewards, nstep_next_states, nstep_dones, nstep_gamma


def compute_cql_q_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_q1_target: torch.Tensor,
    next_q2_target: torch.Tensor,
    next_logits: torch.Tensor,
    alpha: float,
    alpha_cql: float,
    gamma: float,
    dones: torch.Tensor,
    n_step: int = 1,
) -> tuple:
    """CQL-Discrete Q-loss combining Bellman backup with conservative penalty.

    Args:
        q_values: [N, 9] Q-values for current states.
        actions: [N] LongTensor, expert action indices.
        rewards: [N] rewards.
        next_q1_target: [N, 9] target Q1 values for next states.
        next_q2_target: [N, 9] target Q2 values for next states.
        next_logits: [N, 9] policy logits for next states.
        alpha: Entropy temperature (scalar).
        alpha_cql: CQL regularization weight (scalar).
        gamma: Discount factor.
        dones: [N] float tensor (1.0 for terminal, 0.0 otherwise).

    Returns:
        (total_loss, bellman_loss_float, cql_penalty_float):
            total_loss: tensor with grad, bellman + alpha_cql * cql_penalty.
            bellman_loss_float: float, standard Bellman Q-loss value.
            cql_penalty_float: float, CQL penalty value.
    """
    # --- Bellman backup ---
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [N]

    with torch.no_grad():
        next_probs = F.softmax(next_logits, dim=-1)  # [N, 9]
        next_log_probs = F.log_softmax(next_logits, dim=-1)  # [N, 9]
        min_q_target_next = torch.min(next_q1_target, next_q2_target)  # [N, 9]
        next_v = (next_probs * (min_q_target_next - alpha * next_log_probs)).sum(dim=-1)  # [N]
        gamma_n = gamma ** n_step
        target = rewards + gamma_n * (1.0 - dones) * next_v

    bellman_loss = F.mse_loss(q_taken, target)

    # --- CQL conservative penalty ---
    # logsumexp(Q(s, .), dim=actions) - Q(s, a_expert)
    cql_penalty = (torch.logsumexp(q_values, dim=1) - q_taken).mean()

    total_loss = bellman_loss + alpha_cql * cql_penalty

    return total_loss, bellman_loss.item(), cql_penalty.item()
