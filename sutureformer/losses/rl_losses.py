"""RL loss functions: entropy-regularized policy loss and behavior-cloning loss."""

import torch
import torch.nn.functional as F


def compute_policy_loss(
    direction_logits: torch.Tensor,
    q1_values: torch.Tensor,
    q2_values: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Entropy-regularized policy loss for discrete actions.

    J_pi = E_s [ sum_a pi(a|s) * (alpha * log pi(a|s) - min(Q1(s,a), Q2(s,a))) ]

    Args:
        direction_logits: [B, 9] raw logits from DirectionHead.
        q1_values: [B, 9] Q-values from Q-network 1.
        q2_values: [B, 9] Q-values from Q-network 2.
        alpha: Temperature parameter (scalar or tensor).

    Returns:
        Scalar policy loss.
    """
    probs = F.softmax(direction_logits, dim=-1)  # [B, 9]
    log_probs = F.log_softmax(direction_logits, dim=-1)  # [B, 9]
    min_q = torch.min(q1_values, q2_values)  # [B, 9]

    policy_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()
    return policy_loss


def compute_bc_loss(
    direction_logits: torch.Tensor,
    expert_actions: torch.Tensor,
) -> torch.Tensor:
    """Behavior cloning loss (cross-entropy on expert actions).

    Args:
        direction_logits: [B, num_actions] raw logits from DirectionHead.
        expert_actions: [B] LongTensor, expert action indices.

    Returns:
        Scalar BC loss.
    """
    return F.cross_entropy(direction_logits, expert_actions)
