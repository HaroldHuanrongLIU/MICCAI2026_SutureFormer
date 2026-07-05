"""Twin Q-networks for discrete-action offline RL (CQL)."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-network for discrete-action RL. Outputs Q-values for all discrete actions.

    Input:  [B, state_dim]
    Output: [B, num_actions] Q-values
    """

    def __init__(self, state_dim: int = 480, hidden_dim: int = 256, num_actions: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: [B, state_dim]

        Returns:
            q_values: [B, num_actions]
        """
        return self.net(state)
