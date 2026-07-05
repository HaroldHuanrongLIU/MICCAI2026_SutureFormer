"""Direction and magnitude action heads for the RL agent."""

import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """MLP that outputs logits for 9 discrete directions.

    Directions: 0=up, 1=NE, 2=right, 3=SE, 4=down, 5=SW, 6=left, 7=NW, 8=idle

    Input:  [B, state_dim]
    Output: [B, num_actions] (raw logits for categorical distribution)
    """

    def __init__(self, state_dim: int = 480, hidden_dim: int = 256, num_actions: int = 9) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: [B, state_dim]

        Returns:
            logits: [B, num_actions]
        """
        return self.mlp(state)


class MagnitudeHead(nn.Module):
    """MLP that predicts step magnitude (normalized displacement).

    Output range: [0, delta_max] where delta_max is in normalized coordinate units.

    Input:  [B, state_dim]
    Output: [B, 1] in [0, delta_max]
    """

    def __init__(
        self, state_dim: int = 480, hidden_dim: int = 128, delta_max: float = 0.05
    ) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1], scaled by delta_max
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: [B, state_dim]

        Returns:
            magnitude: [B, 1] in [0, delta_max]
        """
        return self.mlp(state) * self.delta_max
