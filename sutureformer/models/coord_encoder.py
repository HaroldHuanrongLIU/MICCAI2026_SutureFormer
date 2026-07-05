"""Sinusoidal coordinate encoder with MLP."""

import torch
import torch.nn as nn


class CoordinateEncoder(nn.Module):
    """Encode 2D coordinates using sinusoidal positional encoding + MLP.

    Input:  [B, H, 2]  (normalized x, y)
    Output: [B, H, embed_dim]
    """

    def __init__(self, coord_dim: int = 2, embed_dim: int = 64, num_frequencies: int = 16) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        # Input dim: coord_dim * 2 * num_frequencies = 2 * 2 * 16 = 64
        input_dim = coord_dim * 2 * num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: [B, H, 2] or [B, 1, 2], normalized to [0, 1].

        Returns:
            [B, H, embed_dim]
        """
        # Sinusoidal encoding
        freqs = 2.0 ** torch.arange(
            self.num_frequencies, device=coords.device, dtype=torch.float32
        )  # [num_freq]

        # [B, H, 2] -> [B, H, 2, num_freq]
        coords_expanded = coords.unsqueeze(-1) * freqs

        # [B, H, 2, 2*num_freq] (sin and cos concatenated)
        encoded = torch.cat([torch.sin(coords_expanded), torch.cos(coords_expanded)], dim=-1)

        # [B, H, 2 * 2 * num_freq] = [B, H, 64]
        encoded = encoded.flatten(-2)

        return self.mlp(encoded)  # [B, H, embed_dim]
