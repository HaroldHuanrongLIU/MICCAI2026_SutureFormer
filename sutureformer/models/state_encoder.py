"""Prediction state encoder combining context, position, guidance, and step info."""

import torch
import torch.nn as nn

from sutureformer.models.coord_encoder import CoordinateEncoder


class PredictionStateEncoder(nn.Module):
    """Encode the state for each prediction step.

    Inputs: z_ctx [B, 256], position [B, 2], guidance [B, 2], step_ratio [B, 1]
    Output: state [B, state_dim]

    The relative displacement (guidance - position) is explicitly encoded
    to provide the model with direct information about the guidance direction.
    """

    def __init__(
        self,
        ctx_dim: int = 256,
        pos_dim: int = 64,
        guid_dim: int = 64,
        rel_dim: int = 64,
        step_dim: int = 32,
        coord_num_frequencies: int = 16,
    ) -> None:
        super().__init__()
        self.pos_encoder = CoordinateEncoder(
            embed_dim=pos_dim, num_frequencies=coord_num_frequencies
        )
        self.guid_encoder = CoordinateEncoder(
            embed_dim=guid_dim, num_frequencies=coord_num_frequencies
        )
        self.rel_encoder = CoordinateEncoder(
            embed_dim=rel_dim, num_frequencies=coord_num_frequencies
        )
        self.step_encoder = nn.Sequential(
            nn.Linear(1, step_dim),
            nn.ReLU(),
        )
        self.total_dim = ctx_dim + pos_dim + guid_dim + rel_dim + step_dim

    def forward(
        self,
        z_ctx: torch.Tensor,
        current_pos: torch.Tensor,
        guidance_pos: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z_ctx:        [B, 256] context from observation encoder.
            current_pos:  [B, 2]   normalized current position.
            guidance_pos: [B, 2]   normalized guidance target position.
            step_ratio:   [B, 1]   normalized step ratio (k / n_pred_i).

        Returns:
            state: [B, state_dim]
        """
        pos_feat = self.pos_encoder(current_pos.unsqueeze(1)).squeeze(1)  # [B, 64]
        guid_feat = self.guid_encoder(guidance_pos.unsqueeze(1)).squeeze(1)  # [B, 64]
        rel_disp = guidance_pos - current_pos  # [B, 2]
        rel_feat = self.rel_encoder(rel_disp.unsqueeze(1)).squeeze(1)  # [B, 64]
        step_feat = self.step_encoder(step_ratio)  # [B, 32]
        state = torch.cat([z_ctx, pos_feat, guid_feat, rel_feat, step_feat], dim=-1)
        return state
