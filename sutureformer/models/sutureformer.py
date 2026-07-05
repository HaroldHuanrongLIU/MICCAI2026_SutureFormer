"""SutureFormer: complete model combining observation encoder and prediction heads."""

import torch
import torch.nn as nn

from sutureformer.models.action_heads import DirectionHead, MagnitudeHead
from sutureformer.models.observation_encoder import ObservationEncoder
from sutureformer.models.state_encoder import PredictionStateEncoder


class SutureFormer(nn.Module):
    """Complete SutureFormer model for surgical trajectory prediction.

    Combines:
    - ObservationEncoder: images + coords + obs_mask -> z_ctx [B, 256]
    - PredictionStateEncoder: z_ctx + pos + guidance + rel + step_ratio -> state [B, 480]
    - DirectionHead: state -> logits [B, 9]
    - MagnitudeHead: state -> magnitude [B, 1]
    """

    def __init__(
        self,
        # Spatial CNN
        cnn_in_channels: int = 4,
        cnn_feature_dim: int = 256,
        # Coordinate Encoder
        coord_embed_dim: int = 64,
        coord_num_frequencies: int = 16,
        # Temporal Transformer
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_layers: int = 4,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
        # State Encoder
        state_dim: int = 480,
        # Action Heads
        num_directions: int = 9,
        delta_max: float = 0.05,
        # Q-Networks
        q_hidden_dim: int = 256,
        # Hydra
        _target_: str = None,
    ) -> None:
        super().__init__()

        self.obs_encoder = ObservationEncoder(
            cnn_in_channels=cnn_in_channels,
            cnn_feature_dim=cnn_feature_dim,
            coord_embed_dim=coord_embed_dim,
            coord_num_frequencies=coord_num_frequencies,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
        )

        ctx_dim = transformer_d_model  # 256
        pos_dim = coord_embed_dim  # 64
        guid_dim = coord_embed_dim  # 64
        rel_dim = coord_embed_dim  # 64
        step_dim = state_dim - ctx_dim - pos_dim - guid_dim - rel_dim  # 32

        self.state_encoder = PredictionStateEncoder(
            ctx_dim=ctx_dim,
            pos_dim=pos_dim,
            guid_dim=guid_dim,
            rel_dim=rel_dim,
            step_dim=step_dim,
            coord_num_frequencies=coord_num_frequencies,
        )
        self.direction_head = DirectionHead(
            state_dim=state_dim,
            hidden_dim=q_hidden_dim,
            num_actions=num_directions,
        )
        self.magnitude_head = MagnitudeHead(
            state_dim=state_dim,
            delta_max=delta_max,
        )

    def encode_observation(
        self, images: torch.Tensor, coords: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode observation frames into context vector.

        Args:
            images: [B, T, 4, 128, 128]
            coords: [B, T, 2]
            obs_mask: [B, T] boolean mask (True = valid, False = pad).

        Returns:
            z_ctx: [B, 256]
        """
        return self.obs_encoder(images, coords, obs_mask)

    def predict_step(
        self,
        z_ctx: torch.Tensor,
        current_pos: torch.Tensor,
        guidance_pos: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> tuple:
        """Predict one step of the trajectory.

        Args:
            z_ctx:        [B, 256]
            current_pos:  [B, 2]
            guidance_pos: [B, 2]
            step_ratio:   [B, 1] normalized step ratio (k / n_pred_i per sample).

        Returns:
            direction_logits: [B, 9]
            magnitude: [B, 1]
        """
        state = self.state_encoder(z_ctx, current_pos, guidance_pos, step_ratio)
        direction_logits = self.direction_head(state)  # [B, 9]
        magnitude = self.magnitude_head(state)  # [B, 1]
        return direction_logits, magnitude
