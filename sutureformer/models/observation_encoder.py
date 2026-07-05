"""Combined observation encoder: SpatialCNN + CoordinateEncoder + TemporalTransformer."""

import torch
import torch.nn as nn

from sutureformer.models.coord_encoder import CoordinateEncoder
from sutureformer.models.spatial_cnn import SpatialCNN
from sutureformer.models.temporal_transformer import TemporalTransformer


class ObservationEncoder(nn.Module):
    """Full observation encoder combining visual, coordinate, and temporal processing.

    Input:  images [B, T, 4, 128, 128], coords [B, T, 2], obs_mask [B, T]
    Output: z_ctx [B, 256]
    """

    def __init__(
        self,
        cnn_in_channels: int = 4,
        cnn_feature_dim: int = 256,
        coord_embed_dim: int = 64,
        coord_num_frequencies: int = 16,
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_layers: int = 4,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_cnn = SpatialCNN(in_channels=cnn_in_channels, feature_dim=cnn_feature_dim)
        self.coord_encoder = CoordinateEncoder(
            embed_dim=coord_embed_dim, num_frequencies=coord_num_frequencies
        )
        self.temporal_transformer = TemporalTransformer(
            input_dim=cnn_feature_dim + coord_embed_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
        )

    def forward(
        self, images: torch.Tensor, coords: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            images: [B, T, 4, 128, 128] (RGB + guidance channel).
            coords: [B, T, 2] (normalized x, y).
            obs_mask: [B, T] boolean mask (True = valid, False = pad).

        Returns:
            z_ctx: [B, 256]
        """
        visual_features = self.spatial_cnn(images)  # [B, T, 256]
        coord_features = self.coord_encoder(coords)  # [B, T, 64]
        combined = torch.cat([visual_features, coord_features], dim=-1)  # [B, T, 320]
        z_ctx = self.temporal_transformer(combined, obs_mask)  # [B, 256]
        return z_ctx
