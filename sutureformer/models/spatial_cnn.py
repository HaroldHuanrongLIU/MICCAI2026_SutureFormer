"""3D CNN spatial feature extractor with temporal kernel size 1."""

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """3D CNN that processes each frame independently (temporal kernel=1).

    Input:  [B, H, 4, 128, 128]  (H frames, 4 channels: RGB + guidance)
    Output: [B, H, 256]           (one feature vector per frame)
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 256) -> None:
        super().__init__()

        # PyTorch Conv3d expects (B, C, D, H, W)
        # We treat the frame dimension H as the depth D
        self.conv_blocks = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Block 2: 64x64 -> 32x32
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Block 3: 32x32 -> 16x16
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        # Output: [B, 128, H, 16, 16]

        # Global average pooling over spatial dims, keep temporal
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, 128, H, 1, 1]
        self.project = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, H, 4, 128, 128]

        Returns:
            features: [B, H, 256]
        """

        # Reshape: [B, H, C, H_img, W_img] -> [B, C, H, H_img, W_img]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv_blocks(x)  # [B, 128, H, 16, 16]
        x = self.spatial_pool(x)  # [B, 128, H, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, 128, H]
        x = x.permute(0, 2, 1)  # [B, H, 128]
        x = self.project(x)  # [B, H, 256]

        return x
