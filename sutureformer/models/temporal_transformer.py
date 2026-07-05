"""Transformer temporal encoder for observation sequence aggregation."""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modeling of the observation sequence.

    Supports variable-length inputs via obs_mask. Extracts the last valid
    frame's output per sample as the context vector z_ctx.

    Input:  [B, T, input_dim] (visual features + coordinate features)
    Output: z_ctx [B, d_model]
    """

    def __init__(
        self,
        input_dim: int = 320,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with variable-length masking.

        Args:
            x: [B, T, input_dim] where T = max_obs in batch.
            obs_mask: [B, T] boolean mask (True = valid, False = pad).

        Returns:
            z_ctx: [B, d_model]
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]  # add positional encoding

        # Causal mask: upper triangular boolean (True = block attention)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )  # [T, T]

        # Key padding mask: True = ignore (PyTorch convention)
        key_padding_mask = ~obs_mask  # [B, T]

        x = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=key_padding_mask
        )  # [B, T, d_model]

        # Extract last valid frame per sample
        n_obs = obs_mask.sum(dim=1).long()  # [B]
        last_idx = (n_obs - 1).clamp(min=0)  # [B]
        z_ctx = x[torch.arange(B, device=x.device), last_idx, :]  # [B, d_model]

        z_ctx = self.output_norm(z_ctx)
        return z_ctx
