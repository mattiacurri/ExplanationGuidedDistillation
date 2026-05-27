"""Token pooling heads for frozen vision-backbone classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdditiveAttentionPooling(nn.Module):
    """Learn token weights with a small additive-attention scorer."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        attention_hidden_size = max(64, hidden_size // 4)
        self.norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden_size),
            nn.GELU(),
            nn.Linear(attention_hidden_size, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_tokens = self.norm(tokens)
        scores = self.scorer(norm_tokens).squeeze(-1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask,
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * norm_tokens, dim=1)
        return pooled, weights


class CrossAttentionPooling(nn.Module):
    """Pool image tokens with a learnable query attending over the sequence."""

    def __init__(self, hidden_size: int, *, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            num_heads = 1
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_tokens = self.norm(tokens)
        query = self.query.expand(tokens.size(0), -1, -1)
        pooled, attention_weights = self.attention(
            query,
            norm_tokens,
            norm_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        token_weights = attention_weights.mean(dim=1).squeeze(1)
        return pooled.squeeze(1), token_weights
