import numpy as np
from einops import rearrange
from MultiHeadAttn import MultiHeadSelfAttention

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Implement transformer block in the paper "Attention is all you need"
    Paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim, num_heads = 8, dim_head = None,
        dim_linear_block = 1024, activation = nn.ReLU, dropout = 0.1, mhsa = None
    ):
        super(TransformerBlock, self).__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(p = dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Positionwise fully connected feed-forward network
        self.linear_transform = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # Default is ReLU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # Residual Dropout connection around each of the sub-layers.
        # LayerNorm(Dropout(x) + Sublayer(x))
        y = self.norm1(self.dropout(self.mhsa(x, mask)) + x)
        return self.norm2(self.linear_transform(y) + y)


class TransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dim_linear_block = 1024,
        dropout = 0
    ):
        super().__init__()
        self.layers = self._make_layer(dim, blocks, num_heads, dim_head, dim_linear_block, dropout = dropout)


    def _make_layer(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dim_linear_block = 1024,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(TransformerBlock(dim, num_heads, dim_head, dim_linear_block, dropout = dropout))
        return nn.Sequential(*layers)


    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

model = TransformerEncoder(dim=64, blocks=6, num_heads=8)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
print(model(x).size())