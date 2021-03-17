import numpy as np
from einops import rearrange
from MultiHeadAttn import MultiHeadSelfAttention

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Implement transformer block in the paper "Attention is all you need"
    Paper: https://arxiv.org/abs/1706.03762

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    dim_linear_block:
        Number of hidden features of the linear transform
    activation:
        Activation function apply in the linear transform layer
    dropout:
        Dropout rate in the linear transform layers
    mhsa:
        Optional[MultiHeadSelfAttention object | None]
    """
    def __init__(self,
        dim,
        num_heads = 8,
        dim_head = None,
        dim_linear_block = 1024,
        activation = nn.GELU,
        dropout = 0.1,
        mhsa = None
    ):
        super(TransformerBlock, self).__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(p = dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Positionwise fully connected feed-forward network
        self.linear_transform = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # Default is GeLU, can try relu、selu、elu...etc
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
        self.blocks = self._make_layer(dim, blocks, num_heads, dim_head, dim_linear_block, dropout = dropout)


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
        for block in self.blocks:
            x = block(x, mask)
        return x
