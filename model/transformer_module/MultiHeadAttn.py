import numpy as np
from einops import rearrange

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".
    Paper: https://arxiv.org/abs/1706.03762

    Args:
        dim: Token's dimension, EX: word embedding vector size
        num_heads: The number of distinct representations to learn
        dim_head: The dimension of the each head
    """
    def __init__(self, dim, num_heads = 8, dim_head = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias = False)

        self.scale_factor = dim ** -0.5

        # Weight matrix for output, Size: num_heads*dim_head X dim
        # Final linear transformation layer
        self.w_out = nn.Linear(_weight_dim, dim, bias = False)

    def forward(self, x, mask = None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)

        # Decomposize to q、k and v
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k = 3, h = self.num_heads))

        # dot_product = q * k.T
        dot_product = torch.einsum('... i d, ... j d -> ... i j', q, k)
        scale_dot_product = dot_product * self.scale_factor

        if mask is not None:
            scale_dot_product = scale_dot_product.masked_fill(mask, -np.inf)

        attention = torch.softmax(scale_dot_product, dim = -1)
        result = torch.einsum('... i j, ... j d -> ... i d', attention, v)

         # re-compose: merge heads with dim_head
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)
