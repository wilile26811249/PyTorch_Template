import einops
from einops import rearrange
import numpy as np

import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Implement self attention layer using the "Einstein summation convention".
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    """
    def __init__(self, dim = 3):
        super(SelfAttention, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # Suspect that for large value of the dim,
        # the dot product grow large in magnitude,
        # to counteract this effect, need to scale the dot product.
        self.scale_factor = dim ** -0.5

        # Initialize weight and bias
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.xavier_normal_(self.to_qkv.weight)


    def forward(self, x, mask = None):
        assert x.dim() == 3
        qkv = self.to_qkv(x)

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, "b t (d k) -> k b t d", k = 3))

        # dot_product = q * k.T
        dot_product = torch.einsum('b i d, b j d -> b i j', q, k)
        scale_dot_product = dot_product * self.scale_factor

        #
        if mask is not None:
            scale_dot_product = scale_dot_product.masked_fill(mask, -np.inf)

        attention = torch.softmax(scale_dot_product, dim = -1)
        return torch.einsum('b i j, b j d -> b i d', attention, v)

# Test
# self_attn = SelfAttention(dim = 3)
# x = torch.randn(4, 6, 3)
# print(self_attn(x).size())
