import einops
from einops.layers.torch import Rearrange

import torch
from torch import nn
from torch.nn import functional as F


def drop_path(x, drop_prob = 0., training = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob)

class SpatialGatingUnit(nn.Module):
    """Pay Attention in MLP
    Paper source: https://arxiv.org/pdf/2105.08050v1.pdf
    Something like GLU here.
    """
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size = 1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim = -1)
        v = self.norm(v)  # Improve stability
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLP_block(nn.Module):
    """Pay Attention in MLP
    Paper source: https://arxiv.org/pdf/2105.08050v1.pdf
    """
    def __init__(self, d_model, d_ffn, seq_len, drop_prob = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.spatial_gating_unit(x)
        x = self.channel_proj2(x)
        out = x + self.drop_path(shortcut)
        return out


class gMLP_Img(nn.Module):
    def __init__(
        self,
        patch_size = 16,
        in_channels = 3,
        num_classes = 1000,
        d_model = 256,
        d_ffn = 512,
        seq_len = 256,
        num_layers = 6,
        drop_prob = 0.
    ):
        super().__init__()
        self.img2patch = nn.Sequential(
            nn.Conv2d(in_channels, d_model, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c")
        )
        self.model = nn.Sequential(
            *[gMLP_block(d_model, d_ffn, seq_len, drop_prob) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        patches = self.img2patch(x)
        embedding = self.model(patches)
        embedding = embedding.mean(dim = 1)
        out = self.classifier(embedding)
        return out


def gMLP_Ti(**kwargs):
    model = gMLP_Img(num_layers = 16, d_model = 128, d_ffn = 768, drop_prob = 0.01, **kwargs)
    return model

def gMLP_S(**kwargs):
    model = gMLP_Img(num_layers = 16, d_model = 256, d_ffn = 1536, drop_prob = 0.05, **kwargs)
    return model

def gMLP_B(**kwargs):
    model = gMLP_Img(num_layers = 16, d_model = 512, d_ffn = 3072, drop_prob = 0.10, **kwargs)
    return model