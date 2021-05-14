import einops
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class GlobalAveragePooling(nn.Module):
    def __init__(self, dim = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim)

class MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout = 0.0):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("batch num_patches channel -> batch channel num_patches"),
            MLP_Block(num_patches, token_dim, dropout),
            Rearrange("batch channel num_patches -> batch num_patches channel")
        )

        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            MLP_Block(dim, channel_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(
        self,
        in_channel = 3,
        dim = 512,
        num_classes = 2,
        patch_size = 16,
        img_size = 256,
        token_dim = 256,
        channel_dim = 2048,
        num_layers = 8,
        dropout = 0.1,
    ):
        super().__init__()
        self.num_patch = (img_size // patch_size) ** 2
        self.img2patch = nn.Sequential(
            nn.Conv2d(in_channel, dim, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c")
        )
        mixer_layer = [
            MixerBlock(dim, self.num_patch, token_dim, channel_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mixer_layer = nn.Sequential(*mixer_layer)
        self.global_pool = GlobalAveragePooling(dim = 1)
        self.mlp_head = nn.Linear(dim, num_classes)
        self.init_weights(classes = num_classes)

    def init_weights(self, classes):
        head_bias = -math.log(classes)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == "mlp_head":
                    nn.init.zeros_(module.weight)
                    nn.init.constant_(module.bias, head_bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std = 1e-6)
                    else:
                        nn.init_zeros(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, x):
        x = self.img2patch(x)
        x = self.mixer_layer(x)
        x = self.global_pool(x)
        return self.mlp_head(x)


def mixer_s16_256(**kwargs):
    """ Mixer-S/16 256x256
    """
    model = MLP_Mixer(patch_size = 16, num_layers = 8 , dim = 512, token_dim = 256, channel_dim = 2048, **kwargs)
    return model

def mixer_s32_256(**kwargs):
    """ Mixer-S/32 256x256
    """
    model = MLP_Mixer(patch_size = 32, num_layers = 8 , dim = 512, token_dim = 256, channel_dim = 2048, **kwargs)
    return model

def mixer_b16_384(**kwargs):
    """ Mixer-B/16 384x384
    """
    model = MLP_Mixer(patch_size = 16, num_layers = 12 , dim = 768, token_dim = 384, channel_dim = 3072, **kwargs)
    return model

def mixer_b32_384(**kwargs):
    """ Mixer-B/32 384x384
    """
    model = MLP_Mixer(patch_size = 32, num_layers = 12 , dim = 768, token_dim = 384, channel_dim = 3072, **kwargs)
    return model

def mixer_l16_512(**kwargs):
    """ Mixer-L/16 512x512
    """
    model = MLP_Mixer(patch_size = 16, num_layers = 24 , dim = 1024, token_dim = 512, channel_dim = 4096, **kwargs)
    return model

def mixer_l32_512(**kwargs):
    """ Mixer-L/32 512x512
    """
    model = MLP_Mixer(patch_size = 32, num_layers = 24 , dim = 1024, token_dim = 512, channel_dim = 4096, **kwargs)
    return model

def mixer_h14_640(**kwargs):
    """ Mixer-H/14 640x640
    """
    model = MLP_Mixer(patch_size = 14, num_layers = 32 , dim = 1280, token_dim = 640, channel_dim = 5120, **kwargs)
    return model
