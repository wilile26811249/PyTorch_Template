import torch
import torch.nn as nn
from einops import rearrange, repeat

from .transformer_block import TransformerEncoder

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

class VIT(nn.Module):
    """
    Parameters
    ----------
    img_dim:
        the spatial image size
    in_channels:
        number of img channels
    patch_dim:
        desired patch dim
    num_classes:
        classification task classes
    dim:
        the linear layer's dim to project the patches for MHSA
    blocks:
        number of transformer blocks
    heads:
        number of heads
    dim_linear_block:
        inner dim of the transformer linear block
    dim_head:
        dim head in case you want to define it. defaults to dim/heads
    dropout:
        for pos emb and transformer
    transformer:
        in case you want to provide another transformer implementation
    classification:
        creates an extra CLS token that we will index in the final classification layer
    """
    def __init__(self,
        img_dim = 224,
        in_channels = 3,
        patch_size = 16,
        num_classes = 10,
        dim = 512,
        blocks = 6,
        num_heads = 8,
        dim_linear_block = 1024,
        dim_head = None,
        dropout = 0.1,
        transformer = None,
        classification = True
    ):
        super(VIT, self).__init__()
        self.p = patch_size
        self.classification = classification
        tokens = (img_dim // patch_size) ** 2
        self.token_dim = in_channels * (patch_size ** 2)
        self.dim = dim
        self.dim_head = (int(dim / num_heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
        self.mlp_head = nn.Linear(dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(
                dim,
                blocks = blocks,
                num_heads = num_heads,
                dim_head = self.dim_head,
                dim_linear_block = dim_linear_block,
                dropout = dropout
            )
        else:
            self.transformer = transformer

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N]
        # tokens = h/p * w/p, N = p * p * c
        # Ex: input_size  = [64, 3, 224, 224], p = 16
        # --> after_patch = [64, 196, 768]
        img_patches = rearrange(
            img,
            'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
            patch_x = self.p, patch_y = self.p
        )
        batch_size, tokens, _ = img_patches.shape  # [64, 196, 768]

        # Linear Projection of Flattened Patches
        img_patches = self.project_patches(img_patches)  # [64, 196, 512]

        # Patch + Position Embedding + dropout
        img_patches = torch.cat(
            (self.expand_cls_to_batch(batch_size), img_patches),
            dim = 1
        )  # [64, 197, 512]
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)
        result = self.mlp_head(y[:, 0, :])
        return result
        # return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1: , :]


def VIT_B_16(img_dim = 224, num_classes = 2):
    return VIT(
        img_dim = 224,
        blocks = 12,
        dim_linear_block = 3072,
        dim = 768,
        num_heads = 12,
        num_classes = num_classes
    )

def VIT_L_16(img_dim = 224, num_classes = 2):
    return VIT(
        img_dim = 224,
        blocks = 24,
        dim_linear_block = 4096,
        dim = 1024,
        num_heads = 16,
        num_classes = num_classes
    )

def VIT_H_16(img_dim = 224, num_classes = 2):
    return VIT(
        img_dim = 224,
        blocks = 32,
        dim_linear_block = 5120,
        dim = 1280,
        num_heads = 16,
        num_classes = num_classes
    )