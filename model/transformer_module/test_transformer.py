import torch

from transformer_block import TransformerBlock, TransformerEncoder


def test_MultiHeadSelfAttention():
    model = TransformerEncoder(dim = 64, blocks = 6, dim_head = 8)
    x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
    mask = torch.zeros(10, 10)  # tokens X tokens
    mask[5 : 8, 5 : 8] = 1
    y = model(x, mask)
    assert y.shape == x.shape
    print("Transformer OK")

    model = TransformerBlock(dim = 64)
    x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
    mask = torch.zeros(10, 10)  # tokens X tokens
    mask[5 : 8, 5 : 8] = 1
    y = model(x, mask)
    assert y.shape == x.shape
    print("Transformer block OK")

test_MultiHeadSelfAttention()