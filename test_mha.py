import torch

from model import MultiHeadAttention, PyTorchMultiHeadAttention


def test_mha():
    context_length = 100
    d_in = 256
    d_out = 16

    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    batch = torch.rand(8, 6, d_in)
    context_vecs = mha(batch)
    assert context_vecs.shape == torch.Size([8, 6, d_out])

    # Test bonus class
    mha = PyTorchMultiHeadAttention(d_in, d_out, num_heads=2)
    batch = torch.rand(8, 6, d_in)
    context_vecs = mha(batch)
    assert context_vecs.shape == torch.Size([8, 6, d_out])
