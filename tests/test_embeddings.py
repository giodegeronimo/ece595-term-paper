import torch

from navit_rf.embeddings import FourierTimeEmbedding, FractionalFourierPositionalEmbedding


def test_fourier_time_shape():
    emb = FourierTimeEmbedding(d_model=64)
    t = torch.rand(4)
    out = emb(t)
    assert out.shape == (4, 64)


def test_fractional_pos_shape():
    emb = FractionalFourierPositionalEmbedding(dim=32)
    out = emb(5, 7)
    assert out.shape == (35, 32)
