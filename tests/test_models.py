import torch
from dooc import models


# def test_MutSmiXAttention():
#     smiles_src = torch.randint(0, 64, [2, 4])
#     smiles_tgt = torch.randint(0, 64, [2, 6])
#     mutations_src = torch.randn(2, 3008, dtype=torch.float32)
#     d_model = 768
#     model = models.MutSmiXAttention(d_model)
#     out = model(smiles_src, smiles_tgt, mutations_src)
#     assert out.shape == (1,)


def test_MutSmiFullConnection():
    smiles_src = torch.randint(0, 64, [2, 4])
    smiles_tgt = torch.randint(0, 64, [2, 6])
    mutations_src = torch.randn(2, 3008, dtype=torch.float32)
    d_model = 768
    model = models.MutSmiFullConnection(d_model)
    out = model(smiles_src, smiles_tgt, mutations_src)
    assert out.shape == (1,)
