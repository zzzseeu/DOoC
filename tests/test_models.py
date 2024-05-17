import torch
from dooc import models, nets
import random


# def test_MutSmiXAttention():
#     smiles_src = torch.randint(0, 64, [2, 4])
#     smiles_tgt = torch.randint(0, 64, [2, 6])
#     mutations_src = torch.randn(2, 3008, dtype=torch.float32)
#     d_model = 768
#     model = models.MutSmiXAttention(d_model)
#     out = model(smiles_src, smiles_tgt, mutations_src)
#     assert out.shape == (1,)


def test_MutSmiFullConnection(datadir):
    smiles_src = torch.randint(0, 64, [2, 200])
    smiles_tgt = torch.randint(0, 64, [2, 200])
    mutations = [[random.choice([0, 1]) for _ in range(3008)],
                [random.choice([0, 1]) for _ in range(3008)]]
    mutations_src = torch.tensor(mutations, dtype=torch.float).to("cpu")
    d_model = 768
    gene_conf = nets.GeneGNNConfig(
        gene_dim=3008,
        drug_dim=2048,
        num_hiddens_genotype=6,
        num_hiddens_drug=[100, 50, 6],
        num_hiddens_final=6,
        gene2ind_path=f"{datadir}/gene2ind.txt",
        ont_path=f"{datadir}/drugcell_ont.txt",
    )
    model = models.MutSmiFullConnection(d_model, gene_conf=gene_conf)
    out = model(smiles_src, smiles_tgt, mutations_src)
    assert out.shape == (2, 1)
