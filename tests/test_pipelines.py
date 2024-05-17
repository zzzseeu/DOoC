import random
from dooc import pipelines, models, nets
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig

 
# def test_MutSmiXAttention():
#     tokenizer = tkz.MoltxTokenizer.from_pretrain(
#         conf=AdaMRTokenizerConfig.Prediction
#         )
#     d_model = 768
#     model = models.MutSmiXAttention(d_model)
#     model.load_ckpt('/path/to/mutsmixattention.ckpt')
#     pipeline = pipelines.MutSmiXAttention(tokenizer, model)
#     mutation = [random.choice([1, 0]) for _ in range(3008)]
#     smiles = "CC[N+](C)(C)Cc1ccccc1Br"
#     predict = pipeline(mutation, smiles)
#     assert isinstance(predict, float)


def test_MutSmiFullConnection(datadir):
    tokenizer = tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
        )
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
    model = models.MutSmiFullConnection(d_model, gene_conf)
    # model.load_ckpt('/path/to/mutsmifullconnection.ckpt')
    pipeline = pipelines.MutSmiFullConnection(smi_tokenizer=tokenizer,
                                              model=model)
    mutation = [[random.choice([1, 0]) for _ in range(3008)]]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"
    predict = pipeline(mutation, smiles)
    assert isinstance(predict, float)
