import random
from dooc import pipelines, models
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig

 
def test_MutSmiXAttention():
    tokenizer = tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
        )
    model = models.MutSmiXAttention()
    # model.load_ckpt('/path/to/mutsmixattention.ckpt')
    pipeline = pipelines.MutSmiXAttention(smi_tokenizer=tokenizer,
                                          model=model)
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"
    predict = pipeline(mutation, smiles)
    assert isinstance(predict, float)


def test_MutSmiFullConnection():
    tokenizer = tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
        )
    model = models.MutSmiFullConnection()
    # model.load_ckpt('/path/to/mutsmifullconnection.ckpt')
    pipeline = pipelines.MutSmiFullConnection(smi_tokenizer=tokenizer,
                                              model=model)
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"
    predict = pipeline(mutation, smiles)
    assert isinstance(predict, float)
