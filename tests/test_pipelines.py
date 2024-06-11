import random
from dooc import pipelines, models


def test_MutSmiReg(smi_tkz):
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"

    model = models.MutSmiReg()
    pipeline = pipelines.MutSmiReg(smi_tokenizer=smi_tkz,
                   model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, float)


def test_MutSmisRank(smi_tkz):
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = ["CC[N+](C)(C)Cc1ccccc1Br", "CC[N+](C)(C)Cc1ccccc1Br", "c1cccc1c"]

    model = models.MutSmiReg()
    pipeline = pipelines.MutSmiPointwise(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"

    model = models.MutSmisPairwise()
    pipeline = pipelines.MutSmisRank(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"
