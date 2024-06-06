import random
from dooc import pipelines, models


def test_DrugcellAdamrMutSmiReg(smi_tkz):

    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"

    class Reg(pipelines._DrugcellAdamrMutSmi, pipelines._MutSmiReg):
        pass

    model = models.MutSmiReg()
    pipeline = Reg(smi_tokenizer=smi_tkz,
                   model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, float)


def test_DrugcellAdamr2MutSmiReg(smi_tkz):
    pass


def test_DrugcellAdamrMutSmisRank(smi_tkz):

    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = ["CC[N+](C)(C)Cc1ccccc1Br", "CC[N+](C)(C)Cc1ccccc1Br", "c1cccc1c"]

    class Pointwise(pipelines._DrugcellAdamrMutSmi, pipelines._MutSmisRank):
        pass

    model = models.MutSmiReg()
    pipeline = Pointwise(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"

    class PairListRank(pipelines._DrugcellAdamrMutSmis, pipelines._MutSmisRank):
        pass

    model = models.MutSmisPairwise()
    pipeline = PairListRank(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"


def test_DrugcellAdamr2MutSmisRank(smi_tkz):
    pass
