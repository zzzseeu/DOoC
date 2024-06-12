import pytest
from dooc import datasets
import random


def test_DrugcellAdamrMutSmi(smi_tkz):
    ds = datasets._DrugcellAdamrMutSmi(smi_tkz)
    smis = ["CC[N+]CCBr", "Cc1ccc1"]
    vals = [0.88, 0.89]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, smis, vals[:1])
    mut_x, smi_src, smi_tgt, out = ds(muts, smis, vals)
    assert smi_src.shape == (2, 200)
    assert smi_tgt.shape == (2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 1)


def test_DrugcellAdamr2MutSmi(smi_tkz):
    ds = datasets._DrugcellAdamr2MutSmi(smi_tkz)
    smis = ["CC[N+]CCBr", "Cc1ccc1"]
    vals = [0.88, 0.89]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, smis, vals[:1])
    mut_x, smi_tgt, out = ds(muts, smis, vals)
    assert smi_tgt.shape == (2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 1)


def test_DrugcellAdamrMutSmis(smi_tkz):
    ds = datasets._DrugcellAdamrMutSmis(smi_tkz)
    lsmis = [["CC[N+]CCBr", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1"]]
    lvals = [[0.88, 0.89], [0.82, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, lsmis, lvals[:1])
    mut_x, smi_src, smi_tgt, out = ds(muts, lsmis, lvals)
    assert smi_src.shape == (2, 2, 200)
    assert smi_tgt.shape == (2, 2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 2)


def test_DrugcellAdamr2MutSmis(smi_tkz):
    ds = datasets._DrugcellAdamr2MutSmis(smi_tkz)
    lsmis = [["CC[N+]CCBr", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1"]]
    lvals = [[0.88, 0.89], [0.82, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, lsmis, lvals[:1])
    mut_x, smi_tgt, out = ds(muts, lsmis, lvals)
    assert smi_tgt.shape == (2, 2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 2)


def test_MutSmiReg(smi_tkz):
    ds = datasets.MutSmiReg(smi_tkz)
    smis = ["CC[N+]CCBr", "Cc1ccc1"]
    vals = [0.88, 0.89]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, smis, vals[:1])
    mut_x, smi_tgt, out = ds(muts, smis, vals)
    assert smi_tgt.shape == (2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 1)


def test_MutSmisPairwiseRank(smi_tkz):
    ds = datasets.MutSmisPairwiseRank(smi_tkz)
    lsmis = [["CC[N+]CCBr", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1"]]
    lvals = [[0.88, 0.89], [0.82, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, lsmis, lvals[:1])
    mut_x, smi_tgt, out = ds(muts, lsmis, lvals)
    assert smi_tgt.shape == (2, 2, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2,)


def test_MutSmisListwiseRank(smi_tkz):
    ds = datasets.MutSmisListwiseRank(smi_tkz)
    lsmis = [["CC[N+]CCBr", "Cc1ccc1", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1", "Cc1ccc1"]]
    lvals = [[0.88, 0.89, 0.89], [0.82, 0.9, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(52)],
            [random.choice([0, 1]) for _ in range(52)]]
    with pytest.raises(AssertionError):
        ds(muts, lsmis, lvals[:1])
    mut_x, smi_tgt, out = ds(muts, lsmis, lvals)
    assert smi_tgt.shape == (2, 3, 200)
    assert mut_x.shape == (2, 52)
    assert out.shape == (2, 3)
