import pytest
import random
from moltx import models as mmodel
from dooc import nets, datasets


@pytest.fixture
def adamr_conf():
    return mmodel.AdaMR.CONFIG_BASE

@pytest.fixture
def adamr2_conf():
    return mmodel.AdaMR2.CONFIG_LARGE

@pytest.fixture
def drugcell_conf():
    return nets.Drugcell.DEFAULT_CONFIG

@pytest.fixture
def drugcell_adamr_mut_smi_ds(smi_tkz):
    smis = ["CC[N+]CCBr", "Cc1ccc1"]
    vals = [0.88, 0.89]
    muts = [[random.choice([0, 1]) for _ in range(3008)],
            [random.choice([0, 1]) for _ in range(3008)]]
    ds = datasets._DrugcellAdamrMutSmi(smi_tkz)
    return ds(muts, smis, vals)

@pytest.fixture
def drugcell_adamr2_mut_smi_ds(smi_tkz):
    smis = ["CC[N+]CCBr", "Cc1ccc1"]
    vals = [0.88, 0.89]
    muts = [[random.choice([0, 1]) for _ in range(3008)],
            [random.choice([0, 1]) for _ in range(3008)]]
    ds = datasets._DrugcellAdamr2MutSmi(smi_tkz)
    return ds(muts, smis, vals)

@pytest.fixture
def drugcell_adamr_mut_smis_ds(smi_tkz):
    lsmis = [["CC[N+]CCBr", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1"]]
    lvals = [[0.88, 0.89], [0.82, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(3008)],
            [random.choice([0, 1]) for _ in range(3008)]]
    ds = datasets._DrugcellAdamrMutSmis(smi_tkz)
    return ds(muts, lsmis, lvals)

@pytest.fixture
def drugcell_adamr2_mut_smis_ds(smi_tkz):
    lsmis = [["CC[N+]CCBr", "Cc1ccc1"], ["CCC[N+]CCBr", "CCc1ccc1"]]
    lvals = [[0.88, 0.89], [0.82, 0.9]]
    muts = [[random.choice([0, 1]) for _ in range(3008)],
            [random.choice([0, 1]) for _ in range(3008)]]
    ds = datasets._DrugcellAdamr2MutSmis(smi_tkz)
    return ds(muts, lsmis, lvals)

def test_DrugcellAdamrMutSmi(adamr_conf, drugcell_conf, drugcell_adamr_mut_smi_ds):
    label = drugcell_adamr_mut_smi_ds[-1]

    model = nets.DrugcellAdamrMutSmiAdd(drugcell_conf, adamr_conf)
    out = model(*drugcell_adamr_mut_smi_ds[:-1])
    assert out.dim() == 2
    assert out.size(0) == label.size(0)

    model = nets.DrugcellAdamrMutSmiXattn(drugcell_conf, adamr_conf)
    out = model(*drugcell_adamr_mut_smi_ds[:-1])
    assert out.dim() == 2
    assert out.size(0) == label.size(0)


def test_DrugcellAdamr2MutSmi(adamr2_conf, drugcell_conf, drugcell_adamr2_mut_smi_ds):
    label = drugcell_adamr2_mut_smi_ds[-1]

    model = nets.DrugcellAdamr2MutSmiAdd(drugcell_conf, adamr2_conf)
    out = model(*drugcell_adamr2_mut_smi_ds[:-1])
    assert out.dim() == 2
    assert out.size(0) == label.size(0)

    model = nets.DrugcellAdamr2MutSmiXattn(drugcell_conf, adamr2_conf)
    out = model(*drugcell_adamr2_mut_smi_ds[:-1])
    assert out.dim() == 2
    assert out.size(0) == label.size(0)


def test_DrugcellAdamrMutSmis(adamr_conf, drugcell_conf, drugcell_adamr_mut_smis_ds):
    label = drugcell_adamr_mut_smis_ds[-1]

    model = nets.DrugcellAdamrMutSmisAdd(drugcell_conf, adamr_conf)
    out = model(*drugcell_adamr_mut_smis_ds[:-1])
    assert out.dim() == 3
    assert out.size(0) == label.size(0) and out.size(1) == label.size(1)

    model = nets.DrugcellAdamrMutSmisXattn(drugcell_conf, adamr_conf)
    out = model(*drugcell_adamr_mut_smis_ds[:-1])
    assert out.dim() == 3
    assert out.size(0) == label.size(0) and out.size(1) == label.size(1)


def test_DrugcellAdamr2MutSmis(adamr2_conf, drugcell_conf, drugcell_adamr2_mut_smis_ds):
    label = drugcell_adamr2_mut_smis_ds[-1]

    model = nets.DrugcellAdamr2MutSmisAdd(drugcell_conf, adamr2_conf)
    out = model(*drugcell_adamr2_mut_smis_ds[:-1])
    assert out.dim() == 3
    assert out.size(0) == label.size(0) and out.size(1) == label.size(1)

    model = nets.DrugcellAdamr2MutSmisXattn(drugcell_conf, adamr2_conf)
    out = model(*drugcell_adamr2_mut_smis_ds[:-1])
    assert out.dim() == 3
    assert out.size(0) == label.size(0) and out.size(1) == label.size(1)
