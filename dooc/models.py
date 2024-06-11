import torch
from moltx import nets as mnets
from moltx import models as mmodels
from dooc import nets as dnets
from dooc.nets import heads, drugcell


"""
Mutations(Individual Sample) and Smiles Interaction

MutSmiReg
MutSmis{Pair/List}
MutsSmi{Pair/List}
"""


class MutSmiReg(dnets.DrugcellAdamr2MutSmiXattn):

    def __init__(self, mut_conf: drugcell.DrugcellConfig = dnets.Drugcell.DEFAULT_CONFIG, smi_conf: mnets.AbsPosEncoderCausalConfig = mmodels.AdaMR2.CONFIG_LARGE) -> None:
        super().__init__(mut_conf, smi_conf)
        self.reg = heads.RegHead(self.smi_conf.d_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.reg(super().forward(*args, **kwargs))  # [b, 1]


class MutSmisPairwise(dnets.DrugcellAdamr2MutSmisXattn):

    def __init__(self, mut_conf: drugcell.DrugcellConfig = dnets.Drugcell.DEFAULT_CONFIG, smi_conf: mnets.AbsPosEncoderCausalConfig = mmodels.AdaMR2.CONFIG_LARGE) -> None:
        super().__init__(mut_conf, smi_conf)
        self.pairwise_rank = heads.PairwiseRankHead(self.smi_conf.d_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.pairwise_rank(super().forward(*args, **kwargs))  # [b, 2]

    def forward_cmp(self, *args, **kwargs) -> float:
        """
        for infer, no batch dim
        """
        out = self.forward(*args, **kwargs)
        return (out[1] - out[0]).item()
