import torch
from moltx import models as mmodels
from dooc import nets as dnets
from dooc.nets import mutations, heads


"""
Mutations(Individual Sample) and Smiles Interaction

MutSmiReg
MutSmis{Pair/List}
MutsSmi{Pair/List}
"""


class MutSmiReg(dnets.DrugcellAdamrMutSmiXattn):

    def __init__(self) -> None:
        super().__init__(mut_conf=mutations.Drugcell.DEFAULT_CONFIG, smi_conf=mmodels.AdaMR.CONFIG_BASE)
        self.reg = heads.RegHead(self.smi_conf.d_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.reg(super().forward(*args, **kwargs))  # [b, 1]


class MutSmisPairwise(dnets.DrugcellAdamrMutSmisXattn):

    def __init__(self) -> None:
        super().__init__(mut_conf=mutations.Drugcell.DEFAULT_CONFIG, smi_conf=mmodels.AdaMR.CONFIG_BASE)
        self.pairwise_rank = heads.PairwiseRankHead(self.smi_conf.d_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.pairwise_rank(super().forward(*args, **kwargs))  # [b, 2]

    def forward_cmp(self, *args, **kwargs) -> float:
        """
        for infer, no batch dim
        """
        out = self.forward(*args, **kwargs)
        return (out[1] - out[0]).item()
