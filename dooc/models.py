import torch
from moltx import nets as mnets
from moltx import models as mmodels
from dooc import nets as dnets
from dooc.nets import heads, drugcell


"""
Mutations(Individual Sample) and Smiles Interaction

MutSmiReg
MutSmisRank
MutsSmiRank
"""


class MutSmiAddReg(dnets.DrugcellAdamr2MutSmiAdd):

    def __init__(self, mut_conf: drugcell.DrugcellConfig = dnets.Drugcell.DEFAULT_CONFIG, smi_conf: mnets.AbsPosEncoderCausalConfig = mmodels.AdaMR2.CONFIG_LARGE) -> None:
        super().__init__(mut_conf, smi_conf)
        self.reg = heads.RegHead(self.smi_conf.d_model)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        return self.reg(super().forward(mut_x, smi_tgt))  # [b, 1]


class MutSmiReg(dnets.DrugcellAdamr2MutSmiXattn):

    def __init__(self, mut_conf: drugcell.DrugcellConfig = dnets.Drugcell.DEFAULT_CONFIG, smi_conf: mnets.AbsPosEncoderCausalConfig = mmodels.AdaMR2.CONFIG_LARGE) -> None:
        super().__init__(mut_conf, smi_conf)
        self.reg = heads.RegHead(self.smi_conf.d_model)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        return self.reg(super().forward(mut_x, smi_tgt))  # [b, 1]


class MutSmisRank(dnets.DrugcellAdamr2MutSmisXattn):

    def __init__(self, mut_conf: drugcell.DrugcellConfig = dnets.Drugcell.DEFAULT_CONFIG, smi_conf: mnets.AbsPosEncoderCausalConfig = mmodels.AdaMR2.CONFIG_LARGE) -> None:
        super().__init__(mut_conf, smi_conf)
        self.reg = heads.RegHead(self.smi_conf.d_model)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        return self.reg(super().forward(mut_x, smi_tgt)).squeeze(-1)  # [b, n]

    def forward_cmp(self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> float:
        """
        for infer, no batch dim
        """
        assert mut_x.dim() == 1 and smi_tgt.dim() == 2
        out = self.forward(mut_x, smi_tgt)  # [2]
        return (out[0] - out[1]).item()
