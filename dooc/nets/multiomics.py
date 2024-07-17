import torch
import torch.nn as nn
from dataclasses import dataclass
from dooc.nets.drugcell import Drugcell, DrugcellConfig


@dataclass
class MultiOmicsConfig:
    d_model: int
    mut_dim: int
    rna_dim: int
    pathway_dim: int
    drug_cell_conf: DrugcellConfig


class MultiOmicsEncoder(nn.Module):

    DEFAULT_CONFIG = MultiOmicsConfig(
        d_model=768,
        mut_dim=3008,
        rna_dim=5537,
        pathway_dim=3793,
        drug_cell_conf=Drugcell.DEFAULT_CONFIG,
    )

    def __init__(self, conf: MultiOmicsConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.conf = conf

        self.mut_encoder = Drugcell(self.conf.drug_cell_conf)
        hidden_dim = self.conf.rna_dim + self.conf.drug_cell_conf.d_model + self.conf.pathway_dim
        self.out_fc = nn.Linear(hidden_dim, self.conf.d_model)

    def forward(
        self, mut_x: torch.Tensor, rna_x: torch.Tensor, pathway_x: torch.Tensor
    ) -> torch.Tensor:
        dim = mut_x.dim()
        mut_x = mut_x.unsqueeze(0) if mut_x.dim() == 1 else mut_x
        rna_x = rna_x.unsqueeze(0) if rna_x.dim() == 1 else rna_x
        pathway_x = pathway_x.unsqueeze(0) if pathway_x.dim() == 1 else pathway_x
        mut_out = self.mut_encoder(mut_x)
        x = torch.concat((mut_out, rna_x, pathway_x), dim=1)
        out = self.out_fc(x)
        if dim == 1:
            out = out.squeeze(0)
        return out

    def load_pretrained_ckpt(self, mut_ckpt: str, freeze_mut: bool = False) -> None:
        self.mut_encoder.load_ckpt(mut_ckpt)
        if freeze_mut:
            self.mut_encoder.requires_grad_(False)
