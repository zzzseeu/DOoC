import torch
import torch.nn as nn
from dataclasses import dataclass
from dooc.nets.drugcell import Drugcell, DrugcellConfig


@dataclass
class PrmoConfig:
    d_model: int
    mut_dim: int
    rna_dim: int
    pathway_dim: int
    drug_cell_conf: DrugcellConfig


class PrmoEncoder(nn.Module):

    DEFAULT_CONFIG = PrmoConfig(
        d_model=768,
        mut_dim=3008,
        rna_dim=5537,
        pathway_dim=3793,
        drug_cell_conf=Drugcell.DEFAULT_CONFIG,
    )

    def __init__(self, conf: PrmoConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.conf = conf

        self.mut_encoder = Drugcell(self.conf.drug_cell_conf)
        hidden_dim = self.conf.rna_dim + self.conf.drug_cell_conf.d_model + self.conf.pathway_dim
        self.out_fc = nn.Linear(hidden_dim, self.conf.d_model)

    def forward(
        self, mut_x: torch.Tensor, rna_x: torch.Tensor, pathway_x: torch.Tensor
    ) -> torch.Tensor:
        mut_out = self.mut_encoder(mut_x)
        x = torch.concat((mut_out, rna_x, pathway_x), dim=-1)
        return self.out_fc(x)

    def load_pretrained_ckpt(self, mut_ckpt: str, freeze_mut: bool = False) -> None:
        self.mut_encoder.load_ckpt(mut_ckpt)
        if freeze_mut:
            self.mut_encoder.requires_grad_(False)
