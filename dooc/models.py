import torch
from torch import nn


class MutSmi(nn.Module):
    """Base

    """
    def __init__(self):
        pass

    def load_ckpt(self, ckpt_file: str) -> None:
        """load check point model.

        Args:
            ckpt_file (str): check point file path.
        """
        return

    def load_pretrained_ckpt(self,
                             drugcell_ckpt: str,
                             moltx_ckpt: str) -> None:
        """load drugcell checkpoint and moltx checkpoint.

        Args:
            drugcell_ckpt (str): ckpt file path.
            moltx_ckpt (str): ckpt file path.
        """
        return


class MutSmiXAttention(MutSmi):
    """Regression model using transformer cross attention.
    """
    def __init__(self):
        pass

    def forward(self,
                smiles_src: torch.Tensor,
                smiles_tgt: torch.Tensor,
                mutations_src: torch.Tensor) -> torch.Tensor:
        pass


class MutSmiFullConnection(MutSmi):
    """Regression model using fully connection.
    """
    def __init__(self):
        pass

    def forward(self,
                smiles_src: torch.Tensor,
                smiles_tgt: torch.Tensor,
                mutations_src: torch.Tensor) -> torch.Tensor:
        pass
