import typing
import torch
from moltx.datasets import Base


class MutSmi(Base):
    """Base datasets, convert smiles and mutations to torch.Tensor.

    """
    def __init__(self):
        pass


class MutSmiXAttention(MutSmi):
    """Regression task datasets.

    """
    def __call__(self,
                 smiles: typing.Sequence[str],
                 mutations: typing.Sequence[list],
                 values: typing.Sequence[float]
                 ) -> typing.Tuple[torch.Tensor]:
        """_summary_

        Args:
            smiles (typing.Sequence[str]): molecule smiles.
            mutations (typing.Sequence[list]): mutations one-hot list.
            values (typing.Sequence[float]): actual inhibitation rate.

        Returns:
            smiles_src, smiles_tgt, mutations_src, out
        """
        return


class MutSmiFullConnection(MutSmiXAttention):
    """_summary_

    Args:
        MutSmi (_type_): _description_
    """
    pass
