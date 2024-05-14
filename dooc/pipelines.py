import typing
import torch


class MutSmi:
    def __init__(self):
        pass


class MutSmiXAttention(MutSmi):
    def _model_args(self,
                    mutation: typing.Sequence[int],
                    smiles: str) -> typing.Tuple[torch.Tensor]:
        return

    def __call__(self, mutation: typing.Sequence[int], smiles: str) -> float:
        return


class MutSmiFullConnection(MutSmi):
    def _model_args(self,
                    mutation: typing.Sequence[int],
                    smiles: str) -> typing.Tuple[torch.Tensor]:
        return

    def __call__(self, mutation: typing.Sequence[int], smiles: str) -> float:
        return
