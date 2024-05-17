import typing
import torch

from moltx import tokenizers, datasets


class MutSmi:
    """Base datasets, convert smiles and genes to torch.Tensor."""

    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.smi_tokenizer = smi_tokenizer
        self.smi_ds = datasets.AdaMRRegression(self.smi_tokenizer)
        self.device = device

    def gen_smi_token(
        self,
        smiles: typing.Sequence[str],
        values: typing.Sequence[float],
        seq_len: int = 200,
    ) -> torch.Tensor:
        src, tgt, out = self.smi_ds(smiles, values, seq_len)
        return src.to(self.device), tgt.to(self.device), out.to(self.device)

    def gen_gene_token(self, genes: typing.Sequence[list]) -> torch.Tensor:
        return torch.tensor(genes, dtype=torch.float).to(self.device)

    def __call__(
        self,
        smiles: typing.Sequence[str],
        genes: typing.Sequence[list],
        values: typing.Sequence[float],
        seq_len: int = 200,
    ) -> typing.Tuple[torch.Tensor]:
        smi_src, smi_tgt, out = self.gen_smi_token(smiles, values, seq_len)
        gene_src = self.gen_gene_token(genes)
        return smi_src, smi_tgt, gene_src, out


class MutSmiXAttention(MutSmi):
    pass


class MutSmiFullConnection(MutSmi):
    pass
