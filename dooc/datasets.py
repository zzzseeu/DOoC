import typing
import torch

from moltx import tokenizers, datasets


class _SmiMutBase:
    def __init__(self, smi_tokenizer: tokenizers.MoltxTokenizer, device: torch.device = torch.device("cpu")) -> None:
        self.smi_ds = datasets.Base(smi_tokenizer, device)
        self.device = device

    def _smi_tokenize(self, smiles: typing.Sequence[str], seq_len: int = None) -> torch.Tensor:
        return self.smi_ds._tokenize(smiles, seq_len)


"""
Mutations(Individual Sample) and Smiles Interaction

{MutationEnc}{SmileEnc}MutSmi: 1 mut with 1 smi
{MutationEnc}{SmileEnc}MutSmis: 1 mut with n smi
{MutationEnc}{SmileEnc}MutsSmi: n mut with 1 smi
"""


class _DrugcellAdamrBase(_SmiMutBase):
    """Base datasets, convert smiles and genes to torch.Tensor."""

    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__(smi_tokenizer, device)
        self.smi_tokenizer = smi_tokenizer

    def _smi_tokens(
        self,
        smiles: typing.Sequence[str],
        seq_len: int = 200,
    ) -> torch.Tensor:
        src = self._smi_tokenize(smiles, seq_len)
        tgt = self._smi_tokenize(
            [f"{self.smi_tokenizer.BOS}{smi}{self.smi_tokenizer.EOS}" for smi in smiles], seq_len)
        return src, tgt

    def _mut_tokens(self, muts: typing.Sequence[list]) -> torch.Tensor:
        return torch.tensor(muts, device=self.device)

    def _out(self, values: typing.Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, device=self.device)


class _DrugcellAdamrMutSmi(_DrugcellAdamrBase):

    def __call__(
        self,
        muts: typing.Sequence[list],
        smis: typing.Sequence[str],
        vals: typing.Sequence[float],
        seq_len: int = 200
    ) -> typing.Tuple[torch.Tensor]:
        assert len(smis) == len(vals) and len(muts) == len(vals)
        mut_x = self._mut_tokens(muts)
        smi_src, smi_tgt = self._smi_tokens(smis, seq_len)
        out = self._out(vals).unsqueeze(-1)
        return mut_x, smi_src, smi_tgt, out


class _DrugcellAdamrMutSmis(_DrugcellAdamrBase):

    def __call__(
        self,
        muts: typing.Sequence[list],
        lsmis: typing.Sequence[typing.Sequence[str]],
        lvals: typing.Sequence[typing.Sequence[float]],
        seq_len: int = 200
    ) -> typing.Tuple[torch.Tensor]:
        """
        muts: [mut1, mut2, ...] mut1: [gene1, gene2, ...]
        bsmiles: [[smi11, smi12], [smi21, smi22], ...]
        bvlaues: [[val11, val12], [val21, val22], ...]
        """
        assert len(lsmis) == len(lvals) and len(muts) == len(lvals)
        mut_x = self._mut_tokens(muts)
        batchlen = len(lsmis)
        listlen = len(lsmis[0])
        smiles = [smi for bsmi in lsmis for smi in bsmi]
        smi_src, smi_tgt = self._smi_tokens(smiles, seq_len)
        smi_src = smi_src.reshape(batchlen, listlen, smi_src.size(-1))
        smi_tgt = smi_tgt.reshape(batchlen, listlen, smi_src.size(-1))
        out = self._out(lvals)
        return mut_x, smi_src, smi_tgt, out


class _DrugcellAdamr2Base(_SmiMutBase):
    pass


class _DrugcellAdamr2MutSmi(_DrugcellAdamr2Base):
    pass


class _DrugcellAdamr2MutSmisPairwiseRank(_DrugcellAdamr2Base):
    pass


"""
Mutations(Individual Sample) and Smiles Interaction

MutSmiReg
MutSmis{Pair/List}
MutsSmi{Pair/List}
"""


class MutSmiReg(_DrugcellAdamrMutSmi):
    pass


class MutSmisPairwise(_DrugcellAdamrMutSmis):
    def __call__(
        self,
        muts: typing.Sequence[list],
        lsmiles: typing.Sequence[typing.Sequence[str]],
        lvalues: typing.Sequence[typing.Sequence[float]],
        seq_len: int = 200
    ) -> typing.Tuple[torch.Tensor]:
        mut_x, smi_src, smi_tgt, rout = super().__call__(muts, lsmiles, lvalues, seq_len)
        out = torch.zeros(rout.size(0), dtype=torch.long, device=self.device)
        out[(rout[:, 0] - rout[:, 1]) > 0.0] = 1
        return mut_x, smi_src, smi_tgt, out.unsqueeze(-1)
