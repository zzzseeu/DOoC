import torch
import torch.nn as nn


class ListNetLoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = - (target.softmax(dim=-1) * predict.log_softmax(dim=-1))
        return getattr(out, self.reduction)()
