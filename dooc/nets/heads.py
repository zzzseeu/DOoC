import torch
from torch import nn


class RegHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        hidden_dim = in_features // 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PairwiseRankHead(nn.Module):
    def __init__(self, d_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(d_features * 2, d_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_features, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, 2, d_features]
        """
        assert x.size(-2) == 2
        return self.mlp(x)  # [b, 2] 1: x1 > x2, 0: x1 <= x2
