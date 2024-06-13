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
