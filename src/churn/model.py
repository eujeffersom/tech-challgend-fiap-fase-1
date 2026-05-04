import random

import numpy as np
import torch
from torch import nn


def set_global_seed(seed: int) -> None:
    # Reduz variacao entre execucoes de treino.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        # MLP simples para dados tabulares ja pre-processados.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
