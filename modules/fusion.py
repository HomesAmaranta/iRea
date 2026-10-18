"""Current-item fusion shared by SID recommendation backbones."""

import torch
from torch import nn


class ItemFusion(nn.Module):
    """Frozen item features fused into one hidden representation."""

    def __init__(self, hidden_dim, embedding_dim, mode):
        super().__init__()
        if mode not in ("add", "concat") or embedding_dim <= 0:
            raise ValueError("Fusion requires add/concat and a positive embedding dimension")
        self.mode = mode
        self.embedding_dim = embedding_dim
        input_dim = embedding_dim + (hidden_dim if mode == "concat" else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden, embedding):
        embedding = embedding.detach().to(device=hidden.device, dtype=hidden.dtype)
        inputs = torch.cat((hidden, embedding), dim=-1) if self.mode == "concat" else embedding
        projected = self.mlp(inputs)
        return hidden + projected if self.mode == "add" else projected
