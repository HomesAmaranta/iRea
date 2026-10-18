"""Next implicit token prediction using detached, shifted shallow states."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_nitp(weight, layer, depth=None):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("nitp_weight must be finite and nonnegative")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 1:
        raise ValueError("nitp_target_layer must be a positive, one-based integer")
    if weight > 0 and depth is not None and layer > depth:
        raise ValueError("nitp_target_layer exceeds backbone depth")


class NITPProjector(nn.Module):
    """Two-stage SwiGLU projector: d -> (gate, value) -> d."""

    def __init__(self, dim):
        super().__init__()
        self.input = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, hidden):
        gate, value = self.input(hidden).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


def configure_nitp(model, dim, depth, weight=0.0, layer=1):
    validate_nitp(weight, layer, depth)
    model.nitp_weight = weight
    model.nitp_target_layer = layer
    if weight > 0:
        model.nitp_projector = NITPProjector(dim)


def nitp_loss(projector, hidden, target):
    if hidden.shape != target.shape or target.requires_grad:
        raise ValueError("NITP requires matching hidden/target shapes and detached targets")
    prediction = projector(hidden)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return (1 - F.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=1e-8
        )).mean()
