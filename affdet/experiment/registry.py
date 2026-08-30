"""Head registry — select the affordance head by string (like the detector's
yolo11s/rtdetr-l switch). Each builder takes input_dim and returns an nn.Module
with the (B, input_dim) -> (B, 6) raw-logit contract."""
from __future__ import annotations
from typing import Callable
import torch.nn as nn

from affdet.model.mlp import AffordanceMLP
from affdet.model.heads import AttentionHead

HEAD_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "mlp":       lambda d: AffordanceMLP(input_dim=d),
    "mlp-big":   lambda d: AffordanceMLP(input_dim=d, hidden=(1024, 512, 256), dropout=0.3),
    "attention": lambda d: AttentionHead(input_dim=d),
}


def build_head(name: str, input_dim: int = 1544) -> nn.Module:
    if name not in HEAD_REGISTRY:
        raise ValueError(f"unknown head '{name}'; choices: {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[name](input_dim)
