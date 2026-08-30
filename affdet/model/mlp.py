"""AffordanceMLP — shared trunk + 6 independent per-predicate heads.

Input:  (B, 1544)  float32  [f_A(768) | f_B(768) | geo(8)]
Output: (B, 6)     float32  raw logits, one per TIER1_PREDICATE

Trunk: 1544 → 512 → 256 → 128 (BatchNorm + ReLU + Dropout between layers)
Heads: 6 × Linear(128 → 1), independent weights

Note: forward() returns raw logits (no sigmoid). Apply sigmoid at inference
time. BCEWithLogitsLoss is used during training for numerically stable
pos_weight reweighting.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from affdet.intersection_matrix import TIER1_PREDICATES

_N_PRED = len(TIER1_PREDICATES)  # 6


class AffordanceMLP(nn.Module):
    def __init__(self, input_dim: int = 1544,
                 hidden: tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:          # no dropout after the last hidden layer
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(_N_PRED)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)
