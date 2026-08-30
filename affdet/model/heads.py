"""Alternative affordance heads (swappable via the experiment registry).

AttentionHead: instead of concatenating the two object feature vectors and
running an MLP, it treats actor and recipient as two tokens and applies one
multi-head self-attention block over them, then combines with geometry. Same
I/O contract as AffordanceMLP: input (B, 1544), output (B, 6) raw logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from affdet.intersection_matrix import TIER1_PREDICATES

_N_PRED = len(TIER1_PREDICATES)


class AttentionHead(nn.Module):
    def __init__(self, input_dim: int = 1544, feat_dim: int | None = None,
                 n_geo: int = 8, embed: int = 256, heads: int = 4):
        super().__init__()
        # feat_dim depends on the DINOv2 backbone; derive it from input_dim
        # (x = [f_A | f_B | geo] => feat = (input_dim - n_geo) / 2) unless the
        # caller pins it explicitly. Keeps the attention head backbone-agnostic.
        if feat_dim is None:
            feat_dim = (input_dim - n_geo) // 2
        assert input_dim == 2 * feat_dim + n_geo, "input_dim must be 2*feat_dim + n_geo"
        self.feat_dim = feat_dim
        self.n_geo = n_geo
        self.proj = nn.Linear(feat_dim, embed)
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.geo_proj = nn.Sequential(nn.Linear(n_geo, embed), nn.ReLU())
        self.norm = nn.LayerNorm(embed)
        self.heads = nn.ModuleList([nn.Linear(embed * 2, 1) for _ in range(_N_PRED)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fa = x[:, : self.feat_dim]
        fb = x[:, self.feat_dim : 2 * self.feat_dim]
        geo = x[:, 2 * self.feat_dim :]
        tokens = torch.stack([self.proj(fa), self.proj(fb)], dim=1)  # (B,2,embed)
        attended, _ = self.attn(tokens, tokens, tokens)
        pooled = self.norm(attended.mean(dim=1))                    # (B,embed)
        combined = torch.cat([pooled, self.geo_proj(geo)], dim=1)   # (B,2*embed)
        return torch.cat([h(combined) for h in self.heads], dim=1)
