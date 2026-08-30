"""Regression tests: _f1_per_predicate and _compute_pos_weights must binarize
soft VLM-score targets ([0,1]) at 0.5 for their internal computations, even
though the BCE loss itself keeps using the soft targets directly.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from affdet.intersection_matrix import TIER1_PREDICATES
from affdet.model.train import _f1_per_predicate, _compute_pos_weights

_N_PRED = len(TIER1_PREDICATES)


def test_f1_binarizes_soft_targets():
    # Single predicate column ("canCut", index 0) with two valid pairs.
    # Soft targets [0.6, 0.4] binarize to [1, 0].
    # Choose logits so preds (threshold 0.0) = [1, 0] -> perfect match -> F1 == 1.0.
    ci = TIER1_PREDICATES.index("canCut")

    logits = torch.zeros(2, _N_PRED)
    logits[0, ci] = 5.0   # >= 0.0 -> pred 1
    logits[1, ci] = -5.0  # <  0.0 -> pred 0

    targets = torch.zeros(2, _N_PRED)
    targets[0, ci] = 0.6  # binarizes to 1
    targets[1, ci] = 0.4  # binarizes to 0

    masks = torch.zeros(2, _N_PRED)
    masks[0, ci] = 1.0
    masks[1, ci] = 1.0

    f1 = _f1_per_predicate(logits, targets, masks)
    assert abs(f1 - 1.0) < 1e-6, f"expected perfect F1 after binarizing, got {f1}"


def test_pos_weights_count_binarized():
    @dataclass
    class _FakeDataset:
        _xs: torch.Tensor
        _ys: torch.Tensor
        _masks: torch.Tensor

    ci = TIER1_PREDICATES.index("canCut")

    # 2 pairs, both valid candidates for canCut only.
    # Soft scores 0.8 and 0.2 binarize to 1 pos / 1 neg -> n_pos=1, n_neg=1 -> weight 1.0.
    # The old (buggy) sum-based code would compute n_pos = 0.8+0.2 = 1.0, n_neg = 1.0,
    # which happens to coincide here, so use asymmetric soft scores that would NOT
    # coincide under the old sum-based computation to make the assertion discriminating.
    ys = torch.zeros(2, _N_PRED)
    ys[0, ci] = 0.9  # binarizes to 1 (pos)
    ys[1, ci] = 0.3  # binarizes to 0 (neg)

    masks = torch.zeros(2, _N_PRED)
    masks[0, ci] = 1.0
    masks[1, ci] = 1.0

    xs = torch.zeros(2, 1544)

    ds = _FakeDataset(_xs=xs, _ys=ys, _masks=masks)
    weights = _compute_pos_weights(ds)

    # Binary counts: n_pos=1, n_neg=1 -> weight = min(1/1, cap) = 1.0
    # Old buggy sum-based code: n_pos = 0.9+0.3 = 1.2, n_neg = 2-1.2 = 0.8
    # -> weight = min(0.8/1.2, cap) = 0.6667, which differs from 1.0.
    assert abs(weights[ci].item() - 1.0) < 1e-6, (
        f"expected binarized weight 1.0, got {weights[ci].item()}"
    )
