"""AffordanceDataset.balance_masks() — true 50/50 per-predicate class balance.

pos_weight reweighting (train.py::_compute_pos_weights) corrects the LOSS but
leaves the DATA skewed: canSupport is ~89% positive in the training split.
balance_masks() instead downsamples the majority class per predicate by zeroing
mask entries, so each predicate sees an actually balanced sample.

Masking (not dropping rows) is what makes this per-PREDICATE: a pair usually
carries several predicates, and dropping the row would distort every other
predicate's balance too. resample() has that flaw — it works per-PAIR, counting
a pair positive if ANY predicate is positive, which barely rebalances anything
when most predicates are majority-positive.
"""
import numpy as np
import torch

from affdet.model.dataset import AffordanceDataset
from affdet.intersection_matrix import TIER1_PREDICATES

_CUT_I = TIER1_PREDICATES.index("canCut")
_CONTAIN_I = TIER1_PREDICATES.index("canContain")
_N_PRED = len(TIER1_PREDICATES)


def _make_dataset(ys: np.ndarray, masks: np.ndarray) -> AffordanceDataset:
    """Build a dataset directly from tensors, bypassing scene/feature IO."""
    ds = AffordanceDataset.__new__(AffordanceDataset)
    n = len(ys)
    ds._xs = torch.zeros(n, 8)
    ds._ys = torch.from_numpy(ys.astype(np.float32))
    ds._masks = torch.from_numpy(masks.astype(np.float32))
    ds._candidate_masks = ds._masks.clone()
    ds._active_idx = list(range(n))
    ds._holdout = None
    return ds


def test_balances_a_majority_positive_predicate_to_5050():
    # canCut: 90 positives, 10 negatives -> should end up 10 vs 10.
    n = 100
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:90, _CUT_I] = 1.0
    masks[:, _CUT_I] = 1.0

    ds = _make_dataset(ys, masks)
    ds.balance_masks(seed=0)

    active = ds._masks[:, _CUT_I] > 0
    pos = ((ds._ys[:, _CUT_I] >= 0.5) & active).sum().item()
    neg = ((ds._ys[:, _CUT_I] < 0.5) & active).sum().item()
    assert pos == neg == 10, f"expected 10/10, got {pos} pos / {neg} neg"


def test_balances_a_majority_negative_predicate_too():
    # canContain: 10 positives, 90 negatives -> 10 vs 10.
    n = 100
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:10, _CONTAIN_I] = 1.0
    masks[:, _CONTAIN_I] = 1.0

    ds = _make_dataset(ys, masks)
    ds.balance_masks(seed=0)

    active = ds._masks[:, _CONTAIN_I] > 0
    pos = ((ds._ys[:, _CONTAIN_I] >= 0.5) & active).sum().item()
    neg = ((ds._ys[:, _CONTAIN_I] < 0.5) & active).sum().item()
    assert pos == neg == 10


def test_each_predicate_balanced_independently():
    """The whole point of masking: balancing canCut must not disturb
    canContain, even though both live on the same rows."""
    n = 100
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:90, _CUT_I] = 1.0        # 90 pos / 10 neg
    ys[:20, _CONTAIN_I] = 1.0    # 20 pos / 80 neg
    masks[:, _CUT_I] = 1.0
    masks[:, _CONTAIN_I] = 1.0

    ds = _make_dataset(ys, masks)
    ds.balance_masks(seed=0)

    for idx, expect in ((_CUT_I, 10), (_CONTAIN_I, 20)):
        active = ds._masks[:, idx] > 0
        pos = ((ds._ys[:, idx] >= 0.5) & active).sum().item()
        neg = ((ds._ys[:, idx] < 0.5) & active).sum().item()
        assert pos == neg == expect, f"predicate {idx}: {pos}/{neg}, want {expect}"


def test_degenerate_predicate_with_no_negatives_is_disabled():
    """canPierce has ZERO negatives corpus-wide. There is no way to balance
    it, so it must be masked out entirely rather than left 100% positive."""
    n = 50
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:, _CUT_I] = 1.0   # every single one positive
    masks[:, _CUT_I] = 1.0

    ds = _make_dataset(ys, masks)
    ds.balance_masks(seed=0)

    assert ds._masks[:, _CUT_I].sum().item() == 0, \
        "a predicate with no negatives cannot be balanced and must be disabled"


def test_is_deterministic_for_a_given_seed():
    n = 100
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:90, _CUT_I] = 1.0
    masks[:, _CUT_I] = 1.0

    a = _make_dataset(ys, masks); a.balance_masks(seed=7)
    b = _make_dataset(ys, masks); b.balance_masks(seed=7)
    c = _make_dataset(ys, masks); c.balance_masks(seed=8)

    assert torch.equal(a._masks, b._masks), "same seed must give same mask"
    assert not torch.equal(a._masks, c._masks), "different seed should differ"


def test_leaves_candidate_masks_untouched():
    """_candidate_masks drives evaluation, so balancing (a TRAIN-time-only
    intervention) must not shrink the eval set."""
    n = 100
    ys = np.zeros((n, _N_PRED))
    masks = np.zeros((n, _N_PRED))
    ys[:90, _CUT_I] = 1.0
    masks[:, _CUT_I] = 1.0

    ds = _make_dataset(ys, masks)
    before = ds._candidate_masks.clone()
    ds.balance_masks(seed=0)

    assert torch.equal(ds._candidate_masks, before)
