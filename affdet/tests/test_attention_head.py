import torch
from affdet.model.heads import AttentionHead
from affdet.intersection_matrix import TIER1_PREDICATES


def test_forward_shape():
    h = AttentionHead().eval()
    out = h(torch.randn(5, 1544))
    assert out.shape == (5, len(TIER1_PREDICATES))


def test_returns_raw_logits_not_bounded():
    # raw logits: no sigmoid inside — values can exceed [0,1]
    h = AttentionHead().eval()
    with torch.no_grad():
        out = h(torch.randn(64, 1544) * 5)
    assert out.abs().max() > 1.0 or out.min() < 0.0


def test_batch_size_one_ok():
    h = AttentionHead().eval()
    assert h(torch.randn(1, 1544)).shape == (1, len(TIER1_PREDICATES))
