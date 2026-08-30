import torch
from affdet.model.mlp import AffordanceMLP
from affdet.intersection_matrix import TIER1_PREDICATES


def test_default_trunk_shapes_unchanged():
    m = AffordanceMLP()
    lin = [mod for mod in m.trunk if isinstance(mod, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lin] == [(1544,512),(512,256),(256,128)]
    assert len(m.heads) == len(TIER1_PREDICATES)
    assert m.heads[0].in_features == 128


def test_forward_default_shape():
    m = AffordanceMLP().eval()
    out = m(torch.randn(4, 1544))
    assert out.shape == (4, len(TIER1_PREDICATES))


def test_custom_hidden_builds_and_forwards():
    m = AffordanceMLP(hidden=(1024, 512, 256)).eval()
    lin = [mod for mod in m.trunk if isinstance(mod, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lin] == [(1544,1024),(1024,512),(512,256)]
    assert m.heads[0].in_features == 256
    assert m(torch.randn(2, 1544)).shape == (2, len(TIER1_PREDICATES))
