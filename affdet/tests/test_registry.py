import pytest
import torch
from affdet.experiment.registry import build_head, HEAD_REGISTRY
from affdet.intersection_matrix import TIER1_PREDICATES


@pytest.mark.parametrize("name", ["mlp", "mlp-big", "attention"])
def test_each_head_builds_and_forwards(name):
    m = build_head(name, 1544).eval()
    assert m(torch.randn(3, 1544)).shape == (3, len(TIER1_PREDICATES))


def test_mlp_big_is_wider_than_mlp():
    from affdet.model.mlp import AffordanceMLP
    big = build_head("mlp-big", 1544)
    first = [mod for mod in big.trunk if isinstance(mod, torch.nn.Linear)][0]
    assert first.out_features == 1024


def test_unknown_head_raises():
    with pytest.raises(ValueError, match="unknown"):
        build_head("resnet", 1544)


def test_registry_keys():
    assert set(HEAD_REGISTRY) == {"mlp", "mlp-big", "attention"}
