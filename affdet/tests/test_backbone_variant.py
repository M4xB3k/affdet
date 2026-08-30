"""Regression test for C1: _build_backbone must thread `variant` into the
torch.hub.load call instead of hardcoding vitb14.

No network access — torch.hub.load is monkeypatched to record the model name
it was called with, and a nn.Identity() stands in for the real backbone.
"""
from __future__ import annotations


def test_build_backbone_uses_variant(monkeypatch):
    import affdet.extract.dinov2 as d
    calls = {}

    def fake_load(repo, name):
        calls["name"] = name
        import torch.nn as nn
        return nn.Identity()

    monkeypatch.setattr(d.torch.hub, "load", fake_load)

    d._build_backbone("vitl14")
    assert calls["name"] == "dinov2_vitl14"

    d._build_backbone("vitb14")
    assert calls["name"] == "dinov2_vitb14"

    d._build_backbone("vits14")
    assert calls["name"] == "dinov2_vits14"


def test_build_backbone_default_variant_is_vitb14(monkeypatch):
    """No args -> default matches DINOv2Extractor's default variant field."""
    import affdet.extract.dinov2 as d
    calls = {}

    def fake_load(repo, name):
        calls["name"] = name
        import torch.nn as nn
        return nn.Identity()

    monkeypatch.setattr(d.torch.hub, "load", fake_load)
    d._build_backbone()
    assert calls["name"] == "dinov2_vitb14"
